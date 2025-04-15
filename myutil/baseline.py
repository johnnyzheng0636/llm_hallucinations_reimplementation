# self check gpt baseline
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
from tqdm import tqdm
from datasets import load_dataset
import functools
import pickle
from pathlib import Path
from string import Template
import traceback
from collections import defaultdict

from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore, SelfCheckNgram
from sklearn.metrics import roc_auc_score
import statistics
import spacy

import torch
import numpy as np
import pandas as pd

class SelfCheckGpt():
    def __init__(
            self, 
            model_name="open_llama_7b", 
            dataset_name="capitals", 
            data_dir="./data/",
            model_dir="./.cache/models/",
            # results_dir="./results/",
            out_dir="./outouts/",
            trex_data_to_question_template = None,
            start=0, 
            end=2500, 
            layer_number=-1
        ):
        
        self.selfcheck_bertscore = SelfCheckBERTScore(rescale_with_baseline=True)
        self.selfcheck_ngram = SelfCheckNgram(n=1) # n=1 means Unigram, n=2 means Bigram, etc.
        self.self_checkgpt_temperature = 1.0
        self.selfcheckgpt_n_trials = 20
        
        self.start = start
        self.end = end

        self.dataset_name =  dataset_name
        if trex_data_to_question_template is None:
            self.trex_data_to_question_template = {  
                "capitals": Template("What is the capital of $source?"),
                "place_of_birth": Template("Where was $source born?"),
                "founders": Template("Who founded $source?"),
            }
        else:
            self.trex_data_to_question_template = trex_data_to_question_template

        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

        # Model
        self.model_name = model_name #"falcon-7b" #"opt-30b"
        self.layer_number = layer_number
        # hardcode below,for now. Could dig into all models but they take a while to load
        model_num_layers = {
            "falcon-40b" : 60,
            "falcon-7b" : 32,
            "open_llama_13b" : 40,
            "open_llama_7b" : 32,
            "opt-6.7b" : 32,
            "opt-30b" : 48,
        }
        assert self.layer_number < model_num_layers[self.model_name]
        self.coll_str = "[0-9]+" if self.layer_number==-1 else str(self.layer_number)
        # find name for transformer llm download
        self.model_repos = {
            "falcon-40b" : ("tiiuae", f".*transformer.h.{self.coll_str}.mlp.dense_4h_to_h", f".*transformer.h.{self.coll_str}.self_attention.dense"),
            "falcon-7b" : ("tiiuae", f".*transformer.h.{self.coll_str}.mlp.dense_4h_to_h", f".*transformer.h.{self.coll_str}.self_attention.dense"),
            "open_llama_13b" : ("openlm-research", f".*model.layers.{self.coll_str}.mlp.up_proj", f".*model.layers.{self.coll_str}.self_attn.o_proj"),
            "open_llama_7b" : ("openlm-research", f".*model.layers.{self.coll_str}.mlp.up_proj", f".*model.layers.{self.coll_str}.self_attn.o_proj"),
            "opt-6.7b" : ("facebook", f".*model.decoder.layers.{self.coll_str}.fc2", f".*model.decoder.layers.{self.coll_str}.self_attn.out_proj"),
            "opt-30b" : ("facebook", f".*model.decoder.layers.{self.coll_str}.fc2", f".*model.decoder.layers.{self.coll_str}.self_attn.out_proj", ),
        }

        # IO
        self.data_dir = Path(data_dir) # Where our data files are stored
        self.model_dir = Path(model_dir) # Cache for huggingface models
        self.outPath = Path(out_dir) / f"{self.model_name}_basline_{self.dataset_name}_{self.start}-{self.end}" # Directory for storing results
        # if IO dir not exist, create it
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.model_dir.mkdir(parents=True, exist_ok=True)
            self.outPath.mkdir(parents=True, exist_ok=True)
        except  Exception as err:
            print("The name may have been used already, try anoter name.")
            print(traceback.format_exc())
        # For storing results
        self.fully_connected_hidden_layers = defaultdict(list)
        self.attention_hidden_layers = defaultdict(list)
        self.attention_forward_handles = {}
        self.fully_connected_forward_handles = {}

        self.dataset = self.load_data(self.dataset_name)

        model_loader = LlamaForCausalLM if "llama" in self.model_name else AutoModelForCausalLM
        token_loader = LlamaTokenizer if "llama" in self.model_name else AutoTokenizer
        self.tokenizer = token_loader.from_pretrained(f'{self.model_repos[self.model_name][0]}/{self.model_name}')
        self.model = model_loader.from_pretrained(f'{self.model_repos[self.model_name][0]}/{self.model_name}',
                                            cache_dir=self.model_dir,
                                            device_map=self.device,
                                            torch_dtype=torch.bfloat16,
                                            # load_in_4bit=True,
                                            trust_remote_code=True)

    def get_stop_token(self):
        if "llama" in self.model_name:
            stop_token = 13
        elif "falcon" in self.model_name:
            stop_token = 193
        else:
            stop_token = 50118
        return stop_token
    

    def get_next_token(self, x, model):
        with torch.no_grad():
            return model(x).logits
        

    def generate_response(self, x, model, *, max_length=100, pbar=False):
        response = []
        bar = tqdm(range(max_length)) if pbar else range(max_length)
        for step in bar:
            logits = self.get_next_token(x, model)
            next_token = logits.squeeze()[-1].argmax()
            x = torch.concat([x, next_token.view(1, -1)], dim=1)
            response.append(next_token)
            if next_token == self.get_stop_token() and step>5:
                break
        return torch.stack(response).cpu().numpy(), logits.squeeze()


    def answer_question(self, question, model, tokenizer, *, max_length=100, pbar=False):
        input_ids = tokenizer(question, return_tensors='pt').input_ids.to(model.device)
        response, logits = self.generate_response(input_ids, model, max_length=max_length, pbar=pbar)
        return response, logits, input_ids.shape[-1]


    def answer_trivia(self, question, targets, model, tokenizer):
        response, logits, start_pos = self.answer_question(question, model, tokenizer)
        str_response = tokenizer.decode(response, skip_special_tokens=True)
        correct = False
        for alias in targets:
            if alias.lower() in str_response.lower():
                correct = True
                break
        return response, str_response, logits, start_pos, correct


    def answer_trex(self, source, targets, model, tokenizer, question_template):
        response, logits, start_pos = self.answer_question(question_template.substitute(source=source), model, tokenizer)
        str_response = tokenizer.decode(response, skip_special_tokens=True)
        correct = any([target.lower() in str_response.lower() for target in targets])
        return response, str_response, logits, start_pos, correct


    def load_data(self, dataset_name):
        if dataset_name in self.trex_data_to_question_template.keys():
            pd_frame = pd.read_csv(self.data_dir / f'{dataset_name}.csv')
            dataset = [(pd_frame.iloc[i]['subject'], pd_frame.iloc[i]['object'].split("<OR>")) for i in range(self.start, min(self.end, len(pd_frame)))]
        elif dataset_name=="trivia_qa":
            trivia_qa = load_dataset('trivia_qa', data_dir='rc.nocontext', cache_dir=str(self.data_dir))
            full_dataset = []
            for obs in tqdm(trivia_qa['train']):
                aliases = []
                aliases.extend(obs['answer']['aliases'])
                aliases.extend(obs['answer']['normalized_aliases'])
                aliases.append(obs['answer']['value'])
                aliases.append(obs['answer']['normalized_value'])
                full_dataset.append((obs['question'], aliases))
            dataset = full_dataset[self.start: self.end]
        else:
            raise ValueError(f"Unknown dataset {dataset_name}.")
        return dataset
    

    def generate_responses(self, question, str_response, tokenizer):

        # generate several responses to the question and (self)check them against the zero temp response
        inputs = tokenizer(question, return_tensors="pt").input_ids.to(self.device)
        start_pos = inputs.size(dim=-1)

        hitemp_str_responses = []
        for i in range(0, self.selfcheckgpt_n_trials):
            current_hitemp = ''
            # may generate meaningless empty string
            # keep generate until a non-empty string is generated
            while current_hitemp == '':
                model_outputs = self.model.generate(
                    inputs, 
                    do_sample=True, 
                    temperature=self.self_checkgpt_temperature, 
                    max_new_tokens=100, 
                    return_dict_in_generate=True, 
                    output_scores=True,
                )
                generated_tokens_ids = model_outputs.sequences[0]
                current_hitemp = tokenizer.decode(generated_tokens_ids[start_pos:]).replace(
                    "\n", " "
                ).replace("</s>", " ").strip()
            hitemp_str_responses.append(current_hitemp)

        # for i in range(len(hitemp_str_responses)):
        #     print(i, ' th sample: ', hitemp_str_responses[i])

        selfcheck_scores_bert_overall = []
        selfcheck_scores_bert_average = []
        selfcheck_ngram_overall = []
        
        sentences = [str_response]
        try:
            overall_bertscore = self.selfcheck_bertscore.predict(
                sentences = sentences,                          # list of sentences
                sampled_passages = hitemp_str_responses, # list of sampled passages
            )
            # print(f"overall_bertscore: {overall_bertscore}")
        except Exception as e:
            print('error at selfcheck_scores_bert_overall')
            print(f"str_response: {sentences}")
            print(f"hitemp_str_responses: {hitemp_str_responses}")
            print(traceback.format_exc())
            overall_bertscore = [-1]
            
        selfcheck_scores_bert_overall.append(overall_bertscore[0])
        
        nlp = spacy.load("en_core_web_sm")
        sentences = [sent for sent in nlp(str_response).sents]
        sentences = [sent.text.strip() for sent in sentences if len(sent) > 3]
        try:
            all_bertscores = self.selfcheck_bertscore.predict(
                sentences = sentences,                          # list of sentences
                sampled_passages = hitemp_str_responses, # list of sampled passages
            )
        except Exception as e:
            print('error at selfcheck_scores_bert_average')
            print(f"sentences: {sentences}")
            print(f"hitemp_str_responses: {hitemp_str_responses}")
            print(traceback.format_exc())
            all_bertscores = [-1]
        average_bertscore = statistics.mean(all_bertscores)
        selfcheck_scores_bert_average.append(average_bertscore)
        
        
        sent_scores_ngram = self.selfcheck_ngram.predict(
            sentences = sentences,   
            passage = str_response,
            sampled_passages = hitemp_str_responses,
        )
        selfcheck_ngram_overall.append(sent_scores_ngram)
        
        return hitemp_str_responses, selfcheck_scores_bert_overall, selfcheck_scores_bert_average, selfcheck_ngram_overall


    def run(self,):
        selfcheck_dict = {
            'question': [],
            'response': [],
            'str_response': [],
            'start_pos': [],
            'correct': [],
            'hitemp_str_responses': [],
            'selfcheck_scores_bert_overall': [],
            'selfcheck_scores_bert_average': [],
            'selfcheck_ngram_overall': []
        }

        selfcheck_arr_overall = []
        selfcheck_arr_average = []
        selfcheck_ngram_average = []
        correct_arr = []

        if self.dataset_name in self.trex_data_to_question_template.keys():
            question_asker = functools.partial(self.answer_trex, question_template=self.trex_data_to_question_template[self.dataset_name])
        elif self.dataset_name == "trivia_qa":
            question_asker = self.answer_trivia
        else:
            raise ValueError(f"Unknown dataset name {self.dataset_name}.")

        error_count = 0
        for idx in tqdm(range(len(self.dataset))):

            question, answers = self.dataset[idx]
            response, str_response, logits, start_pos, correct = question_asker(question, answers, self.model, self.tokenizer)
            hitemp_str_responses, selfcheck_scores_bert_overall, selfcheck_scores_bert_average, selfcheck_ngram_overall\
                = self.generate_responses(
                    question if self.dataset_name=="trivia_qa" else self.trex_data_to_question_template[self.dataset_name].substitute(source=question),
                    str_response, 
                    self.tokenizer
                )
            # print('selfcheck_scores_bert_overall', selfcheck_scores_bert_overall)
            # print('selfcheck_scores_bert_average', selfcheck_scores_bert_average)
            if selfcheck_scores_bert_overall[0] == -1 or selfcheck_scores_bert_overall[0] == -1:
                # print(f"selfcheckgpt bugs out, skipping this example")
                error_count += 1
                continue

            selfcheck_dict['question'].append(question)
            selfcheck_dict['response'].append(response)
            selfcheck_dict['str_response'].append(str_response)
            selfcheck_dict['start_pos'].append(start_pos)
            selfcheck_dict['correct'].append(correct)
            selfcheck_dict['hitemp_str_responses'].append(hitemp_str_responses)
            selfcheck_dict['selfcheck_scores_bert_overall'].append(selfcheck_scores_bert_overall)
            selfcheck_dict['selfcheck_scores_bert_average'].append(selfcheck_scores_bert_average)
            selfcheck_dict['selfcheck_ngram_overall'].append(selfcheck_ngram_overall)

            selfcheck_arr_overall.append(1.0-selfcheck_scores_bert_overall[0]) #bert score flipped
            selfcheck_arr_average.append(1.0-selfcheck_scores_bert_average[0]) #bert score flipped
            selfcheck_ngram_average.append(1.0-np.exp(-selfcheck_ngram_overall[0]['doc_level']['avg_neg_logprob']))
            correct_arr.append(int(correct))
            
        #print(selfcheck_arr_overall)
        #print(correct_arr)
        baseline_results = {}
        roc_score = roc_auc_score(correct_arr, selfcheck_arr_overall)
        print(f"AUROC for self check overall: {roc_score}")
        baseline_results['selfcheck_overall'] = roc_score

        #print(selfcheck_arr_average)
        #print(correct_arr)
        roc_score = roc_auc_score(correct_arr, selfcheck_arr_average)
        print(f"AUROC for self check average: {roc_score}")
        baseline_results['selfcheck_average'] = roc_score

        roc_score = roc_auc_score(correct_arr, selfcheck_ngram_average)
        print(f"AUROC for self check ngram: {roc_score}")
        baseline_results['selfcheck_ngram'] = roc_score

        df = pd.DataFrame.from_dict(baseline_results, orient="index", columns=[self.model_name])
        csv_path = self.outPath / 'eval.csv'
        df.to_csv(str(csv_path))

        tmp_path = Path(self.outPath) / f"selfcheckgpt_{self.model_name}_{self.dataset_name}.pickle"
        with open(str(tmp_path), "wb") as outfile:
                outfile.write(pickle.dumps(selfcheck_dict))

        print(selfcheck_dict['hitemp_str_responses'][0])
        print("Total error: ", error_count)