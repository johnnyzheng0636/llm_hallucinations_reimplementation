import functools
from datetime import datetime
from typing import Any, Dict
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, LlamaTokenizer, LlamaForCausalLM
from tqdm import tqdm
from datasets import load_dataset
from collections import defaultdict, Counter
from functools import partial
import re
from captum.attr import IntegratedGradients
from string import Template
import copy
import traceback


class hookLayer():
    def __init__(
            self, 
            model_name="open_llama_7b", 
            dataset_name="capitals", 
            data_dir="./data/",
            model_dir="./.cache/models/",
            results_dir="./results/",
            trex_data_to_question_template = None,
            start=0, 
            end=2500, 
            chunk_sz=50, 
            ig_steps=64,
            internal_batch_size=4,
            layer_number=-1,
            demo=False,
            train_exist=False,
        ):
        # if demo is True, print intermediate results
        self.train_exist = train_exist
        self.demo = demo
        print('starting')
        # Data related params
        # self.iteration = iteration
        # self.interval = interval # We run the inference on these many examples at a time to achieve parallelization
        self.chunk_sz = chunk_sz # chunk in pickle
        # test with smaller size
        # interval = 10
        # chunk_sz = 4
        # self.start = self.iteration * self.interval
        # self.end = self.start + self.interval
        
        self.start = start
        self.end = end

        # dataset_name =  "trivia_qa"
        # dataset_name = "place_of_birth"
        # dataset_name = "founders"
        self.dataset_name =  dataset_name
        if trex_data_to_question_template is None:
            self.trex_data_to_question_template = {  
                "capitals": Template("What is the capital of $source?"),
                "place_of_birth": Template("Where was $source born?"),
                "founders": Template("Who founded $source?"),
            }
        else:
            self.trex_data_to_question_template = trex_data_to_question_template

        # Hardware
        # gpu = "0"
        # device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

        # Integrated Grads
        self.ig_steps = ig_steps
        self.internal_batch_size = internal_batch_size

        # Model
        self.model_name = model_name #"falcon-7b" #"opt-30b"
        self.layer_number = layer_number
        # hardcode below,for now. Could dig into all models but they take a while to load
        model_num_layers = {
            "falcon-40b" : 60,
            "falcon-7b" : 32,
            "open_llama_13b" : 40,
            "open_llama_7b" : 32,
            "Llama-3.1-8B": 32,
            "Llama-3.2-3B": 28,
            "Llama-3.2-1B": 16,
            "opt-6.7b" : 32,
            "opt-30b" : 48,
        }
        assert self.layer_number < model_num_layers[self.model_name]
        self.coll_str = "[0-9]+" if self.layer_number==-1 else str(self.layer_number)
        self.model_repos = {
            "falcon-40b" : ("tiiuae", f".*transformer.h.{self.coll_str}.mlp.dense_4h_to_h", f".*transformer.h.{self.coll_str}.self_attention.dense"),
            "falcon-7b" : ("tiiuae", f".*transformer.h.{self.coll_str}.mlp.dense_4h_to_h", f".*transformer.h.{self.coll_str}.self_attention.dense"),
            "open_llama_13b" : ("openlm-research", f".*model.layers.{self.coll_str}.mlp.up_proj", f".*model.layers.{self.coll_str}.self_attn.o_proj"),
            "open_llama_7b" : ("openlm-research", f".*model.layers.{self.coll_str}.mlp.up_proj", f".*model.layers.{self.coll_str}.self_attn.o_proj"),
            "Llama-3.1-8B" : ("meta-llama", f".*model.layers.{self.coll_str}.mlp.up_proj", f".*model.layers.{self.coll_str}.self_attn.o_proj"),
            "Llama-3.2-3B" : ("meta-llama", f".*model.layers.{self.coll_str}.mlp.up_proj", f".*model.layers.{self.coll_str}.self_attn.o_proj"),
            "Llama-3.2-1B" : ("meta-llama", f".*model.layers.{self.coll_str}.mlp.up_proj", f".*model.layers.{self.coll_str}.self_attn.o_proj"),
            "opt-6.7b" : ("facebook", f".*model.decoder.layers.{self.coll_str}.fc2", f".*model.decoder.layers.{self.coll_str}.self_attn.out_proj"),
            "opt-30b" : ("facebook", f".*model.decoder.layers.{self.coll_str}.fc2", f".*model.decoder.layers.{self.coll_str}.self_attn.out_proj", ),
        }

        # IO
        self.data_dir = Path(data_dir) # Where our data files are stored
        self.model_dir = Path(model_dir) # Cache for huggingface models
        self.results_dir = Path(results_dir) / f"{self.model_name}_{self.chunk_sz}chunk_{self.dataset_name}_{self.start}-{self.end}" # Directory for storing results

        # if IO dir not exist, create it
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.model_dir.mkdir(parents=True, exist_ok=True)
            self.results_dir.mkdir(parents=True, exist_ok=True)
        except  Exception as err:
            print("The name may have been used already, try anoter name.")
            print(traceback.format_exc())
        # For storing results
        self.fully_connected_hidden_layers = defaultdict(list)
        self.attention_hidden_layers = defaultdict(list)
        self.attention_forward_handles = {}
        self.fully_connected_forward_handles = {}

    def demo_mate_data(self):
        # print(self.demo)
        if self.demo:
            print(self.results_dir)
            return self.results_dir


    def save_fully_connected_hidden(self, layer_name, mod, inp, out):
        self.fully_connected_hidden_layers[layer_name].append(out.squeeze().detach().to(torch.float32).cpu().numpy())


    def save_attention_hidden(self, layer_name, mod, inp, out):
        self.attention_hidden_layers[layer_name].append(out.squeeze().detach().to(torch.float32).cpu().numpy())


    def get_stop_token(self):
        if "llama" in self.model_name:
            stop_token = 13
        elif "Llama-3.2" in self.model_name:
            stop_token = 128001
        elif "Llama" in self.model_name:
            stop_token = 128009
        elif "falcon" in self.model_name:
            stop_token = 193
        else:
            stop_token = 50118
        return stop_token


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


    def get_start_end_layer(self, model):
        if "llama" in self.model_name.lower():
            layer_count = model.model.layers
        elif "falcon" in self.model_name:
            layer_count = model.transformer.h
        else:
            layer_count = model.model.decoder.layers
        layer_st = 0 if self.layer_number == -1 else self.layer_number
        layer_en = len(layer_count) if self.layer_number == -1 else self.layer_number + 1
        return layer_st, layer_en


    def collect_fully_connected(self, token_pos, layer_start, layer_end):
        layer_name = self.model_repos[self.model_name][1][2:].split(self.coll_str)
        first_activation = np.stack([self.fully_connected_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][-1][token_pos] \
                                    for i in range(layer_start, layer_end)])
        final_activation = np.stack([self.fully_connected_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][-1][-1] \
                                    for i in range(layer_start, layer_end)])
        return first_activation, final_activation


    def collect_attention(self, token_pos, layer_start, layer_end):
        layer_name = self.model_repos[self.model_name][2][2:].split(self.coll_str)
        first_activation = np.stack([self.attention_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][-1][token_pos] \
                                    for i in range(layer_start, layer_end)])
        final_activation = np.stack([self.attention_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][-1][-1] \
                                    for i in range(layer_start, layer_end)])
        return first_activation, final_activation


    def normalize_attributes(self, attributes: torch.Tensor) -> torch.Tensor:
            # attributes has shape (batch, sequence size, embedding dim)
            attributes = attributes.squeeze(0)

            # if aggregation == "L2":  # norm calculates a scalar value (L2 Norm)
            norm = torch.norm(attributes, dim=1)
            attributes = norm / torch.sum(norm)  # Normalize the values so they add up to 1
            
            return attributes


    def model_forward(self, input_: torch.Tensor, model, extra_forward_args: Dict[str, Any]) \
                -> torch.Tensor:
            output = model(inputs_embeds=input_, **extra_forward_args)
            return torch.nn.functional.softmax(output.logits[:, -1, :], dim=-1)


    def get_embedder(self, model):
        if "falcon" in self.model_name:
            return model.transformer.word_embeddings
        elif "opt" in self.model_name:
            return model.model.decoder.embed_tokens
        elif "llama" in self.model_name.lower():
            return model.model.embed_tokens
        else:
            raise ValueError(f"Unknown model {self.model_name}")

    def get_ig(self, prompt, forward_func, tokenizer, embedder, model):
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
        prediction_id = self.get_next_token(input_ids, model).squeeze()[-1].argmax()
        encoder_input_embeds = embedder(input_ids).detach() # fix this for each model
        ig = IntegratedGradients(forward_func=forward_func)
        attributes = self.normalize_attributes(
            ig.attribute(
                encoder_input_embeds,
                target=prediction_id,
                n_steps=self.ig_steps,
                internal_batch_size=self.internal_batch_size
            )
        ).detach().cpu().numpy()
        return attributes


    def save_results(self):
        # Dataset
        dataset = self.load_data(self.dataset_name)
        if self.dataset_name in self.trex_data_to_question_template.keys():
            question_asker = functools.partial(self.answer_trex, question_template=self.trex_data_to_question_template[self.dataset_name])
        elif self.dataset_name == "trivia_qa":
            question_asker = self.answer_trivia
        else:
            raise ValueError(f"Unknown dataset name {self.dataset_name}.")

        # Model
        model_loader = LlamaForCausalLM if "llama" in self.model_name else AutoModelForCausalLM
        token_loader = LlamaTokenizer if "llama" in self.model_name else AutoTokenizer
        tokenizer = token_loader.from_pretrained(f'{self.model_repos[self.model_name][0]}/{self.model_name}')
        model = model_loader.from_pretrained(f'{self.model_repos[self.model_name][0]}/{self.model_name}',
                                            cache_dir=self.model_dir,
                                            device_map=self.device,
                                            torch_dtype=torch.bfloat16,
                                            # load_in_4bit=True,
                                            trust_remote_code=True)
        # edit the layers parameter based on this
        # self.model_repos and model_num_layers
        print(f"{self.model_repos[self.model_name][0]}/{self.model_name}")
        print(model)

        # restart from failures if exist
        existmax = 0
        if not self.train_exist:
            existData = list(Path(f"./{self.results_dir}").glob(f"{self.model_name}_{self.dataset_name}_start-{self.start}_end-{self.end}*.pickle"))
            for e in existData:
                tmpmax = int(str(e).split('_')[-3].split('-')[-1])
                if tmpmax > existmax:
                    existmax = tmpmax + 1
            # skip if done
            if existmax >= self.end:
                print(f"Already completed {self.model_name} on {self.dataset_name} from {self.start} to {self.end}. Process to classification.")
                return self.results_dir
            # get a subset from exsit data

        forward_func = partial(self.model_forward, model=model, extra_forward_args={})
        embedder = self.get_embedder(model)

        # Prepare to save the internal states
        for name, module in model.named_modules():
            if re.match(f'{self.model_repos[self.model_name][1]}$', name):
                self.fully_connected_forward_handles[name] = module.register_forward_hook(
                    partial(self.save_fully_connected_hidden, name))
            if re.match(f'{self.model_repos[self.model_name][2]}$', name):
                self.attention_forward_handles[name] = module.register_forward_hook(
                    partial(self.save_attention_hidden, name))

        # Generate results
        results = defaultdict(list)
        for idx in tqdm(range(existmax, len(dataset))):
            self.fully_connected_hidden_layers.clear()
            self.attention_hidden_layers.clear()

            question, answers = dataset[idx]
            # print(question)
            # return
            response, str_response, logits, start_pos, correct = question_asker(question, answers, model, tokenizer)
            layer_start, layer_end = self.get_start_end_layer(model)
            first_fully_connected, final_fully_connected = self.collect_fully_connected(start_pos, layer_start, layer_end)
            first_attention, final_attention = self.collect_attention(start_pos, layer_start, layer_end)

            if self.dataset_name in self.trex_data_to_question_template.keys():
                full_question = self.trex_data_to_question_template[self.dataset_name].substitute(
                    source=question)
            elif self.dataset_name == "trivia_qa":
                full_question = question
            else:
                raise ValueError(f"Unknown dataset name {self.dataset_name}.")
            attributes_first = self.get_ig(full_question, forward_func, tokenizer, embedder, model)

            results['question'].append(question)
            results['answers'].append(answers)
            results['response'].append(response)
            results['str_response'].append(str_response)
            results['logits'].append(logits.to(torch.float32).cpu().numpy())
            results['start_pos'].append(start_pos)
            results['correct'].append(correct)
            results['first_fully_connected'].append(first_fully_connected)
            results['final_fully_connected'].append(final_fully_connected)
            results['first_attention'].append(first_attention)
            results['final_attention'].append(final_attention)
            results['attributes_first'].append(attributes_first)
            # debug
            # break

            # chunking the pickle since one pickle is too large and give error
            if (idx+1) % self.chunk_sz == 0:
                with open(self.results_dir/f"{self.model_name}_{self.dataset_name}_start-{self.start}_end-{self.end}_{idx+1-self.chunk_sz}-{idx}_{datetime.now().month}_{datetime.now().day}.pickle", "wb") as outfile:
                    outfile.write(pickle.dumps(results))
                results.clear()

        # ending chunk check
        if (len(dataset)) % self.chunk_sz != 0:
            with open(self.results_dir/f"{self.model_name}_{self.dataset_name}_start-{self.start}_end-{self.end}_{len(dataset)-len(dataset)%self.chunk_sz}-{len(dataset)-1}_{datetime.now().month}_{datetime.now().day}.pickle", "wb") as outfile:
                outfile.write(pickle.dumps(results))

        return self.results_dir
