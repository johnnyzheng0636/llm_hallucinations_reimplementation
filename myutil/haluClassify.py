import pickle
from pathlib import Path
import numpy as np
import scipy as sp
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import random

from tqdm import tqdm
import traceback
from string import Template
import matplotlib.pyplot as plt
import time

# For custom classifier, notice the input and output dimension should be identical to the default model
# For attention, one may consider pad input to a constnat length first or use linear to scale down dimension.

class defaultMLP(torch.nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        
        self.linear_relu_stack =torch.nn.Sequential(
            torch.nn.Linear(input_shape, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(256, 2)
            )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
class defaultGRU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = 128
        num_layers = 4
        self.gru = torch.nn.GRU(1, hidden_dim, num_layers, dropout=0.25, batch_first=True, bidirectional=False)
        self.linear = torch.nn.Linear(hidden_dim, 2)
    
    def forward(self, seq):
        gru_out, _ = self.gru(seq)
        return self.linear(gru_out)[-1, -1, :]

class haluClassify():
    def __init__(
            self, 
            layerDataPath, # path has data for training
            out_dir = "./outouts/", # path storing output(eval and model) 
            cache_model_dir="./.cache/models/",
            cls_IG=defaultGRU, 
            cls_logit=defaultMLP, 
            cls_att=defaultMLP, 
            cls_linear=defaultMLP,
            lr=1e-4,
            weight_decay=1e-2,
            batch_size = 128, # should be no larger than total number of sample
            epochs = 1000,
            train_exist = False,
            demo=False,
            chunk_sz=50, # only used for demo
            dataset = "capitals", # only used for demo
            seed = 42,
            model_statistic=False,
            llm_name="open_llama_7b",
            k=10,
        ):
        self.k = k
        self.llm_name = llm_name
        self.model_statistic = model_statistic # only get mode statistic if true(task performance and heuristic accuracy)
        self.dataset = dataset
        self.seed = seed
        self.chunk_sz = chunk_sz
        self.demo = demo
        self.train_exist = train_exist # true => train classifier regardless if eval.csv exists
        self.layerDataFiles = list(Path(layerDataPath).glob("*.pickle"))
        self.cls_IG = cls_IG
        self.cls_logit = cls_logit
        self.cls_att = cls_att
        self.cls_linear = cls_linear
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs

        self.cache_model_dir = Path(cache_model_dir) # Cache for huggingface models


        self.trex_data_to_question_template = {  
            "capitals": Template("What is the capital of $source?"),
            "place_of_birth": Template("Where was $source born?"),
            "founders": Template("Who founded $source?"),
        }

        self.model_repos = {
            "falcon-40b" : ("tiiuae"),
            "falcon-7b" : ("tiiuae"),
            "open_llama_13b" : ("openlm-research"),
            "open_llama_7b" : ("openlm-research"),
            "Llama-3.1-8B" : ("meta-llama"),
            "Llama-3.2-3B" : ("meta-llama"),
            "Llama-3.2-1B" : ("meta-llama"),
            "opt-6.7b" : ("facebook"),
            "opt-30b" : ("facebook"),
        }


        # create output directory
        self.model_name = str(layerDataPath).split('/')[-1]
        self.outPath = Path(out_dir) / self.model_name
        
        # print(self.outPath)

        self.output_eval_dir = self.outPath / 'eval'
        self.output_model_dir = self.outPath / 'model_stat'
        self.output_demo_dir = self.outPath / 'demo'

        # print(self.outPath)
        # print(self.output_eval_dir)
        # print(self.output_model_dir)

        try:
            self.output_eval_dir.mkdir(parents=True, exist_ok=True)
            self.output_model_dir.mkdir(parents=True, exist_ok=True)
            self.output_demo_dir.mkdir(parents=True, exist_ok=True)
        except  Exception as err:
            print("The name may have been used already, try anoter name.")
            print(traceback.format_exc())

        self.output_eval = self.output_eval_dir / f'eval.csv'
        self.output_model_ig = self.output_model_dir / f'{self.cls_IG.__name__}_ig_model.pth'
        self.output_model_logit = self.output_model_dir / f'{self.cls_logit.__name__}_logit_model.pth'
        self.output_model_att = self.output_model_dir / f'{self.cls_att.__name__}_att_model.pth'
        self.output_model_linear = self.output_model_dir / f'{self.cls_linear.__name__}_linear_model.pth'

        random.seed(self.seed)
        
        # print('output path')
        # print(layerDataPath)
        # print(self.output_eval_dir)
        # print(self.output_model_dir)
        # print(self.output_eval)
        # print(self.output_model_ig)
        # print(self.output_model_logit)
        # print(self.output_model_att)
        # print(self.output_model_linear)

    def get_embedder(self, model):
        if "falcon" in self.model_name:
            return model.transformer.word_embeddings
        elif "opt" in self.model_name:
            return model.model.decoder.embed_tokens
        elif "llama" in self.model_name.lower():
            return model.model.embed_tokens
        else:
            raise ValueError(f"Unknown model {self.model_name}")

    # def gen_classifier_roc(self, X, y, cls, lr, weight_decay, batch_size):
    def gen_classifier_roc(self, X, label, cls, save_path):
        X_train, X_test, y_train, y_test = train_test_split(X, label.astype(int), test_size = 0.2, random_state=42)
        # print(X_train, X_test, y_train, y_test)
        classifier_model = cls(X_train.shape[1]).to(self.device)
        X_train = torch.tensor(X_train).to(self.device)
        y_train = torch.tensor(y_train).to(torch.long).to(self.device)
        X_test = torch.tensor(X_test).to(self.device)
        y_test = torch.tensor(y_test).to(torch.long).to(self.device)

        optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for _ in range(self.epochs):
            optimizer.zero_grad()
            sample = torch.randperm(X_train.shape[0])[:self.batch_size]
            pred = classifier_model(X_train[sample])
            loss = torch.nn.functional.cross_entropy(pred, y_train[sample])
            loss.backward()
            optimizer.step()
        classifier_model.eval()
        with torch.no_grad():
            pred = torch.nn.functional.softmax(classifier_model(X_test), dim=1)
            prediction_classes = (pred[:,1]>0.5).type(torch.long).cpu()
            # TODO
            # save the model
            torch.save(classifier_model.state_dict(), save_path)
            return roc_auc_score(y_test.cpu(), pred[:,1].cpu()), (prediction_classes.numpy()==y_test.cpu().numpy()).mean()

    def demo_classifier(self, X, cls, cls_type):
        # print(X.shape)
        if cls_type.lower() == 'attention':
            load_path = self.output_model_att
        elif cls_type.lower() == 'linear':
            load_path = self.output_model_linear
        elif cls_type.lower() == 'softmax':
            load_path = self.output_model_logit
        # print(load_path)
        # print(cls_type)
        cls = cls(X.shape[1]).to(self.device)
        cls.load_state_dict(torch.load(load_path, map_location=self.device, weights_only=True))
        cls.eval()
        X = torch.tensor(X).to(self.device)

        with torch.no_grad():
            start_time = time.time()
            pred = cls(X)
            # print('pred raw shape', pred.shape)
            # print('pred raw first', pred[0])
            pred = torch.nn.functional.softmax(pred,dim=1)
            # print('pred prob', pred)
            # print('pred prob first', pred[0])
            # 0=hallucination, 1= not hallucination
            prediction_classes = torch.argmax((pred[0]>0.5).type(torch.long).cpu(),dim=0)
            end_time = time.time()
        print('{} hallucination classifier overhead: {}s'.format(cls_type, end_time - start_time))
        print('{} prediction, is hallucination?: {}.'.format(cls_type, prediction_classes==0))

    def model_task_accuracy(
            output_path, #path to store the accuracy
            data,
        ):
        # get task accuracy (based on the question)
        # find statistic in hidden data
        # store in file, in different file since this is not hallucination performance
        pass


    def heuristic_accuracy(
            question,
            answer,
            pred
    ):
        # get and save heuristic data, 
        # just output 100 random data from hidden data and their correct answer and model output
        # need count accuracy manually in the saved data
        pass

         # print(self.output_eval_dir)
        # print(self.output_model_dir)
        # print(self.output_eval)
        # print(self.output_model_ig)
        # print(self.output_model_logit)
        # print(self.output_model_att)
        # print(self.output_model_linear)


    def train_and_eval(self):
        # skip if finished
        if not self.demo:
            cls_found = self.output_eval.exists() and self.output_model_ig.exists() and self.output_model_logit.exists() and self.output_model_att.exists() and self.output_model_linear.exists()
            if cls_found and not self.train_exist:
                print('Found result, skipping')
                return
            else:
                print('Proceed to training clasifier')
        
        all_results = {}
        # for idx, results_file in enumerate(tqdm(self.layerDataFiles)):
        # add a llm input output for demo 
        # results['question'].append(question)
        # results['answers'].append(answers)
        # results['response'].append(response)
        # results['str_response'].append(str_response)
        hidden_data = {
                'correct': [],
                'attributes_first': [],
                'logits': [],
                'start_pos': [],
                'first_fully_connected': [],
                'first_attention': [],
                'question': [],
                'answers': [],
                'response': [],
                'str_response': [],
        }
        print('Loading hidden data')
        
        # print('classifier demo mode 1: ', self.demo)
        # print('datadir: ', self.layerDataFiles)
        for results_file in tqdm(self.layerDataFiles):
            # print('classifier demo mode 3: ', self.demo)
            # merge all chunk for a given hidden data
            try:
                # print('classifier demo mode 2: ', self.demo)
                if self.demo:
                    # select a random sample for demo
                    print("Using demo mode")
                    rand_chunk = random.choice(self.layerDataFiles)
                    rand_idx = random.randint(0, self.chunk_sz-1)
                    print('chunk for demo: ', rand_chunk)
                    print(f"Using {rand_idx+1} th data of {rand_chunk} for demo")
                    with open(rand_chunk, "rb") as infile:
                        results = pickle.loads(infile.read())
                        for k in hidden_data.keys():
                            hidden_data[k].append(results[k][rand_idx])
                    break
                else:
                    with open(results_file, "rb") as infile:
                        results = pickle.loads(infile.read())
                        # print('result key')
                        # for k in results.keys():
                        #     print(k)
                        # print('='*50)
                        # print('hidden data key')
                        for k in hidden_data.keys():
                            # print(k)
                            hidden_data[k].extend(results[k])
                        # print('='*50)
                    # used keys
                    # results['attributes_first']
                    # results['logits'], results['start_pos']
                    # results['first_fully_connected']
                    # results['first_attention']
            except Exception as err:
                print(traceback.format_exc())
        # print('hidden data')
        # for k, v in hidden_data.items():
        #     print(k, len(v))
        #     try:
        #         print(v[0].shape)
        #     except:
        #         print(v[0])
        # use the merged data to train classifier
        # for results_file in tqdm(self.layerDataFiles):
        # merge all chunk for a given hidden data will be used by classifier

        #     if results_file not in all_results.keys():
        #         try:
        #             del results
        #         except:
        #             pass
        if self.demo:
            # TODO
            # print(hidden_data)
            # print LLM input and output and halu label
            
            # model_loader = LlamaForCausalLM if "llama" in self.model_name else AutoModelForCausalLM
            token_loader = LlamaTokenizer if "llama" in self.llm_name else AutoTokenizer
            print(f'{self.model_repos[self.llm_name]}/{self.llm_name}')
            tokenizer = token_loader.from_pretrained(f'{self.model_repos[self.llm_name]}/{self.llm_name}')
            # model = model_loader.from_pretrained(f'{self.model_repos[self.llm_name]}/{self.llm_name}',
            #                                     cache_dir=self.cache_model_dir,
            #                                     device_map=self.device,
            #                                     torch_dtype=torch.bfloat16,
            #                                     # load_in_4bit=True,
            #                                     trust_remote_code=True)

            # embedder = self.get_embedder(model)
            if self.dataset == 'trivia_qa':
                prompt_input = hidden_data['question']
            else:
                prompt_input = self.trex_data_to_question_template[self.dataset].substitute(
                    source=hidden_data['question'][0])
                
            print('Prompt input: ', prompt_input)
            print('LLM output: ', hidden_data['str_response'][0])
            print('Correct answer: ', hidden_data['answers'][0][0])
            print('Hallucination? ', (not hidden_data['correct'][0]))
            # print input (softmax and IG only, since linear and attention is not meaningful to human)
            # and plot graph
            # output softmax
            # for first generation token
            first_softmax = torch.nn.functional.softmax(torch.tensor(hidden_data['logits'][0][0]),dim=0)
            print('output softmax: ', first_softmax)
            print('softmax max: ', torch.max(first_softmax))
            print('output softmax shape: ', first_softmax.shape)
            # get top n
            torch_topk = torch.topk(first_softmax, k=self.k, dim=0)
            print('top 10 softmax: ', torch_topk)
            topk_logit_idx = np.argpartition(first_softmax, -self.k)[-self.k:]
            topk_logit_idx = torch.flip(topk_logit_idx[np.argsort(first_softmax[topk_logit_idx])], dims=(0,))
            topk_logit = first_softmax[topk_logit_idx]
            print(f"top10_logit: {topk_logit}, top10_logit_idx: {topk_logit_idx}")
            topk_str = []
            for tok in topk_logit_idx:
                topk_str.append(tokenizer.decode(tok, skip_special_tokens=False))
            # tokenizer.decode
            print('topk_str: ', topk_str)
            # plot
            # Create figure with two subplots
            plt.figure(figsize=(12, 5))

            # First bar chart
            plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
            plt.bar(topk_str, topk_logit)
            plt.title(f'Top {self.k} Softmax of First Output Token Bar Chart')
            plt.xlabel('Softmax token')
            plt.ylabel('Softmax')
            plt.xticks(rotation=45, ha='right')

            # input IG weight to output
            # print('input feature weight: ', hidden_data['attributes_first'])
            print('input feature weight: ', hidden_data['attributes_first'][0])
            input_tok = tokenizer(prompt_input, return_tensors='pt').input_ids.cpu()
            print('input tokens: ', input_tok[0])
            input_tok_str = tokenizer.decode(input_tok[0], skip_special_tokens=True)
            print('input tokens str: ', input_tok_str)
            print('input feature weight length: ', len(hidden_data['attributes_first'][0]))
            print('input tokens length: ', len(input_tok[0]))
            # print('special tok: ', tok)
            print('tok to str map')
            input_tok_str_with_special = []
            for tok in input_tok[0]:
                print(tok)
                cur_input_tok_str_spe = tokenizer.decode(tok, skip_special_tokens=False)
                input_tok_str_with_special.append(cur_input_tok_str_spe)
                cur_input_tok_str = tokenizer.decode(tok, skip_special_tokens=True)
                print('tok: {}, tok_str: {}, tok_str_spe: {}'.format(tok, cur_input_tok_str, cur_input_tok_str_spe))
            # map with input question
            # embedding = embedder(input_tok).detach()
            # print('embedding: ', embedding)
            # print('embedding shape: ', embedding.shape)

            # print('length of input: ', hidden_data['question'][0])
            # plot

            # Second bar chart
            plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
            plt.bar(input_tok_str_with_special, hidden_data['attributes_first'][0])
            plt.title(f'Intergrated Gradient(IG) of Input Token Bar Chart')
            plt.xlabel('Input token')
            plt.ylabel('Normalized IG')
            plt.xticks(rotation=45, ha='right')

            plt.tight_layout()
            tmp_path = self.output_demo_dir / f'Demo.png'
            plt.savefig(str(tmp_path), bbox_inches='tight')

            # load the 4 model and give prediction
            # IG
            rnn_model = self.cls_IG()
            X_demo = torch.tensor(hidden_data['attributes_first'][0]).to(self.device).view(1, -1, 1).to(torch.float)
            rnn_model = rnn_model.to(self.device)
            rnn_model.load_state_dict(torch.load(self.output_model_ig, map_location=self.device, weights_only=True))
            rnn_model.eval()
            # demo_input = torch.tensor(hidden_data['attributes_first'][0]).to(self.device)
            # print('demo torch input: ', X_train)
            # X_train = torch.tensor(demo_input).view(1, -1, 1).to(torch.float)
            # print('demo torch input transformed: ', X_train)
            # print('X train shape', X_train.shape)

            with torch.no_grad():
                start_time = time.time()
                pred = rnn_model(X_demo)
                # print('pred prob', pred)
                pred = torch.nn.functional.softmax(pred,dim=0)
                # 0=hallucination, 1= not hallucination
                prediction_classes = torch.argmax((pred>0.5).type(torch.long).cpu(),dim=0)
                end_time = time.time()
            print('IG hallucination classifier overhead: {}s'.format(end_time - start_time))
            print('IG prediction, is hallucination?: {}.'.format(prediction_classes==0))

            # logit/softmax
            X_demo = hidden_data['logits'][0]
            self.demo_classifier(X_demo, self.cls_logit, 'Softmax')

            # linear
            X_demo = hidden_data['first_fully_connected'][0]
            self.demo_classifier(X_demo, self.cls_logit, 'Linear')

            # attention
            X_demo = hidden_data['first_attention'][0]
            self.demo_classifier(X_demo, self.cls_logit, 'attention')
            
            return
        # train classifier
        try:
            classifier_results = {}
            # with open(results_file, "rb") as infile:
            #     results = pickle.loads(infile.read())
            label = np.array(hidden_data['correct'])
    
            # attributes
            # print('IG')
            # print(hidden_data['attributes_first'])                    
            X_train, X_test, y_train, y_test = train_test_split(hidden_data['attributes_first'], label.astype(int), test_size = 0.2, random_state=self.seed)

            # print(X_train, X_test, y_train, y_test)
            rnn_model = self.cls_IG()
            optimizer = torch.optim.AdamW(rnn_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            for _ in range(self.epochs):
                # print(X_train)
                # print(len(X_train))
                if self.batch_size > len(X_train):
                    tmp_batch_size = len(X_train)
                    x_sub, y_sub = zip(*random.sample(list(zip(X_train, y_train)), tmp_batch_size))
                else:
                    x_sub, y_sub = zip(*random.sample(list(zip(X_train, y_train)), self.batch_size))
                y_sub = torch.tensor(y_sub).to(torch.long)
                optimizer.zero_grad()
                preds = torch.stack([rnn_model(torch.tensor(i).view(1, -1, 1).to(torch.float)) for i in x_sub])
                loss = torch.nn.functional.cross_entropy(preds, y_sub)
                loss.backward()
                optimizer.step()
            # save the model
            torch.save(rnn_model.state_dict(), self.output_model_ig)
            preds = torch.stack([rnn_model(torch.tensor(i).view(1, -1, 1).to(torch.float)) for i in X_test])
            preds = torch.nn.functional.softmax(preds, dim=1)
            prediction_classes = (preds[:,1]>0.5).type(torch.long).cpu()
            classifier_results['attribution_rnn_roc'] = roc_auc_score(y_test, preds[:,1].detach().cpu().numpy())
            classifier_results['attribution_rnn_acc'] = (prediction_classes.numpy()==y_test).mean()

            # logits
            # print('logits and start_pos')
            # print(hidden_data['logits'], hidden_data['start_pos'])
            first_logits = np.stack([sp.special.softmax(i[j]) for i,j in zip(hidden_data['logits'], hidden_data['start_pos'])])
            # print('logits')
            first_logits_roc, first_logits_acc = self.gen_classifier_roc(
                first_logits, 
                label, 
                self.cls_logit, 
                self.output_model_logit,
            )
            classifier_results['first_logits_roc'] = first_logits_roc
            classifier_results['first_logits_acc'] = first_logits_acc

            # fully connected
            # print('linear')
            for layer in range(hidden_data['first_fully_connected'][0].shape[0]):
                # print('='*50)
                # print('layer: ', layer)
                # print(len(hidden_data['first_fully_connected']))
                # print(hidden_data['first_fully_connected'][0].shape)
                # print(hidden_data['first_fully_connected'])
                layer_roc, layer_acc = self.gen_classifier_roc(
                    np.stack([i[layer] for i in hidden_data['first_fully_connected']]),
                    label,
                    self.cls_linear,
                    self.output_model_linear,
                )
                classifier_results[f'first_fully_connected_roc_{layer}'] = layer_roc
                classifier_results[f'first_fully_connected_acc_{layer}'] = layer_acc

            # attention
            # print('attention')
            for layer in range(hidden_data['first_attention'][0].shape[0]):
                # print('='*50)
                # print('layer: ', layer)
                # print(len(hidden_data['first_attention']))
                # print(hidden_data['first_attention'][0].shape)
                # print(hidden_data['first_attention'])
                layer_roc, layer_acc = self.gen_classifier_roc(
                    np.stack([i[layer] for i in hidden_data['first_attention']]),
                    label,
                    self.cls_att,
                    self.output_model_att,
                )
                classifier_results[f'first_attention_roc_{layer}'] = layer_roc
                classifier_results[f'first_attention_acc_{layer}'] = layer_acc
            
            # all_results[results_file] = classifier_results.copy()
            # TODO
            # Done
            # save the eval result as csv
            # print(classifier_results.keys())
            for k,v in classifier_results.items():
                print(k, v)

            df = pd.DataFrame.from_dict(classifier_results, orient="index", columns=[self.model_name])
            df.to_csv(self.output_eval)
        except Exception as err:
            print(traceback.format_exc())
            print("\"ValueError: Sample larger than population or is negative\" maybe due to batch size too large, reduce it use --cls_batch_size")