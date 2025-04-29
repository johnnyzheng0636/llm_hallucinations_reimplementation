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

class deafaultCombined(torch.nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        
        self.linear_relu_stack =torch.nn.Sequential(
            torch.nn.Linear(input_shape, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(256, 6)
            )
        hidden_dim = 128
        num_layers = 4
        self.gru = torch.nn.GRU(1, hidden_dim, num_layers, dropout=0.25, batch_first=True, bidirectional=False)
        self.gru_linear = torch.nn.Linear(hidden_dim, 2)
        self.mlp = torch.nn.Linear(8, 2)
    def forward(self, x, seq):
        linear_logits = self.linear_relu_stack(x)
        # print('linear_logits shape: ', linear_logits.size())
        gru_out, _ = self.gru(seq)
        # print('gru_out shape: ', gru_out.size())
        gru_logits = self.gru_linear(gru_out)
        # print('gru_logits shape: ', gru_logits.size())
        gru_logits = gru_logits[:, -1, :]
        # print('gru_logits shape: ', gru_logits.size())
        logit = torch.cat((linear_logits, gru_logits), dim=1)
        final_x = self.mlp(logit)
        return final_x

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
            cls_combined=deafaultCombined,
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
        self.cls_combined = cls_combined
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
            "gemma-3-4b-it" : ("google"),
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
        self.output_model_combined = self.output_model_dir / f'{self.cls_combined.__name__}_linear_model.pth'

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
        elif "gemma" in self.model_name.lower():
            return model.language_model.model.embed_tokens
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

        for _ in tqdm(range(self.epochs)):
            optimizer.zero_grad()
            sample = torch.randperm(X_train.shape[0])[:self.batch_size]
            # print(sample)
            # print('X_train size', X_train[sample].size())
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


    def gen_combined_classifier_roc(self, X_linear, X_softmax, X_att, X_ig, label, cls, save_path):
        # print('='*50)
        # print('X_linear len: ', len(X_linear))
        # print('X_linear shape: ', X_linear[0].shape)
        # print('X_softmax shape: ', X_softmax.shape)
        # print('X_att len: ', len(X_att))
        # print('X_att shape: ', X_att[0].shape)
        # print('X_ig len: ', len(X_ig))
        # print('X_ig shape: ', X_ig[0].shape)
        # shape(the same data number, sum of wide of three data)
        # only use the last layer of att and linear
        last_layer_first_att = np.stack([i[-1] for i in X_att])
        # print('transformed att shape: ', last_layer_first_att.shape)
        last_layer_first_linear = np.stack([i[-1] for i in X_linear])
        # print('transformed att shape: ', last_layer_first_linear.shape)
        X = np.concatenate((X_softmax, last_layer_first_att, last_layer_first_linear), axis=1)
        # print('final X shape: ', X.shape)
        # print('final X data size: ', X.shape[0])
        # print('dummy index: ', np.arange(X.shape[0]))
        X_train_idx, X_test_idx, y_train, y_test = train_test_split(np.arange(X.shape[0]), label.astype(int), test_size = 0.2, random_state=42)
        print(X_train_idx, X_test_idx, y_train, y_test)

        # X_ig = torch.nn.utils.rnn.pad_sequence(X_ig, batch_first=True)
        # print('X_ig shape: ', X_ig.size())

        X_mat_train = np.take(X, X_train_idx, axis=0)
        # print('X_mat_train shape: ', X_mat_train.shape)
        X_rnn_train = [torch.tensor(X_ig[i]) for i in X_train_idx]
        # print('X_rnn_train shape: ', len(X_rnn_train), X_rnn_train[0].shape, X_rnn_train[1].shape)
        X_mat_test = np.take(X, X_test_idx, axis=0)
        # print('X_mat_test shape: ', X_mat_test.shape)
        X_rnn_test = [torch.tensor(X_ig[i]) for i in X_test_idx]
        # print('X_rnn_test shape: ', len(X_rnn_test), X_rnn_test[0].shape, X_rnn_test[1].shape)

        
        
        classifier_model = cls(X_mat_train.shape[1]).to(self.device)

        X_mat_train = torch.tensor(X_mat_train).to(torch.float).to(self.device)
        X_rnn_train = torch.unsqueeze(torch.nn.utils.rnn.pad_sequence(X_rnn_train, batch_first=True),dim=2).to(torch.float).to(self.device)
        y_train = torch.tensor(y_train).to(torch.long).to(self.device)
        X_mat_test = torch.tensor(X_mat_test).to(torch.float).to(self.device)
        X_rnn_test = torch.unsqueeze(torch.nn.utils.rnn.pad_sequence(X_rnn_test, batch_first=True),dim=2).to(torch.float).to(self.device)
        y_test = torch.tensor(y_test).to(torch.long).to(self.device)

        # print('X_mat_train shape: ', X_mat_train.size())
        # print('X_rnn_train shape: ', X_rnn_train.size())
        # print('y_train shape: ', y_train.size())
        # print('X_mat_test shape: ', X_mat_test.size())
        # print('X_rnn_test shape: ', X_rnn_test.size())
        # print('y_test shape: ', y_test.size())

        optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for _ in tqdm(range(self.epochs)):
            optimizer.zero_grad()
            sample = torch.randperm(X_mat_train.shape[0])[:self.batch_size]
            pred = classifier_model(X_mat_train[sample], X_rnn_train[sample])
            # print('pred shape: ', pred.size())
            loss = torch.nn.functional.cross_entropy(pred, y_train[sample])
            loss.backward()
            optimizer.step()
    
        classifier_model.eval()
        with torch.no_grad():
            pred = torch.nn.functional.softmax(classifier_model(X_mat_test, X_rnn_test), dim=1)
            # print('pred size: ', pred.size())
            prediction_classes = (pred[:,1]>0.5).type(torch.long).cpu()
            # print('pred size: ', prediction_classes.size())
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
            
            # print('pred softmax shape', pred.shape)
            # print('pred prob', pred)
            # print('pred prob first', pred[0])
            # 0=hallucination, 1= not hallucination
            # print('select: ', pred[0])
            # print('select transformed: ', (pred[0]>0.5).type(torch.long).cpu())
            prediction_classes = torch.argmax((pred[-1]>0.5).type(torch.long).cpu(),dim=0)
            end_time = time.time()
        print('{} hallucination classifier overhead: {}s'.format(cls_type, end_time - start_time))
        print('{} prediction, is hallucination?: {}.'.format(cls_type, prediction_classes==0))


    def demo_combined_classifier(self, X_linear, X_softmax, X_att, X_ig, cls):
        load_path = self.output_model_combined

        # print('='*50)
        # print('X_linear len: ', len(X_linear))
        # print('X_linear shape: ', X_linear.shape)
        # print('X_softmax shape: ', X_softmax.shape)
        # print('X_att len: ', len(X_att))
        # print('X_att shape: ', X_att.shape)
        # print('X_ig len: ', len(X_ig))
        # print('X_ig shape: ', X_ig.shape)
        # shape(the same data number, sum of wide of three data)
        # only use the last layer of att and linear
        last_layer_first_att = X_att[-1]
        # print('transformed att shape: ', last_layer_first_att.shape)
        last_layer_first_linear = X_linear[-1]
        # print('transformed att shape: ', last_layer_first_linear.shape)
        last_layer_first_softmax = X_softmax[-1]
        # print('transformed softmax shape: ', last_layer_first_softmax.shape)
        X = np.concatenate((last_layer_first_softmax, last_layer_first_att, last_layer_first_linear), axis=0)
        # X = np.concatenate((X_softmax, last_layer_first_att, last_layer_first_linear), axis=1)
        # print('final X shape: ', X.shape)
        # print('final X data size: ', X.shape[0])
        # print('dummy index: ', np.arange(X.shape[0]))


        # last_layer_first_att = np.stack([i[-1] for i in X_att])
        # last_layer_first_linear = np.stack([i[-1] for i in X_linear])
        # X = np.concatenate((X_softmax, last_layer_first_att, last_layer_first_linear), axis=1)
        
        # print('X. shape: ', X.shape)

        # print(load_path)
        # print(cls_type)
        cls = cls(X.shape[0]).to(self.device)
        cls.load_state_dict(torch.load(load_path, map_location=self.device, weights_only=True))
        cls.eval()
        # TODO
        # combined three into one
        X_mat_train = torch.tensor(X).to(torch.float).view(1, -1).to(self.device)
        # print('X_mat_train shape: ', X_mat_train.size())

        # print('X_ig: ', X_ig)

        X_rnn_train = torch.tensor(X).view(1, -1, 1)

        # print('X_rnn_train: ', X_rnn_train)
        # X = torch.tensor(X).to(self.device)

        with torch.no_grad():
            start_time = time.time()
            pred = cls(X_mat_train, X_rnn_train)
            # print('pred raw shape', pred.shape)
            # print('pred raw first', pred[0])
            pred = torch.nn.functional.softmax(pred,dim=1)
            # print('pred prob', pred)
            # print('pred prob first', pred[0])
            # 0=hallucination, 1= not hallucination
            prediction_classes = torch.argmax((pred[-1]>0.5).type(torch.long).cpu(),dim=0)
            end_time = time.time()
        print('{} hallucination classifier overhead: {}s'.format('Combined', end_time - start_time))
        print('{} prediction, is hallucination?: {}.'.format('Combined', prediction_classes==0))

    # TODO add these two and  add a switch for only doing them and forcely redo all 
    def model_task_accuracy(
            self,
            output_path, #path to store the accuracy
            data,
        ):
        # get task accuracy (based on the question)
        sum = np.sum(data)
        acc = sum/data.shape[0]
        task_acc = {'Accuracy': acc}
        # store in file, in different file since this is not hallucination performance
        df = pd.DataFrame.from_dict(task_acc, orient="index", columns=[self.model_name])
        df.to_csv(output_path)
        print(f"Task accuracy saved to {output_path}")


    def heuristic_accuracy(
        self,
        question,
        answer,
        correct,
        response_str,
    ):
        # get and save heuristic data, 
        # just output 100 random data from hidden data and their correct answer and model output
        # need count accuracy manually in the saved data

        if self.dataset != 'trivia_qa':
            for q in range(len(question)):
                question[q] = self.trex_data_to_question_template[self.dataset].substitute(
                    source=question[q])

        heuristic_dict = {
            'question': question,
            'answer': answer,
            'correct': correct,
            'response_str': response_str,
        }

        df = pd.DataFrame(heuristic_dict)
        output_path = self.output_eval_dir / 'heuristic.csv'
        df.to_csv(output_path)


    def train_and_eval(self):
        # skip if finished
        if not self.demo:
            cls_found = self.output_eval.exists() and self.output_model_ig.exists() and self.output_model_logit.exists() and self.output_model_att.exists() and self.output_model_linear.exists()
            if cls_found and not self.train_exist:
                print('Found classification result, skipping')
                return
            else:
                print('Proceed to training clasifier')
        
        all_results = {}
        # for idx, results_file in enumerate(tqdm(self.layerDataFiles)):
        # add a llm input output for demo 
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
                    # print('chunk for demo: ', rand_chunk)
                    # print(f"Using {rand_idx+1} th data of {rand_chunk} for demo")
                    with open(rand_chunk, "rb") as infile:
                        results = pickle.loads(infile.read())
                        for k in hidden_data.keys():
                            hidden_data[k].append(results[k][rand_idx])
                        found_data_correct = hidden_data['correct'][0]
                        # print('found_data_correct: ', found_data_correct)
                        for i in range(len(results['correct'])):
                            # print(f'{i} th element in chunk')
                            # print('current correct: ', results['correct'][i])
                            if results['correct'][i] == found_data_correct:
                                continue
                            else:
                                # print('second hidden data adding')
                                for k in hidden_data.keys():
                                    hidden_data[k].append(results[k][i])
                                break
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
            # print(hidden_data)
            token_loader = LlamaTokenizer if "llama" in self.llm_name else AutoTokenizer
            print(f'{self.model_repos[self.llm_name]}/{self.llm_name}')
            tokenizer = token_loader.from_pretrained(f'{self.model_repos[self.llm_name]}/{self.llm_name}')

            # First demo
            if self.dataset == 'trivia_qa':
                prompt_input_1 = hidden_data['question'][0]
            else:
                prompt_input_1 = self.trex_data_to_question_template[self.dataset].substitute(
                    source=hidden_data['question'][0])
                
            print('='*50, 'Demo 1', '='*50)
            print('Prompt input: ', prompt_input_1)
            print('='*50)
            print('LLM output: ', hidden_data['str_response'][0])
            print('='*50)
            print('Correct answer: ', hidden_data['answers'][0])
            print('='*50)
            print('Hallucination? ', (not hidden_data['correct'][0]))
            print('='*50)
            # and plot graph
            # output softmax
            # for first generation token
            # print(hidden_data['logits'][0][hidden_data['start_pos'][0]])
            first_softmax = torch.nn.functional.softmax(torch.tensor(hidden_data['logits'][0][hidden_data['start_pos'][0]-1]),dim=0)
            topk_logit_idx = np.argpartition(first_softmax, -self.k)[-self.k:]
            topk_logit_idx = torch.flip(topk_logit_idx[np.argsort(first_softmax[topk_logit_idx])], dims=(0,))
            topk_logit = first_softmax[topk_logit_idx]
            # print(f"top10_logit: {topk_logit}, top10_logit_idx: {topk_logit_idx}")
            topk_str = []
            for tok in topk_logit_idx:
                topk_str.append(tokenizer.decode(tok, skip_special_tokens=False))
            # tokenizer.decode
            # print('topk_str: ', topk_str)
            # plot
            # Create figure with two subplots
            title_halu = '(Hallucination)' if not hidden_data['correct'][0] else '(Correct)'
            plt.figure(figsize=(12, 10))

            # First bar chart
            plt.subplot(2, 2, 1)  # 1 row, 2 columns, first plot
            plt.bar(topk_str, topk_logit)
            plt.title(f'{title_halu} Top {self.k} Softmax of First Output Token Bar Chart')
            plt.xlabel('Softmax token')
            plt.ylabel('Softmax')
            plt.xticks(rotation=45, ha='right')

            # input IG weight to output
            input_tok = tokenizer(prompt_input_1, return_tensors='pt').input_ids.cpu()
            # input_tok_str = tokenizer.decode(input_tok[0], skip_special_tokens=True)
            
            input_tok_str_with_special = []
            for tok in input_tok[0]:
                # print(tok)
                cur_input_tok_str_spe = tokenizer.decode(tok, skip_special_tokens=False)
                input_tok_str_with_special.append(cur_input_tok_str_spe)
                # cur_input_tok_str = tokenizer.decode(tok, skip_special_tokens=True)
            plt.subplot(2, 2, 2)  # 1 row, 2 columns, second plot
            plt.bar(input_tok_str_with_special, hidden_data['attributes_first'][0])
            plt.title(f'{title_halu} Intergrated Gradient(IG) of Input Token Bar Chart')
            plt.xlabel('Input token')
            plt.ylabel('Normalized IG')
            plt.xticks(rotation=45, ha='right')

            # plt.tight_layout()
            # tmp_path = self.output_demo_dir / f'Demo.png'
            # plt.savefig(str(tmp_path), bbox_inches='tight')
            # print(f'graph saved to {tmp_path}')

            # load the 4 model and give prediction
            # IG
            rnn_model = self.cls_IG()
            X_demo = torch.tensor(hidden_data['attributes_first'][0]).to(self.device).view(1, -1, 1).to(torch.float)
            rnn_model = rnn_model.to(self.device)
            rnn_model.load_state_dict(torch.load(self.output_model_ig, map_location=self.device, weights_only=True))
            rnn_model.eval()

            with torch.no_grad():
                start_time = time.time()
                pred = rnn_model(X_demo)
                pred = torch.nn.functional.softmax(pred,dim=0)
                # 0=hallucination, 1= not hallucination
                prediction_classes = torch.argmax((pred>0.5).type(torch.long).cpu(),dim=0)
                end_time = time.time()
            print('IG hallucination classifier overhead: {}s'.format(end_time - start_time))
            print('IG prediction, is hallucination?: {}.'.format(prediction_classes==0))

            # logit/softmax
            X_demo = np.expand_dims(hidden_data['logits'][0][hidden_data['start_pos'][0]-1], axis=0)
            # print(X_demo)
            self.demo_classifier(X_demo, self.cls_logit, 'Softmax')

            # linear
            X_demo = hidden_data['first_fully_connected'][0]
            self.demo_classifier(X_demo, self.cls_logit, 'Linear')

            # attention
            X_demo = hidden_data['first_attention'][0]
            self.demo_classifier(X_demo, self.cls_logit, 'attention')

            # combined
            X_linear = hidden_data['first_fully_connected'][0]
            X_softmax = np.expand_dims(hidden_data['logits'][0][hidden_data['start_pos'][0]-1], axis=0)
            X_att = hidden_data['first_attention'][0]
            X_ig = hidden_data['attributes_first'][0]
            self.demo_combined_classifier(X_linear, X_softmax, X_att, X_ig, self.cls_combined)
            print('='*50)
            print('\n\n')

            # Second demo
            if self.dataset == 'trivia_qa':
                prompt_input_2 = hidden_data['question'][1]
            else:
                prompt_input_2 = self.trex_data_to_question_template[self.dataset].substitute(
                    source=hidden_data['question'][1])
                
            print('='*50, 'Demo 2', '='*50)
            print('Prompt input: ', prompt_input_2)
            print('='*50)
            print('LLM output: ', hidden_data['str_response'][1])
            print('='*50)
            print('Correct answer: ', hidden_data['answers'][1])
            print('='*50)
            print('Hallucination? ', (not hidden_data['correct'][1]))
            print('='*50)
            # and plot graph
            # output softmax
            # for first generation token
            # print(hidden_data['logits'][1][hidden_data['start_pos'][1]])
            first_softmax = torch.nn.functional.softmax(torch.tensor(hidden_data['logits'][1][hidden_data['start_pos'][1]-1]),dim=0)
            topk_logit_idx = np.argpartition(first_softmax, -self.k)[-self.k:]
            topk_logit_idx = torch.flip(topk_logit_idx[np.argsort(first_softmax[topk_logit_idx])], dims=(0,))
            topk_logit = first_softmax[topk_logit_idx]
            # print(f"top10_logit: {topk_logit}, top10_logit_idx: {topk_logit_idx}")
            topk_str = []
            for tok in topk_logit_idx:
                topk_str.append(tokenizer.decode(tok, skip_special_tokens=False))
            # tokenizer.decode
            # print('topk_str: ', topk_str)
            # plot
            # Create figure with two subplots
            # plt.figure(figsize=(12, 5))
            
            title_halu = '(Hallucination)' if not hidden_data['correct'][1] else '(Correct)'
            # First bar chart
            plt.subplot(2, 2, 3) 
            plt.bar(topk_str, topk_logit)
            plt.title(f'{title_halu} Top {self.k} Softmax of First Output Token Bar Chart')
            plt.xlabel('Softmax token')
            plt.ylabel('Softmax')
            plt.xticks(rotation=45, ha='right')

            # input IG weight to output
            input_tok = tokenizer(prompt_input_2, return_tensors='pt').input_ids.cpu()
            # input_tok_str = tokenizer.decode(input_tok[0], skip_special_tokens=True)
            
            input_tok_str_with_special = []
            for tok in input_tok[0]:
                # print(tok)
                cur_input_tok_str_spe = tokenizer.decode(tok, skip_special_tokens=False)
                input_tok_str_with_special.append(cur_input_tok_str_spe)
                # cur_input_tok_str = tokenizer.decode(tok, skip_special_tokens=True)
            plt.subplot(2, 2, 4)
            plt.bar(input_tok_str_with_special, hidden_data['attributes_first'][1])
            plt.title(f'{title_halu} Intergrated Gradient(IG) of Input Token Bar Chart')
            plt.xlabel('Input token')
            plt.ylabel('Normalized IG')
            plt.xticks(rotation=45, ha='right')

            plt.tight_layout()
            tmp_path = self.output_demo_dir / f'Demo.png'
            plt.savefig(str(tmp_path), bbox_inches='tight')
            print(f'graph saved to {tmp_path}')

            # load the 4 model and give prediction
            # IG
            rnn_model = self.cls_IG()
            X_demo = torch.tensor(hidden_data['attributes_first'][1]).to(self.device).view(1, -1, 1).to(torch.float)
            rnn_model = rnn_model.to(self.device)
            rnn_model.load_state_dict(torch.load(self.output_model_ig, map_location=self.device, weights_only=True))
            rnn_model.eval()

            with torch.no_grad():
                start_time = time.time()
                pred = rnn_model(X_demo)
                pred = torch.nn.functional.softmax(pred,dim=0)
                # 0=hallucination, 1= not hallucination
                prediction_classes = torch.argmax((pred>0.5).type(torch.long).cpu(),dim=0)
                end_time = time.time()
            print('IG hallucination classifier overhead: {}s'.format(end_time - start_time))
            print('IG prediction, is hallucination?: {}.'.format(prediction_classes==0))

            # logit/softmax
            X_demo = np.expand_dims(hidden_data['logits'][1][hidden_data['start_pos'][1]-1], axis=0)
            self.demo_classifier(X_demo, self.cls_logit, 'Softmax')

            # linear
            X_demo = hidden_data['first_fully_connected'][1]
            # print('linear shape', X_demo.shape)
            self.demo_classifier(X_demo, self.cls_logit, 'Linear')

            # attention
            X_demo = hidden_data['first_attention'][1]
            # print('att shape', X_demo.shape)
            self.demo_classifier(X_demo, self.cls_logit, 'attention')

            # combined
            X_linear = hidden_data['first_fully_connected'][1]
            X_softmax = np.expand_dims(hidden_data['logits'][1][hidden_data['start_pos'][1]-1], axis=0)
            X_att = hidden_data['first_attention'][1]
            X_ig = hidden_data['attributes_first'][1]
            self.demo_combined_classifier(X_linear, X_softmax, X_att, X_ig, self.cls_combined)
            print('='*50)
            
            return
        # train classifier
        try:
            classifier_results = {}
            # with open(results_file, "rb") as infile:
            #     results = pickle.loads(infile.read())
            # 0=hallucination, 1= not hallucination
            label = np.array(hidden_data['correct'])

            Task_acc_path = self.output_eval_dir / 'Task_accuracy.csv'
            self.model_task_accuracy(Task_acc_path, label)
            self.heuristic_accuracy(
                hidden_data['question'][:100],
                hidden_data['answers'][:100],
                hidden_data['correct'][:100],
                hidden_data['str_response'][:100],
            )
            # return #for debug
    
            # attributes
            # print('IG')
            # print(hidden_data['attributes_first'])                    
            X_train, X_test, y_train, y_test = train_test_split(hidden_data['attributes_first'], label.astype(int), test_size = 0.2, random_state=self.seed)

            # print(X_train, X_test, y_train, y_test)
            rnn_model = self.cls_IG()
            optimizer = torch.optim.AdamW(rnn_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            print('training IG classifier')
            for _ in tqdm(range(self.epochs)):
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
            first_logits = np.stack([sp.special.softmax(i[j-1]) for i,j in zip(hidden_data['logits'], hidden_data['start_pos'])])
            print('logits shape', first_logits.shape)
            first_logits_roc, first_logits_acc = self.gen_classifier_roc(
                first_logits, 
                label, 
                self.cls_logit, 
                self.output_model_logit,
            )
            classifier_results['first_logits_roc'] = first_logits_roc
            classifier_results['first_logits_acc'] = first_logits_acc

            # fully connected
            print('linear')
            print(len(hidden_data['first_fully_connected']))
            print(hidden_data['first_fully_connected'][0].shape)
            for layer in range(hidden_data['first_fully_connected'][0].shape[0]):
                # print('='*50)
                # print('layer: ', layer)
                # print(len(hidden_data['first_fully_connected']))
                # print(hidden_data['first_fully_connected'][0].shape)
                # print(hidden_data['first_fully_connected'])
                print(f'training {layer} th linear classifier')
                layer_roc, layer_acc = self.gen_classifier_roc(
                    np.stack([i[layer] for i in hidden_data['first_fully_connected']]),
                    label,
                    self.cls_linear,
                    self.output_model_linear,
                )
                classifier_results[f'first_fully_connected_roc_{layer}'] = layer_roc
                classifier_results[f'first_fully_connected_acc_{layer}'] = layer_acc

            # attention
            print('attention')
            print(len(hidden_data['first_attention']))
            print(hidden_data['first_attention'][0].shape)
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

            # combined
            combined_roc, combined_acc = self.gen_combined_classifier_roc(
                hidden_data['first_fully_connected'],
                first_logits,
                hidden_data['first_attention'],
                hidden_data['attributes_first'],
                label,
                self.cls_combined,
                self.output_model_combined,                
            )
            classifier_results['combined_roc'] = combined_roc
            classifier_results['combined_acc'] = combined_acc

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