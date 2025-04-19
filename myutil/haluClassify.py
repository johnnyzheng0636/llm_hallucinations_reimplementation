import pickle
from pathlib import Path
import numpy as np
import scipy as sp
import pandas as pd

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import random

from tqdm import tqdm
import traceback

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
        ):
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

        # create output directory
        self.model_name = str(layerDataPath).split('/')[-1]
        self.outPath = Path(out_dir) / self.model_name
        
        # print(self.outPath)

        self.output_eval_dir = self.outPath / 'eval'
        self.output_model_dir = self.outPath / 'model_stat'

        # print(self.outPath)
        # print(self.output_eval_dir)
        # print(self.output_model_dir)

        try:
            self.output_eval_dir.mkdir(parents=True, exist_ok=True)
            self.output_model_dir.mkdir(parents=True, exist_ok=True)
        except  Exception as err:
            print("The name may have been used already, try anoter name.")
            print(traceback.format_exc())

        self.output_eval = self.output_eval_dir / f'eval.csv'
        self.output_model_ig = self.output_model_dir / f'{self.cls_IG.__name__}_ig_model.pth'
        self.output_model_logit = self.output_model_dir / f'{self.cls_logit.__name__}_logit_model.pth'
        self.output_model_att = self.output_model_dir / f'{self.cls_att.__name__}_att_model.pth'
        self.output_model_linear = self.output_model_dir / f'{self.cls_linear.__name__}_linear_model.pth'
        
        # print('output path')
        # print(layerDataPath)
        # print(self.output_eval_dir)
        # print(self.output_model_dir)
        # print(self.output_eval)
        # print(self.output_model_ig)
        # print(self.output_model_logit)
        # print(self.output_model_att)
        # print(self.output_model_linear)

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

    def demo_classifier(self, X, label, cls, save_path, dataType):
        # load exist classifier model and print LLM input, output and hidden layers
        # expect only one data
        print(f'LLM {dataType}: ', X[0])
        print('Hallucination ground truth: ', label[0])
        # load model

        # print predicted model output

    def train_and_eval(self):
        # skip if finished

        if not self.demo:
            if self.output_eval.exists() and not self.train_exist:
                print('Found result, skipping')
                return
        
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
        for results_file in tqdm(self.layerDataFiles):
            # merge all chunk for a given hidden data
            try:
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

            # demo classifier
            # TODO
            # load the model
            # self.demo_classifier(X, label, cls, save_path, dataType)
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
            X_train, X_test, y_train, y_test = train_test_split(hidden_data['attributes_first'], label.astype(int), test_size = 0.2, random_state=42)

            # print(X_train, X_test, y_train, y_test)
            rnn_model = self.cls_IG()
            optimizer = torch.optim.AdamW(rnn_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            for _ in range(self.epochs):
                x_sub, y_sub = zip(*random.sample(list(zip(X_train, y_train)), self.batch_size))
                y_sub = torch.tensor(y_sub).to(torch.long)
                optimizer.zero_grad()
                preds = torch.stack([rnn_model(torch.tensor(i).view(1, -1, 1).to(torch.float)) for i in x_sub])
                loss = torch.nn.functional.cross_entropy(preds, y_sub)
                loss.backward()
                optimizer.step()
            # TODO
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
        except Exception as err:
            print(traceback.format_exc())
            print("\"ValueError: Sample larger than population or is negative\" maybe due to batch size too large, reduce it use --cls_batch_size")

        # TODO
        # save the eval result as csv
        # print(classifier_results.keys())
        for k,v in classifier_results.items():
            print(k, v)

        df = pd.DataFrame.from_dict(classifier_results, orient="index", columns=[self.model_name])
        df.to_csv(self.output_eval)