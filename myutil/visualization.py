import os
import pickle
from pathlib import Path
import glob
import numpy as np
import scipy as sp
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
import traceback

# TODO save figs and modify to python

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class visualization():
    def __init__(
            self, 
            layerDataPath, # path has data for training            
            out_dir = "./outouts/",
            alpha = 0.2,
            only_bar = False,
        ):
        self.layerDataFiles = list(Path(layerDataPath).glob("*.pickle"))
        self.model_name = str(layerDataPath).split('/')[-1]
        self.outPath = Path(out_dir) / self.model_name
        self.alpha = alpha
        
        # print(self.outPath)

        self.output_fig_dir = self.outPath / 'fig'
        self.csv_input_dir = self.outPath / 'eval' / 'eval.csv'

        try:
            self.output_fig_dir.mkdir(parents=True, exist_ok=True)
        except  Exception as err:
            print("The name may have been used already, try anoter name.")
            print(traceback.format_exc())

        dataChunk = list(self.layerDataFiles)

        # initialize useful list
        # skip those before kill due to memory
        if not only_bar:
            first_attribute_entropy_ls = []
            first_logits_entropy_ls = []
            last_logits_entropy_ls = []
            first_logit_ls = []
            last_logit_ls = []
            first_token_layer_activations_ls = []
            final_token_layer_activations_ls = []
            first_token_layer_attention_ls = []
            final_token_layer_attention_ls = []
            correct_ls = []
            
            final_attention_ls = []
            final_linear_ls = []
            attr_ls = []

            for i in tqdm(range(len(dataChunk))):
                try:
                    with open(dataChunk[i], "rb") as infile:
                        chunk = pickle.loads(infile.read())

                    num_layers = chunk['first_fully_connected'][0].shape[0]
                    layer_pos = num_layers-2
                    first_attribute_entropy_ls.append(np.array([sp.stats.entropy(i) for i in chunk['attributes_first']]))
                    first_logits_entropy_ls.append(np.array([sp.stats.entropy(sp.special.softmax(i[j])) for i,j in zip(chunk['logits'], chunk['start_pos'])]))
                    last_logits_entropy_ls.append(np.array([sp.stats.entropy(sp.special.softmax(i[-1])) for i in chunk['logits']]))
                    # first_logit_decomp_ls.append(PCA(n_components=2).fit_transform(np.array([i[j] for i,j in zip(chunk['logits'], chunk['start_pos'])])))
                    # last_logit_decomp_ls.append(PCA(n_components=2).fit_transform(np.array([i[-1] for i in chunk['logits']])))
                    first_logit_ls.append(np.array([i[j] for i,j in zip(chunk['logits'], chunk['start_pos'])]))
                    # used by tsne too
                    last_logit_ls.append(np.array([i[-1] for i in chunk['logits']]))
                    first_token_layer_activations_ls.append(np.array([i[layer_pos] for i in chunk['first_fully_connected']]))
                    final_token_layer_activations_ls.append(np.array([i[layer_pos] for i in chunk['final_fully_connected']]))
                    first_token_layer_attention_ls.append(np.array([i[layer_pos] for i in chunk['first_attention']]))
                    final_token_layer_attention_ls.append(np.array([i[layer_pos] for i in chunk['final_attention']]))
                    correct_ls.append(np.array(chunk['correct']))

                    final_attention_ls.append(np.array([i[-1] for i in chunk['final_attention']]))
                    final_linear_ls.append(np.array([i[-1] for i in chunk['final_fully_connected']]))
                    for attr in chunk['attributes_first']:
                        if len(attr)<=20:
                            attr_ls.append(np.pad(attr, (0,20-len(attr))))
                        elif len(attr)>20:
                            attr_ls.append(np.array(attr[:20]))

                    del chunk
                except:
                    print(traceback.format_exc())

            self.curve_data = {
                'first_attribute_entropy': np.concatenate(first_attribute_entropy_ls),
                'correct': np.concatenate(correct_ls),
                'first_logits_entropy': np.concatenate(first_logits_entropy_ls),
                'last_logits_entropy': np.concatenate(last_logits_entropy_ls),
                'first_logit_decomp': PCA(n_components=2).fit_transform(np.concatenate(first_logit_ls)),
                'last_logit_decomp': PCA(n_components=2).fit_transform(np.concatenate(last_logit_ls)),
                'first_token_layer_activations': np.concatenate(first_token_layer_activations_ls),
                'final_token_layer_activations': np.concatenate(final_token_layer_activations_ls),
                'first_token_layer_attention': np.concatenate(first_token_layer_attention_ls),
                'final_token_layer_attention': np.concatenate(final_token_layer_attention_ls),
            }

            self.scatter_data = {
                'final_attention': np.concatenate(final_attention_ls),
                'final_linear': np.concatenate(final_linear_ls),
                'final_softmax': sp.special.softmax(np.concatenate(last_logit_ls), axis=1),
                'final_attr': np.vstack(attr_ls),
            }
        
        # after kill due to memory, run anyway
        self.eval_df = pd.read_csv(self.csv_input_dir, index_col=0)
        result = self.eval_df.to_dict()
        linear_acc_ls = []
        linear_roc_ls = []
        attention_acc_ls = []
        attention_roc_ls = []
        result = result[list(result.keys())[0]]
        # print(result.keys())
        for key in result.keys():
            # print(key)
            if 'acc' in key:
                if 'attention' in key:
                    attention_acc_ls.append(result[key])
                elif 'fully_connected' in key:
                    linear_acc_ls.append(result[key])
            elif 'roc' in key:
                if 'attention' in key:
                    attention_roc_ls.append(result[key])
                elif 'fully_connected' in key:
                    linear_roc_ls.append(result[key])

        self.linear_acc_ls = np.array(linear_acc_ls)
        self.linear_roc_ls = np.array(linear_roc_ls)
        self.attention_acc_ls = np.array(attention_acc_ls)
        self.attention_roc_ls = np.array(attention_roc_ls)

        print('Graph saved to: ', f'{self.output_fig_dir}')
        
        # data load may transform to class methods since together is too large for memmory load all data
            
    def entropy_graph(self, data, label, title, save_file, ):
        _, axes = plt.subplots()
        plt.title(title)
        plt.xlabel('Entropy')
        sns.ecdfplot(data[np.where(label==True)[0]], ax=axes, label="Non-Hallucination", linewidth = 2)
        sns.ecdfplot(data[np.where(label==False)[0]], ax=axes, label="Hallucination", linewidth = 2)
        tmp_path = self.output_fig_dir / save_file
        print(tmp_path)
        plt.legend()
        plt.savefig(str(tmp_path), bbox_inches='tight')

    def pca_graph(self, data, label, title, save_file, ):
        _, axes = plt.subplots()
        plt.title(title)
        mylabels=['Hallucination', 'Non-Hallucination', ]
        sns.scatterplot(x=data[:,0], y=data[:,1], hue=label, ax=axes, alpha=self.alpha)
        tmp_path = self.output_fig_dir / save_file
        print(tmp_path)
        plt.legend(labels=mylabels)
        plt.savefig(str(tmp_path), bbox_inches='tight')

    def tsne_graph(self, data, label, title, save_file, ):
        _, axes = plt.subplots()
        tsne_2d = TSNE(n_components=2).fit_transform(data)
        plt.title(title)
        mylabels=['Hallucination', 'Non-Hallucination', ]
        sns.scatterplot(x=tsne_2d[:,0], y=tsne_2d[:,1], hue=label, ax=axes, alpha=self.alpha)
        tmp_path = self.output_fig_dir / save_file
        print(tmp_path)
        plt.legend(labels=mylabels)
        plt.savefig(str(tmp_path), bbox_inches='tight')

    def roc_graph(self, data, title, save_file):
        pass

    def plot_curev(self):
        self.entropy_graph(
            self.curve_data['first_attribute_entropy'],
            self.curve_data['correct'],
            'Cumulative Distribution of Entropy of Integrated Gradients of Input Tokens',
            'ecdf_ig.png'
        )
        self.entropy_graph(
            self.curve_data['first_logits_entropy'],
            self.curve_data['correct'],
            'Cumulative Distribution of Entropy of Softmax of first token',
            'ecdf_first_tok_softmax.png'
        )
        self.entropy_graph(
            self.curve_data['last_logits_entropy'],
            self.curve_data['correct'],
            'Cumulative Distribution of Entropy of Softmax for last token',
            'ecdf_last_tok_softmax.png'
        )

        # print(len(self.curve_data['first_attribute_entropy']))
        # print(self.curve_data['first_attribute_entropy'][0].shape)
        # print(self.curve_data['first_attribute_entropy'][0])
        # print(len(self.curve_data['correct']))
        # print(self.curve_data['correct'])
        
        # entropy = sp.stats.entropy(self.curve_data['first_token_layer_activations'] - self.curve_data['first_token_layer_activations'].min(axis=0), axis=1)
        # print(entropy.shape)
        # print(entropy)
        # print(self.curve_data['first_token_layer_activations'].min(axis=0).shape)
        # print((self.curve_data['first_token_layer_activations'] - self.curve_data['first_token_layer_activations'].min(axis=0)).shape)

        # self.entropy_graph(
        #     entropy,
        #     self.curve_data['correct'],
        #     'Cumulative Distribution of Entropy of layer activations for first token',
        #     'ecdf_first_tok_linear.png'
        # )
        
        # ensure non-negative
        entropy = sp.stats.entropy(self.curve_data['first_token_layer_activations'] - self.curve_data['first_token_layer_activations'].min(axis=1, keepdims=True), axis=1)
        self.entropy_graph(
            entropy,
            self.curve_data['correct'],
            'Cumulative Distribution of Entropy of layer activations for first token origin',
            'ecdf_first_tok_linear.png'
        )
        
        # ensure non-negative
        entropy = sp.stats.entropy(self.curve_data['first_token_layer_attention'] - self.curve_data['first_token_layer_attention'].min(axis=1, keepdims=True), axis=1)
        self.entropy_graph(
            entropy,
            self.curve_data['correct'],
            'Cumulative Distribution of Entropy of layer attentions for first token origin',
            'ecdf_first_tok_attention.png'
        )

        # _, axes = plt.subplots()

        # plt.title('Cumulative Distribution of Entropy of Integrated Gradients of Input Tokens')
        # plt.xlabel('Entropy')
        # correct = self.curve_data['correct']
        # entropy = self.curve_data['first_attribute_entropy']
        # sns.ecdfplot(entropy[np.where(correct==True)[0]], ax=axes, label="Non-Hallucination", linewidth = 2)
        # sns.ecdfplot(entropy[np.where(correct==False)[0]], ax=axes, label="Hallucination", linewidth = 2)
        # tmp_path = self.output_fig_dir / 'ecdf_ig.png'
        # print(tmp_path)
        # plt.legend()
        # plt.savefig(str(tmp_path), bbox_inches='tight')
        # plt.cla()

        # plt.title('Cumulative Distribution of Entropy of Softmax of first token')
        # plt.xlabel('Entropy')
        # entropy = self.curve_data['first_logits_entropy']
        # sns.ecdfplot(entropy[np.where(correct==True)[0]], ax=axes, label="Non-Hallucination", linewidth = 2)
        # sns.ecdfplot(entropy[np.where(correct==False)[0]], ax=axes, label="Hallucination", linewidth = 2)
        # tmp_path = self.output_fig_dir / 'ecdf_softmax.png'
        # print(tmp_path)
        # plt.legend()
        # plt.savefig(str(tmp_path), bbox_inches='tight')
        # plt.cla()

    def plot_scatter(self):
        self.pca_graph(
            self.curve_data['first_logit_decomp'],
            self.curve_data['correct'],
            'PCA Clustering of Softmax output for first token',
            'pca_softmax_first_tok.png'
        )

        self.pca_graph(
            self.curve_data['last_logit_decomp'],
            self.curve_data['correct'],
            'PCA Clustering of Softmax output for last token',
            'pca_softmax_last_tok.png'
        )

        self.pca_graph(
            self.curve_data['first_token_layer_activations'],
            self.curve_data['correct'],
            'PCA Clustering of Final Linear for First Token',
            'pca_linear_first_tok.png'
        )

        self.pca_graph(
            self.curve_data['final_token_layer_activations'],
            self.curve_data['correct'],
            'PCA Clustering of Final Linear for last token',
            'pca_linear_last_tok.png'
        )

        self.tsne_graph(
            self.scatter_data['final_linear'],
            self.curve_data['correct'],
            'TSNE clustring of Final Linear for Last Token',
            'tsne_linear_last_tok.png',
        )

        self.tsne_graph(
            self.scatter_data['final_attr'],
            self.curve_data['correct'],
            'TSNE clustring of IG for Last Token',
            'tsne_IG.png',
        )

        self.tsne_graph(
            self.scatter_data['final_softmax'],
            self.curve_data['correct'],
            'TSNE clustring of Final Softmax for Last Token',
            'tsne_softmax_last_tok.png',
        )

        self.tsne_graph(
            self.scatter_data['final_attention'],
            self.curve_data['correct'],
            'TSNE clustring of Final Attention for Last Token',
            'tsne_attention_last_tok.png',
        )



    def plot_bar(self):
        
        plt.clf()
        plt.ylim(0,1)
        plt.title(f'Hallucination Detection Accuracy for {self.model_name}')
        plt.xlabel('Accuracy')
        plt.ylabel('layers')
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline (random)')
        plt.plot(self.linear_acc_ls, label='Linear')
        plt.plot(self.attention_acc_ls, label='Attention')
        plt.legend()
        tmp_path = self.output_fig_dir / 'Accuracy.png'
        print(tmp_path)
        plt.savefig(str(tmp_path), bbox_inches='tight')
        
        plt.clf()
        plt.ylim(0,1)
        plt.title(f'Hallucination Detection ROCAUC for {self.model_name}')
        plt.xlabel('ROCAUC')
        plt.ylabel('layers')
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline (random)')
        plt.plot(self.linear_roc_ls, label='Linear')
        plt.plot(self.attention_roc_ls, label='Attention')
        plt.legend()
        tmp_path = self.output_fig_dir / 'ROCAUC.png'
        print(tmp_path)
        plt.savefig(str(tmp_path), bbox_inches='tight')

