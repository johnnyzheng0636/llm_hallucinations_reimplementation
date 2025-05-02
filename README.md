# llm_hallucinations_reimplementation

This is based on the paper "On Early Detection of Hallucinations in Factual Question Answering". All rights are reserved by the authors of the paper.

# How to use
To download the repo, run: `git clone https://github.com/johnnyzheng0636/llm_hallucinations_reimplementation.git`

Move into the repo: `cd llm_hallucinations_reimplementation`

Initialize Conda on Superpod run: `conda`

If not exist, follow the instruction given by the os after inputing `conda` or download anaconda, else process

To set up environment, run: `source setup.sh`

Before runing any python code, please easure the correct virtual environment (venv) is activate, i.e.

```
(hallucination) user_name@host_name:$
```

If the starting `(hallucination)` is missing, run `conda activate hallucination` to activate the correct venv. If success, you will see the starting `(hallucination)`

***Important! Ensure you have at least 400GB of available disk space for artifact hidden layer data to be collect. Depending on model used, you may need 20+ TB of storage available. If no enough space is available, the job will be killed when it used up all space. After allocating more space, you can rerun the same command and it will resume from where it is killed last time.***

To collect data and train the hallucination classifier, run: `python main.py`. To run with GPU on SuperPod run `sbatch test.sbatch`, else if GPU is reachable on `pwd`, you can run `python main.py` directly. 

Notice to use `.sbatch` you just modify the python part with different parameters and switch between `main.py`, `graph.cpu.py`, and `demo.py`. The `main.py` must run on GPU, while `graph.cpu.py` and `demo.py` can run on CPU.

To avoid the [pull issues](#git-pull-issue), you can create a new file `local.sbatch`, which is in `.gitignore` already, and modify it as your will.

To plot graph, run: `python graph_cpu.py` 

Notice that the parameter of graph_cpu.py and main.py should be identical for the same dataset and model

e.g

```
python main.py --model open_llama_7b --dataset capitals 
python graph_cpu.py --model open_llama_7b --dataset capitals
```

Notice flag should be ignored (those with out parameters)

e.g

```
python main.py --model open_llama_7b --dataset capitals --train_exist --run_baseline
python graph_cpu.py --model open_llama_7b --dataset capitals
```

~~Also, to plot new figures, we need to manually delete the old figure in `./outouts`. This can't be done with code, because some model have more than 180 GB data for plotting and cause forced exit halfway due to memory limitation. To avoid this, the `graph_cpu.py` have a forced fallback for this memory problem. Hence we need an empty `./outouts/<model_name>/fig` to plot all new figures.~~

To address the job killed due to memory used up problem for `graph_cpu.py`, please manually check which part is missing and run only that part using the corresponding flag. `--bar_only` only run the Accuracy and ROCAUC curve. `--curve_only` only run the entropy curve. `scatter_only` only run on TSNE and PCA dimension reduction graph. If the problem still presist, try use GPU for ploting with `.sbatch`

To run a demo of halluccination classifier. run: `python demo.py` with the same parameters as `python main.py`

To rerun a model for updated code add flag ~~`--train_exist`~~ `--overwrite_all`. This flag will overwrite all  data or files or classifier if exist. Notice this flag is in the new `.sbatch` by default. Without this flag, related process will be skipped if corresponding files exist. So, if you are running the code for the first time for a certain model, remove it. Similarly, if you only want to overwrite the hidden data collected by the forward hook, use only `--overwrite_data` flag, if you only want to overwrite the hallucination classifier trained, use only `--overwrite_cls` flag.

To train classifier on all 4 dataset, after you finished all 4 dataset, run `main.py` for the 5th time using dataset arg `--combined`

# Usage of each files

The ./myutil custom Python package forms the backbone of the implementation, with each module addressing specific aspects of hallucination detection in large language models (LLMs).  

The hookLayer.py module is a critical utility designed to extract hidden-layer activations and self-attention scores from PyTorch models. By leveraging PyTorch’s forward_hook mechanism, it intercepts and caches intermediate states—such as outputs from fully connected layers or attention weights—during model inference. This capability enables granular analysis of internal model dynamics, which is essential for identifying hallucination artifacts. 

Building on this, haluClassify.py trains classifiers to detect hallucinations using the extracted layer data. It implements architectures like Gated Recurrent Units (GRUs) and Multi-Layer Perceptrons (MLPs), optimized to correlate layer-wise activations with hallucination labels, achieving high detection accuracy. 

For benchmarking, baseline.py evaluates the performance of our artifact-based classifiers against SelfCheckGPT, a state-of-the-art baseline. The comparison uses SelfCheckGPT’s default configuration, and the setting of generates 20 random passages per query to ensure statistically robust results. 

The trexLoad.py module automates the preprocessing of the TREX dataset. It first verifies the dataset’s presence in the cache directory, downloading it if necessary, and then extracts three key knowledge domains: capitals, founders, and places of birth. These structured facts are transformed into question-answer pairs, stored as CSV files, and used by hookLayer.py to query LLMs and generate responses for analysis. 

Finally, visualization.py generates publication-ready figures to interpret results. These include accuracy and ROCAUC curves for classifiers, t-SNE/UMAP projections of artifact embeddings, token-wise integrated gradients (IG) and softmax entropy across layers, and cumulative entropy distributions for hallucinated versus non-hallucinated outputs. The visualizations not only replicate all graphs from the original paper but also extend them with supplementary analyses. 

To streamline usability, we provide three Command Line Interface (CLI) scripts: main.py, graph_cpu.py, and demo.py 

main.py: Orchestrates the end-to-end pipeline for training hallucination classifiers, integrating dataset preprocessing (via trexLoad.py), activation extraction (hookLayer.py), and model training (haluClassify.py). graph_cpu.py: Generates analytical graphs (e.g., ROCAUV and accuracy curves, entropy visualizations, low dimension visualization of artifacts) from training results, usable for CPU environments to ensure accessibility on resource-constrained hardware. This script complements visualization.py by focusing on post-training insights. demo.py: Demonstrates real-time hallucination classification using trained models, enabling users to interactively test the system. 

Lastly, there are three auxiliary scripts: README.md, setup.sh, and test.sbatch. README.md give a short introduction to the code, including how to use and common mistakes. setup.sh setup the python virtual environment for the project, notice it assumed anaconda is available by default. test.sbatch is for allocating GPU resources for jobs. 

# output

Figures and evaluation can be found in `./outouts`

`./outouts/<model_name>/demo` contains the demo figure

`./outouts/<model_name>/eval` contains the hallucination classifier evaluation, LLM task evaluation and heuristic wait for human evaluation

`./outouts/<model_name>/fig` contains the graphs

`./outouts/<model_name>/model_stat` contains the hallucination classifier model parameters

# OS

We run the codes on HKUST SuperPod, the spec is:

```
Operating System: Ubuntu 22.04.3 LTS              
          Kernel: Linux 5.19.0-45-generic
    Architecture: x86-64
 Hardware Vendor: Dell Inc.
  Hardware Model: PowerEdge R750
```

# Git pull issue

After modifying the test.sbatch, you can't pull directly due to conflict in git tree. You can create a new file called `local.sbatch` and copy the content of `test.sbatch` into it. Notice that `local.sbatch` is in `.gitignore`, so the change is not tracked by git, and modifying it will not cause any conflict in git tree. Now restore the `test.sbatch` to the state record in the last change in the git tree by pressing the button as follow

![Find green or blue beside line number\label{git_tut_1}](./fig/git_tut_1.png)
![Click it and found the undo like an u arrow\label{git_tut_2}](./fig/git_tut_2.png)
![No changes(green or blue) pull now\label{git_tut_3}](./fig/git_tut_3.png)

If no green or blue left, you can pull now
