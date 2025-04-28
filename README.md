# llm_hallucinations_reimplementation

This is based on the paper "On Early Detection of Hallucinations in Factual Question Answering". All rights are reserved by the authors of the paper.

# How to use
To download the repo, run: `git clone https://github.com/johnnyzheng0636/llm_hallucinations_reimplementation.git`

Move into the repo: `cd llm_hallucinations_reimplementation`

Initialize Conda on Superpod run: `conda`

If not exist, follow the instruction, else process

To set up environment, run: `source setup.sh`

Before runing any python code, please easure the correct virtual environment (venv) is activate, i.e.

```
(hallucination) user_name@host_name:$
```

If the starting `(hallucination)` is missing, run `conda activate hallucination` to activate the correct venv. If success, you will see the starting `(hallucination)`

To collect data and train the hallucination classifier, run: `python main.py`. To run with GPU on SuperPod run `sbatch test.sbatch`

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

To rerun a model for updated code add flag ~~`--train_exist`~~ `--overwrite_all`. This flag will overwrite all  data/files/classifier if exist. Notice this flag is in the new `.sbatch` by default. Without this flag, related process will be skipped if corresponding files exist. So, if you are running the code for the first time for a certain model, remove it. Similarly, if you only want to overwrite the hidden data collected by the forward hook, use only `--overwrite_data` flag, if you only want to overwrite the hallucination classifier trained, use only `--overwrite_cls` flag.

# output

Figures and evaluation can be found in `./outouts`

`./outouts/<model_name>/demo` contains the demo figure

`./outouts/<model_name>/eval` contains the hallucination classifier evaluation, LLM task evaluation and heuristic wait for human evaluation

`./outouts/<model_name>/fig` contains the graphs

`./outouts/<model_name>/model_stat` contains the hallucination classifier model parameters

# Git pull issue

After modifying the test.sbatch, you can't pull directly due to conflict in git tree. You can create a new file called `local.sbatch` and copy the content of `test.sbatch` into it. Notice that `local.sbatch` is in `.gitignore`, so the change is not tracked by git, and modifying it will not cause any conflict in git tree. Now restore the `test.sbatch` to the state record in the last change in the git tree by pressing the button as follow

![Find green or blue beside line number\label{git_tut_1}](./fig/git_tut_1.png)
![Click it and found the undo like an u arrow\label{git_tut_2}](./fig/git_tut_2.png)
![No changes(green or blue) pull now\label{git_tut_3}](./fig/git_tut_3.png)

If no green or blue left, you can pull now
