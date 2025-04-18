# llm_hallucinations_reimplementation

This is based on the paper "On Early Detection of Hallucinations in Factual Question Answering". All rights are reserved by the authors of the paper.

# How to use
To download the repo, run: "git clone https://github.com/johnnyzheng0636/llm_hallucinations_reimplementation.git"

Move into the repo: "cd llm_hallucinations_reimplementation"

Initialize Conda on Superpod run: "conda"

If not exist, follow the instruction, else process

To set up environment, run: source setup.sh

To collect data and train the hallucination classifier, run: "python main.py" To run with GPU on SuperPod run "sbatch test.sbatch"

To plot graph, run: "python graph_cpu.py"

Notice that the parameter of graph_cpu.py and main.py should be identical for the same dataset and model

e.g

```
python main.py --model open_llama_7b --dataset capitals
python graph_cpu.py --model open_llama_7b --dataset capitals
```
