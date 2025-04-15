import argparse
import os

from myutil import trexLoad, hookLayer, haluClassify, baseline

# This main perfomre all job can be boosted by GPU
def main():
    parser = argparse.ArgumentParser()

    # add choices later
    parser.add_argument('--model', type=str, default='open_llama_7b', help='model to load; for example `open_llama_7b`.')
    parser.add_argument('--dataset', type=str, default='capitals', help='dataset to load; for example `triviaqa.`')
    # parser.add_argument('--classifier', type=str, default='mlp', help='classifier for the hallucination')
    parser.add_argument('--trex_dest', type=str, default='./trex/', help='path to store the t-rex dataset')
    parser.add_argument('--csv_dest', type=str, default='./data/', help='path to store the csv extracted from t-rex dataset')

    args = parser.parse_args()

    # download trex if not found
    if (not os.path.isdir(args.csv_dest)) or (not os.path.isdir(args.trex_dest)):
        print('Downloading trex')
        trexLoader = trexLoad.trexLoad(CSV_DEST=args.csv_dest, TREX_PATH=args.trex_dest)
        trexLoader.trexToCsv()
    else:
        print('trex found, skipping download')

    my_baseline = baseline.SelfCheckGpt(model_name=args.model, dataset_name=args.dataset, start=2, end=4, chunk_sz=10,)
    my_baseline.run()

    # forwardHook = hookLayer.hookLayer(model_name=args.model, dataset_name=args.dataset, start=0, end=5, chunk_sz=3,)
    forwardHook = hookLayer.hookLayer(model_name=args.model, dataset_name=args.dataset, start=0, end=50, chunk_sz=50,)
    # forwardHook = hookLayer.hookLayer(model_name=args.model, dataset_name=args.dataset, start=0, end=2500, chunk_sz=50,)
    # forwardHook = hookLayer.hookLayer(model_name=args.model, dataset_name=args.dataset)
    layerDataPath = forwardHook.save_results()

    # next train the classifier based on hooke data
    # graph are separeted from this main since GPU din't accelerate ploting
    classify = haluClassify.haluClassify(layerDataPath, batch_size=2, epochs=2,)
    classify.train_and_eval()

if __name__ == "__main__":
    main()