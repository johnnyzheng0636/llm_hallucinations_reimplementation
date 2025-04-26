# This demo prints out the input, output and if it is halluciantion
# equivalent to the main.py but only run the classifier for 1 data, load exist model and print all content

import argparse
import os

from myutil import trexLoad, hookLayer, haluClassify, baseline

# This main perfomre all job can be boosted by GPU
def main():
    parser = argparse.ArgumentParser()

    # add choices later
    # model available:
    # model_num_layers = {
    #         "falcon-40b" : 60,
    #         "falcon-7b" : 32,
    #         "open_llama_13b" : 40,
    #         "open_llama_7b" : 32,
    #         "opt-6.7b" : 32,
    #         "opt-30b" : 48,
    #     }
    # 
    # please only use model < 10b due to hardware limitation
    # To add new model, on need to hardcode new model_num_layers and self.model_repos in
    # ./myutil/hookLayer.py
    parser.add_argument('--model', type=str, default='open_llama_7b', help='model to load; for example `open_llama_7b`.')
    parser.add_argument('--dataset', type=str, default='capitals', help='dataset to load; for example `triviaqa.`')
    # parser.add_argument('--classifier', type=str, default='mlp', help='classifier for the hallucination')
    parser.add_argument('--model_cache', type=str, default='./.cache/models/', help='path to store the model cache')
    parser.add_argument('--trex_dest', type=str, default='./trex/', help='path to store the t-rex dataset')
    parser.add_argument('--data_dir', type=str, default='./data/', help='path to store the csv extracted from t-rex dataset and trivialaqa dataset')
    parser.add_argument('--hidden_data_dir', type=str, default='./results/', help='path to the hidden data')
    parser.add_argument('--out_dir', type=str, default='./outouts/', help='path to the output directory(model, evaluation, figures)')
    
    parser.add_argument('--chunk_sz', type=int, default=50, help='chunk size for the hook layer')
    parser.add_argument('--start', type=int, default=0, help='start index for dataset')
    parser.add_argument('--end', type=int, default=2500, help='end index for dataset')

    parser.add_argument('--cls_lr', type=float, default=1e-4, help='learning rate for the classifier')
    parser.add_argument('--cls_weight_decay', type=float, default=1e-2, help='weight decay for the classifier')
    parser.add_argument('--cls_batch_size', type=int, default=1, help='batch size for the classifier')
    parser.add_argument('--cls_epochs', type=int, default=1, help='number of epochs for the classifier')

    # set to ture when using a new or custom model for classifier
    parser.add_argument('--train_exist', action='store_true', help='if True then must train classifier regardless if a classifier exist')
    # set this flag to run baseline, by default it doesn't run baseline
    # parser.add_argument('--run_baseline', action='store_true', help='if True then run the baseline model')

    parser.add_argument('--seed', type=int, default=42, help='reproducibility')

    args = parser.parse_args()

    demo = True

    # download trex if not found
    if (not os.path.isdir(args.data_dir)) or (not os.path.isdir(args.trex_dest)):
        print('Downloading trex')
        trexLoader = trexLoad.trexLoad(CSV_DEST=args.data_dir, TREX_PATH=args.trex_dest)
        trexLoader.trexToCsv()
    else:
        print('trex found, skipping download')

    # forwardHook = hookLayer.hookLayer(model_name=args.model, dataset_name=args.dataset, start=0, end=5, chunk_sz=3,)
    forwardHook = hookLayer.hookLayer(
        model_name=args.model, 
        dataset_name=args.dataset, 
        data_dir=args.data_dir,
        model_dir=args.model_cache,
        results_dir=args.hidden_data_dir,
        start=args.start, 
        end=args.end, 
        chunk_sz=args.chunk_sz,
        demo=demo,
    )
    # forwardHook = hookLayer.hookLayer(model_name=args.model, dataset_name=args.dataset, start=0, end=2500, chunk_sz=50,)
    # forwardHook = hookLayer.hookLayer(model_name=args.model, dataset_name=args.dataset)
    # Using hook to collect hidden layer data is slow, takes more than 1 hour for 2500 data
    layerDataPath = forwardHook.demo_mate_data()

    # next train the classifier based on hooke data
    # graph are separeted from this main since GPU din't accelerate ploting
    # the classifier training is rather faster, for 2500 data took less than 10 minutes
    classify = haluClassify.haluClassify(
        layerDataPath, 
        out_dir=args.out_dir,
        # for custom classifier, change the model here
        # cls_IG=haluClassify.defaultGRU, 
        # cls_logit=haluClassify.defaultMLP, 
        # cls_att=haluClassify.defaultMLP, 
        # cls_linear=haluClassify.defaultMLP,
        lr=args.cls_lr,
        weight_decay=args.cls_weight_decay,
        batch_size=args.cls_batch_size, 
        epochs=args.cls_epochs, 
        train_exist=args.train_exist,
        chunk_sz=args.chunk_sz,
        demo=demo,
        dataset=args.dataset,
        llm_name=args.model,
        seed=args.seed,
    )
    classify.train_and_eval()

if __name__ == "__main__":
    main()