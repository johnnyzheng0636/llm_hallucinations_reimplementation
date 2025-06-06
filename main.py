import argparse
import os
import sys

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
    parser.add_argument('--model', type=str, default='open_llama_7b', help='model to load; for example `open_llama_7b`. Available: ' \
        '`falcon-40b`, `falcon-7b`, `open_llama_13b`, `open_llama_7b`, `Llama-3.1-8B`, `Llama-3.2-3B`, `Llama-3.2-1B`, `gemma-3-4b-it`, `opt-6.7b`, `opt-30b`.\n' \
        'Notice Large model above 13B is no tested due to hardware constraint, it is there just for the sake of completeness of reimplenmentation')
    parser.add_argument('--dataset', type=str, default='capitals', help='dataset to load; for example `triviaqa`. Available: ' \
        '`capitals`, `trivia_qa`, `place_of_birth`, `founders`, `combined`.')
    # parser.add_argument('--classifier', type=str, default='mlp', help='classifier for the hallucination')
    parser.add_argument('--model_cache', type=str, default='./.cache/models/', help='path to store the model cache')
    parser.add_argument('--trex_dest', type=str, default='./trex/', help='path to store the t-rex dataset')
    parser.add_argument('--data_dir', type=str, default='./data/', help='path to store the csv extracted from t-rex dataset and trivialaqa dataset')
    parser.add_argument('--hidden_data_dir', type=str, default='./results/', help='path to the hidden data')
    parser.add_argument('--out_dir', type=str, default='./outouts/', help='path to the output directory(model, evaluation, figures)')
    
    parser.add_argument('--chunk_sz', type=int, default=50, help='chunk size for the hook layer')
    parser.add_argument('--start', type=int, default=0, help='start index for dataset')
    parser.add_argument('--end', type=int, default=2500, help='end index for dataset')

    parser.add_argument('--start_b', type=int, default=0, help='start index for dataset for baseline')
    parser.add_argument('--end_b', type=int, default=50, help='end index for dataset for baseline')

    parser.add_argument('--cls_lr', type=float, default=1e-4, help='learning rate for the classifier')
    parser.add_argument('--cls_weight_decay', type=float, default=1e-2, help='weight decay for the classifier')
    parser.add_argument('--cls_batch_size', type=int, default=128, help='batch size for the classifier')
    parser.add_argument('--cls_epochs', type=int, default=1000, help='number of epochs for the classifier')

    # set to ture when using a new or custom model for classifier
    # TODO 
    # modify into a flag for forcely redo all 
    parser.add_argument('--overwrite_all', action='store_true', help='if True then overwrite all saved files if exist')
    parser.add_argument('--overwrite_data', action='store_true', help='if True then only overwrite hidden date collected by forward hook if exist')
    parser.add_argument('--overwrite_cls', action='store_true', help='if True then only overwrite classifier if exist')

    parser.add_argument('--model_statistic', action='store_true', help='if True then only fetch model statistic')
    # set this flag to run baseline, by default it doesn't run baseline
    parser.add_argument('--run_baseline', action='store_true', help='if True then run the baseline model')

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # download trex if not found
    if (not os.path.isdir(args.data_dir)) or (not os.path.isdir(args.trex_dest)):
        print('Downloading trex')
        trexLoader = trexLoad.trexLoad(CSV_DEST=args.data_dir, TREX_PATH=args.trex_dest)
        trexLoader.trexToCsv()
    else:
        print('trex found, skipping download')

    layerDataPath = "combined"
    if args.dataset != "combined":
        overwrite_flag = False
        if args.overwrite_all or args.overwrite_data:
            overwrite_flag = True
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
            train_exist=overwrite_flag,
        )
        # forwardHook = hookLayer.hookLayer(model_name=args.model, dataset_name=args.dataset, start=0, end=2500, chunk_sz=50,)
        # forwardHook = hookLayer.hookLayer(model_name=args.model, dataset_name=args.dataset)
        # Using hook to collect hidden layer data is slow, takes more than 1 hour for 2500 data
        layerDataPath = forwardHook.save_results()

        
        # SelfCheckGpt is very slow due to it's incontext learning, 50 data takes more than 1 hour
        # so we limit the data used with start and enc
        if args.run_baseline:
            my_baseline = baseline.SelfCheckGpt(
                model_name=args.model, 
                dataset_name=args.dataset, 
                data_dir=args.data_dir,
                model_dir=args.model_cache,
                # results_dir=args.hidden_data_dir,
                out_dir=args.out_dir,
                start=args.start_b, 
                end=args.end_b,
            )
            my_baseline.run()

    overwrite_flag = False
    if args.overwrite_all or args.overwrite_cls:
        overwrite_flag = True
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
        chunk_sz=args.chunk_sz,
        lr=args.cls_lr,
        weight_decay=args.cls_weight_decay,
        batch_size=args.cls_batch_size, 
        epochs=args.cls_epochs, 
        train_exist=overwrite_flag,
        model_statistic=args.model_statistic,
        dataset=args.dataset,
        start=args.start,
        end=args.end,
        llm_model_name=args.model,
        hidden_data_dir=args.hidden_data_dir,
    )
    classify.train_and_eval()

if __name__ == "__main__":
    main()