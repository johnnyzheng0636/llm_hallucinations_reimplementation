from myutil import visualization
from pathlib import Path
import argparse

# This main perfomre all job can be boosted by GPU
def main():
    parser = argparse.ArgumentParser()

    # use these to consgruct the path
    parser.add_argument('--model', type=str, default='open_llama_7b', help='model to load; for example `open_llama_7b`.')
    parser.add_argument('--dataset', type=str, default='capitals', help='dataset to load; for example `triviaqa.`')
    # parser.add_argument('--classifier', type=str, default='mlp', help='classifier for the hallucination')
    # parser.add_argument('--trex_dest', type=str, default='./trex/', help='path to store the t-rex dataset')
    # parser.add_argument('--csv_dest', type=str, default='./data/', help='path to store the csv extracted from t-rex dataset')
    parser.add_argument('--hidden_data_dir', type=str, default='./results', help='path to the hidden data')
    parser.add_argument('--chunk_sz', type=int, default=50, help='chunk size for the hook layer')
    parser.add_argument('--start', type=int, default=0, help='start index for dataset')
    parser.add_argument('--end', type=int, default=2500, help='end index for dataset')

    # sometime mem too large >170 GB after scatter and curve done, the process is killed immediately
    # need to print those not run yet
    parser.add_argument('--bar_only', action='store_true', help='if True then only print accuracy and ROCAUC curve')

    args = parser.parse_args()

    # construct the path to hidden data
    input_dir = f'{args.hidden_data_dir}/{args.model}_{args.chunk_sz}chunk_{args.dataset}_{args.start}-{args.end}'
    print(f'input from to: {input_dir}')

    g = visualization.visualization(
        input_dir, 
        only_bar = args.bar_only,
    )
    if not args.bar_only:
        g.plot_curev()
        g.plot_scatter()
    g.plot_bar()


if __name__ == "__main__":
    main()