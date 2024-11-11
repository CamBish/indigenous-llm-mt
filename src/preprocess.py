import argparse
import os

from typing import Literal

from utils import SOURCE_LANGUAGES, SPLITS, GOLD_STANDARD_MODES
from utils import serialize_parallel_corpus, serialize_gold_standards

DATA_MODES = Literal['parallel_corpus', 'gold_standard']

project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
dotenv_path = os.path.join(project_dir, ".env")

#TODO add code to preprocess and serialize data on remote cluster


#TODO add preprocessing code for data -- original pipeline has not worked well
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Define parameters for preprocessing and serializing data")
    parser.add_argument('source_language', type=SOURCE_LANGUAGES, default='inuktitut', help='Language to perform experiments on')
    input_path_help_string = '''
    Path to data files. Format depepnds on whether importing Inuktitut or Cree data. 
    Inuktitut data should point to data folder, with the file prefix for desired split. Ex: /Nunavut-Hansard-Inuktitut-English-Parallel-Corpus-3.0/split/test
    Cree data should point to parent directory of Cree data. Ex: /Plains-Cree-Corpora/PlainsCree
    '''
    parser.add_argument('input_path', type=str, help=input_path_help_string)
    parser.add_argument('output_dir', type=str, help='Directory to store serialized data. Output file will be named based on relevant flags.')
    parser.add_argument('-s', '--split', type=SPLITS, help='Split for loading Inuktitut data')
    parser.add_argument('-d', '--data_mode', type=DATA_MODES, default='parallel_corpus', help='What type of data to load and serialize. Only for Inuktitut.')
    parser.add_argument('-gs','--gold_standard_mode', type=GOLD_STANDARD_MODES, default='consensus', help="For Gold Standards, select which type to load. Only for Inuktitut. Requires data_mode to be 'gold_standard'")

    args = parser.parse_args()

    # make output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.source_language == 'inuktitut':
        if args.data_mode == 'parallel_corpus':
            output_path = f"{args.output_dir}/{args.source_language}-{args.data_mode}-{args.split}.parquet"
            serialize_parallel_corpus(input_path=args.input_path, output_path=output_path, split=args.split, language_mode=args.source_language)
        elif args.data_mode == 'gold_standard':
            output_path = f"{args.output_dir}/{args.source_language}-{args.data_mode}-{args.gold_standard_mode}.parquet"
            serialize_gold_standards(input_path=args.input_path, output_path=output_path, mode=args.gold_standard_mode)

    if args.source_language == 'cree':
        output_path = f"{args.output_dir}/{args.source_language}-{args.data_mode}.parquet"
        serialize_parallel_corpus(input_path=args.input_path, output_path=output_path, language_mode=args.source_language)