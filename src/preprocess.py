from utils import serialize_parallel_corpus


project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
dotenv_path = os.path.join(project_dir, ".env")
load_dotenv(dotenv_path)

#TODO add code to preprocess and serialize data on remote cluster
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Define parameters for preprocessing and serializing data")
    parser.add_argument('source_language', type=str, default='Inuktitut', choices=['Inuktitut', 'Cree'] , help='Language to perform experiments on')
    data_path_help_string = '''
    Path to data files. Format depepnds on whether importing Inuktitut or Cree data. 
    Inuktitut data should point to data folder, with the file prefix for desired split. Ex: /Nunavut-Hansard-Inuktitut-English-Parallel-Corpus-3.0/split/test
    Cree data should point to parent directory of Cree data. Ex: /Plains-Cree-Corpora/PlainsCree
    '''
    parser.add_argument('data_path', type=str, help=data_path_help_string)