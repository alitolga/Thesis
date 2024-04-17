# SentEval imports
from __future__ import absolute_import, division

import os
import sys
import logging

# Transformers (HuggingFace) imports
import argparse
import torch
from transformers import AutoModelForCausalLM, DebertaV2ForMaskedLM, AutoTokenizer
from transformers import AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoModelForTokenClassification
from transformers import DebertaV2ForQuestionAnswering, DebertaV2ForSequenceClassification, DebertaV2ForTokenClassification

# Peft (HuggingFace) imports
from peft import PeftConfig, PeftModel, TaskType
from datasets import load_dataset


# Set PATHs
PATH_TO_SENTEVAL = 'SentEval/'
PATH_TO_DATA = 'SentEval/data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)

# Get the absolute path to the Senteval directory
# current_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where the script is located
# path_to_senteval = os.path.join(current_dir, 'SentEval')
# sys.path.insert(0, path_to_senteval)
import senteval


# SentEval prepare and batcher
def prepare(params, samples):
    return

def batcher(params, batch):
    se_sentences = [sent if sent != [] else ['.'] for sent in batch]
    sentences_str = [' '.join(sentence) for sentence in se_sentences]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(sentences_str, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    embeddings = torch.mean(hidden_states[0], dim=1)
    return embeddings


# Command line Arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default='facebook/opt-350m', help="base model name", 
        choices=['bert-base-uncased', 'gpt2', 'bart-base', 'roberta-base', 'deberta-base', 'deberta-v3-base', 'electra-base-generator'] )
    return parser.parse_args()    
    

if __name__ == "__main__":
    # Parse the command line arguments
    args = parse_args()

    # Initialize the repository name to fetch the corresponding model from HuggingFace
    repo_name = "alitolga/" + args.base_model + "-large-peft"

    # Set the corresponding Peft config and model for that reponame
    config = PeftConfig.from_pretrained(repo_name)
    if args.base_model == 'deberta-base' or args.base_model == 'deberta-v3-base':
        base_model = DebertaV2ForMaskedLM.from_pretrained(config.base_model_name_or_path)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, repo_name)

    # Specify the dataset name
    dataset_name = "helena-balabin/pereira_fMRI_sentences"

    # Specify the path to save or load the dataset
    save_path = "./data"

    # Load the dataset, use the cache if available
    dataset = load_dataset(dataset_name, cache_dir=save_path)
    sentences = dataset["train"]["sentences"]
    sentences = sentences[0] # 0th subject
    

    # Set params for SentEval
    params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
    params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                    'tenacity': 3, 'epoch_size': 2}

    # Set up logger
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      'Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    results = se.eval(transfer_tasks)
    print(results)