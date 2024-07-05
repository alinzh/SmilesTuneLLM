import os
import sys
sys.path.append(os.getcwd())

import pickle

import pandas as pd
import torch
from tokenizers import Tokenizer, trainers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace


os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def make_tokenizer(
    data_path: str = None, max_length: int = 128, vocab_size: int = 600, path_to_saved: str = 'xLSTM/tokenizer.pickle'
) -> Tokenizer:
    """
    Create a tokenizer

    Parameters
    ----------
    data_path: str
        Path to data on which training will be carried out
    max_length: int
        Max len of sequence. Can be less than len of max sequence in dataset
    vocab_size: int
        Vocab size
    path_to_saved: str
        Path to saved tokenizer

    Returns
    -------
    tokenizer: Tokenizer
    """
    if path_to_saved:
        with open(path_to_saved, "rb") as f:
            tokenizer = pickle.load(f)
            return tokenizer

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.enable_truncation(max_length=max_length)
    tokenizer.enable_padding(direction="right", length=max_length)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2, special_tokens=["[PAD]", "[UNK]"]
    )

    df = pd.read_csv(data_path)
    tokenizer.train_from_iterator(df["output"], trainer)
    with open(path_to_saved, "wb") as f:
        pickle.dump(tokenizer, f)

    return tokenizer
