from typing import List

import pandas as pd
import torch
from tokenizers import Tokenizer


class SmilesDataset:
    """Dataset with SMILES structures"""
    def __init__(self, tokenizer: Tokenizer, path: str):
        self.texts = self.formatting_prompts_func(
            pd.read_csv(path)
        )
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        y = self.tokenizer.encode(text).ids
        return torch.tensor(y)

    def formatting_prompts_func(self, examples) -> List[str]:
        """Add service tokens to sequence"""
        outputs = examples["output"]
        texts = []
        for output in outputs:
            text = "[PAD]" + output + "[UNK]"
            texts.append(text)
        return texts