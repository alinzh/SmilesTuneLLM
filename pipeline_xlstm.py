import sys
import os
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer, trainers
import pandas as pd
from tqdm import tqdm
import pickle

sys.path.append('..')
from omegaconf import OmegaConf
from dacite import from_dict
from dacite import Config as DaciteConfig
import torch
import torch.nn as nn

from xlstm.xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def make_tokenizer(data_path: str, max_length=64, vocab_size=600, path_to_saved=False) -> Tokenizer:
    if path_to_saved:
        with open(path_to_saved, 'rb') as f:
            tokenizer = pickle.load(f)
            return tokenizer

    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.enable_truncation(max_length=max_length)
    tokenizer.enable_padding(direction='right', length=max_length)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2, special_tokens=['[PAD]', '[UNK]']
    )

    df = pd.read_csv(data_path)
    tokenizer.train_from_iterator(df['output'], trainer)
    with open('./tokenizer.pickle', 'wb') as f:
        pickle.dump(tokenizer, f)

    return tokenizer


class Dataset:
    def __init__(self):
        self.texts = self.formatting_prompts_func(
            pd.read_csv('/home/alina/data/projects/SmilesTuneLLM/chembl_alpaca.txt'))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        y = self.tokenizer.encode(text).ids
        return torch.tensor(y)

    def formatting_prompts_func(self, examples):
        outputs = examples["output"]
        texts = []
        for output in outputs:
            text = '[PAD]' + output + '[UNK]'
            texts.append(text)
        return texts


class xLSTM():
    def __init__(self, cfg):
        super().__init__()
        self.model = xLSTMLMModel(cfg).to(DEVICE)

    def train(self, dataloader, epoch=5, total_steps=50):
        cnt = 0
        mean_loss = 0

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.0006479739574204421, weight_decay=5e-4
        )

        for i in tqdm(range(epoch), desc="Epochs"):
            for batch_num, inputs in enumerate(dataloader):
                self.model.train()
                optimizer.zero_grad()

                inputs = inputs.to(DEVICE)
                outputs = self.model(inputs)

                # Move for one token in right to predict next
                targets = inputs[:, 1:].contiguous().view(-1)
                outputs = outputs[:, :-1].contiguous().view(-1, outputs.size(-1))

                loss = nn.functional.cross_entropy(outputs, targets)
                loss.backward()
                optimizer.step()
                mean_loss += loss.item()
                cnt += inputs.size(0)

                print(f"--------Mean loss for step {(batch_num + 1) * (i + 1)} is {mean_loss / cnt}--------")
                # if batch_num >= total_steps:
                #     return

    def saver(self):
        torch.save(model.model.state_dict(), "./weights/xlstm_parms.pth")


if __name__ == "__main__":
    data_path='/data/alina_files/projects/SmilesTuneLLM/chembl_alpaca copy.txt'
    xlstm_cfg =""" 
    vocab_size: 600
    context_length: 64      
    num_blocks: 24
    embedding_dim: 64
    tie_weights: false
    weight_decay_on_embedding: false
    mlstm_block:
      mlstm:
        conv1d_kernel_size: 4
        qkv_proj_blocksize: 4
        num_heads: 4
    """
    cfg = OmegaConf.create(xlstm_cfg)
    cfg = from_dict(data_class=xLSTMLMModelConfig, data=OmegaConf.to_container(cfg), config=DaciteConfig(strict=True))

    tokenizer = make_tokenizer(
        data_path=data_path, max_length=64, vocab_size=600,
        path_to_saved='/data/alina_files/projects/SmilesTuneLLM/xlstm/experiments/tokenizer.pickle'
    )

    train_ds = Dataset()
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=False)

    model = xLSTM(cfg)
    model.train(train_loader)
    model.saver()