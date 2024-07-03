import os
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer, trainers
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
import pickle
from typing import List
from omegaconf import OmegaConf
from dacite import from_dict
from dacite import Config as DaciteConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from xlstm.xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def make_tokenizer(
    data_path: str = None, max_length: int = 64, vocab_size: int = 600, path_to_saved: str = "./tokenizer.pickle"
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


class xLSTM:
    def __init__(self, conf_path: str):
        super().__init__()
        with open(conf_path, "r", encoding="utf8") as fp:
            config_yaml = fp.read()
        cfg = OmegaConf.create(config_yaml)
        cfg = from_dict(
            data_class=xLSTMLMModelConfig,
            data=OmegaConf.to_container(cfg),
            config=DaciteConfig(strict=True),
        )
        self.model = xLSTMLMModel(cfg).to(DEVICE)

    def train(self, dataloader: DataLoader, epoch: int = 5):
        """Train model. Every sequence is passed to model separately"""
        cnt = 0
        mean_loss = 0

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.0006, weight_decay=5e-4
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

                if batch_num % 200 == 0:
                    print(
                        f"--------Mean loss for step {(batch_num + 1) * (i + 1)} is {mean_loss / cnt}--------"
                    )
                    self.saver()

        self.saver()

    def saver(self):
        """Save model weights"""
        torch.save(self.model.state_dict(), "./weights/xlstm_parms.pth")


class Generator(xLSTM):
    """Generator of sequence"""
    def __init__(self, cfg: str, weights_path: str = "./weights/xlstm_parms.pth"):
        super().__init__(cfg)
        self.model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        self.model.eval()
        self.tokenizer = make_tokenizer()

    def generate(self, start_token: str, max_length: int = 64, num_answers: int = 1, temperature: float = 0.8):
        """
        Generate sequence

        Parameters
        ----------
        start_token: str
            Served token
        max_length: int
            Max len of sequence
        num_answers: int
            Number of answers
        temperature: float
            Range of values between 0.5 and 2.0.
            The more value the more varied the answers

        Returns
        -------
        store: str
            Generated sequence
        """
        store = []

        with torch.no_grad():
            for num in tqdm(range(num_answers), desc="Answer generation:"):
                generated_sequence = self.tokenizer.encode(start_token).ids[:1]
                input_tensor = torch.tensor(
                    [self.tokenizer.encode(start_token).ids[:1]]
                ).to(DEVICE)
                for _ in range(max_length - 1):
                    output = self.model(input_tensor)
                    logits = output[:, -1, :] / temperature
                    probabilities = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probabilities, num_samples=1).item()
                    generated_sequence.append(next_token)
                    input_tensor = torch.cat(
                        [input_tensor, torch.tensor([[next_token]]).to(DEVICE)], dim=1
                    )
                    if next_token == self.tokenizer.token_to_id(
                        "[UNK]"
                    ):  # [UNK] is the end token
                        break
                store.append(self.tokenizer.decode(generated_sequence))

            return store


def run(
    cfg_path: str,
    is_fine_tuning: bool = False,
    is_generating: bool = True,
    dataset_path: str = None,
    num_answers: int = 100,
    num_epoch: int = 1
):
    """
    Run training or generation

    Parameters
    __________
    cfg_path: str
        Path to config file
    is_fine_tuning: bool
        True if fine-tuning
    is_generating: bool
        True if generating
    dataset_path: str
        Path to dataset
    num_answers: int, optional
        Number of answers for generation
    num_epoch: int, optional
        Number of epochs for training
    """
    if is_fine_tuning:
        tokenizer = make_tokenizer(
            data_path=dataset_path, max_length=64, vocab_size=600
        )
        train_ds = SmilesDataset(tokenizer, dataset_path)
        train_loader = DataLoader(
            train_ds, batch_size=1, shuffle=False
        )

        model = xLSTM(cfg_path)
        model.train(train_loader, num_epoch)

    if is_generating:
        model_gen = Generator(cfg_path)
        results = model_gen.generate(start_token="[PAD]", num_answers=num_answers)
        [print(result) for result in results]
        with open(f"examples/out_2.txt", "w") as f:
            for res in results:
                f.write(res + "\n")


if __name__ == "__main__":
    data_path = "/data/alina_files/projects/2/SmilesTuneLLM/data/chembl_alpaca.txt"
    cfg_path = "/data/alina_files/projects/2/SmilesTuneLLM/xLSTM/cfg/mLSTM_cfg.yaml"
    run(
        cfg_path, is_fine_tuning=False, is_generating=True, dataset_path=data_path
    )
