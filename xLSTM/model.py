import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dacite import Config as DaciteConfig
from dacite import from_dict
from omegaconf import OmegaConf

from torch.utils.data import DataLoader
from tqdm import tqdm
from xlstm.xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig
from xLSTM.tokenizer import make_tokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class xLSTM():
    def __init__(self, conf_path: str, path_to_weights: str = "./weights/xlstm_parms.pth"):
        with open(conf_path, "r", encoding="utf8") as fp:
            config_yaml = fp.read()
        cfg = OmegaConf.create(config_yaml)
        cfg = from_dict(
            data_class=xLSTMLMModelConfig,
            data=OmegaConf.to_container(cfg),
            config=DaciteConfig(strict=True),
        )
        self.model = xLSTMLMModel(cfg).to(DEVICE)
        if path_to_weights:
            self.model.load_state_dict(torch.load(path_to_weights))

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

    def generate(self, start_token: str, max_length: int = 128, num_answers: int = 1, temperature: float = 0.8):
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
