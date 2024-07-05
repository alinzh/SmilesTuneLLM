import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
from dacite import Config as DaciteConfig
from dacite import from_dict
from omegaconf import OmegaConf

from torch.utils.data import DataLoader
from tqdm import tqdm
from xLSTM.xlstm.xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig
from xLSTM.tokenizer import make_tokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def make_conf(path: str) -> xLSTMLMModelConfig:
    with open(path, "r", encoding="utf8") as fp:
        config_yaml = fp.read()
    cfg = OmegaConf.create(config_yaml)
    cfg = from_dict(
        data_class=xLSTMLMModelConfig,
        data=OmegaConf.to_container(cfg),
        config=DaciteConfig(strict=True),
    )
    return cfg


class Encoder(nn.Module):
    def __init__(self, conf_encoder, hidden_dim=128, latent_dim=64):
        super(Encoder, self).__init__()
        self.xlstm = xLSTMLMModel(make_conf(conf_encoder)).to(DEVICE)

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = self.xlstm(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, conf_decoder, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = xLSTMLMModel(make_conf(conf_decoder)).to(DEVICE)
        self.output_dim = output_dim

    def forward(self, x):
        output = self.lstm(x)
        return output
    
    def generate_new_data(self, num_samples: int, latent_dim: int = 128):
        with torch.no_grad():  
            # make samples from std
            z = torch.randn(num_samples, latent_dim).to(DEVICE)
            generated_data = self.forward(z)
        return generated_data
       
    
class AutoEncoder(nn.Module):
    def __init__(self, conf_encoder:str, conf_decoder: str):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(conf_encoder, 64)
        self.decoder = Decoder(conf_decoder, 128, 64, 600)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)

    def compute_kl_divergence(self, mu, log_var):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mu, log_var = self.encoder(inputs)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decoder(z)
        kld = self.compute_kl_divergence(mu, log_var)
        return reconstructed, kld



class EncoderDecoder():
    def __init__(self, conf_path: str, path_to_weights: str = "xLSTM/weights/xlstm_parms.pth"):
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
        torch.save(self.model.state_dict(), "xLSTM/weights/xlstm_parms.pth")


class Generator(EncoderDecoder):
    """Generator of sequence"""
    def __init__(self, cfg: str, weights_path: str = "xLSTM/weights/xlstm_parms.pth"):
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
