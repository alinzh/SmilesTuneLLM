import os
import sys
sys.path.append(os.getcwd())

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn as nn

from xLSTM.tokenizer import make_tokenizer
from xLSTM.model import xLSTM, Generator, AutoEncoder
from xLSTM.smiles_dataset import SmilesDataset


os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def run_auto_encoder(conf_encoder, conf_decoder, dataset_path, epoch=10):
    tokenizer = make_tokenizer(
        data_path=dataset_path, max_length=128, vocab_size=600
    )
    train_ds = SmilesDataset(tokenizer, dataset_path)
    dataloader = DataLoader(
        train_ds, batch_size=1, shuffle=False
    )

    model = AutoEncoder(conf_encoder, conf_decoder).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    cnt, mean_loss = 0, 0

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0006, weight_decay=5e-4)

    for i in tqdm(range(epoch), desc="Epochs"):
        for batch_num, inputs in enumerate(dataloader):
            model.train()
            optimizer.zero_grad()

            inputs = inputs.to(DEVICE)
            outputs = model(inputs)

            # Flatten outputs and inputs for computing loss
            outputs_flat = outputs.view(-1, outputs.size(-1))  # [batch_size * sequence_length, num_tokens]
            inputs_flat = inputs.view(-1)  # [batch_size * sequence_length]

            # Calculate сross-entropy loss
            loss = criterion(outputs_flat, inputs_flat)

            loss.backward()
            optimizer.step()

            mean_loss += loss.item()
            cnt += 1
            
            if cnt % 50 == 0:
                print(
                    f"--------Mean loss for step {batch_num} is {mean_loss / cnt}--------"
                )
                torch.save(model.state_dict(), "xLSTM/weights/xlstm_autoencoder_2.pth")



def run_encoder_decoder(
    cfg_path: str,
    is_fine_tuning: bool = False,
    is_generating: bool = True,
    dataset_path: str = None,
    num_answers: int = 500,
    num_epoch: int = 50
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
            data_path=dataset_path, max_length=128, vocab_size=600
        )
        train_ds = SmilesDataset(tokenizer, dataset_path)
        train_loader = DataLoader(
            train_ds, batch_size=128, shuffle=False
        )

        model = xLSTM(cfg_path)
        model.train(train_loader, num_epoch)

    if is_generating:
        model_gen = Generator(cfg_path)
        results = model_gen.generate(start_token="[PAD]", num_answers=num_answers)
        [print(result) for result in results]
        with open(f"xLSTM/examples/out_74ep.txt", "w") as f:
            for res in results:
                f.write(res + "\n")


if __name__ == "__main__":
    data_path = "/home/alina/data/projects/2/SmilesTuneLLM/xLSTM/cocrys_alpaca.txt"
    cfg_path = "xLSTM/cfg//mLSTM_cfg.yaml"
    # run Encoder-Decoder mLSTM
    # run_encoder_decoder(
    #     cfg_path, is_fine_tuning=True, is_generating=True, dataset_path=data_path
    # )

    # run AutoEncoder (LSTM encoder + mLSTM decoder)
    encod_conf = '/data/alina_files/projects/2/SmilesTuneLLM/xLSTM/cfg/mLSTM_encoder_cfg.yaml'
    decode_conf = '/data/alina_files/projects/2/SmilesTuneLLM/xLSTM/cfg/mLSTM_decoder_cfg.yaml'
    run_auto_encoder(encod_conf, decode_conf, data_path)
