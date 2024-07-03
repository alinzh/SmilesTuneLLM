import os

import torch
from torch.utils.data import DataLoader

from xLSTM.tokenizer import make_tokenizer
from xLSTM.model import xLSTM, Generator
from xLSTM.smiles_dataset import SmilesDataset


os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def run(
    cfg_path: str,
    is_fine_tuning: bool = False,
    is_generating: bool = True,
    dataset_path: str = None,
    num_answers: int = 500,
    num_epoch: int = 5
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
            train_ds, batch_size=256, shuffle=False
        )

        model = xLSTM(cfg_path)
        model.train(train_loader, num_epoch)

    if is_generating:
        model_gen = Generator(cfg_path)
        results = model_gen.generate(start_token="[PAD]", num_answers=num_answers)
        [print(result) for result in results]
        with open(f"examples/out_500_2.txt", "w") as f:
            for res in results:
                f.write(res + "\n")


if __name__ == "__main__":
    data_path = "/data/alina_files/projects/2/SmilesTuneLLM/data/chembl_alpaca.txt"
    cfg_path = "/data/alina_files/projects/2/SmilesTuneLLM/xLSTM/cfg/mLSTM_cfg.yaml"
    run(
        cfg_path, is_fine_tuning=False, is_generating=True, dataset_path=data_path
    )
