import os
import sys
sys.path.append(os.getcwd())
import math

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from xLSTM.tokenizer import make_tokenizer
from xLSTM.model import EncoderDecoder, Generator, AutoEncoder
from xLSTM.smiles_dataset import SmilesDataset
from xLSTM.metrics import metrics

os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def run_auto_encoder(
        conf_encoder: str, conf_decoder: str, dataset_path: str, 
        epoch: int = 1, is_fine_tuning: bool = True, is_generating: bool = False,
        load_predtrain: bool = True
        ):
    """
    Run training and generation for mLSTM-AutoEncoder
    """
    if is_fine_tuning:
        tokenizer = make_tokenizer(
            data_path=dataset_path, max_length=128, vocab_size=600
        )
        train_ds = SmilesDataset(tokenizer, dataset_path)
        dataloader = DataLoader(
            train_ds, batch_size=100, shuffle=True
        )
        model = AutoEncoder(conf_encoder, conf_decoder).to(DEVICE)

        if load_predtrain:
            model_dict_pred_train = torch.load("xLSTM/weights/xlstm_autoencoder.pth", map_location=torch.device(DEVICE))
            model_dict = model.state_dict()
            dict_matched = [i for i,k in zip(model_dict_pred_train,model_dict) if model_dict_pred_train[i].shape==model_dict[k].shape]
            test_dict = {i:model_dict_pred_train[i] for i in dict_matched}
            model_dict.update(test_dict)
            model.load_state_dict(model_dict)

        cnt, mean_loss = 0, 0

        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, weight_decay=5e-4)

        for i in tqdm(range(epoch), desc="Epochs"):
            for _, inputs in enumerate(dataloader):
                model.train(), optimizer.zero_grad()

                inputs = inputs.to(DEVICE)
                reconstructed, kld = model(inputs)

                if math.isnan(model.state_dict()['encoder.xlstm.xlstm_block_stack.blocks.2.xlstm.proj_down.weight'][0][0].item()):
                    print('FOUND NAN in weights of model')
                    break
                if math.isnan(reconstructed[0][0][0].item()):
                    print('FOUND NAN in weights of model')
                    continue

                loss, recon_loss, kld = metrics.loss_function(reconstructed, inputs, kld, beta=1.0)

                loss.backward()
                optimizer.step()

                mean_loss += loss.item()
                cnt += 1
                
                if cnt % 10 == 0:
                    print(
                        f"--------Mean loss for step {cnt} is {mean_loss / cnt}, Recon_loss id {recon_loss}, KLD is {kld}--------"
                    )
                    torch.save(model.state_dict(), "xLSTM/weights/xlstm_autoencoder.pth")
    if is_generating:
        decode_store = []
        model = AutoEncoder(conf_encoder, conf_decoder).to(DEVICE)
        model.load_state_dict(torch.load("xLSTM/weights/xlstm_autoencoder.pth", map_location=DEVICE))

        res = model.decoder.generate_new_data(1000)
        tokenizer = make_tokenizer(
            data_path=dataset_path, max_length=64, vocab_size=600
        )
        for r in res:
            sampled_ids = torch.argmax(torch.nn.functional.softmax(r), dim=-1)
            decode_store.append(tokenizer.decode([i for i in sampled_ids.tolist()]))

        with open(f"xLSTM/examples/generated_molstxt", "w") as f:
            for mol in decode_store:
                f.write(mol + "\n")


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

        model = EncoderDecoder(cfg_path)
        model.train(train_loader, num_epoch)

    if is_generating:
        model_gen = Generator(cfg_path)
        results = model_gen.generate(start_token="[PAD]", num_answers=num_answers)
        [print(result) for result in results]
        with open(f"xLSTM/examples/out_74ep.txt", "w") as f:
            for res in results:
                f.write(res + "\n")


if __name__ == "__main__":
    data_path = "/home/alina/data/projects/2/SmilesTuneLLM/xLSTM/chembl_alpaca.txt"
    cfg_path = "xLSTM/cfg//mLSTM_cfg.yaml"

    # run AutoEncoder (mLSTM encoder + mLSTM decoder)
    encod_conf = '/data/alina_files/projects/2/SmilesTuneLLM/xLSTM/cfg/mLSTM_encoder_cfg.yaml'
    decode_conf = '/data/alina_files/projects/2/SmilesTuneLLM/xLSTM/cfg/mLSTM_decoder_cfg.yaml'
    run_auto_encoder(encod_conf, decode_conf, data_path, is_fine_tuning=True)
