"""
Run tuning/inference mLSTM EnocderDecoder, AutoEncoder
"""
import os
import sys
sys.path.append(os.getcwd())

import math

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from xLSTM.tokenizer import make_tokenizer
from xLSTM.model import EncoderDecoder, EncoderDecoderGenerator, AutoEncoder
from xLSTM.smiles_dataset import SmilesDataset
from xLSTM.metrics import metrics
import run_measuring_metrics

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def val(model, tokenizer, parallel: bool):
    model.eval() 
    decode_store = []
    if parallel:
        res = model.module.decoder.generate_new_data(500)
    else:
        res = model.decoder.generate_new_data(500)

    for r in res:
        sampled_ids = torch.argmax(torch.nn.functional.softmax(r), dim=-1)
        decode_store.append(tokenizer.decode([i for i in sampled_ids.tolist()]))

    with open(f"xLSTM/examples/generated_mols.txt", "w") as f:
        for mol in decode_store:
            f.write(mol + "\n")

    valid, nov, dup = run_measuring_metrics.main()
    return valid, nov, dup


def run_auto_encoder(
        conf_encoder: str, conf_decoder: str, dataset_path: str, 
        epoch: int = 30, is_fine_tuning: bool = True, is_generating: bool = False,
        load_predtrain: bool = False, vocab_size: int = 140, max_length: int = 101,
        bs: int = 100, n_answers: int = 1000, parallels: bool = False, chkp_path: str = None, old_step = 0,
        accumulation_steps = 4
        ) -> None:
    """
    Run training and generation for mLSTM-AutoEncoder. Save answer
    in txt format

    Parameters
    __________
    conf_encoder: str
        Path to config file for encoder
    conf_decoder: str
        Path to config file for decoder
    dataset_path: str
        Path to dataset in csv format. SMILES string in column 'output'
    epoch: int
        Number of epoch
    is_fine_tuning: bool
        True if need to tune
    is_generating:
        True if need to run inference
    load_predtrin:
        True if need to load weights
    vocab_size: int
        vocab_size of tokenizer
    max_lenght: int
        Max len of sequence
    bs: int
        Batch size
    n_answers: int
        Number of generated answers
    parallels: bool
        True, if need to parallel between some GPU
    chkp_path: str, optional
        Path to weights, if need to load pretrained model
    """
    if is_fine_tuning:
        tokenizer = make_tokenizer(
            data_path=dataset_path, max_length=max_length, vocab_size=vocab_size,
            path_from='xLSTM/tokenizer.pickle'
        )
        train_ds = SmilesDataset(tokenizer, dataset_path)
        dataloader = DataLoader(
            train_ds, batch_size=bs, shuffle=False
        )
        model = AutoEncoder(conf_encoder, conf_decoder).to(DEVICE)

        if load_predtrain:
            if parallels:
                model_dict_pred_train = torch.load(chkp_path)
                model.load_state_dict(model_dict_pred_train['model_state_dict'])
                model = torch.nn.DataParallel(model)
                model.to(DEVICE)
            else:
                model_dict_pred_train = torch.load(chkp_path, map_location=torch.device(DEVICE))
                model.load_state_dict(model_dict_pred_train['model_state_dict'])

            optimizer = torch.optim.RMSprop(model.parameters())
            optimizer.load_state_dict(model_dict_pred_train['optimizer_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        else:
            optimizer = torch.optim.AdamW(model.parameters(), weight_decay=5e-4)
            if parallels:
                model = torch.nn.DataParallel(model)
                model.to(DEVICE)

        cnt, mean_loss = 0, 0
        old_step = old_step

        for i in tqdm(range(epoch), desc="Epochs"):
            for _, inputs in enumerate(dataloader):
                model.train()

                inputs = inputs.to(DEVICE)
                reconstructed, kld = model(inputs)

                # check for problem with optimizer. Can be remove
                if math.isnan(reconstructed[0][0][0].item()):
                    print('FOUND NAN in output of model')
                    break

                loss, recon_loss, kld = metrics.loss_function(reconstructed, inputs, kld, beta=1.0)

                if parallels:
                    loss = loss.mean()

                loss.backward()
                
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()  
                    optimizer.zero_grad()  

                mean_loss += loss.item() 
                cnt += 1
                old_step += 1
                
                if cnt % 100 == 0:
                    if parallels:
                        print(
                        f"--------Mean loss for step {old_step} is {mean_loss / cnt}, Recon_loss id {recon_loss}, KLD is {kld.mean().item()}--------"
                        )
                        torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.module.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                }, f"xLSTM/weights/xlstm_autoencoder_ep{i+1}_step{old_step}.pth")
                    else:
                        print(
                            f"--------Mean loss for step {old_step} is {mean_loss / cnt}, Recon_loss id {recon_loss}, KLD is {kld}--------"
                        )
                        torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                }, f"xLSTM/weights/xlstm_autoencoder_ep{i+1}_step{old_step}.pth")
                    writer.add_scalar("Loss", mean_loss / cnt, old_step)
                    writer.add_scalar("Recon loss", recon_loss, old_step)
                    writer.add_scalar("KLD", kld.mean().item(), old_step)
                if cnt % 1000 == 0:
                    valid, nov, _ = val(model, tokenizer, parallels)
                    writer.add_scalar("Valid mols", valid, old_step)
                    writer.add_scalar("New mols", nov, old_step)
            # if number of steps less then gradient store
            if (i + 1) % accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()
            writer.add_scalar("Loss per epoch", mean_loss / cnt, i)

                
    if is_generating:
        decode_store = []
        
        if parallels:
            model = AutoEncoder(conf_encoder, conf_decoder)
            model_dict_pred_train = torch.load(chkp_path)
            model.load_state_dict(model_dict_pred_train['model_state_dict'])
            model = torch.nn.DataParallel(model)
            model.to(DEVICE)
            res = model.module.decoder.generate_new_data(n_answers)
        else:
            model = AutoEncoder(conf_encoder, conf_decoder).to(DEVICE)
            model.load_state_dict(torch.load(chkp_path, map_location=DEVICE))
            res = model.decoder.generate_new_data(n_answers)

        
        tokenizer = make_tokenizer(
            data_path=dataset_path, max_length=max_length, vocab_size=vocab_size,
            path_from='xLSTM/tokenizer.pickle'
        )
        
        for r in res:
            sampled_ids = torch.argmax(torch.nn.functional.softmax(r), dim=-1)
            decode_store.append(tokenizer.decode([i for i in sampled_ids.tolist()]))

        with open(f"xLSTM/examples/generated_mols.txt", "w") as f:
            for mol in decode_store:
                f.write(mol + "\n")

        valid, nov, dup = run_measuring_metrics.main()

def run_encoder_decoder(
    cfg_path: str,
    is_fine_tuning: bool = False,
    is_generating: bool = True,
    dataset_path: str = None,
    num_answers: int = 500,
    num_epoch: int = 15
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
        model_gen = EncoderDecoderGenerator(cfg_path)
        results = model_gen.generate(start_token="[PAD]", num_answers=num_answers)
        [print(result) for result in results]
        with open(f"xLSTM/examples/out.txt", "w") as f:
            for res in results:
                f.write(res + "\n")


if __name__ == "__main__":
    data_path = "xLSTM/chembl_alpaca.txt"

    # run AutoEncoder (mLSTM encoder + mLSTM decoder)
    encod_conf = 'xLSTM/cfg/mLSTM_encoder_cfg.yaml'
    decode_conf = 'xLSTM/cfg/mLSTM_decoder_cfg.yaml'    
    run_auto_encoder(
        encod_conf, decode_conf, data_path, 
        is_fine_tuning=True, is_generating=False, epoch = 100,
        load_predtrain=False,
        parallels=True, bs=120
    )
    
    # uncomment for inference
    # run_auto_encoder(
    #     encod_conf, decode_conf, data_path, 
    #     is_fine_tuning=False, is_generating=True, parallels=True,
    #     chkp_path='/root/projects/SmilesTuneLLM/xLSTM/weights/xlstm_autoencoder_ep1_step4150.pth'
    # )
