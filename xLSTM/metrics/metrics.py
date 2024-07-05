import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn.functional as F


def loss_function(reconstructed: torch.Tensor, target: torch.Tensor, kld: float, beta: float):
    # simple cross entropy loss between input seq and pred sequence
    recon_loss = F.cross_entropy(reconstructed.view(-1, reconstructed.size(-1)), target.view(-1), reduction='sum')
    loss = recon_loss + beta * kld
    return loss, recon_loss, kld