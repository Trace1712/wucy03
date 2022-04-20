import math
import torch
from torch import nn
from torch.nn import functional as F
import ensemble

from aux_loss import get_aux_loss


class Auxagent(nn.Module):
    def __init__(self, action_dim, aux_lst, args, state_dim, hidden_size):
        # convs = [Conv(state_dim, action_dim) for i in range(10)]
        super().__init__()
        auxs = []
        for i, aux_names in enumerate(aux_lst):
            if aux_names == 'none':
                auxs.append(nn.ModuleList([]))
                continue
            head_aux = nn.ModuleList(
                [get_aux_loss(aux_name, i, action_dim, state_dim, hidden_size) for aux_name in
                 aux_names.split('+')]
            )
            auxs.append(head_aux)
        self.auxs = nn.ModuleList(auxs)
        self.auxs_lst = auxs

    def forward(self, x, log=False):
        return ensemble.forward(self, x, log=log)


if __name__ == '__main__':
    aux_agent = Auxagent(6, ['MyInverseDynamicLoss'], None, 17, 200)
