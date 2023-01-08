from audioop import avg
from typing import OrderedDict
from utils import *
import mnist
import torch
import numpy as np


def average_parameters(num_nodes: int):
    list_state_dict = [torch.load(f'src/parameters/client/node_{i+1}_parameters.pth') for i in range(num_nodes)]
    avg_params = OrderedDict()
    layer_names = [name for name, _ in list_state_dict[0].items()]
    list_params = []
    for state_dict in list_state_dict:
        list_params.append([v for _, v in state_dict.items()])

    for i, param in enumerate(zip(*list_params)):
        avg_params[layer_names[i]] = torch.mean(torch.stack(param), axis=0)
    return avg_params
    

if __name__ == '__main__':
    average_parameters(1)
    print()
    # li = [torch.tensor([1, 2])]
    # for i in zip(*li):
    #     print(np.mean(i, axis=0))