from utils import *
import mnist
import torch


def EstablishConnection():
    return 1

def Train(node_params_path: str):
    model = mnist.Net()
    model.load_state_dict(torch.load(node_params_path))
    return