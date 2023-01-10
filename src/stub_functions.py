from typing import OrderedDict
from utils import *
import torch
import transport_pb2

from connect import *

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
    

def establish(i):
    info = transport_pb2.Info(nodeNumber=i+1, address=addresses[i])
    respond = stubs[addresses[i]].EstablishConnection(info)
    if respond.value == 1:
        print(f'{info.address}: ok')

def train(i):
    respond = stubs[addresses[i]].Train(transport_pb2.Message(status=0, message=''))
    if respond.status == 0:
        print(f'[{addresses[i]}]: trained successfully -- acc: {respond.message}')
    elif respond.status == 1:
        print(f'[{addresses[i]}]: {respond.message}')

def set_parameters(i, global_params_path):
    chunk_generator = get_file_chunks(global_params_path)
    respond = stubs[addresses[i]].SetParameters(chunk_generator)
    if respond.value is not None:
        print(f'[{addresses[i]}: parameters is set')

def get_parameters(i):
    stubs[addresses[i]].GetParameters(transport_pb2.Empty(value=1))


def average(i):
    respond = stubs[addresses[i]].GetParameters(transport_pb2.Empty(value=1))

    # receives from nodes
    path = f'src/parameters/client/node_{i+1}_parameters.pth'
    tmp_path = f'src/parameters/tmp/node_{i+1}_tmp'
    save_chunks_to_file(respond, tmp_path)
    file_chunks = get_file_chunks(tmp_path)
    # saves locally
    save_chunks_to_file(file_chunks, path)

