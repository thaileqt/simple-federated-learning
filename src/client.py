import transport_pb2
import concurrent.futures
from multiprocessing import freeze_support
from utils import *
import functions
import mnist
import torch
import os
from connect import *

def _establish(i):
    info = transport_pb2.Info(nodeNumber=i+1, address=addresses[i])
    respond = stubs[addresses[i]].EstablishConnection(info)
    if respond.value == 1:
        print(f'{info.address}: ok')

def _train(i):
    respond = stubs[addresses[i]].Train(transport_pb2.Message(message=''))
    print(f'[{addresses[i]}: trained successfully -- acc: {respond.message}')

def _set_parameters(i, global_params_path):
    chunk_generator = get_file_chunks(global_params_path)
    respond = stubs[addresses[i]].SetParameters(chunk_generator)
    if respond.value is not None:
        print(f'[{addresses[i]}: parameters is set')

def _get_parameters(i):
    stubs[addresses[i]].GetParameters(transport_pb2.Empty(value=1))


def _average(i):
    respond = stubs[addresses[i]].GetParameters(transport_pb2.Empty(value=1))

    # receives from nodes
    path = f'src/parameters/client/node_{i+1}_parameters.pth'
    tmp_path = f'src/parameters/tmp/node_{i+1}_tmp'
    save_chunks_to_file(respond, tmp_path)
    file_chunks = get_file_chunks(tmp_path)
    # saves locally
    save_chunks_to_file(file_chunks, path)

class Client:

    def __init__(self):
        self.global_params_path = 'src/parameters/client/global_parameters.pth'
        self.model = mnist.Net()
        try:
            self.model.load_state_dict(torch.load(self.global_params_path))
        except:
            global_params_path = 'src/parameters/client/global_parameters.pth'
            if not os.path.isfile(global_params_path):
                torch.save(mnist.Net().state_dict(), global_params_path)
        self.avg_params = None

    def establish_connection(self):
        executor = concurrent.futures.ProcessPoolExecutor(n)
        futures = [executor.submit(_establish, i) for i in range(n) if stubs[addresses[i]] in active_stubs.values()]
        concurrent.futures.wait(futures)
        

    def set_parameters(self):
        executor = concurrent.futures.ProcessPoolExecutor(n)
        futures = [executor.submit(_set_parameters, i, self.global_params_path) for i in range(n) if stubs[addresses[i]] in active_stubs.values()]
        concurrent.futures.wait(futures)
        
    def train(self):
        print('TRAINING')
        executor = concurrent.futures.ProcessPoolExecutor(n)
        futures = [executor.submit(_train, i) for i in range(n) if stubs[addresses[i]] in active_stubs.values()]
        concurrent.futures.wait(futures)

    def get_parameters(self):
        executor = concurrent.futures.ProcessPoolExecutor(n)
        futures = [executor.submit(_get_parameters, i) for i in range(n) if stubs[addresses[i]] in active_stubs.values()]
        concurrent.futures.wait(futures)

    def average_parameters(self):
        self.get_parameters()
        executor = concurrent.futures.ProcessPoolExecutor(n)
        futures = [executor.submit(_average, i) for i in range(n) if stubs[addresses[i]] in active_stubs.values()]
        concurrent.futures.wait(futures)

        # calculate average parameters
        self.avg_params = functions.average_parameters(n)
        
        # test
        test_dataset = mnist.load_test_dataset()
        ## test current params
        global_acc = mnist.test(self.model, test_dataset)
        
        ## test new params
        temp_model = mnist.Net()
        temp_model.load_state_dict(self.avg_params)
        acc = mnist.test(temp_model, test_dataset)
        print(
            f'{"global acc":<15}avg acc',
            f'\n{round(global_acc, 3):<15}{round(acc, 3)}')

        return self.avg_params

    def save_averaged_parameters(self):
        ## test new params
        test_dataset = mnist.load_test_dataset()
        self.model.load_state_dict(self.avg_params)
        acc = mnist.test(self.model, test_dataset)
        print(f'Set averaged parameters to global (acc on test set: {round(acc, 3)})')

        torch.save(self.avg_params, self.global_params_path)

def get_input() -> int:
    print()
    print(f'{"":>5}{"":-^30}')
    print(f'{"":>7}1. Set parameters')
    print(f'{"":>7}2. Train')
    print(f'{"":>7}3. Average parameters')
    print(f'{"":>7}4. Save averaged parameters')
    print(f'{"":>5}{"":-^30}')
    i = int(input(''))
    return i
        
if __name__ == '__main__':
    freeze_support()

    client = Client()
    client.establish_connection()
    while True:
        i = get_input()
        if i == 1:
            client.set_parameters()
        elif i == 2:
            client.train()
        elif i == 3:
            client.average_parameters()
        elif i == 4:
            client.save_averaged_parameters()
        else:
            break

