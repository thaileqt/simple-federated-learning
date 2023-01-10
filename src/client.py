import mnist
import torch
import os

import concurrent.futures
from multiprocessing import freeze_support

from utils import *
from stub_functions import *
from connect import *

class Client:

    def __init__(self):
        if not os.path.isdir(os.path.join('src/parameters')):
            os.mkdir(os.path.join('src/parameters'))
            os.mkdir(os.path.join('src/parameters/nodes'))
            os.mkdir(os.path.join('src/parameters/client'))
            os.mkdir(os.path.join('src/parameters/tmp'))
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
        futures = [executor.submit(establish, i) for i in range(n) if stubs[addresses[i]] in active_stubs.values()]
        concurrent.futures.wait(futures)
        

    def set_parameters(self):
        executor = concurrent.futures.ProcessPoolExecutor(n)
        futures = [executor.submit(set_parameters, i, self.global_params_path) for i in range(n) if stubs[addresses[i]] in active_stubs.values()]
        concurrent.futures.wait(futures)
        
    def train(self):
        print('TRAINING')
        executor = concurrent.futures.ProcessPoolExecutor(n)
        futures = [executor.submit(train, i) for i in range(n) if stubs[addresses[i]] in active_stubs.values()]
        concurrent.futures.wait(futures)

    def get_parameters(self):
        executor = concurrent.futures.ProcessPoolExecutor(n)
        futures = [executor.submit(get_parameters, i) for i in range(n) if stubs[addresses[i]] in active_stubs.values()]
        concurrent.futures.wait(futures)

    def average_parameters(self):
        self.get_parameters()
        executor = concurrent.futures.ProcessPoolExecutor(n)
        futures = [executor.submit(average, i) for i in range(n) if stubs[addresses[i]] in active_stubs.values()]
        concurrent.futures.wait(futures)

        # calculate average parameters
        self.avg_params = average_parameters(n)
        
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

