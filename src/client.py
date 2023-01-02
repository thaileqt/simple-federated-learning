import grpc
import transport_pb2
import transport_pb2_grpc

import concurrent.futures
from multiprocessing import freeze_support
from utils import *
import functions
import mnist
import torch


addresses = ['localhost:8081', 'localhost:8082']
# addresses = ['localhost:8081']
stubs = {addr: transport_pb2_grpc.FederatedAppStub(grpc.insecure_channel(addr)) for addr in addresses}

#number of nodes on the network
n = len(stubs)


def _establish(i):
    info = transport_pb2.Info(nodeNumber=i+1, address=addresses[i])
    respond = stubs[info.address].EstablishConnection(info)
    if respond.value == 1:
        print(f'\tnode {i+1}: connected')
    else:
        print(i)
    

def _get_status(i, silence=True):
    respond = stubs[addresses[i]].GetNodeStatus(transport_pb2.Empty(value=1))
    if respond.value == 1:
        if not silence:
            print(f'\t{addresses[i]} | node {i+1}: ready')
        return 1
    if not silence:
        print(f'\t{addresses[i]} | node {i+1}: inactive')
    return 0
    


def _train(i):
    chunk_generator = get_file_chunks('src/parameters/client/global_parameters.pth')
    respond = stubs[addresses[i]].Train(chunk_generator)
    if respond.value is not None:
        print(f'\t[{addresses[i]} | node {i+1}]: trained successfully')



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
            pass

    def establish_connection(self):
        print('ESTABLISHING CONNECTION')
        executor = concurrent.futures.ProcessPoolExecutor(n)
        futures = [executor.submit(_establish, i) for i in range(n)]
        concurrent.futures.wait(futures)


    def get_node_status(self, silence=True):
        if not silence:
            print('CHECK CONNECTION')
        executor = concurrent.futures.ProcessPoolExecutor(n)
        futures = [executor.submit(_get_status, i, silence) for i in range(n)]
        concurrent.futures.wait(futures)
        status = [future.result() for future in futures]
        return sum(status) == len(status)
        

    def train(self):
        print('TRAINING')
        if self.get_node_status() == 1:
            executor = concurrent.futures.ProcessPoolExecutor(n)
            futures = [executor.submit(_train, i) for i in range(n)]
            concurrent.futures.wait(futures)
        else:
            print('Nodes are not all ready, please establish connection first')


    def average_parameters(self):
        print('AVERAGING PARAMETERS')
        executor = concurrent.futures.ProcessPoolExecutor(n)
        futures = [executor.submit(_average, i) for i in range(n)]
        concurrent.futures.wait(futures)

        # calculate average parameters
        avg_params = functions.average_parameters(n)
        
        # test
        test_dataset = mnist.load_test_dataset()
        ## test current params
        global_acc = mnist.test(self.model, test_dataset)
        
        ## test new params
        temp_model = mnist.Net()
        temp_model.load_state_dict(avg_params)
        acc = mnist.test(temp_model, test_dataset)
        
        if acc > global_acc:
            print('\tsaved new parameters')
            torch.save(avg_params, self.global_params_path)

        print(f'\tglobal acc: {round(global_acc, 3)} ----- avg acc: {round(acc, 3)}')

        return avg_params

        
if __name__ == '__main__':
    freeze_support()

    client = Client()
    client.establish_connection()
    client.train()
    client.average_parameters()