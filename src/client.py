import grpc
import transport_pb2
import transport_pb2_grpc

import concurrent.futures
from multiprocessing import freeze_support
from utils import *


addresses = ['localhost:8081', 'localhost:8802']
stubs = {addr: transport_pb2_grpc.FederatedAppStub(grpc.insecure_channel(addr)) for addr in addresses}

#number of nodes on the network
n = len(stubs)


def _establish(i):
    info = transport_pb2.Info(nodeNumber=i+1, address=addresses[i])
    respond = stubs[info.address].EstablishConnection(info)
    if respond.value == 1:
        print(f'node {i+1}: connected')
    else:
        print(i)
    

def establish_connection():
    executor = concurrent.futures.ProcessPoolExecutor(n)
    futures = [executor.submit(_establish, i) for i in range(n)]
    concurrent.futures.wait(futures)
    # print(futures[0].result()+futures[1].result())

def train():
    executor = concurrent.futures.ProcessPoolExecutor(n)
    futures = [executor.submit(_train, i) for i in range(n)]
    concurrent.futures.wait(futures)

def _train(i):
    chunk_generator = get_file_chunks('src/parameters/global_parameters.pth')
    respond = stubs[addresses[i]].Train(chunk_generator)
    if respond.value is not None:
        print(f'[{addresses[i]} | node {i+1}]: received params')

        
if __name__ == '__main__':
    freeze_support()
    train()