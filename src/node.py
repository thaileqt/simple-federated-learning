import grpc
from concurrent import futures

import transport_pb2
import transport_pb2_grpc

import argparse

from utils import get_file_chunks, save_chunks_to_file
import mnist

import torch
from abstracts.stub import Stub, Node

class Servicer(Stub):

    def __init__(self, hostname, port):
        self.tmp_file_name = 'src/parameters/tmp/node_tmp'
        self.is_active = None
        self.model = None
        self.addr = hostname + '_' + str(port)


    def EstablishConnection(self, request_iterator, context):
        self.is_active = True
        self.node_number = request_iterator.nodeNumber
        self.address = request_iterator.address
        self.params_path = f'src/parameters/nodes/{self.addr}.pth'
        return transport_pb2.Empty(value=1)
    
    def Train(self, request_iterator, context):
        if self.is_active is None:
            return transport_pb2.Message(status=1, message='Node is not active')
        if self.model is None:
            return transport_pb2.Message(status=1, message='No parameters is set. Please do it first...')
        
        # Load the data for training
        train_dl, val_dl = mnist.load_data()

        try:
            mnist.train(model=self.model, num_epochs=5, train_dl=train_dl)
            print('Training completed')
            print()
            torch.save(self.model.state_dict(), self.params_path)
            _, acc = mnist.evaluate(self.model, val_dl)
            return transport_pb2.Message(status=0, message=f'{acc}')
        except Exception as e:
            return transport_pb2.Message(status=0, message=e)

    def GetParameters(self, request_iterator, context):
        chunk_generator = get_file_chunks(self.params_path)
        return chunk_generator

    def SetParameters(self, request_iterator, context):
        # receives parameters
        save_chunks_to_file(request_iterator, self.tmp_file_name)
        file_chunks = get_file_chunks(self.tmp_file_name)
        # saves locally
        save_chunks_to_file(file_chunks, self.params_path)

        self.model = mnist.Net()
        self.model.load_state_dict(torch.load(self.params_path))
        print('Parameters is set')
        print()
        return transport_pb2.Empty(value=1)

    def Evaluate(self, request_iterator, context):
        temp_model = mnist.Net()
        temp_model.load()


class MyNode(Node):
    def __init__(self, hostname, port):
        self.hostname = hostname
        self.port = port
        self.servicer = Servicer(hostname, port)
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        transport_pb2_grpc.add_FederatedAppServicer_to_server(self.servicer, self.server)

    def serve(self):
        self.server.add_insecure_port(f'{self.hostname}:{self.port}')
        self.server.start()
        print(f'{"Node started at port %d":_^50}' % port)
        print()
        self.server.wait_for_termination()

def args():
    parser = argparse.ArgumentParser(
        prog = 'Federated Learning',
        description = 'Node for execute federated learning')
    parser.add_argument('port', type=int, help='port of the node')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = args()
    port = args.port

    MyNode('localhost', port).serve()
