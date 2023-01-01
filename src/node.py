import grpc
from concurrent import futures

import transport_pb2
import transport_pb2_grpc
import functions

import argparse

from utils import get_file_chunks, save_chunks_to_file



class Node(transport_pb2_grpc.FederatedAppServicer):
    def __init__(self):

        class Servicer(transport_pb2_grpc.FederatedAppServicer):

            def __init__(self):
                self.tmp_file_name = 'src/parameters/node_tmp'

            def EstablishConnection(self, request_iterator, context):
                self.nodeNumber = request_iterator.nodeNumber
                self.port = request_iterator.port
                response = transport_pb2.Empty()
                response.value = 1
                return response

            def Train(self, request_iterator, context):
                save_chunks_to_file(request_iterator, self.tmp_file_name)
                file_chunks = get_file_chunks(self.tmp_file_name)

                file_path = 'src/parameters/node_parameters.pth'
                save_chunks_to_file(file_chunks, file_path)
                functions.Train(file_path)
                return transport_pb2.Empty(value=1)

        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        transport_pb2_grpc.add_FederatedAppServicer_to_server(Servicer(), self.server)

    def serve(self, port):
        self.server.add_insecure_port(f'[::]:{port}')
        self.server.start()
        print(f'=== Node started at port {port} ===')
        self.server.wait_for_termination()

def args():
    parser = argparse.ArgumentParser(
        prog = 'Federated Learning',
        description = 'Node for execute federated learning')
    parser.add_argument('port', type=int, help='port of  the node')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = args()
    port = args.port
    Node().serve(port)
