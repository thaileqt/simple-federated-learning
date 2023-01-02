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
                self.tmp_file_name = 'src/parameters/tmp/node_tmp'
                self.is_active = None

            def EstablishConnection(self, request_iterator, context):
                self.is_active = True
                self.node_number = request_iterator.nodeNumber
                self.address = request_iterator.address
                self.params_path = f'src/parameters/nodes/node_{self.node_number}_parameters.pth'
                return transport_pb2.Empty(value=1)
            
            def GetNodeStatus(self, request_iterator, context):
                if self.is_active:
                    return transport_pb2.Empty(value=1)
                return transport_pb2.Empty(value=0)


            def Train(self, request_iterator, context):
                if self.is_active is None:
                    print('node is not active')
                    return
                # receives parameters
                save_chunks_to_file(request_iterator, self.tmp_file_name)
                file_chunks = get_file_chunks(self.tmp_file_name)
                # saves locally
                save_chunks_to_file(file_chunks, self.params_path)
                return transport_pb2.Empty(value=functions.Train(self.params_path))

            def GetParameters(self, request_iterator, context):
                chunk_generator = get_file_chunks(self.params_path)
                return chunk_generator

            

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
