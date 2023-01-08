import grpc
import transport_pb2_grpc


def is_ready(addr, TIMEOUT_SEC=5):
    try:
        grpc.channel_ready_future(channels[addr]).result(timeout=TIMEOUT_SEC)
        active_stubs[addr] = stubs[addr]
        return True
    except grpc.FutureTimeoutError:
        return False
        

addresses = ['localhost:8081', 'localhost:8082', 'localhost:8083']
# addresses = ['localhost:8081']
channels = {addr: grpc.insecure_channel(addr) for addr in addresses}
stubs = {addr: transport_pb2_grpc.FederatedAppStub(channel) for addr, channel in channels.items()}
active_stubs = {}

#number of nodes on the network

for addr in addresses:
    is_ready(addr)
    
n = len(active_stubs)