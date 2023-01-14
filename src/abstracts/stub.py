from abc import ABC, abstractmethod
import transport_pb2_grpc


class Stub(ABC, transport_pb2_grpc.FederatedAppServicer):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def EstablishConnection(self, request, context):
        return super().EstablishConnection(request, context)
    
    def Train(self, request, context):
        return super().Train(request, context)

    def GetParameters(self, request, context):
        return super().GetParameters(request, context)

    def SetParameters(self, request_iterator, context):
        return super().SetParameters(request_iterator, context)

    def Evaluate(self, request, context):
        return super().Evaluate(request, context)

class Node(ABC):
    @abstractmethod
    def serve(self):
        pass
