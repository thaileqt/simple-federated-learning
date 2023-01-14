from abc import ABC, abstractmethod
import os

class Server(ABC):

    def set_parameters(self):
        pass

    def get_parameters(self):
        pass

    def train(self):
        pass

    def average_parameters(self):
        pass

    def save_averaged_parameters(self):
        pass
    