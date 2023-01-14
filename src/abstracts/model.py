from abc import ABC, abstractmethod

class Model(ABC):

    @abstractmethod
    def forward(self, x):
        pass