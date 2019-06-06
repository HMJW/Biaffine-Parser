import abc
import torch

class Base(torch.nn.Module, metaclass=abc.ABCMeta):
    # @abc.abstractmethod
    # def train_model(self):
    #     raise NotImplementedError

    @abc.abstractclassmethod
    def load(self):
        raise NotImplementedError

    # @abc.abstractmethod
    # def save(self):
    #     raise NotImplementedError

    @abc.abstractmethod
    def predict(self):
        raise NotImplementedError

    # @abc.abstractmethod
    # def predict_file(self):
    #     raise NotImplementedError

    # @abc.abstractmethod
    # def evaluate(self):
    #     raise NotImplementedError
