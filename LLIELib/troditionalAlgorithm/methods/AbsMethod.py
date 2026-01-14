from abc import ABC, abstractmethod


class AbsMethod(ABC):
    def __init__(self, param=None, pipeline=None):
        self.param = param
        self.pipeline = pipeline
        self.pipeline_tuple = ()

    @abstractmethod
    def check_way(self):
        pass
