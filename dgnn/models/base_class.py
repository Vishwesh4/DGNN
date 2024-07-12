from typing import Any
import torch

class BaseModel(torch.nn.Module):
    """
    To register the subclasses based on the name
    To be used as
    @BaseModel.register("some_name")
    class SomeClass(BaseModel)

    To build that class, one can use
    BaseModel.create("some_name")
    """
    subclasses = {}

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def register(cls, subclass_name: str):
        def decorator(subclass: Any):
            subclass.subclasses[subclass_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, subclass_name: str, **params):
        if subclass_name not in cls.subclasses:
            raise ValueError("Unknown subclass name {}".format(subclass_name))
        print("-"*50)
        print(f"For class: {cls.__name__}, Selected subclass: ({subclass_name}):{cls.subclasses[subclass_name]}")
        print("-"*50)

        return cls.subclasses[subclass_name](**params)