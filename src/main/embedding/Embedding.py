from abc import ABC, abstractmethod
from typing import List

import numpy as np


class Embedding(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def embed(self, texts: List[str], **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def dimensions(self) -> int:
        pass

