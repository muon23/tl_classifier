from typing import List

import numpy as np

from embedding.Embedding import Embedding


class AdaEmbedding(Embedding):
    def __init__(self, model_name: str):
        super().__init__(model_name)

        # TODO: Place holder
        raise NotImplementedError(f"Embedding model {model_name} not supported")

    def embed(self, texts: List[str]) -> np.ndarray:
        pass

    def dimensions(self) -> int:
        return 1536
