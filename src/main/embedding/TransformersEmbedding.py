from typing import List

import numpy as np
import torch
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertModel

from embedding.Embedding import Embedding


class TransformersEmbedding(Embedding):
    SUPPORTED_MODELS = [
        "distilbert-base-uncased",
        "distilbert-base-cased",
        "distilbert-base-multilingual-cased",
    ]

    SUPPORTED_MODEL_ALIASES = {
        "bert": "distilbert-base-uncased"
    }

    def __init__(self, model_name: str, batch_size: int = 20):
        if model_name in self.SUPPORTED_MODEL_ALIASES:
            model_name = self.SUPPORTED_MODEL_ALIASES[model_name]

        if model_name not in self.SUPPORTED_MODELS:
            raise NotImplementedError(f"Embedding model {model_name} not supported")

        super().__init__(model_name)

        if model_name.startswith("distilbert"):
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
            self.__dimensions = 768
        else:
            raise AssertionError(f"This line shall not be reached {model_name}")

        self.batch_size = batch_size

    def embed(self, texts: List[str], **kwargs) -> np.ndarray:
        show_progress = kwargs.get("show_progress", False)

        embeddings = torch.Tensor()
        for i in tqdm(range(0, len(texts), self.batch_size), disable=not show_progress):
            batch = texts[i: i + self.batch_size]
            encoded = self.tokenizer(batch, return_tensors='pt', max_length=512, truncation=True, padding=True)
            with torch.no_grad():
                output = self.model(**encoded)
                embeddings = torch.cat((embeddings, output.last_hidden_state.mean(dim=1)))

        return embeddings.numpy()

    def dimensions(self) -> int:
        return self.__dimensions



