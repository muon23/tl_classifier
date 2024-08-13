from embedding import Embedding
from embedding.AdaEmbedding import AdaEmbedding
from embedding.TransformersEmbedding import TransformersEmbedding


def using(model_name: str) -> Embedding:
    try:
        return TransformersEmbedding(model_name)
    except NotImplementedError:
        pass

    try:
        return AdaEmbedding(model_name)
    except NotImplementedError:
        pass

    raise NotImplementedError(f"Embedding model {model_name} not supported")