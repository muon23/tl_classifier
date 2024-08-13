import json
import logging
import os
import random
from dataclasses import dataclass
from typing import List

import embedding
from classifier.NNClassifier import NNClassifier
from embedding.Embedding import Embedding


class TextClassifier:
    logger: logging.Logger = None

    @dataclass
    class Prediction:
        label: str
        prob: float

    def __init__(
            self,
            labels: List[str],
            embedding_model_name="bert",
            confidence: float = 0.9,
            other_label: str = "other"
    ):
        self.labels = labels
        self.embedding_model_name = embedding_model_name
        self.other_label = other_label

        self.embedding_model: Embedding = embedding.using(embedding_model_name)
        self.classifier = NNClassifier(
            num_features=self.embedding_model.dimensions(),
            num_classes=len(labels),
            hidden_dims=[256, 256],
            confidence=confidence,
        )

        if not self.logger:
            self.logger = logging.getLogger(__class__.__name__)

    def train_with(self, dataset: List[dict], validation_split: float = 0.2):
        show_progress = logging.getLogger().level <= logging.INFO

        # Remove data with unsupported labels
        dataset = [d for d in dataset if d["label"] in self.labels]

        if validation_split:
            if validation_split > 1 or validation_split < 0:
                raise ValueError(f"validation_split must be between 0 and 1 (was {validation_split})")
            train_data_size = int(len(dataset) * (1 - validation_split))
            random.shuffle(dataset)
            train_data = dataset[:train_data_size]
            validate_data = dataset[train_data_size:]
        else:
            train_data = dataset
            validate_data = None

        self.logger.info("Embedding the training set...")
        train_embeddings = self.embedding_model.embed(
            [d["content"] for d in train_data],
            show_progress=show_progress
        )
        train_labels = [self.labels.index(d["label"]) for d in train_data]

        if validate_data:
            self.logger.info("Embedding the validation set...")
            validate_embeddings = self.embedding_model.embed(
                [d["content"] for d in validate_data],
                show_progress=show_progress
            )
            validate_labels = [self.labels.index(d["label"]) for d in validate_data]
        else:
            validate_embeddings = None
            validate_labels = None

        self.classifier.train_with(
            features=train_embeddings,
            labels=train_labels,
            validate_features=validate_embeddings,
            validate_labels=validate_labels,
        )

    def infer(self, texts: List[str]) -> List[Prediction]:
        embeddings = self.embedding_model.embed(texts)
        results = self.classifier.infer(features=embeddings)
        return [
            self.Prediction(self.labels[int(r[0])], r[1]) if r[0] >= 0 else self.Prediction(self.other_label, r[1])
            for r in results
        ]

    @classmethod
    def __metadata_file(cls, path: str) -> str:
        return os.path.join(path, f"{__class__.__name__}.json")

    @classmethod
    def __model_file(cls, path: str) -> str:
        return os.path.join(path, "model")

    def _metadata(self):
        return {
            "labels": self.labels,
            "embedding_model_name": self.embedding_model_name,
            "other_label": self.other_label,
        }

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)

        with open(self.__metadata_file(path), "w") as fp:
            json.dump(self._metadata(), fp)

        self.classifier.save(path)

    @classmethod
    def load(cls, path: str) -> "TextClassifier":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory {path} not found.")

        metadata_file = cls.__metadata_file(path)
        try:
            with open(metadata_file, "r") as fp:
                metadata = json.load(fp)
        except Exception as e:
            cls.logger.error(f"Problem accessing {metadata_file}: {e}")

        # Restore TextClassifier
        classifier = TextClassifier(**metadata)
        classifier.embedding_model = embedding.using(classifier.embedding_model_name)
        classifier.classifier = NNClassifier.load(path)

        return classifier








