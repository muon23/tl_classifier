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
    """
    A class for text classification using a neural network classifier and embeddings.

    This class initializes a text classifier that uses a specified embedding model
    to convert text into feature vectors and then classifies these vectors into
    predefined labels. It supports a confidence threshold for predictions and
    includes an option for an "other" label for out-of-scope inputs.

    Attributes:
        logger (logging.Logger): Logger for logging information and errors.
        labels (List[str]): List of labels for classification.
        embedding_model_name (str): Name of the embedding model to use (default is "bert").
        other_label (str): Label for inputs that do not fit into any of the predefined labels (default is "other").
        embedding_model (Embedding): Instance of the embedding model used for feature extraction.
        classifier (NNClassifier): Instance of the neural network classifier.
    """

    logger: logging.Logger = None

    @dataclass
    class Prediction:
        """A data class to hold the prediction results."""
        label: str
        prob: float

    def __init__(
            self,
            labels: List[str],
            embedding_model_name="bert",
            confidence: float = 0.9,
            other_label: str = "other"
    ):
        """
        Initializes the TextClassifier with specified labels and parameters.

        Args:
            labels (List[str]): List of labels for classification.
            embedding_model_name (str, optional): Name of the embedding model to use. Defaults to "bert".
            confidence (float, optional): Confidence threshold for predictions. Defaults to 0.9.
            other_label (str, optional): Label for inputs that do not fit into any of the predefined labels. Defaults to "other".
        """

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
        """
        Trains the text classifier using the provided dataset.

        This method processes the input dataset, splits it into training and validation sets
        based on the given validation split ratio, and then embeds the text content before
        training the classifier.

        Args:
            dataset (List[dict]): A list of dictionaries containing training data, where
                each dictionary must have a "label" and "content" key.
            validation_split (float, optional): The proportion of the dataset to be used for
                validation. Should be between 0 and 1. Defaults to 0.2.

        Raises:
            ValueError: If the validation_split is not between 0 and 1.
        """
        # Determine if progress should be shown based on logger level
        show_progress = logging.getLogger().level <= logging.INFO

        # Remove data with unsupported labels
        dataset = [d for d in dataset if d["label"] in self.labels]

        # Split the dataset into training and validation sets
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

        # Embed the training set
        self.logger.info("Embedding the training set...")
        train_embeddings = self.embedding_model.embed(
            [d["content"] for d in train_data],
            show_progress=show_progress
        )
        train_labels = [self.labels.index(d["label"]) for d in train_data]

        # Embed the validation set if exist
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

        # Train the classifier with the embedded training data and optional validation data
        self.classifier.train_with(
            features=train_embeddings,
            labels=train_labels,
            validate_features=validate_embeddings,
            validate_labels=validate_labels,
        )

    def infer(self, texts: List[str]) -> List[Prediction]:
        """
        Infers labels for the provided texts using the trained classifier.

        This method takes a list of input texts, embeds them using the embedding model,
        and then uses the classifier to predict the labels and probabilities.

        Args:
            texts (List[str]): A list of text strings to classify.

        Returns:
            List[Prediction]: A list of Prediction objects, each containing the predicted
                label and its probability.
        """
        # Embed the input texts to get feature representations
        embeddings = self.embedding_model.embed(texts)

        # Perform inference using the classifier
        results = self.classifier.infer(features=embeddings)

        # Create and return a list of Prediction objects
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
        """
        Saves the TextClassifier model and its metadata to the specified directory.

        This method creates the directory if it does not exist, saves the model's
        metadata to a JSON file, and delegates the saving of the classifier to its
        own save method.

        Args:
            path (str): The directory path where the model and metadata will be saved.
        """
        # Create the directory if it does not exist
        os.makedirs(path, exist_ok=True)

        # Save metadata to a JSON file
        with open(self.__metadata_file(path), "w") as fp:
            json.dump(self._metadata(), fp)

        # Save the classifier's state
        self.classifier.save(path)

    @classmethod
    def load(cls, path: str) -> "TextClassifier":
        """
        Loads a previously saved TextClassifier model and its metadata from the specified directory.

        This method reconstructs the TextClassifier instance by loading its metadata and
        model state from the specified path. It raises an error if the directory does not exist.

        Args:
            path (str): The directory path from which to load the model and metadata.

        Returns:
            TextClassifier: An instance of the TextClassifier with the loaded model and metadata.

        Raises:
            FileNotFoundError: If the specified directory does not exist.
        """
        # Check if the specified directory exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory {path} not found.")

        # Load metadata from the JSON file
        metadata_file = cls.__metadata_file(path)
        try:
            with open(metadata_file, "r") as fp:
                metadata = json.load(fp)
        except Exception as e:
            cls.logger.error(f"Problem accessing {metadata_file}: {e}")

        # Restore TextClassifier instance using the loaded metadata
        classifier = TextClassifier(**metadata)

        # Reinitialize the embedding model using the specified model name
        classifier.embedding_model = embedding.using(classifier.embedding_model_name)

        # Load the classifier's state from the specified path
        classifier.classifier = NNClassifier.load(path)

        return classifier








