import json
import logging
import os
from typing import List, Union

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from classifier.ForwardNN import ForwardNN
from classifier.SupervisedDataset import SupervisedDataset


class NNClassifier:

    logger: logging.Logger = None

    def __init__(
            self,
            num_features: int,
            num_classes: int,
            hidden_dims: List[int] = None,
            dropout: float = None,
            default_epochs=60,
            default_patience=3,
            confidence: float = None,
    ):
        super(NNClassifier, self).__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.confidence = confidence

        self.model = ForwardNN(
            input_dim=num_features,
            output_dim=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
            # softmax=(confidence is not None)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.default_epochs = default_epochs
        self.default_patience = default_patience

        self.best_loss = float('inf')
        self.patience_counter = 0

        if not self.logger:
            self.logger = logging.getLogger(self.__class__.__name__)

    def train_with(
            self,
            features: Union[List[List[float]], np.ndarray],
            labels: Union[List[int], np.ndarray],
            validate_features: Union[List[List[float]], np.ndarray] = None,
            validate_labels: Union[List[int], np.ndarray] = None,
            batch_size: int = 32,
            num_epochs: int = None,
            patience: int = None,
    ):
        if not num_epochs:
            num_epochs = self.default_epochs
        if not patience:
            patience = self.default_patience

        num_features = len(features[0]) if isinstance(features, list) else features.shape[1]
        if num_features != self.num_features:
            raise ValueError(
                f"Number of features ({features.size()[1]}) does not match input dimensions of the model "
                f"({self.num_features})"
            )

        train_loader = DataLoader(SupervisedDataset(features, labels), batch_size=batch_size, shuffle=True)

        test_loader = None
        if validate_features is not None and validate_labels is not None:
            test_loader = DataLoader(SupervisedDataset(validate_features, validate_labels), batch_size=batch_size)

        # Turn on training mode
        self.model.train()

        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            running_loss /= len(train_loader)
            # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss:.4f}')
            self.logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss:.4f}')

            if test_loader:
                # Validation loss for early stopping
                val_loss = self._validate_model(test_loader)
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                if self.patience_counter >= patience:
                    self.logger.info(f'Early stopping on epoch {epoch + 1}')
                    break

    def _validate_model(self, test_loader: DataLoader) -> float:
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(test_loader)
        self.logger.info(f'Validation Loss: {val_loss:.4f}')
        return val_loss

    def infer(self, features: Union[List[List[float]], np.ndarray]) -> np.ndarray:
        # Ensure the model is in evaluation mode
        self.model.eval()

        # Check if this is a single example or a batch of them
        single = (
            (isinstance(features, list) and isinstance(features[0], float)) or
            (len(features.shape) == 1)
        )

        # Convert the numpy array to a PyTorch tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)

        if single:
            features_tensor = features_tensor.unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            outputs = self.model.predict(features_tensor)

        # Get the predicted classes in (max value, index of max value)
        max_probs, predicted = torch.max(outputs, 1)

        if self.confidence:
            unknown_class = max_probs < self.confidence
            predicted[unknown_class] = -1

        if single:
            predicted = predicted.item()
            max_probs = max_probs.item()

        return np.c_[predicted.numpy(), max_probs.numpy()]  # Convert the tensor back to a numpy array if needed

    def _metadata(self):
        return {
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "confidence": self.confidence,
            "default_epochs": self.default_epochs,
            "default_patience": self.default_patience,
        }

    @classmethod
    def __metadata_file(cls, path: str) -> str:
        return os.path.join(path, f"{__class__.__name__}.json")

    @classmethod
    def __model_file(cls, path: str) -> str:
        return os.path.join(path, "model")

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)

        with open(self.__metadata_file(path), "w") as fp:
            json.dump(self._metadata(), fp)

        torch.save(self.model.state_dict(), self.__model_file(path))

    @classmethod
    def load(cls, path: str) -> "NNClassifier":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory {path} not found.")

        metadata_file = cls.__metadata_file(path)
        try:
            with open(metadata_file, "r") as fp:
                metadata = json.load(fp)
        except Exception as e:
            cls.logger.error(f"Problem accessing {metadata_file}: {e}")

        # Reconstruct NNClassifier
        classifier = NNClassifier(**metadata)

        # Load back torch model
        model_file = cls.__model_file(path)
        try:
            classifier.model.load_state_dict(torch.load(model_file))
        except Exception as e:
            cls.logger.error(f"Problem loading model from {model_file}: {e}")

        classifier.criterion = nn.CrossEntropyLoss()
        classifier.optimizer = optim.Adam(classifier.model.parameters(), lr=0.001)

        return classifier


