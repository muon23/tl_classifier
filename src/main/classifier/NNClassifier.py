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
    """
    A neural network classifier for multi-class classification tasks.

    This class encapsulates a feedforward neural network model, allowing for
    customizable architecture, dropout regularization, and training parameters.
    It utilizes the ForwardNN model for classification and manages the training
    process, including loss calculation and optimization.

    Attributes:
        logger (logging.Logger): Logger instance for logging messages.
        num_features (int): The number of input features.
        num_classes (int): The number of output classes.
        hidden_dims (List[int]): A list of integers specifying the number of
            neurons in each hidden layer.
        dropout (float): The dropout probability for regularization.
        confidence (float): Confidence threshold for predictions.
        model (ForwardNN): Instance of the ForwardNN model.
        criterion (nn.Module): Loss function used for training.
        optimizer (optim.Optimizer): Optimizer for updating model parameters.
        default_epochs (int): Default number of epochs for training.
        default_patience (int): Default patience for early stopping.
    """

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
        """
        Initializes the NNClassifier model.

        Args:
            num_features (int): The number of input features.
            num_classes (int): The number of output classes.
            hidden_dims (List[int], optional): A list of integers specifying
                the number of neurons in each hidden layer. Defaults to
                [64, 32] if not provided.
            dropout (float, optional): The dropout probability for regularization.
                If None, dropout is not applied.
            default_epochs (int): The default number of epochs for training.
            default_patience (int): The default patience for early stopping.
            confidence (float, optional): Confidence threshold for predictions.
        """
        super(NNClassifier, self).__init__()

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

        """
        Trains the neural network classifier using the provided features and labels.

        This method allows for training the model on a dataset represented as
        features and labels. It supports optional validation data for monitoring
        performance and implementing early stopping based on validation loss.

        Args:
            features (Union[List[List[float]], np.ndarray]): Input features for training.
            labels (Union[List[int], np.ndarray]): Corresponding labels for the input features.
            validate_features (Union[List[List[float]], np.ndarray], optional):
                Input features for validation. If provided, the model will be evaluated
                on this dataset during training.  Early stop will also be performed if this is given.
            validate_labels (Union[List[int], np.ndarray], optional):
                Corresponding labels for the validation features.
            batch_size (int, optional): Number of samples per gradient update. Defaults to 32.
            num_epochs (int, optional): Number of epochs to train the model. If None,
                defaults to the value set in `self.default_epochs`.
            patience (int, optional): Number of epochs with no improvement after which
                training will be stopped. If None, defaults to `self.default_patience`.

        Raises:
            ValueError: If the number of features does not match the expected input dimensions.
        """

        # Set default values for num_epochs and patience if not provided
        if not num_epochs:
            num_epochs = self.default_epochs
        if not patience:
            patience = self.default_patience

        # Check if the number of features matches the model's expected input dimensions
        num_features = len(features[0]) if isinstance(features, list) else features.shape[1]
        if num_features != self.num_features:
            raise ValueError(
                f"Number of features ({features.size()[1]}) does not match input dimensions of the model "
                f"({self.num_features})"
            )

        # Create DataLoader for the training dataset
        train_loader = DataLoader(SupervisedDataset(features, labels), batch_size=batch_size, shuffle=True)

        # Create DataLoader for the validation dataset if provided
        test_loader = None
        if validate_features is not None and validate_labels is not None:
            test_loader = DataLoader(SupervisedDataset(validate_features, validate_labels), batch_size=batch_size)

        # Set the model to training mode
        self.model.train()

        # Training loop
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

                running_loss += loss.item()  # Accumulate loss

            # Average loss for the epoch
            running_loss /= len(train_loader)
            self.logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss:.4f}')

            # Validation step if validation data is provided
            if test_loader:
                val_loss = self._validate_model(test_loader)  # Validate the model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss  # Update best loss
                    self.patience_counter = 0  # Reset patience counter
                else:
                    self.patience_counter += 1  # Increment patience counter

                # Check for early stopping
                if self.patience_counter >= patience:
                    self.logger.info(f'Early stopping on epoch {epoch + 1}')
                    break

    def _validate_model(self, test_loader: DataLoader) -> float:
        """
        Validates the neural network model on the provided test dataset.

        This method evaluates the model's performance on a validation or test dataset
        by calculating the average loss. It sets the model to evaluation mode and
        disables gradient computation to save memory and improve performance.

        Args:
            test_loader (DataLoader): DataLoader for the test dataset.

        Returns:
            float: The average validation loss over the test dataset.
        """
        self.model.eval()  # Set the model to evaluation mode
        val_loss = 0.0  # Initialize validation loss accumulator

        # Disable gradient calculation for validation
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)  # Forward pass
                loss = self.criterion(outputs, labels)  # Compute loss
                val_loss += loss.item()  # Accumulate loss

        # Calculate average validation loss
        val_loss /= len(test_loader)
        self.logger.info(f'Validation Loss: {val_loss:.4f}')  # Log the validation loss
        return val_loss  # Return the average validation loss

    def infer(self, features: Union[List[List[float]], np.ndarray]) -> np.ndarray:
        """
        Performs inference on the provided features using the trained model.

        This method takes input features, processes them, and returns the predicted
        classes along with their corresponding probabilities. It handles both single
        examples and batches of examples.

        Args:
            features (Union[List[List[float]], np.ndarray]): Input features for inference,
                which can be a single example or a batch of examples.

        Returns:
            np.ndarray: An array containing predicted classes and their corresponding
                probabilities. Each row corresponds to a sample, with the first column
                being the predicted class and the second column being the probability.
        """

        # Ensure the model is in evaluation mode
        self.model.eval()

        # Check if this is a single example or a batch of them
        single = (
            (isinstance(features, list) and isinstance(features[0], float)) or
            (len(features.shape) == 1)
        )

        # Convert the numpy array to a PyTorch tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)

        # If it's a single example, add a batch dimension
        if single:
            features_tensor = features_tensor.unsqueeze(0)

        # Perform inference without tracking gradients
        with torch.no_grad():
            outputs = self.model.predict(features_tensor)

        # Get the predicted classes and their maximum probabilities
        max_probs, predicted = torch.max(outputs, 1)

        # If a confidence threshold is set, mark predictions below the threshold as unknown
        if self.confidence:
            unknown_class = max_probs < self.confidence
            predicted[unknown_class] = -1

        # If it's a single example, convert the predicted class and probability to scalar values
        if single:
            predicted = predicted.item()
            max_probs = max_probs.item()

        # Return the predicted classes and their probabilities as a numpy array
        return np.c_[predicted.numpy(), max_probs.numpy()]  # Combine predictions and probabilities

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
        """
        Saves the model and its metadata to the specified directory.

        This method creates the directory if it does not exist, saves the model's
        state dictionary, and writes metadata about the model to a JSON file.

        Args:
            path (str): The directory path where the model and metadata will be saved.
        """
        # Create the directory if it does not exist
        os.makedirs(path, exist_ok=True)

        # Save metadata to a JSON file
        with open(self.__metadata_file(path), "w") as fp:
            json.dump(self._metadata(), fp)

        # Save the model
        torch.save(self.model.state_dict(), self.__model_file(path))

    @classmethod
    def load(cls, path: str) -> "NNClassifier":
        """
        Loads a previously saved NNClassifier model and its metadata from the specified directory.

        This method reconstructs the NNClassifier instance by loading its metadata and
        model state from the specified path. It raises an error if the directory does not exist.

        Args:
            path (str): The directory path from which to load the model and metadata.

        Returns:
            NNClassifier: An instance of the NNClassifier with the loaded model and metadata.

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

        # Reconstruct NNClassifier instance using the loaded metadata
        classifier = NNClassifier(**metadata)

        # Load the model's state dictionary from the specified file
        model_file = cls.__model_file(path)
        try:
            classifier.model.load_state_dict(torch.load(model_file))
        except Exception as e:
            cls.logger.error(f"Problem loading model from {model_file}: {e}")

        # Reinitialize the loss function and optimizer
        classifier.criterion = nn.CrossEntropyLoss()
        classifier.optimizer = optim.Adam(classifier.model.parameters(), lr=0.001)

        return classifier


