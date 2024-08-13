import argparse
import logging
import sys

from classifier.TextClassifier import TextClassifier
from training.TrainingData import TrainingData

DEFAULT_EMBEDDING = "bert"
DEFAULT_CONFIDENCE = 0.9
DEFAULT_OTHER = "other"


def parse_arguments() -> argparse.Namespace:
    # Create a new ArgumentParser object
    parser = argparse.ArgumentParser(description="Train a text classifier.")

    # Add mandatory URL argument
    parser.add_argument("data_dir", type=str, help="Path to the directory containing training dataset.")
    parser.add_argument("model_dir", type=str, help="Path to the directory to output the trained model.")

    # Optional argument to specify embedding model
    parser.add_argument("--embedding", type=str, default=DEFAULT_EMBEDDING,
                        help=f"Text embedding model to use. (default: '{DEFAULT_EMBEDDING}'.")

    # Optional argument for confidence threshold when inferring with the model
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE,
                        help=f"Confidence threshold when inferring with the model")

    # Optional argument for labeling unrecognized texts
    parser.add_argument("--other", type=str, default=DEFAULT_OTHER,
                        help=f"What to label a low-confidence classification")

    # Set log level
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')

    # Parse the arguments and return the result as a Namespace object
    try:
        return parser.parse_args()
    except SystemExit:
        sys.exit(1)


def requirements_satisfied(args: argparse.Namespace):
    # Place holder
    return True


def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Ensure all system requirements and dependencies are satisfied
    try:
        requirements_satisfied(args)
    except AttributeError as e:
        print(f"Missing requirements: {e}")
        return

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    data = TrainingData(args.data_dir)
    classifier = TextClassifier(
        data.labels,
        confidence=args.confidence,
        embedding_model_name=args.embedding,
        other_label=args.other,
    )
    classifier.train_with(data.dataset)

    classifier.save(args.model_dir)


if __name__ == '__main__':
    main()
