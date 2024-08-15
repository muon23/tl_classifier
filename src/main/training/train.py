import argparse
import configparser
import logging
import sys

from classifier.TextClassifier import TextClassifier
from training.TrainingData import TrainingData

DEFAULT_EMBEDDING = "bert"
DEFAULT_CONFIDENCE = 0.9
DEFAULT_OTHER = "other"
DEFAULT_CONFIG_FILE = "config.properties"

DEFAULT_CONFIG_SECTION = "DEFAULT"
TRAIN_CONFIG_SECTION = "TRAIN"


def parse_arguments() -> argparse.Namespace:
    # Create a new ArgumentParser object
    parser = argparse.ArgumentParser(description="Train a text classifier.")

    # Add arguments
    parser.add_argument("-c", "--config", type=str, default=DEFAULT_CONFIG_FILE,
                        help=f"Configura (default: '{DEFAULT_CONFIG_FILE}').")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to the directory containing training dataset.")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to the directory to output the trained model.")
    parser.add_argument("--embedding", type=str, default=DEFAULT_EMBEDDING,
                        help=f"Text embedding model to use. (default: '{DEFAULT_EMBEDDING}'.")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE,
                        help=f"Confidence threshold when inferring with the model")
    parser.add_argument("--other", type=str, default=DEFAULT_OTHER,
                        help=f"What to label a low-confidence classification")
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')

    # Parse the arguments and return the result as a Namespace object
    try:
        return parser.parse_args()
    except SystemExit:
        sys.exit(1)


def get_property(config: configparser.ConfigParser, key: str, argument: str = None, default=None) -> str:
    if argument:
        return argument

    value = config[TRAIN_CONFIG_SECTION].get(key)
    if not value:
        value = config[DEFAULT_CONFIG_SECTION].get(key, default)
    return value


def main():

    # Parse command-line arguments
    args = parse_arguments()

    # Parse configuration file if given
    config = configparser.ConfigParser()
    if args.config:
        config.read(args.config)

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    confidence = float(get_property(config, "confidence", str(args.confidence), DEFAULT_CONFIDENCE))
    embedding_model_name = get_property(config, "embedding", args.embedding, DEFAULT_EMBEDDING)
    other_label = get_property(config, "other", args.other, DEFAULT_OTHER)
    data_dir = get_property(config, "data", args.data, None)
    model_dir = get_property(config, "model", args.model, None)

    if not data_dir:
        sys.stderr.write("Training data not given.")
        sys.stderr.write("(Either use '--data' option or specify the 'data' property in the properties file.)")
        sys.exit(1)
    if not model_dir:
        sys.stderr.write("Model output not given.")
        sys.stderr.write("(Either use '--model' option or specify the 'model' property in the properties file.)")
        sys.exit(1)

    print(f"Start training with:")
    print(f"   - data set from {data_dir}")
    print(f"   - embedding model '{embedding_model_name}'")
    print(f"   - unclassified label '{other_label}'")
    print(f"   - confidence level {confidence}")
    print(f"   - output model to {model_dir}")

    data = TrainingData(data_dir)
    classifier = TextClassifier(
        data.labels,
        confidence=confidence,
        embedding_model_name=embedding_model_name,
        other_label=other_label,
    )
    classifier.train_with(data.dataset)

    classifier.save(model_dir)


if __name__ == '__main__':
    main()
