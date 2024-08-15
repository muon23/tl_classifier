import argparse
import configparser
import logging
import sys

from classifier.TextClassifier import TextClassifier

DEFAULT_CONFIG_FILE = "config.properties"

DEFAULT_CONFIG_SECTION = "DEFAULT"
PLAY_CONFIG_SECTION = "PLAY"


def parse_arguments() -> argparse.Namespace:
    # Create a new ArgumentParser object
    parser = argparse.ArgumentParser(description="Classifying a user input.")

    # Add arguments
    parser.add_argument("-c", "--config", type=str, default=DEFAULT_CONFIG_FILE,
                        help=f"Configura (default: '{DEFAULT_CONFIG_FILE}'.")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to the directory to output the trained model.")

    # Parse the arguments and return the result as a Namespace object
    try:
        return parser.parse_args()
    except SystemExit:
        sys.exit(1)


def get_property(config: configparser.ConfigParser, key: str, argument: str = None, default=None) -> str:
    if argument:
        return argument

    value = config[PLAY_CONFIG_SECTION].get(key)
    if not value:
        value = config[DEFAULT_CONFIG_SECTION].get(key, default)
    return value


def main():
    logger = logging.getLogger()

    # Parse command-line arguments
    args = parse_arguments()

    # Parse configuration file if given
    config = configparser.ConfigParser()
    if args.config:
        config.read(args.config)

    model_dir = get_property(config, "model", args.model, None)

    if not model_dir:
        sys.stderr.write("No model given.")
        sys.stderr.write("(Either use '--model' option or specify the 'model' property in the properties file.)")
        sys.exit(1)

    print(f"Playground started with:")
    print(f"   - classifier model at {model_dir}")
    print()

    try:
        classifier = TextClassifier.load(model_dir)
    except Exception as e:
        logger.error(f"Failed to load model from {model_dir}. {e}")
        sys.exit(1)

    print("Type or paste anything to classify.  Type 'q' to exist.")
    while True:
        user_input = input(">> ")
        if user_input.lower() == "q":
            print("Goodbye.")
            break

        results = classifier.infer([user_input])
        print(f"It is {results[0].label} (confidence: {results[0].prob})")


if __name__ == '__main__':
    main()
