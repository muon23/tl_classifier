import argparse
import logging
import sys

from classifier.TextClassifier import TextClassifier


def parse_arguments() -> argparse.Namespace:
    # Create a new ArgumentParser object
    parser = argparse.ArgumentParser(description="Classifying a user input.")

    # Add mandatory arguments
    parser.add_argument("model_dir", type=str, help="Path to the directory to output the trained model.")

    # Parse the arguments and return the result as a Namespace object
    try:
        return parser.parse_args()
    except SystemExit:
        sys.exit(1)


def main():
    logger = logging.getLogger()

    # Parse command-line arguments
    args = parse_arguments()

    try:
        classifier = TextClassifier.load(args.model_dir)
    except Exception as e:
        logger.error(f"Failed to load model from {args.model_dir}. {e}")
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
