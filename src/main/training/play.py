import argparse
import os
import sys

from classifier.TextClassifier import TextClassifier

REQUIRED_MODEL_FILES = ["model", "NNClassifier.json", "TextClassifier.json"]

def parse_arguments() -> argparse.Namespace:
    # Create a new ArgumentParser object
    parser = argparse.ArgumentParser(description="Classifying a user input.")

    # Add mandatory URL argument
    parser.add_argument("model_dir", type=str, help="Path to the directory to output the trained model.")

    # Parse the arguments and return the result as a Namespace object
    try:
        return parser.parse_args()
    except SystemExit:
        sys.exit(1)


def requirements_satisfied(args: argparse.Namespace):
    # Check if the model directory has proper model files
    try:
        files = os.listdir(args.model_dir)
    except FileNotFoundError as e:
        raise AttributeError(f"Model directory {args.model_dir} not found.")

    if not all([(f in files) for f in REQUIRED_MODEL_FILES]):
        raise AttributeError(f"Missing data in {args.model_dir}")

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

    classifier = TextClassifier.load(args.model_dir)

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
