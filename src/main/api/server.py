import argparse
import configparser
import logging
import sys

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from classifier.TextClassifier import TextClassifier

app = FastAPI()
logger = logging.getLogger()

DEFAULT_CONFIG_FILE = "config.properties"
DEFAULT_SERVER_PORT = 8000

CONFIG_SECTION = "text_classify"

classifier: TextClassifier | None = None


# Define the request body
class Document(BaseModel):
    document_text: str


# Define the response body
class ClassificationResponse(BaseModel):
    message: str
    label: str


# Load property file
def load_config(file_path: str) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read(file_path)
    return config


def parse_arguments() -> argparse.Namespace:
    # Create a new ArgumentParser object
    parser = argparse.ArgumentParser(description="Classifier API server")

    # Add options
    parser.add_argument("-c", "--config", type=str, default=DEFAULT_CONFIG_FILE,
                        help=f"Configura (default: '{DEFAULT_CONFIG_FILE}'.")
    parser.add_argument("-m", "--model", type=str, default=None,
                        help="Path to the directory to output the trained model.")
    parser.add_argument("-r", "--root", type=str, default=None,
                        help=f"Root of the endpoints (default '')")
    parser.add_argument("-p", "--port", type=int, default=None,
                        help=f"Server port (default: {DEFAULT_SERVER_PORT} or from the config file.)")
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')

    # Parse the arguments and return the result as a Namespace object
    try:
        return parser.parse_args()
    except SystemExit:
        sys.exit(1)


# Parse command-line arguments and configuration file
args = parse_arguments()
config = load_config(args.config)
root: str = config[CONFIG_SECTION].get('root', '')
if args.root:
    root = args.root


@app.post(f"{root}/classify_document", response_model=ClassificationResponse)
def classify_document(document: Document):
    # For demonstration, we'll just return a dummy label
    if not document.document_text:
        raise HTTPException(status_code=400, detail="Document text is required")

    # Classify the text
    logger.info(f"Receive request: {document.document_text}")
    results = classifier.infer([document.document_text])
    label = results[0].label
    confidence = results[0].prob
    logger.info(f"Classification: {label}, Confidence: {confidence}")

    # Respond
    message = "Class uncertain" if label == classifier.other_label else "Classification successfully"

    return ClassificationResponse(
        message=message,
        label=label
    )


def main():
    global root, classifier, args, config

    # Get parameters from the config file
    port = int(config[CONFIG_SECTION].get('port', str(DEFAULT_SERVER_PORT)))
    model = config[CONFIG_SECTION].get('model', None)

    # Override parameters with command line arguments
    if args.port:
        port = args.port
    if args.model:
        model = args.model
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    if not model:
        logger.error(f"No classifier model was given (not in config file nor in argument)")
        sys.exit(1)

    try:
        classifier = TextClassifier.load(model)
    except Exception as e:
        logger.error(f"Problem loading classifier model from {model}. {e}")
        sys.exit(1)

    uvicorn.run(app, host="0.0.0.0", port=port)


# Run the application using uvicorn
if __name__ == "__main__":
    main()
