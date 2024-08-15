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

DEFAULT_CONFIG_SECTION = "DEFAULT"
API_CONFIG_SECTION = "API"

classifier: TextClassifier | None = None


# Define the request body
class Document(BaseModel):
    document_text: str


# Define the response body
class ClassificationResponse(BaseModel):
    message: str
    label: str


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
config = configparser.ConfigParser()
config.read(args.config)


def get_property(section: str, key: str, default=None) -> str:
    value = config[section].get(key)
    if not value:
        value = config[DEFAULT_CONFIG_SECTION].get(key, default)
    return value


root: str = get_property(API_CONFIG_SECTION, 'root', '')
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
    port = int(get_property(API_CONFIG_SECTION, 'port', str(DEFAULT_SERVER_PORT)))
    model = get_property(API_CONFIG_SECTION, 'model')

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
