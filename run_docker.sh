#!/bin/bash

docker run "$@" -v "$(realpath data/training):/app/classifier/data/training" \
           -v "$(realpath data/model):/app/classifier/data/model" \
           -p 8000:8000 text_classifier_api