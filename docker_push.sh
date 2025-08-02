#!/bin/bash
# Build and push image to Docker Hub
# Make sure you're logged in with `docker login`

IMAGE_NAME="yourdockerhubusername/kaggle-ml-pipeline"

docker build -t $IMAGE_NAME .
docker push $IMAGE_NAME
