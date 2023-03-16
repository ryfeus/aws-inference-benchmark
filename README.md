# aws-inference-benchmark
Repository with the code for running deep learning inference benchmarks on different AWS instances and service types.

## Copilot example

This example demonstrates how to deploy a deep learning model for image inference using ONNX on Amazon ECS/Fargate with AWS Copilot. This project provides an easy-to-follow example and a scalable solution for serving deep learning models in the cloud.

### Requirements

- Python 3.6 or later
- Docker
- AWS CLI
- AWS Copilot

### Deploy

Clone repository
```bash
git clone https://github.com/ryfeus/aws-inference-benchmark.git
cd copilot/cpu/aws-copilot-inference-service
```

Initialize the environment and deploy the application.

```bash
copilot env init
copilot deploy
```

### Run locally

#### Build the Docker image

```bash
docker build -t image-inference .
```

#### Run the Docker container

```bash
docker run --rm -p 80:80 image-inference
```

#### Make a prediction using the REST API

```bash
curl -X POST -F "image=@panda.png" http://localhost:80/predict
```

### Test

#### Install the development dependencies

```bash
pip install -r dev-requirements.txt
```

#### Run the tests

```bash
pytest -v test_inference.py
```