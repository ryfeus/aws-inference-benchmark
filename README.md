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

Make single prediction
```bash
curl -X POST -H "Content-Type: image/jpeg" --data-binary "@flower.png" http://<prefix>.us-east-1.elb.amazonaws.com/predict
```


Benchmark using apache benchmark
```bash
ab -n 10 -c 10 -p flower.png -T image/jpeg http://<prefix>.us-east-1.elb.amazonaws.com/predict
```


### Run locally

#### Build the Docker image

```bash
docker build -t image-inference .
```

#### Run the Docker container

```bash
docker run --rm -p 8080:8080 image-inference
```

#### Make a prediction using the REST API

```bash
curl -X POST -H "Content-Type: image/jpeg" --data-binary "@flower.png" http://localhost:8080/predict
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

## Copilot LLM example

This example demonstrates how to deploy large language model for text generation using transformers library on Amazon ECS/Fargate with AWS Copilot. This project provides an easy-to-follow example and a scalable solution for serving deep learning models in the cloud.

### Deploy

Clone repository
```bash
git clone https://github.com/ryfeus/aws-inference-benchmark.git
cd copilot/transformers/aws-copilot-inference-service
```

Clone model from Hugging Face repo. Example - LaMini T5 223M
```bash
git lfs install
git clone https://huggingface.co/MBZUAI/LaMini-T5-223M.git
mv LaMini-T5-223M model
```

Initialize the environment and deploy the application.

```bash
copilot env init
copilot deploy
```

Make single prediction
```bash
curl -X POST -H "Content-Type: application/json" -d '{"instruction":"Main tour attractions in Rome:?"}' http://<prefix>.us-east-1.elb.amazonaws.com/predict
```

### Run locally

#### Build the Docker image

```bash
docker build -t llm-inference .
```

#### Run the Docker container

```bash
docker run --rm -p 8080:8080 llm-inference
```

#### Make a prediction using the REST API

```bash
curl -X POST -H "Content-Type: application/json" -d '{"instruction":"Main tour attractions in Rome:?"}' http://localhost:8080/predict
```
