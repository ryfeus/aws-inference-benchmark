# aws-inference-benchmark
Repository with the code for running deep learning inference benchmarks on different AWS instances and service types.

## Copilot example

ONNX example deployed to ECS/Fargate using AWS Copilot.

### Deploy

```bash
copilot env init
copilot deploy
```

### Run locally

```bash
docker build -t image-inference .
docker run --rm -p 80:80 image-inference
curl -X POST -F "image=@panda.png" http://localhost:80/predict
```

### Test

```bash
pip install -r dev-requirements.txt
pytest -v test_inference.py
```