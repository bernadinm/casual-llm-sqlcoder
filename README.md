# How to Deploy Hugging Face Models in a Docker Container

This is the repository for the tutorial on how to deploy Hugging Face models in a Docker container and to expose it as a web service endpoint using Flask.

This is a manual deployment method and will be superceeded by an automated way with CICD and infra tools

### How To Build

```bash
make # it will build and download the LLM locally within the container
```

### How to Run

```bash
curl -X POST http://localhost:6000/generate \
    -H "Content-Type: application/json" \
    -d '{
          "question": "What are the total sales for each product category?"
        }'
```

### Note

This model is to large to run, please deploy to a server with a GPU
