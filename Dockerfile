# Use Ubuntu's current LTS
FROM ubuntu:jammy-20230804

# Make sure to not install recommends and to clean the
# install to minimize the size of the container as much as possible.
RUN apt-get update && \
    apt-get install -y gnupg && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys BA6932366A755776 && \
    apt-get install --no-install-recommends -y python3=3.10.6-1~22.04 python3-pip python3-venv=3.10.6-1~22.04  && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory within the container
WORKDIR /app

# Copy necessary files to the container
COPY requirements.txt .
COPY main.py .
COPY download_models.py .
COPY metadata.sql .
COPY prompt.md .

# Create a virtual environment in the container
RUN python3 -m venv .venv

# Activate the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Install Python dependencies from the requirements file
RUN pip install --no-cache-dir -r requirements.txt && \
    # Get the models from Hugging Face to bake into the container
    python3 download_models.py

# Make port 6000 available to the world outside this container
EXPOSE 6000

ENTRYPOINT [ "python3" ]

# Run main.py when the container launches
CMD [ "main.py" ]
