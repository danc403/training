FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Set up the mandatory non-root user (ID 1000)
RUN useradd -m -u 1000 user
USER user

# Set environment variables for the user's home and path
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Establish the working directory inside the home folder
WORKDIR $HOME/app

# Copy requirements and install (ensures ownership is correct)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the iDragonfly stack
COPY --chown=user . .

# Launch the server binding to 0.0.0.0 and port 7860
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "7860"]
