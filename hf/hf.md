hf.md: Hosting Your iDragonfly Suite

Follow these instructions to deploy your trained models to a Hugging Face Space.

1\. Preparation

Ensure you have the following files inside the ./hf/ directory:

• 

server.py (The Multi-Model API)

• 

index.html (The Web Interface)

• 

Dockerfile (The Container Config)

• 

style.css (The Carbon Fiber Theme)

• 

Your Models: Copy nymph.pt, sprite.pt, etc., into this folder.

2\. Initialize Deployment

Open a terminal inside the ./hf/ directory and run:

Bash

Download codeCopy code

git init

git lfs install

git lfs track "\*.pt"

git add .

git commit -m "Initial iDragonfly Deployment"

3\. Connect to Hugging Face

1\. 

Create a new Space on Hugging Face. Select Docker as the SDK.

2\. 

Copy the remote URL provided by Hugging Face.

3\. 

Run the following commands in your terminal:

Bash

Download codeCopy code

git remote add origin https://huggingface.co/spaces/YOUR\_USERNAME/YOUR\_SPACE\_NAME

git push -u origin main

4\. Updates

• 

To update the Code: Pull the latest changes from the main iDragonfly GitHub repo. The updates to ./hf/ will appear automatically.

• 

To update the Live Space: Copy the new files/models into ./hf/, commit, and push from inside that directory.





