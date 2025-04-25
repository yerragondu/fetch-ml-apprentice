Fetch ML Apprentice – Take-Home Challenge
Hi! This repository contains my solution to the Machine Learning Engineer Apprentice take-home assessment for Fetch.

The project demonstrates how to build a sentence transformer using BERT, expand it into a multi-task learning (MTL) model, and implement relevant training strategies. The goal was not just to build something that works, but to explain the decisions behind it clearly and practically.

Tasks Overview
Task 1 – Sentence Transformer

Built a sentence encoder using HuggingFace’s bert-base-uncased model.

Used mean pooling across token embeddings to generate sentence-level embeddings instead of relying on the [CLS] token.

Tested with a few sentences and verified the output shape is [batch size, 768], as expected from BERT.

Task 2 – Multi-Task Learning Model

Extended the sentence transformer to support two tasks:

Task A: Sentence classification (example categories include News, Sports, Tech, Health)

Task B: Sentiment analysis (Positive, Neutral, Negative)

Kept the encoder shared and added separate classification heads for each task, so both tasks benefit from the same language understanding while learning independently.

Task 3 – Training Considerations

Outlined different training strategies:

Freezing the full network

Freezing just the transformer encoder

Freezing only one of the task-specific heads

Proposed a simple transfer learning plan where the lower BERT layers are frozen and the upper layers are fine-tuned along with the heads. This helps balance general language understanding with task-specific learning.

Task 4 – Bonus: Training Loop

Implemented a multi-task training loop using PyTorch.

Created a dummy dataset with five example sentences and fake labels.

Logged predictions and computed both losses and accuracy during training.

Demonstrated how to combine task-specific losses and jointly train the model in a clean and scalable way.

How to Run

1. First, install the required packages:
   pip install -r requirements.txt

2. To generate embeddings using the sentence transformer:
   %run sentence_transformer.py
   This will print the embeddings to the console and export them to a file called sentence_embeddings.json.

3. To test the multi-task model:
   %run multitask_model.py
   This will:

Run a forward pass using dummy sentences.

Print predictions for both tasks (topic and sentiment).

Show the mapped class names for better readability.

4. To run the training loop and simulate multi-task training:
   %run training.py

This will:

Simulate training on a small labeled dataset.

Track and print loss and accuracy for each task.

Show predicted vs. actual classes during training.

Notes:
All training data in this project is synthetic and for demonstration only. The architecture is fully functional and can easily be adapted to real datasets. Mean pooling was used to better capture full-sentence context across varying lengths.

## Running with Docker

You can run this project entirely in Docker.

### Build the image (only once):

In terminal run:
docker build -t fetch .

and then run: docker run --rm fetch
