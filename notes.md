Task 1 – Sentence Transformer
For this task, I built a sentence transformer using HuggingFace's bert-base-uncased model. Instead of relying on just the [CLS] token for sentence embeddings, I used mean pooling across all token embeddings — which tends to give a more balanced representation of the entire sentence.

I tested it on a few example sentences and confirmed that the model returns consistent 768-dimensional embeddings, just as expected from BERT. Tokenization and encoding were handled using HuggingFace's AutoTokenizer.

Task 2 – Multi-Task Learning Model
The next step was to extend the transformer into a multi-task model. I kept the encoder shared and added two separate heads:

Task A: Sentence classification (e.g., classifying into topics like Sports, Tech, Politics, Health)

Task B: Sentiment analysis (e.g., Positive, Neutral, Negative)

Just like before, I applied mean pooling to get sentence-level representations. Then I passed those through the respective task-specific heads. This setup lets the model share general language understanding from the encoder, while still learning distinct features for each task.

Task 3 – Training Considerations
Here, I explored different strategies for training and transfer learning in a multi-task setup. I broke it down into a few key scenarios:

1. Freezing the Entire Network
   If we freeze everything (the encoder and both heads), we're essentially using the model in inference mode. It's super fast, but it won't adapt to your specific tasks. Useful as a baseline — but not ideal for real learning.

2. Freezing Just the Transformer
   This keeps the core language model intact while letting the heads learn. It's a great choice if you’re working with a small dataset or if your tasks are similar to what BERT already understands. But you might miss out on deeper task-specific nuance.

3. Freezing One Head
   This comes in handy in continual learning. Maybe one task (like sentiment) is well-trained and you want to keep it stable, while allowing the other (like topic classification) to keep learning.

Transfer Learning Strategy
My transfer learning plan would look like this:

Start with bert-base-uncased

Freeze the lower layers (which capture general language patterns)

Fine-tune the top layers + heads, which are more task-specific

This way, the model retains its foundational knowledge while still adapting to the new tasks — a solid balance between stability and flexibility.

Task 4 – Multi-Task Training Loop (Bonus)
Finally, I built a training loop that simulates learning on both tasks.

I used 5 sample sentences with synthetic labels for both topic and sentiment.

The model computes losses for each task separately using CrossEntropyLoss, then combines them (with a balance weight α = 0.5).

Each batch gives predictions for both tasks, and the loop handles optimization and backpropagation jointly.

Even though the training is simulated (tiny dataset, few epochs), it demonstrates how a real MTL pipeline would be structured.
