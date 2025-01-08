import os
import warnings
from transformers import AutoTokenizer, T5EncoderModel
import numpy as np
import torch


# Load the flan-t5 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = T5EncoderModel.from_pretrained("google/flan-t5-base")

# Suppress all warnings
warnings.filterwarnings("ignore")

story_embeddings = []
for i in range(1, 5):
    print(f"Processing story{i}")
    text_embeddings = {0: []}
    with open(f"stimuli/text/story{i}.txt", 'r', encoding='utf-8') as f:
        input_texts = f.read().split()
    print(len(input_texts))

    for j in range(len(input_texts)):
        word = input_texts[j]

        # Tokenize the word
        inputs = tokenizer(word, return_tensors="pt", truncation=True, padding=True)

        try:
            # Process the text and extract features
            with torch.no_grad():
                outputs = model(**inputs)
            # Get the hidden states of the last layer
            hidden_states = outputs.last_hidden_state

            # Average across the sequence dimension
            mean_hidden_states = hidden_states.mean(dim=1).cpu().numpy()
            text_embeddings[0].append(mean_hidden_states[0])
        
        except Exception as e:
            print(f"Error processing word '{word}': {e}")
            continue

    story_embeddings.append(text_embeddings)

# Convert the list of embeddings into a numpy array
text_embeddings_array = np.array(story_embeddings, dtype=object)

# Save the embeddings array to a .npy file
np.save("flan_t5_joint_embeddings.npy", text_embeddings_array)

# Load the embeddings for testing the output format
ae = np.load("flan_t5_joint_embeddings.npy", allow_pickle=True)
print(ae.shape)
print(ae[0].keys())
print(np.array(ae[0][0]).shape)
