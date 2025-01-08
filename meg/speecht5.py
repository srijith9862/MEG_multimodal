import os
import warnings
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
import numpy as np
import torch

# Load SpeechT5 model and processor for text
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

# Suppress all warnings
warnings.filterwarnings("ignore")

# Initialize list to store story embeddings
story_text_embeddings = []

# Function to extract text features
def extract_text_features(input_texts):
    text_embeddings = {}
    for k in range(1):
        text_embeddings[k] = []
    
    for word in input_texts:
        if word.endswith('.'):
            word = word[:-1]
        
        # Process text input
        inputs = processor(text=word, return_tensors="pt", padding=True)
        
        # Extract text features using the model
        with torch.no_grad():
            outputs = model.get_input_embeddings()(inputs.input_ids)
        
        # Store the text embeddings
        outputs = outputs.mean(dim=1).cpu().numpy()
        text_embeddings[0].append(outputs[0])
    
    return text_embeddings

# Iterate through each story
for i in range(1, 5):
    print(f"running story{i}")
    
    with open(f"stimuli/text/story{i}.txt", 'r', encoding='utf-8') as f:
        input_texts = f.read().split()
    
    print(len(input_texts))
    
    text_embeddings = extract_text_features(input_texts)
    
    # Append the text embeddings of the current story
    story_text_embeddings.append(text_embeddings)

# Convert the list of embeddings into a numpy array
text_embeddings_array = np.array(story_text_embeddings, dtype=object)

# Save the embeddings array to a .npy file
np.save("speech_joint_embeddings.npy", text_embeddings_array)

# Code to test the output format
ae = np.load("speech_joint_embeddings.npy", allow_pickle=True)
print(ae.shape)
print(ae[0].keys())
print(np.array(ae[0][0]).shape)
