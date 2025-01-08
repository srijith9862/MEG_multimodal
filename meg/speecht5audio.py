import os
import warnings
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
import numpy as np
import librosa
import torch

# Load SpeechT5 model and processor for audio
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

# Suppress all warnings
warnings.filterwarnings("ignore")

# Initialize list to store story embeddings
story_audio_embeddings = []

# Function to extract audio features
def extract_audio_features(input_texts, story_num):
    audio_embeddings = {}
    for k in range(1):
        audio_embeddings[k] = []
    
    for j, word in enumerate(input_texts):
        if word.endswith('.'):
            word = word[:-1]
        
        audio_file = f"Samantha_Wavfiles/story{story_num}/{story_num-1}_{j}/{word}_s-Samantha_r-150.wav"
        
        # Attempt to load the audio file with different capitalizations if not found
        try:
            audio, sr = librosa.load(audio_file, sr=None)
        except FileNotFoundError:
            try:
                audio_file = f"Samantha_Wavfiles/story{story_num}/{story_num-1}_{j}/{word.lower()}_s-Samantha_r-150.wav"
                audio, sr = librosa.load(audio_file, sr=None)
            except FileNotFoundError:
                try:
                    audio_file = f"Samantha_Wavfiles/story{story_num}/{story_num-1}_{j}/{word.upper()}_s-Samantha_r-150.wav"
                    audio, sr = librosa.load(audio_file, sr=None)
                except FileNotFoundError:
                    audio_file = f"Samantha_Wavfiles/story{story_num}/{story_num-1}_{j}/{word.capitalize()}_s-Samantha_r-150.wav"
                    audio, sr = librosa.load(audio_file, sr=None)
        
        # Resample the audio to 16,000 Hz
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Process audio input
        inputs = processor(audio=audio, sampling_rate=sr, return_tensors="pt", padding=True)
        
        # Extract audio features using the model's hidden states
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        print(outputs.keys())
        exit(0)
        # Store the audio embeddings (hidden states)
        audio_embeddings[0].append(outputs.hidden_states[-1][0].cpu().numpy())
    
    return audio_embeddings

# Iterate through each story
for i in range(1, 5):
    print(f"running story{i}")
    
    with open(f"stimuli/text/story{i}.txt", 'r', encoding='utf-8') as f:
        input_texts = f.read().split()
    
    print(len(input_texts))
    
    audio_embeddings = extract_audio_features(input_texts[:1], i)
    
    # Append the audio embeddings of the current story
    story_audio_embeddings.append(audio_embeddings)

# Convert the list of embeddings into a numpy array
audio_embeddings_array = np.array(story_audio_embeddings, dtype=object)

# Save the embeddings array to a .npy file
np.save("speech_audio_joint_embeddings.npy", audio_embeddings_array)

# Code to test the output format
ae = np.load("speech_audio_joint_embeddings.npy", allow_pickle=True)
print(ae.shape)
print(ae[0].keys())
print(np.array(ae[0][0]).shape)
