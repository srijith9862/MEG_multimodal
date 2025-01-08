import os
import warnings
from datasets import Dataset, Audio
from transformers import ClapProcessor, ClapModel
import numpy as np
import librosa
import torch
# Suppress all warnings
warnings.filterwarnings("ignore")

# Load CLAP model and processor for the 'fused' version
model = ClapModel.from_pretrained("laion/clap-htsat-fused")
processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")

story_embeddings = []

# Loop through each story
for i in range(1, 5):
    print(f"running story{i}")
    text_embeddings = {0: []}

    # Read input texts from file
    with open(f"stimuli/text/story{i}.txt", 'r', encoding='utf-8') as f:
        input_texts = f.read().split()

    print(len(input_texts))

    for j, word in enumerate(input_texts):
        word_cleaned = word.rstrip('.')

        # Define possible paths for audio files
        audio_file_variants = [
            f"Samantha_Wavfiles/story{i}/{i-1}_{j}/{word_cleaned}_s-Samantha_r-150.wav",
            f"Samantha_Wavfiles/story{i}/{i-1}_{j}/{word_cleaned.lower()}_s-Samantha_r-150.wav",
            f"Samantha_Wavfiles/story{i}/{i-1}_{j}/{word_cleaned.upper()}_s-Samantha_r-150.wav",
            f"Samantha_Wavfiles/story{i}/{i-1}_{j}/{word_cleaned.capitalize()}_s-Samantha_r-150.wav"
        ]

        # Attempt to load audio file
        audio = None
        for audio_file in audio_file_variants:
            if os.path.exists(audio_file):
                audio, _ = librosa.load(audio_file, sr=None)
                break

        if audio is None:
            print(f"Audio file for word '{word_cleaned}' not found.")
            continue

        # Process the audio and text
        inputs = processor(audios=audio, return_tensors="pt", padding=True)

        audio_embed = model.get_audio_features(**inputs)
        text_embeddings[0].append(audio_embed[0].cpu().detach().numpy())

    story_embeddings.append(text_embeddings)

# Convert the list of embeddings into a numpy array
text_embeddings_array = np.array(story_embeddings, dtype=object)

# Save the embeddings array to a .npy file
np.save("fused_audio_joint_embeddings.npy", text_embeddings_array)

# Code to test the output format
ae = np.load("fused_audio_joint_embeddings.npy", allow_pickle=True)
print(ae.shape)
print(ae[0].keys())
print(np.array(ae[0][0]).shape)
