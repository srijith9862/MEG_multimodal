import os
import warnings
from transformers import AutoProcessor, WavLMModel
import numpy as np
import torch
import librosa

# Load WavLM model and processor
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
model = WavLMModel.from_pretrained("facebook/wav2vec2-base-960h")

# Suppress all warnings
warnings.filterwarnings("ignore")

story_embeddings = []
for i in range(1, 5):
    print(f"Running story{i}")
    text_embeddings = {0: []}
    with open(f"stimuli/text/story{i}.txt", 'r', encoding='utf-8') as f:
        input_texts = f.read().split()
    print(len(input_texts))

    for j in range(len(input_texts)):
        # Normalize the punctuation in input_texts[j]
        word = input_texts[j].rstrip('.')

        # Try different cases to locate the audio file
        audio_file_found = False
        for variant in [word, word.lower(), word.upper(), word.capitalize()]:
            audio_file = f"Samantha_Wavfiles/story{i}/{i-1}_{j}/{variant}_s-Samantha_r-150.wav"
            if os.path.exists(audio_file):
                audio_file_found = True
                break
        
        if not audio_file_found:
            print(f"Audio file for '{word}' not found in story{i}, index {j}.")
            continue

        try:
            audio, _ = librosa.load(audio_file, sr=None)
        except Exception as e:
            print(f"Error loading audio file '{audio_file}': {e}")
            continue

        # Process the audio and extract features
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
        
        extract_features = outputs.extract_features
        # print(f"Extracted features shape for word '{word}':", extract_features.shape)

        # Average across the time dimension
        mean_extract_features = extract_features.mean(dim=1).cpu().numpy()
        text_embeddings[0].append(mean_extract_features[0])

    story_embeddings.append(text_embeddings)

# Convert the list of embeddings into a numpy array
text_embeddings_array = np.array(story_embeddings, dtype=object)

# Save the embeddings array to a .npy file
np.save("extract_features_wavlm.npy", text_embeddings_array)

# Load the embeddings for testing the output format
ae = np.load("extract_features_wavlm.npy", allow_pickle=True)
print(ae.shape)
print(ae[0].keys())
print(np.array(ae[0][0]).shape)
