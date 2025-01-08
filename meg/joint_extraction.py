import os
import warnings
from datasets import Dataset, Audio
from transformers import AutoProcessor, ClapModel
import numpy as np
import librosa


# Load CLAP model and processor
model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
# Suppress all warnings
# warnings.filterwarnings("ignore")
story_embeddings = []
for i in range(1,5):
    print(f"running story{i}")
    text_embeddings = {}
    for k in range(1):
        text_embeddings[k]=[]
    with open(f"stimuli/text/story{i}.txt", 'r', encoding='utf-8') as f:
        input_texts = f.read().split()
    print(len(input_texts))
    for j in range(len(input_texts)):
        if(input_texts[j].endswith('.')):
            input_texts[j] = input_texts[j][:-1]
        # print(input_texts[j])
        audio_file = f"Samantha_Wavfiles/story{i}/{i-1}_{j}/{input_texts[j]}_s-Samantha_r-150.wav" 
        try:
            audio, _ = librosa.load(audio_file, sr=None) 
        except FileNotFoundError:
            try:
                audio_file = f"Samantha_Wavfiles/story{i}/{i-1}_{j}/{input_texts[j].lower()}_s-Samantha_r-150.wav" 
                audio, _ = librosa.load(audio_file, sr=None)
            except FileNotFoundError:
                try:
                    audio_file = f"Samantha_Wavfiles/story{i}/{i-1}_{j}/{input_texts[j].upper()}_s-Samantha_r-150.wav" 
                    audio, _ = librosa.load(audio_file, sr=None)
                except FileNotFoundError:
                    audio_file = f"Samantha_Wavfiles/story{i}/{i-1}_{j}/{input_texts[j].capitalize()}_s-Samantha_r-150.wav" 
                    audio, _ = librosa.load(audio_file, sr=None)
        # try:
        #     audio_file = f"Samantha_Wavfiles/story{i}/{i-1}_{j}/{input_texts[j]}_s-Samantha_r-150.wav" 
        # except:
        #     audio_file = f"Samantha_Wavfiles/story{i}/{i-1}_{j}/{input_texts[j].lower()}_s-Samantha_r-150.wav" 
        # audio, _ = librosa.load(audio_file, sr=None)

        inputs = processor(text=input_texts[j], audios=audio,
                           return_tensors="pt", padding=True)

        outputs = model(**inputs, output_hidden_states=True)
        text_embeddings[0].append(outputs.audio_embeds[0].cpu().detach().numpy())
        # text_embeddings[0].append(outputs.text_embeds[0].cpu().detach().numpy())
    story_embeddings.append(text_embeddings)
# Convert the list of embeddings into numpy array
text_embeddings_array = np.array(story_embeddings)
# Save the embeddings array to a .npy file
np.save("audio_joint_embeddings.npy", text_embeddings_array)
# np.save("text_joint_embeddings.npy", text_embeddings_array)
# Code to test the output format
ae = np.load("audio_joint_embeddings.npy", allow_pickle=True)
# ae = np.load("text_joint_embeddings.npy", allow_pickle=True)
print(ae.shape)  
print(ae[0].keys())
print(np.array(ae[0][0]).shape)  



