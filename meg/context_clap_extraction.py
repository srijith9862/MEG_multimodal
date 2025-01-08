import os
import warnings
from datasets import Dataset, Audio
from transformers import AutoProcessor, ClapModel
import numpy as np
import librosa
import soundfile as sf
import string 

# Load CLAP model and processor
model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")

# Suppress all warnings
# warnings.filterwarnings("ignore")

def remove_punctuation(word):
    return word.translate(str.maketrans('', '', string.punctuation))

story_embeddings = []
for i in range(1, 5):
    print(f"running story{i}")
    text_embeddings = {}
    for k in range(1):
        text_embeddings[k] = []
    with open(f"stimuli/text/story{i}.txt", 'r', encoding='utf-8') as f:
        input_texts = f.read().split()
    print(len(input_texts))
    
    context_length = 5
    c=0
    for j in range(len(input_texts) - context_length + 1):
        c=c+1
        context = input_texts[j:j + context_length]
        context = [remove_punctuation(word) for word in context]
        context_str = " ".join(context)
        
        # Concatenate audio files for the context
        concatenated_audio = []
        temp=j
        for word in context:
            # Prepare file path for the word
            audio_file_base = f"Samantha_Wavfiles/story{i}/{i-1}_{temp}/"
            temp+=1
            file_variants = [
                audio_file_base +f"{word}"+ "_s-Samantha_r-150.wav",
                audio_file_base+ f"{word}".lower() + "_s-Samantha_r-150.wav",
                audio_file_base+f"{word}".upper() + "_s-Samantha_r-150.wav",
                audio_file_base+f"{word}".capitalize() + "_s-Samantha_r-150.wav"
            ]
            audio_data = None
            fl=0
            for variant in file_variants:
                if os.path.exists(variant):
                    fl=1
                    audio_data, _ = librosa.load(variant, sr=None)
                    break
            if fl==0:
                print(i,c,word, context, temp-1)
                exit(0)
            if audio_data is not None:
                concatenated_audio.append(audio_data)
        
        # Ensure we have some audio data before processing
        if concatenated_audio:
            concatenated_audio = np.concatenate(concatenated_audio)
        
            inputs = processor(text=context_str, audios=concatenated_audio, return_tensors="pt", padding=True)
            outputs = model(**inputs, output_hidden_states=True)
            text_embeddings[0].append(outputs.text_embeds[0].cpu().detach().numpy())
        else:
            print(f"Missing audio for context: '{context_str}'")
            print(c, i, context)
            exit(0)
    for j in range(len(input_texts) - context_length + 1, len(input_texts)):
        context_length-=1
        c=c+1
        context = input_texts[j:j + context_length]
        context = [remove_punctuation(word) for word in context]
        context_str = " ".join(context)
        
        # Concatenate audio files for the context
        concatenated_audio = []
        temp=j
        for word in context:
            # Prepare file path for the word
            audio_file_base = f"Samantha_Wavfiles/story{i}/{i-1}_{temp}/"
            temp+=1
            file_variants = [
                audio_file_base +f"{word}"+ "_s-Samantha_r-150.wav",
                audio_file_base+ f"{word}".lower() + "_s-Samantha_r-150.wav",
                audio_file_base+f"{word}".upper() + "_s-Samantha_r-150.wav",
                audio_file_base+f"{word}".capitalize() + "_s-Samantha_r-150.wav"
            ]
            audio_data = None
            fl=0
            for variant in file_variants:
                if os.path.exists(variant):
                    fl=1
                    audio_data, _ = librosa.load(variant, sr=None)
                    break
            if fl==0:
                print(i,c,word, context, temp-1)
                exit(0)
            if audio_data is not None:
                concatenated_audio.append(audio_data)
        
        # Ensure we have some audio data before processing
        if concatenated_audio:
            concatenated_audio = np.concatenate(concatenated_audio)
        
            inputs = processor(text=context_str, audios=concatenated_audio, return_tensors="pt", padding=True)
            outputs = model(**inputs, output_hidden_states=True)
            text_embeddings[0].append(outputs.text_embeds[0].cpu().detach().numpy())
        else:
            print(f"Missing text for context: '{context_str}'")
            print(c, i, context)
            exit(0)
    story_embeddings.append(text_embeddings)

# Convert the list of embeddings into numpy array
text_embeddings_array = np.array(story_embeddings)
# Save the embeddings array to a .npy file
np.save("context_text_joint_embeddings.npy", text_embeddings_array)

# Code to test the output format
ae = np.load("context_text_joint_embeddings.npy", allow_pickle=True)
print(ae.shape)  
print(ae[0].keys())
print(np.array(ae[0][0]).shape)
