import os
import numpy as np
from transformers import AutoTokenizer, ClapTextModel

# # Define the folder to save embeddings
# output_folder = "text_embeddings"
# os.makedirs(output_folder, exist_ok=True)  # Create directory if it doesn't exist

for i in range(1, 5):
    # Define the folder containing the audio files
    audio_files_folder = "Samantha_Wavfiles/story" + str(i)

    # Create a list of paths to audio files and corresponding input_text (words)
    audio_files = []
    input_texts = []
    for root, dirs, files in os.walk(audio_files_folder):
        for file in files:
            if file.endswith(".wav"):
                audio_files.append(os.path.join(root, file))
                # word = file.split("_")[0]
                # input_texts.append(word)
    
    # Load CLAP model and tokenizer
    model = ClapTextModel.from_pretrained("laion/clap-htsat-unfused")
    tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

    # Initialize an empty list to store text embeddings for each word
    embeddings_list = []

    # Process each input text and store the text embeddings
    for input_text in input_texts:
        inputs = tokenizer([input_text], padding=True, return_tensors="pt")
        outputs = model(**inputs)
        text_embeds = outputs.text_embeds
        embeddings_list.append(text_embeds.detach().numpy())

    # Concatenate embeddings into a single array
    # embeddings_array = np.concatenate(embeddings_list, axis=0)

    # Save embeddings to a single .npy file
    # np.save(os.path.join(f"text_embeddings_story{i}.npy"), embeddings_array)


# import os
# import json
# from transformers import AutoTokenizer, ClapTextModelWithProjection

# for i in range(1,5):
#     # Define the folder containing the audio files
#     audio_files_folder = "Samantha_Wavfiles/story" + str(i)

#     # Create a list of paths to audio files and corresponding input_text (words)
#     audio_files = []
#     input_texts = []
#     for root, dirs, files in os.walk(audio_files_folder):
#         for file in files:
#             if file.endswith(".wav"):
#                 audio_files.append(os.path.join(root, file))
#                 word = file.split("_")[0]
#                 input_texts.append(word)
#     # Load CLAP model and tokenizer
#     model = ClapTextModelWithProjection.from_pretrained("laion/clap-htsat-unfused")
#     tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

#     # Initialize an empty dictionary to store text embeddings for each word
#     embeddings_dict = {}

#     # Process each input text and store the text embeddings
#     for input_text in input_texts:
#         inputs = tokenizer([input_text], padding=True, return_tensors="pt")
#         outputs = model(**inputs)
#         text_embeds = outputs.text_embeds
#         embeddings_dict[input_text] = text_embeds.tolist()

#     # Write embeddings to a JSON file
#     output_file = "text_embeddings_story" + str(i) + ".json"
#     with open(output_file, "w") as f:
#         json.dump(embeddings_dict, f)
