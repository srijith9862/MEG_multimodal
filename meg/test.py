import os
import numpy as np
from transformers import AutoTokenizer, ClapTextModel
import torch
# # Load CLAP model and tokenizer
# model = ClapTextModel.from_pretrained("laion/clap-htsat-unfused")
# tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

# # Define the paths to your stories
# stories_folder = "./stimuli/text"
# story_files = [os.path.join(stories_folder, f) for f in os.listdir(stories_folder) if f.endswith('new.txt')]



# story_embeddings = []
# for i in range(1,5):
#     print(f"running story{i}")
#     text_embeddings = {}
#     for j in range(13):
#         text_embeddings[j]=[]
#     with open(f"stimuli/text/story{i}.txt", 'r', encoding='utf-8') as f:
#         input_texts = f.read().split()
#     for input_text in input_texts:
#         inputs = tokenizer(input_text, padding=True, return_tensors="pt")
#         # print(inputs)
#         outputs = model(**inputs, output_hidden_states=True)
#         # print(dir(outputs))
#         # print(len(outputs.hidden_states))
#         for k,layer in enumerate(outputs.hidden_states):
#             average_vector = torch.mean(layer, dim=1)
#             # print(average_vector[0].shape)
#             text_embeddings[k].append(average_vector[0].cpu().detach().numpy())
#     # print(layer.shape)
#     # print(np.array(text_embeddings).shape)
#     # print(text_embeddings.keys())
#     # print(np.array(text_embeddings[0]).shape)
#     # exit(0)
#     story_embeddings.append(text_embeddings)


# # Convert the list of embeddings into numpy array
# text_embeddings_array = np.array(story_embeddings)


# # Save the embeddings array to a .npy file
# np.save("text_embeddings.npy", text_embeddings_array)

# Code to test the output format
ae = np.load("text_embeddings.npy", allow_pickle=True)
print(ae.shape)  
print(ae[0].keys())
print(np.array(ae[0][0]).shape)  
