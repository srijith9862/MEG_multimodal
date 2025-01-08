import numpy as np
import os
import librosa
ae = np.load("context_audio_joint_embeddings.npy", allow_pickle=True)
print(ae.shape)  
# print(ae[0].keys())
print(np.array(ae[0][0]).shape)  
print(np.array(ae[1][0]).shape)  
print(np.array(ae[2][0]).shape)  
print(np.array(ae[3][0]).shape)  
# for i in np.arange(5,9):
#     print(i)x``