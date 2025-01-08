#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import argparse
import os
import numpy as np
import mne
import pandas as pd
import mne_bids
import time as tm
import matplotlib.pyplot as plt

ph_info = pd.read_csv("phoneme_info.csv")
subjects = pd.read_csv("participants.tsv", sep="\t")
subjects = subjects.participant_id.apply(lambda x: x.split("-")[1]).values


def meg_preprocessing(df_new,raw):
    meta = list()
    for annot in raw.annotations:
        d = eval(annot.pop("description"))
        for k, v in annot.items():
            assert k not in d.keys()
            d[k] = v
        meta.append(d)
    meta = pd.DataFrame(meta)
    meta["intercept"] = 1.0

    # compute voicing
    phonemes = meta.query('kind=="phoneme"')
    assert len(phonemes)
    for ph, d in phonemes.groupby("phoneme"):
        ph = ph.split("_")[0]
        match = ph_info.query("phoneme==@ph")
        assert len(match) == 1
        meta.loc[d.index, "voiced"] = match.iloc[0].phonation == "v"

    # compute word frquency and merge w/ phoneme
    meta["is_word"] = False
    words = meta.query('kind=="word"').copy()
    assert len(words) > 10
    # assert np.all(meta.loc[words.index + 1, "kind"] == "phoneme")
    meta.loc[words.index + 1, "is_word"] = True

    meta = meta.query('kind=="phoneme"')
    #assert len(meta.wordfreq.unique()) > 2

    # segment
    events = np.c_[
        meta.onset * raw.info["sfreq"], np.ones((len(meta), 2))
    ].astype(int)

    epochs = mne.Epochs(
        raw,
        events,
        tmin=-0.2,
        tmax=0.6,
        decim=10,
        baseline=(-0.2, 0.0),
        metadata=meta,
        preload=True,
        event_repeated="drop",
        picks=["meg"]
    )

    # threshold
    th = np.percentile(np.abs(epochs._data), 95)
    epochs._data[:] = np.clip(epochs._data, -th, th)
    epochs.apply_baseline()
    th = np.percentile(np.abs(epochs._data), 95)
    epochs._data[:] = np.clip(epochs._data, -th, th)
    epochs.apply_baseline()
    
    return epochs


def load_data(sub,ses,task):
    bids_path = mne_bids.BIDSPath(
    subject = sub, session = ses, task=task, datatype= "meg",
    root = './meg_listening/')
    
    raw = mne_bids.read_raw_bids(bids_path)
    raw.load_data().filter(0.5, 30.0, n_jobs=1)
    
    df = raw.annotations.to_data_frame()
    df_new = pd.DataFrame(df.description.apply(eval).to_list())
    
    ep = meg_preprocessing(df_new, raw)
    
    ep.metadata["half"] = np.round(
                np.linspace(0, 1.0, len(ep))
            ).astype(int)
    ep.metadata["task"] = ses
    ep.metadata["session"] = task
    
    phonemes = ep["not is_word"]
    X = phonemes.get_data() * 1e13
    return ep, X

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "CheXpert NN argparser")
    # parser.add_argument("subjectNum", help="Choose subject", type = int)
    # parser.add_argument("layers", help="Choose layers", type = int)
    # parser.add_argument("outputdir", help="Choose layers", type = int)
    args = parser.parse_args()
    for j in np.arange(27,28):
        if j<10:
            subj = '0'+str(j)
        else:
            subj = str(j)
        megdata = []
        epochs = []
        for i in np.arange(4):
            temp = []
            ep, xx = load_data(subj,'0',str(i))
            megdata.append(xx)
            epochs.append(ep)
        np.save('sub'+str(j)+'-meg-data-ses0-phoneme', megdata)
