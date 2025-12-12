
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

DATA_DIR = "data/raw"

st.title("Acoustic Dataset Explorer")

# File selector
files = [f for f in os.listdir(DATA_DIR) if f.endswith(".wav")]
selected_file = st.selectbox("Choose an audio file", files)

if selected_file:
    file_path = os.path.join(DATA_DIR, selected_file)
    y, sr = librosa.load(file_path, sr=None)

    st.subheader("Waveform")
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    st.pyplot(fig)

    st.subheader("Spectrogram (dB)")
    S = librosa.stft(y)
    S_db
