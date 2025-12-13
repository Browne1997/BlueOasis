import streamlit as st
import os
import pandas as pd
import random

# Import helper functions
from visualisation import (   
    plot_waveform,
    plot_spectrogram,
    plot_mfcc,
    plot_class_distribution,
    plot_duration_distribution,
)
from preprocessing import (
    load_audio,
    extract_spectrogram,
    extract_mfcc,
)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(BASE_DIR, "data", "raw", "audio")
META_FILE = os.path.join(BASE_DIR, "data", "raw", "meta", "esc50.csv")

# Load metadata
meta = pd.read_csv(META_FILE)

st.title("ESC-50 Acoustic Dataset Explorer")

# Dataset-level exploration
st.header("Dataset Overview")
st.pyplot(plot_class_distribution())
st.pyplot(plot_duration_distribution())

# File selector
files = meta["filename"].tolist()
selected_file = st.selectbox("Choose an audio file", files)

# Random sample button
if st.button("🎲 Random Sample"):
    selected_file = random.choice(files)

if selected_file:
    file_path = os.path.join(AUDIO_DIR, selected_file)
    y, sr = load_audio(file_path)   # <-- fixed

    st.subheader(f"Audio Playback: {selected_file}")
    st.audio(file_path, format="audio/wav")

    st.subheader("Waveform")
    st.pyplot(plot_waveform(y, sr))

    st.subheader("Spectrogram (dB)")
    S_db = extract_spectrogram(y, sr)
    st.pyplot(plot_spectrogram(S_db, sr))

    st.subheader("MFCCs")
    mfccs = extract_mfcc(y, sr)
    st.pyplot(plot_mfcc(mfccs))


