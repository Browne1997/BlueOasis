import streamlit as st
import os
import pandas as pd
import random
import librosa
import numpy as np

from visualisation import (
    load_raw_audio,
    load_processed_audio,
    load_mfcc_feature,
    plot_waveform,
    plot_spectrogram,
    plot_mfcc,
    plot_class_distribution,
    plot_duration_distribution,
)

from visualisation import extract_spectrogram, extract_mfcc

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_AUDIO_DIR = os.path.join(BASE_DIR, "data", "raw", "audio")
PROC_AUDIO_DIR = os.path.join(BASE_DIR, "data", "processed", "audio")
FEATURE_DIR = os.path.join(BASE_DIR, "data", "processed", "features")
META_FILE = os.path.join(BASE_DIR, "data", "raw", "meta", "esc50.csv")

meta = pd.read_csv(META_FILE)

st.title("ESC‑50 Dataset Explorer")

# -----------------------------
# Dataset Overview
# -----------------------------
st.header("Dataset Overview")
st.pyplot(plot_class_distribution())
st.pyplot(plot_duration_distribution())

# -----------------------------
# File Selection
# -----------------------------
files = meta["filename"].tolist()
selected_file = st.selectbox("Choose an audio file", files)

if st.button("🎲 Random Sample"):
    selected_file = random.choice(files)

if selected_file:
    st.header(f"Exploring: {selected_file}")

    # Tabs for RAW / PROCESSED / FEATURES
    tab1, tab2, tab3 = st.tabs(["Raw Audio", "Processed Audio", "MFCC Features"])

    # -----------------------------
    # RAW AUDIO TAB
    # -----------------------------
    with tab1:
        st.subheader("Raw Audio Playback")
        raw_path = os.path.join(RAW_AUDIO_DIR, selected_file)
        y_raw, sr_raw = load_raw_audio(selected_file)
        st.audio(raw_path)

        st.subheader("Waveform (Raw)")
        st.pyplot(plot_waveform(y_raw, sr_raw))

        st.subheader("Spectrogram (Raw)")
        S_db_raw = extract_spectrogram(y_raw, sr_raw)
        st.pyplot(plot_spectrogram(S_db_raw, sr_raw))

        st.subheader("MFCCs (Raw)")
        mfcc_raw = extract_mfcc(y_raw, sr_raw)
        st.pyplot(plot_mfcc(mfcc_raw))

    # -----------------------------
    # PROCESSED AUDIO TAB
    # -----------------------------
    with tab2:
        st.subheader("Processed Audio Playback")
        proc_path = os.path.join(PROC_AUDIO_DIR, selected_file)

        if os.path.exists(proc_path):
            y_proc, sr_proc = load_processed_audio(selected_file)
            st.audio(proc_path)

            st.subheader("Waveform (Processed)")
            st.pyplot(plot_waveform(y_proc, sr_proc))

            st.subheader("Spectrogram (Processed)")
            S_db_proc = extract_spectrogram(y_proc, sr_proc)
            st.pyplot(plot_spectrogram(S_db_proc, sr_proc))

            st.subheader("MFCCs (Processed)")
            mfcc_proc = extract_mfcc(y_proc, sr_proc)
            st.pyplot(plot_mfcc(mfcc_proc))
        else:
            st.warning("Processed audio not found. Run preprocessing.py first.")

    # -----------------------------
    # MFCC FEATURE TAB
    # -----------------------------
    with tab3:
        st.subheader("Extracted MFCC Feature File")
        mfcc_file = selected_file.replace(".wav", "_mfcc.npy")
        mfcc_path = os.path.join(FEATURE_DIR, mfcc_file)

        if os.path.exists(mfcc_path):
            mfcc = np.load(mfcc_path)
            st.write(f"MFCC shape: {mfcc.shape}")
            st.pyplot(plot_mfcc(mfcc, title="Heatmap of MFCCs"))
        else:
            st.warning("MFCC feature file not found. Run feature_extraction.py first.")



