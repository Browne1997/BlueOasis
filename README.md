## Data 
Passive Acoustic Data was collected from NOAA's GCP. Data comes from the ADEON Project which is an integrated network of deep water observatories across U.S (mid - and south atlantic outer continental shelf).
Audio data, metadata and documents specifically for this take were from this platform 'adeon/audio/ble/adeon_ble_amar384.1-2-3-4.16/' in which I used a subset of the .flac files

Pipeline
- convert .flac to raw uncompressed .wav files (preprocessing step)
- convert raw .wav into MFCC features (or spectrograms)
  - 2D vector array of time and feature - treat as a sequence of feature vectors like a spectrogram "image"

## Model Architecture 
- Use a CNN based architecture
  - Treat MFCCs/spectrograms as 2D images
  - CNNs learn local patterns (e.g. harmonic, formants) for audio classification
  - ResNet, Yolo or lightweight MobileNet (yolo-tiny) for edge deployment
Or use a Transformers (e.g. AST) to handle long-rage dependencies but heavier computational requirements
Or use RRNs treat the MFCCs as a time series of feature vectors - good for handling temporal dynamics matter (event detection)

model.py:
📝 Notes
Input shape: MFCCs are 2D arrays (coefficients × time). We treat them as grayscale images with one channel.
Conv layers: Learn local spectral patterns.
Pooling: Reduces dimensionality and captures invariances.
Fully connected layers: Map learned features to class probabilities.
Adjust dimensions: The fc1 input size depends on your MFCC shape — you’ll need to calculate (n_mfcc//pool_factor) × (time_frames//pool_factor) based on your preprocessing.

Decision reasoning:
I preprocess raw .wav files into MFCCs and spectrograms. These 2D feature representations are well‑suited for convolutional neural networks, which can learn local spectral patterns. For tasks requiring temporal modeling, I would extend this with recurrent layers or consider transformer architectures. For deployment, I would prioritize lightweight CNNs to balance accuracy and efficiency. In model.py I chose a CNN architecture because spectrograms/MFCCs resemble images, and CNNs are effective at learning local spectral features. For deployment, this architecture can be scaled down (MobileNet‑style) or extended with recurrent layers for temporal modeling.

## 📝 README outline
- Project overview: Acoustic ML pipeline demo for job application.

- Dataset: Source link, subset size, preprocessing notes.

- Setup: pip install -r requirements.txt

- Usage:
  - Run preprocessing: python src/preprocessing.py
  - Launch web app: python src/app.py

- Notes: Splitting strategy, model architecture reasoning, limitations.

## 🚀 How to run web application
Utilising GitHub Codespace - pull the repository 'locally' 

Check / Install dependencies:
- python version X.X
- pip versoin X.X

Create a python venv
> python env -m venv
> soucre ./bin/activate # activate into venv (linux)
Install all relevant python packages (see requirements.txt)
pip install -t requirements.txt

### Run the app:
> streamlit run src/app.py
Open the link Streamlit prints (usually http://localhost:8501) in your browser.

### ✨ Web Application Features included
File selector for .wav files in data/raw/.

Waveform visualization.

Spectrogram (STFT in dB).

MFCC feature visualization.

Audio playback in browser.


Raw audio (.wav) → load with librosa or torchaudio.

Feature extraction → compute MFCCs (or spectrograms).

Dataset pairing → each MFCC array is paired with its label (e.g. class ID).

Data split → train/validation/test sets (careful to avoid leakage).

Model input → MFCCs are fed into your CNN/RNN/Transformer as tensors.

Training → model learns to map MFCC patterns → labels.

Evaluation → accuracy, confusion matrix, etc.


## 🔎 Loading & Exploring
This is your exploratory data analysis (EDA) stage:

Load audio files with librosa or torchaudio.

Inspect metadata (esc50.csv) → check class distribution, folds, clip durations.

Visualize:

Waveforms (time domain).

Spectrograms (frequency domain).

Histograms of clip lengths or class counts.

Document issues:

Are some clips noisy, clipped, or silent?

Are classes imbalanced (ESC‑50 has 40 clips per class, so it’s balanced — but note if you subset)?
Any sample rate inconsistencies? (ESC‑50 is standardized at 44.1 kHz, so you can mention that).

##⚙️ Preprocessing → ML Features
This is the feature engineering stage:

Convert raw .wav → MFCCs (or Mel spectrograms).

Normalize features (per‑clip mean/variance).

Pad or truncate to fixed length (ESC‑50 clips are all 5s, so you’re safe).

Save features into data/processed/ for reuse.

👉 For CNNs: treat MFCCs or spectrograms as 2D “images” (coefficients × time). 👉 For RNNs/Transformers: treat MFCCs as sequential feature vectors over time.

### preprocessing script
✅ What this script does
Loads metadata (esc50.csv) and prints dataset stats.

Explores: class distribution, clip durations, sample rates.

Loads audio: returns waveform + sample rate.

Extracts features: spectrograms and MFCCs.

Documents: ESC‑50 is balanced (40 clips per class), clips are 5s long, sample rate is 44.1 kHz.

### visualisation script 
✅ What this gives you
File‑level exploration: Waveform, spectrogram, MFCC plots.

Dataset‑level exploration: Class distribution and duration histograms.

Reusable functions: Can be imported into app.py for interactive visualization.

### app script 
✅ What’s this gives you
Uses metadata (esc50.csv) to list files instead of scanning the folder.

Imports functions from preprocessing.py and visualization.py to keep code modular.

Adds dataset-level plots: class distribution and clip duration histograms.

Interactive file selector: lets you pick a clip, play it, and see waveform, spectrogram, and MFCCs.

Random Sample button: Picks a random file from the dataset when clicked.

Keeps the dropdown for manual selection, but adds a quick way to explore.

Displays the filename above the audio player so you know what you’re listening to.

To run app 
> streamlit run src/app.py
in browser run:
http://localhost:8501
