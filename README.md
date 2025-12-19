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

### feature extraction script
🔎 What the script produces
Spectrograms (dB) → 2D arrays of shape (freq_bins, time_frames)

MFCCs → 2D arrays of shape (n_mfcc, time_frames)

Labels → the class name (e.g. "dog", "rain", "engine") for each clip

These are saved both:

Individually (clipname_spec.npy, clipname_mfcc.npy)

As combined arrays (all_specs.npy, all_mfccs.npy, labels.npy)

🧑‍💻 How this becomes a training/test dataset
You’ll load all_mfccs.npy (or all_specs.npy) and labels.npy.

Each entry is a 2D feature matrix for one clip.

You then split them into train/test sets (e.g. 80/20 split).

Depending on the model, you might:

Flatten the 2D arrays into 1D vectors (for classical ML like SVM, RandomForest).

Keep them as 2D “images” (for CNNs).

Pad/truncate to a fixed length if needed.

### Preprocessing reasoning
✅ How to document your preprocessing
1. Describe the transformation
Raw .wav clips (5s, 44.1 kHz) were converted into:

Spectrograms (dB): 2D time–frequency arrays.

MFCCs (13 coefficients): compact representations of timbre.

Features saved as .npy arrays for efficient loading in ML models.

2. Note preprocessing decisions
Sampling rate: kept original 44.1 kHz (no resampling needed).

Clip length: dataset standardized at ~5s, so no trimming/padding required.

Feature parameters:

STFT window size = 2048, hop length = 512.

MFCCs = 13 coefficients.

Normalization: spectrograms converted to decibel scale, MFCCs left in raw form.

3. Document quality issues/artifacts
Even though ESC‑50 is clean, you can still mention:

Background noise: some clips contain environmental noise (e.g. “dog bark” with traffic).

Overlapping sounds: certain categories may have secondary sounds (e.g. “engine” with voices).

Class ambiguity: some categories are perceptually similar (e.g. “chainsaw” vs “engine”).

No missing/corrupted files: verified all 2000 clips load successfully.

4. Summarize outcome
Dataset is balanced (40 clips per class, 50 classes).

Preprocessing produced consistent 2D feature arrays ready for ML training.

No major quality issues requiring correction; minor artifacts noted but left intact to preserve dataset realism.

✍️ Example write‑up
We preprocessed the ESC‑50 dataset by converting raw .wav clips into spectrograms and MFCCs. Each clip (5s, 44.1 kHz) was transformed into 2D feature arrays and saved as .npy files. We retained the original sampling rate and clip length, as the dataset is standardized and clean. Spectrograms were computed using STFT (window=2048, hop=512) and converted to decibel scale; MFCCs were extracted with 13 coefficients.

The dataset is well curated, with no missing or corrupted files. Minor artifacts such as background noise and overlapping sounds were observed but not corrected, as they reflect real‑world acoustic conditions. Overall, preprocessing produced consistent ML‑ready features without the need for heavy cleaning.

You did the raw → feature transformation.

You made conscious preprocessing choices.

You documented any issues, even if minor.

### checklist on decisions made
🎯 Preprocessing Checklist Template
1. Data Ingestion
[ ] Dataset source: __________________________

[ ] File types: __________________________

[ ] Number of samples: __________________________

[ ] Classes / labels: __________________________

2. Audio Handling
[ ] Sampling rate: kept at ____ Hz / resampled to ____ Hz

[ ] Clip length: standardized at ____ seconds / padded / trimmed

[ ] Channels: mono / stereo → converted to mono?

[ ] Normalization: amplitude normalized / spectrograms converted to dB

3. Feature Extraction
[ ] Spectrogram parameters:

FFT window size = ____

Hop length = ____

[ ] MFCC parameters:

Number of coefficients = ____

[ ] Other features: __________________________

[ ] Storage format: .npy / .csv / database

4. Quality Issues / Artifacts
[ ] Missing files?

[ ] Corrupted files?

[ ] Background noise noted?

[ ] Overlapping sounds?

[ ] Class ambiguity?

[ ] Decision: kept as‑is / removed / flagged

5. Preprocessing Decisions
[ ] Why you chose your sampling rate

[ ] Why you chose your feature parameters

[ ] Why you kept or corrected artifacts

[ ] Any trade‑offs (accuracy vs efficiency)

6. Outcome
[ ] Dataset integrity confirmed (balanced, complete)

[ ] Features extracted and saved

[ ] Ready for train/test split

✍️ Example (filled for ESC‑50)
Sampling rate: kept at 44.1 kHz (dataset standard).

Clip length: ~5s, no padding/trimming needed.

Features: spectrograms (FFT=2048, hop=512), MFCCs (13 coefficients).

Normalization: spectrograms converted to dB scale.

Quality issues: minor background noise and overlapping sounds noted, left intact to preserve realism.

Outcome: 2000 clips processed, features saved as .npy, dataset balanced (40 clips per class).

👉 This way you can show assessors that you didn’t just run code — you thought about preprocessing choices, documented them, and checked for issues.

## Section 2: Data Splitting Strategy
### Load features script overview
Purpose is to reduce RAM required to create merge features file and perform data splitting strategy.
✅ What this does
1. Reads metadata (esc50.csv) to know which clips exist.
2. Loads the corresponding _mfcc.npy or _spec.npy file for each clip generated by feature_extraction.py
3. Builds arrays X (features) and y (labels) by using the metadata in esc50.csv to map them to the feature arrays.
  - X -> list/array of feature matrices
  - Y -> list/array of class labels
4. Splits into train/test sets with stratification.
  - This is done by the train_test_split() from the python pkg sklearn.model_selection and ensures:
    - balanced classes 
    - no leakage
    - no bias 
What its actually doing:
You provide it with X and Y arrays its returns X and Y train and test arrays whereby:
- test_size=0.2 splits train and test 80/20
- random_state=42 shuffles dataset to prevent ordering bias and ensure they are representative. 42 choosen for reproducibility and widely used, any integer would work.
- stratify=y ensure class proportions in train and test match original dataset i.e. no over- or under- representation in either split
- the above utility ensures no sample appears in both sets and no lable or feature leaks across the two sets.

### Explanation to explain the following:
#### 1. Avoiding Data Leakage
What leakage means in audio:  
This happens when information from the test set “bleeds” into the training set — for example, if the same recording session appears in both sets.ESC‑50 is a benchmark dataset where each clip is an independent 5‑second recording. There are no repeated clips, no overlapping segments, and no multi‑segment recordings.

The strategy is then as follows:
 - Splitting is done after feature extraction.
 - train_test_split(..., stratify=y) is used to ensure labels are separated cleanly.
 - No clip appears in both sets.
 - No metadata or feature normalization is computed using test data.

More information: 
Leakages can occur in the following scenarios:
- poor splitting of dataset
- shared preprocessing 
- overlapping samples
Repercussion of this are the following:
- inflated model performance due to model seeing data already when trained and then being tested on it.
- poor generalisation - doesn't learn off features its been supervised to learn off 
- contamination of accuracy, CM, loss curves etc,. 
- overfitting to recording conditions with the audio rather than actual sound class (similiar to poor generalisation)

#### 2. Avoiding Temporal Bias
What temporal bias means:  
In time‑series datasets, you must ensure training data occurs before test data. ESC‑50 is not a time‑series dataset.
Clips are independent and not ordered in time. Therefore:
- Random stratified split is appropriate.
- No temporal ordering exists to preserve.

Repercussion of temporal bias and what I would do if present in a dataset:
Concept:
Occurs if the order of the data in time influences the model in a misleading way because if you randomly shuffle and split this data you can train data that comes later than the data you test with.
Reprecussions:
- inflated model performance due to indirectly seeing future data 
- poor generalisation to new data as it has learned pattern based on the future data portion if the dataset
- temporal drift in audio occurs due to enviornmental changes, new noise, seasons etc,. and if models mixed time periods incorrectly the relationships between the features can be learnt incorrectly.
- overfitting if acoustic features in the training audio comes from a time window completely different from the test. Therefore you could fail to recognise a class in a future deployment.

What I would have done given my dataset did not have this issue would be:
- use chronological splitting 
- group recordings by session/device/location 
- utilised the following utilites in sklearn (TimeSeriesSplit, GroupShuffleSplit and GroupKFold) to prevent temporal overlap
- Disable shuffling (unlike what I done with random_state)

3. Avoiding Class Imbalance Issues
ESC‑50 dataset is balanced with 50 classes, 40 clips per class and 2000 total clips.

Approach used:
- stratify=y is used in train_test_split, which ensures:
  - Each class appears in train and test in the same proportion.
  - No class is over‑ or under‑represented.

Repercussions of class imbalance:
- Majority class dominance (FP)
- Minority class misclassified (FN)
- Skewed loss function as model optimising for the wrong objective 
- Metrics are misleading (Precision, recall, F1 score etc,.)
- Overfitting for majority class patterns 

Approaches to reduce class imbalance:
- augmentation (pitch shift, add audio artifacts)
- synthetic samples 
- unsampled majority class (adjust model architecture size accordingly)
- class weighted loss function 
- collect more data

4. Avoiding Overfitting to Recording Conditions / Equipment
ESC‑50 recordings come from many sources e.g. microphones, environments, and noise levels.

Approach used:
- Random stratified split ensures:
  - Each class’s variability (different microphones, environments) is represented in both train and test.
  - You do not group by recording session (ESC‑50 doesn’t provide session IDs anyway).
Therefore -> split distributes recording conditions across train/test, reducing overfitting risk.
- Used MFCCs feature (domain-invariant) i.e. compress spectral info and reduce sensitivity to equiptment frequency artifacts

Repercussions:
- Similiar outcomes as above issues in terms of model performances as if class representation over recording conditions/equiptment is uneven it increases potential of model learning incorrect audio features over the target features. Therefore, when using the model on new devices or environments the model under performs.

Further improvements beyond representative classes over different instruments:
- Collate contextual metadata (device ID, location, seesion ID any other relevant deployment information)
- Reduce influence of equiptment by normalising volume, amplitudes etc,. 
- Perform cross-device validation
