
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
