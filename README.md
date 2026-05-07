# Real-Time Sign Language Translator

Let's connect people together.

A final-year B.Tech level Python MVP that recognizes selected useful hand signs using webcam input and converts them into text, emojis, simple sentences, speech output, and quiz feedback.

This is a demo-ready learning project. It is not a complete sign language translation system.

## Problem Statement

People who are new to sign language may find it difficult to understand even basic hand signs. A full sign language translator is a complex system because real sign languages include hand movement, facial expression, body posture, grammar, and regional variations.

This project solves a smaller problem: it recognizes a selected set of useful hand signs from webcam input and converts them into simple text output for demonstration and beginner learning.

## Objective

The objective is to build a local real-time sign language translator MVP that can:

- detect one hand using webcam input
- extract hand landmarks using MediaPipe Hands
- collect custom sign datasets
- train a machine learning classifier
- predict signs in real time
- build simple sentence output
- show emoji mapping
- speak the sentence using browser text-to-speech
- provide a quiz mode for practice

## Features

- Real-time webcam sign prediction
- MediaPipe 21-hand-landmark extraction
- CSV dataset collection for custom signs
- RandomForestClassifier model training
- Label encoding using LabelEncoder
- Confidence-based prediction filtering
- Sentence builder with duplicate control and cooldown
- Emoji output for each detected sign
- Browser text-to-speech using the Web Speech API
- Browser webcam support using WebRTC for public deployment
- Beginner quiz mode with score and accuracy
- Streamlit app with clean pages
- Clear error messages for missing model, webcam issues, no hand, and low confidence

## Tech Stack

- Python
- OpenCV
- MediaPipe Hands
- scikit-learn
- Streamlit
- streamlit-webrtc
- NumPy
- pandas
- joblib

## Supported Signs

- hello
- yes
- no
- help
- water
- food
- medicine
- thank_you
- please
- stop
- good
- bad
- doctor
- pain
- toilet
- home
- phone
- sorry
- more
- emergency

## Folder Structure

```text
real_time_sign_language_translator/
├── app.py
├── requirements.txt
├── README.md
├── config.py
├── data/
│   ├── raw_images/
│   ├── keypoints/
│   └── labels.json
├── models/
│   ├── gesture_model.pkl
│   └── label_encoder.pkl
├── src/
│   ├── hand_tracker.py
│   ├── dataset_collector.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── predictor.py
│   ├── sentence_builder.py
│   ├── text_to_speech.py
│   ├── quiz_mode.py
│   └── utils.py
├── assets/
│   ├── emojis.json
│   └── sample_gestures.md
└── docs/
    ├── project_report_outline.md
    ├── system_architecture.md
    └── demo_script.md
```

`gesture_model.pkl` and `label_encoder.pkl` are generated after training.

## Installation Steps

Open a terminal in the project folder:

```powershell
cd "C:\Users\Ntesh Singh Mehra\Documents\New project\real_time_sign_language_translator"
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Dataset Collection Steps

Open the project folder in terminal and run the collector for one label:

```powershell
python src/dataset_collector.py --label water
```

Another example:

```powershell
python src/dataset_collector.py --label help
```

Controls:

- Press `s` to save the current hand sample.
- Press `q` to quit the collection window.

Recommended sample count:

- Minimum: 50 samples per sign
- Better for demo: 100 samples per sign

Collected CSV files are saved in:

```text
data/keypoints/
```

Each CSV row has:

```text
label, f1, f2, f3, ... f63
```

## Model Training Steps

After collecting data for at least two signs, train the classifier:

```powershell
python src/train_model.py
```

The training script:

- loads all CSV files from `data/keypoints/`
- combines them into one dataframe
- separates features `X = f1...f63` and labels `y = label`
- encodes labels using `LabelEncoder`
- splits data into train and test sets
- trains `RandomForestClassifier`
- prints accuracy, classification report, and confusion matrix
- saves the model and label encoder

Generated files:

```text
models/gesture_model.pkl
models/label_encoder.pkl
```

## Running App Steps

Start the Streamlit application:

```powershell
streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

App pages:

- Live Translator
- Dataset Collector Guide
- Quiz Mode
- About Project

## Public Deployment Steps

This app is now web-ready for Streamlit Community Cloud because Live Translator and Quiz Mode use the visitor's browser webcam through WebRTC.

1. Push the latest code to GitHub.
2. Open Streamlit Community Cloud: `https://share.streamlit.io`
3. Click `Create app`.
4. Select this repository and branch `main`.
5. Set main file path to `app.py`.
6. Use Python `3.10` in advanced settings if available.
7. Click `Deploy`.

After deployment, open the app URL, go to `Live Translator` or `Quiz Mode`, click `START` in the webcam box, and allow camera permission in the browser.

## Demo Workflow

1. Run `streamlit run app.py`.
2. Open `Dataset Collector Guide`.
3. Collect samples from terminal for at least two signs, such as `water` and `help`.
4. Train the model using `python src/train_model.py`.
5. Open `Live Translator`.
6. Click `START` inside the browser webcam box.
7. Perform a trained sign.
8. Check detected sign, confidence, emoji, and generated sentence.
9. Click `Speak Sentence`.
10. Open `Quiz Mode`.
11. Perform the target sign and check score and accuracy.

## Limitations

- This MVP supports selected useful signs only.
- It is not full sign language translation.
- It mainly works with one visible hand.
- It performs better for static or simple signs.
- Accuracy depends on dataset quality, lighting, camera position, and background.
- It does not handle full sign language grammar, facial expressions, or continuous conversation.

## Future Scope

- Add more signs
- Collect a larger and better dataset
- Support two-hand gestures
- Add deep learning models for motion-based gestures
- Add multilingual speech output
- Build a mobile app version
- Add user profiles and progress tracking
- Improve sentence grammar using NLP

## Troubleshooting

### ModuleNotFoundError

Make sure you are inside the project folder and the virtual environment is active.

```powershell
cd "C:\Users\Ntesh Singh Mehra\Documents\New project\real_time_sign_language_translator"
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run scripts from the project root:

```powershell
python src/train_model.py
python src/dataset_collector.py --label water
streamlit run app.py
```

### Webcam Not Opening

For the Streamlit web app, click `START` inside the webcam box and allow browser camera permission. On mobile or desktop, camera access usually requires `https://` or `localhost`.

For terminal tools such as dataset collection, close other apps that may be using the camera, such as Zoom, Teams, or the Windows Camera app. If needed, try a different camera index in `config.py`:

```python
CAMERA_INDEX = 1
```

### No Hand Detected

Use a plain background, good lighting, and keep your full hand inside the frame. Avoid moving too fast.

### Model File Missing

The model files are created only after training. Collect samples and run:

```powershell
python src/train_model.py
```

### Low Accuracy

Collect more samples with consistent gestures. Keep sample counts balanced across signs. Use similar lighting and background during collection and demo.

### Browser Voice Not Working

The deployed app uses the browser Web Speech API. If voice does not play, check browser audio permission, unmute the tab, and click `Speak Sentence` after a sentence is generated.

### Streamlit Rerun Issues

If Streamlit shows stale model behavior after training, refresh the browser or restart the server:

```powershell
Ctrl+C
streamlit run app.py
```

## Note

The gesture examples are for project demonstration only. Real sign languages are complete languages with their own grammar and regional differences.
