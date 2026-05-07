# Project Report Outline

## Title

Real-Time Sign Language Translator Using Webcam and Machine Learning

## 1. Abstract

This project presents a real-time sign language translator MVP using Python, OpenCV, MediaPipe Hands, scikit-learn, and Streamlit. The system detects hand landmarks from webcam input, predicts selected useful signs, converts them into simple sentence output, shows emoji mapping, provides text-to-speech output, and includes a quiz mode for beginners. The project is designed as a final-year B.Tech demonstration and supports a limited set of signs instead of full sign language translation.

## 2. Introduction

Communication can be difficult when a person does not understand basic hand signs. Computer vision and machine learning can help recognize gestures from a camera feed. In this project, a webcam is used to capture hand signs, MediaPipe is used to extract hand landmarks, and a machine learning model is used to classify the sign. The output is shown in a simple Streamlit interface.

## 3. Problem Statement

To design and develop a local real-time application that recognizes selected hand signs from webcam input and converts them into readable text, emoji output, speech output, and quiz feedback for beginner practice.

## 4. Objectives

- Detect hand landmarks using MediaPipe Hands
- Collect custom landmark data for selected signs
- Store landmark data in CSV format
- Train a RandomForestClassifier
- Predict signs in real time
- Display sign label, confidence, emoji, and generated sentence
- Speak the generated sentence using pyttsx3
- Provide quiz mode with score and accuracy
- Keep the system simple and demo-ready

## 5. Existing System

Existing sign language recognition systems often use large datasets, deep learning models, and complex gesture recognition pipelines. Some systems require high-end hardware or cloud processing. Many online tools are either limited, not customizable, or not suitable for small academic demonstrations. Full sign language translation also requires grammar, motion, facial expression, and context understanding.

## 6. Proposed System

The proposed system is a local MVP that runs on a normal laptop with a webcam. It supports selected useful signs such as `hello`, `water`, `help`, and `thank_you`. The system uses MediaPipe Hands to extract 21 hand landmarks, stores 63 feature values per sample, trains a RandomForestClassifier, and predicts signs in real time. It also includes sentence generation, emoji mapping, text-to-speech, and quiz mode.

## 7. System Architecture

```text
Webcam
  -> OpenCV frame capture
  -> MediaPipe Hands landmark detection
  -> 63 landmark features
  -> Preprocessing
  -> RandomForestClassifier
  -> Predicted sign and confidence
  -> Sentence builder, emoji mapping, text-to-speech, quiz mode
```

Main components:

- Webcam input
- Hand landmark extraction
- Dataset collector
- Preprocessing module
- Model training module
- Prediction module
- Sentence builder
- Text-to-speech module
- Quiz manager
- Streamlit user interface

## 8. Methodology

1. Capture frames from webcam using OpenCV.
2. Detect one hand using MediaPipe Hands.
3. Extract 21 hand landmarks.
4. Convert landmarks into 63 features: `x`, `y`, and `z` for each landmark.
5. Normalize landmarks relative to the wrist.
6. Save collected samples in CSV files.
7. Combine all CSV files into one dataframe.
8. Encode labels using LabelEncoder.
9. Split data into train and test sets.
10. Train RandomForestClassifier.
11. Predict signs from live webcam frames.
12. Convert predictions into sentence, emoji, speech, and quiz feedback.

## 9. Modules

`hand_tracker.py`

Detects hand landmarks using MediaPipe Hands and returns 63 normalized values.

`dataset_collector.py`

Collects webcam samples for a selected sign and saves them as CSV files.

`preprocess.py`

Loads CSV files, combines data, validates features, and prepares `X` and `y`.

`train_model.py`

Trains the RandomForest model and saves the model and label encoder.

`predictor.py`

Loads the trained model and predicts sign labels with confidence.

`sentence_builder.py`

Converts detected signs into simple meaningful sentence output.

`text_to_speech.py`

Speaks the generated sentence using pyttsx3.

`quiz_mode.py`

Manages target signs, answer checking, score, wrong attempts, and accuracy.

`utils.py`

Provides helper functions such as emoji mapping and label loading.

`app.py`

Main Streamlit application with Live Translator, Dataset Collector Guide, Quiz Mode, and About Project pages.

## 10. Result and Discussion

The system can recognize trained signs from webcam input and display the predicted label with confidence. It can build simple sentences such as `I need water`, show related emojis, and speak the sentence. Quiz mode helps beginners practice signs and shows accuracy. The accuracy depends mainly on dataset quality, number of samples, lighting, background, and consistency of hand gestures.

Suggested result content for report:

- screenshots of dataset collection
- number of samples per sign
- model accuracy
- classification report
- confusion matrix
- screenshots of Live Translator
- screenshots of Quiz Mode

## 11. Limitations

- Supports selected useful signs only
- Not full sign language translation
- Mostly suitable for one-hand signs
- Accuracy depends on webcam quality and lighting
- Does not understand full grammar or continuous signing
- Does not use facial expression or body posture
- Motion-based signs may require better temporal models

## 12. Future Scope

- Add more signs
- Collect a larger dataset
- Support two-hand gestures
- Use deep learning models such as CNN, LSTM, or Transformers
- Add multilingual text and speech output
- Build a mobile application
- Add user login and learning progress tracking
- Improve grammar using NLP

## 13. Conclusion

This project successfully demonstrates a complete real-time gesture recognition workflow. It includes webcam input, hand landmark extraction, dataset collection, model training, real-time prediction, sentence generation, emoji output, speech output, and quiz-based learning. The project is simple, local, and suitable for a final-year B.Tech demonstration, while leaving scope for future improvements.
