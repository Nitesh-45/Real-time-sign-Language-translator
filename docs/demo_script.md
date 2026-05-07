# 2-Minute Viva/Demo Script

Good morning/afternoon respected teacher. My project title is **Real-Time Sign Language Translator**.

The problem is that many people cannot understand even basic hand signs. A complete sign language translator is very complex because it needs grammar, movement, facial expression, and context. So, in this project, I have built a small MVP that recognizes selected useful signs and converts them into text, emoji, speech, and quiz feedback.

This project is built using Python. For webcam input I used OpenCV. For hand landmark detection I used MediaPipe Hands. MediaPipe gives 21 hand landmarks, and each landmark has x, y, and z values, so one sample contains 63 features. For model training I used scikit-learn RandomForestClassifier. The user interface is made with Streamlit, and text-to-speech is done using pyttsx3.

The project has four main pages. The first page is **Live Translator**, where the webcam detects the sign, shows the predicted label, confidence percentage, emoji, and generated sentence. The sentence can also be spoken using the Speak Sentence button.

The second page is **Dataset Collector Guide**. Here, I show the terminal commands used to collect data. For example:

```text
python src/dataset_collector.py --label water
python src/dataset_collector.py --label help
```

During collection, pressing `s` saves the hand landmark sample, and pressing `q` quits. The data is saved as CSV files in `data/keypoints`.

The third page is **Quiz Mode**. It randomly shows a target sign. The user performs the sign, and the system checks whether the predicted sign matches the target. It shows correct or try again feedback and updates score and accuracy.

The fourth page is **About Project**, where I explain the objective, tech stack, features, limitations, and future scope.

The main limitation is that this MVP supports only selected useful signs like hello, water, help, food, medicine, yes, no, and thank you. It is not full sign language translation.

In future, this project can be improved by adding more signs, collecting a larger dataset, supporting two-hand gestures, using deep learning models, adding multilingual speech output, and creating a mobile app.

So overall, this project demonstrates a complete machine learning workflow: webcam capture, landmark extraction, dataset collection, model training, real-time prediction, sentence generation, speech output, and quiz-based learning. Thank you.
