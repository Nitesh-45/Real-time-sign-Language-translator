from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_IMAGES_DIR = DATA_DIR / "raw_images"
KEYPOINTS_DIR = DATA_DIR / "keypoints"
LABELS_PATH = DATA_DIR / "labels.json"

MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "gesture_model.pkl"
ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

ASSETS_DIR = BASE_DIR / "assets"
EMOJIS_PATH = ASSETS_DIR / "emojis.json"

REQUIRED_SIGNS = [
    "hello",
    "yes",
    "no",
    "help",
    "water",
    "food",
    "medicine",
    "thank_you",
    "please",
    "stop",
    "good",
    "bad",
    "doctor",
    "pain",
    "toilet",
    "home",
    "phone",
    "sorry",
    "more",
    "emergency",
]

NUM_LANDMARKS = 21
FEATURES_PER_LANDMARK = 3
FEATURE_VECTOR_SIZE = NUM_LANDMARKS * FEATURES_PER_LANDMARK

CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.70
MIN_STABLE_PREDICTIONS = 4
MAX_SENTENCE_SIGNS = 10

DEFAULT_COLLECTION_SAMPLES = 80
COLLECTION_DELAY_SECONDS = 0.08

RANDOM_STATE = 42
