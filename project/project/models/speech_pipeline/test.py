import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

from utils.dataset_loader import build_dataframe
from utils.audio_utils import extract_mfcc

AUDIO_ROOT = "/kaggle/input/toronto-emotional-speech-set-tess"

print("Loading dataset...")
df = build_dataframe(AUDIO_ROOT)

print("Extracting features...")
X = []
for path in df["path"]:
    X.append(extract_mfcc(path))

X = np.array(X)

print("Loading model...")
model = load_model("models/speech_pipeline/speech_model.h5")
le = joblib.load("models/speech_pipeline/label_encoder.pkl")

y_true = le.transform(df["emotion"])
y_pred = np.argmax(model.predict(X), axis=1)

print("\n=== Speech Model Results ===")
print(classification_report(y_true, y_pred, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))