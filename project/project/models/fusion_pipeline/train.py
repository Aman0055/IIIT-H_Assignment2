import numpy as np
import joblib
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from utils.dataset_loader import build_dataframe
from utils.audio_utils import extract_mfcc

AUDIO_ROOT = "/kaggle/input/toronto-emotional-speech-set-tess"

df = build_dataframe(AUDIO_ROOT)

# Speech features
X_speech = []
for path in df["path"]:
    X_speech.append(extract_mfcc(path))

X_speech = np.array(X_speech)
X_speech = X_speech.mean(axis=1)

# Text features
vectorizer = joblib.load("models/text_pipeline/vectorizer.pkl")
X_text = vectorizer.transform(df["text"]).toarray()

le = joblib.load("models/text_pipeline/label_encoder.pkl")
y = to_categorical(le.transform(df["emotion"]))

Xs_train, Xs_test, Xt_train, Xt_test, y_train, y_test = train_test_split(
    X_speech, X_text, y, test_size=0.2, random_state=42
)

input_s = Input(shape=(Xs_train.shape[1],))
input_t = Input(shape=(Xt_train.shape[1],))

merged = Concatenate()([input_s, input_t])
dense = Dense(128, activation="relu")(merged)
output = Dense(y.shape[1], activation="softmax")(dense)

fusion_model = Model([input_s, input_t], output)

fusion_model.compile(optimizer="adam",
                     loss="categorical_crossentropy",
                     metrics=["accuracy"])

fusion_model.fit([Xs_train, Xt_train], y_train,
                 validation_data=([Xs_test, Xt_test], y_test),
                 epochs=20,
                 batch_size=32)

fusion_model.save("fusion_model.h5")