import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from utils.dataset_loader import build_dataframe

AUDIO_ROOT = "/kaggle/input/toronto-emotional-speech-set-tess"

df = build_dataframe(AUDIO_ROOT)

vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(df["text"])

le = LabelEncoder()
y = le.fit_transform(df["emotion"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Text Accuracy:", model.score(X_test, y_test))

joblib.dump(model, "text_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(le, "label_encoder.pkl")