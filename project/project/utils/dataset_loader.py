import os
import pandas as pd

def build_dataframe(audio_root):

    data = []

    for root, dirs, files in os.walk(audio_root):
        for file in files:
            if file.endswith(".wav"):

                file_path = os.path.join(root, file)

                # Emotion from folder name
                emotion = os.path.basename(root).split("_")[-1]

                # Extract sentence from filename
                # Example: OAF_back_angry.wav
                parts = file.replace(".wav", "").split("_")

                if len(parts) >= 2:
                    sentence_word = parts[1]
                else:
                    sentence_word = "unknown"

                sentence = f"The word is {sentence_word}"

                data.append([file_path, sentence, emotion])

    df = pd.DataFrame(data, columns=["path", "text", "emotion"])

    return df