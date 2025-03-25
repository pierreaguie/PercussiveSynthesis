import numpy as np
import librosa

notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
octaves = ["1", "2", "3", "4", "5", "6"]

idx2note = {i + 36 : f"{notes[i % 12]}{octaves[i // 12]}" for i in range(12 * 6)}
note2idx = {v : k for k, v in idx2note.items()}


def load_piano_sound(pitch : str,
                     velocity : str,
                     sr : int) -> np.ndarray:
    idx = note2idx[pitch]
    path = f"data/yamaha/0{idx}_{pitch}_{velocity}.wav"
    s = librosa.load(path, sr = sr)[0]
    return s


def load_ikembe_sound(pitch : str,
                      sr : int) -> np.ndarray:
    if pitch not in ["A1", "A2", "B2", "C2", "C3", "E1", "E2", "F1"]:
        raise ValueError("Pitch not available. The only available pitches are A1, A2, B2, C2, C3, E1, E2, F1.")
    path = f"data/ikembe/Ikembe {pitch}.aif"
    s = librosa.load(path, sr = sr)[0]
    return s