"""Feature extraction for the MUStARD++ dataset.

Extracts text, visual, and audio embeddings using Data2Vec models and writes
a pickled DataFrame (features.pkl) that downstream steps consume.

Text:  facebook/data2vec-text-base  → CLS-token vector [1, 768]
Video: facebook/data2vec-vision-base → last hidden state [1, 197, 768]
Audio: facebook/data2vec-audio-base-960h → last hidden state [1, T, 768]

Usage:
    python src/feature_extraction.py --config config.yaml
"""

import argparse
import glob
import os
import pickle

import cv2
import librosa
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoProcessor,
    AutoTokenizer,
    Data2VecAudioModel,
    Data2VecTextModel,
    Data2VecVisionModel,
)

import yaml

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


# ---------------------------------------------------------------------------
# Text embeddings
# ---------------------------------------------------------------------------

def load_text_models(device):
    tokenizer = AutoTokenizer.from_pretrained("facebook/data2vec-text-base")
    model = Data2VecTextModel.from_pretrained("facebook/data2vec-text-base").to(device)
    model.eval()
    return tokenizer, model


def get_text_embedding(sentence: str, tokenizer, model, device) -> torch.Tensor:
    """Return the CLS-token vector for a sentence as [1, 768]."""
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu()  # [1, 768]


def extract_text_embeddings(df: pd.DataFrame, tokenizer, model, device) -> pd.DataFrame:
    """Populate df['text_embeddings'] for utterance rows (KEY contains 'u').

    Context rows are handled separately by concatenating all sentences in
    the scene with a </s> separator.
    """
    separator = " </s> "

    # Utterance rows
    df["text_embeddings"] = df.apply(
        lambda row: get_text_embedding(row["SENTENCE"], tokenizer, model, device)
        if "u" in row["KEY"]
        else None,
        axis=1,
    )

    # Context rows: concatenate all 'c'-type sentences per scene
    prev_scene = None
    concatenated = ""
    prev_index = None

    for index, row in df.iterrows():
        if row["SCENE"] != prev_scene:
            if prev_scene is not None and concatenated:
                df.at[prev_index, "text_embeddings"] = get_text_embedding(
                    concatenated, tokenizer, model, device
                )
                concatenated = ""
            prev_scene = row["SCENE"]
            prev_index = index

        if "c" in row["KEY"]:
            concatenated += row["SENTENCE"] + separator

    if concatenated:
        df.at[prev_index, "text_embeddings"] = get_text_embedding(
            concatenated, tokenizer, model, device
        )

    return df


# ---------------------------------------------------------------------------
# Visual embeddings — keyframes
# ---------------------------------------------------------------------------

def load_vision_models(device):
    processor = AutoImageProcessor.from_pretrained("facebook/data2vec-vision-base")
    model = Data2VecVisionModel.from_pretrained("facebook/data2vec-vision-base").to(device)
    model.eval()
    return processor, model


def embed_image(img_path: str, processor, model, device) -> torch.Tensor:
    """Return last_hidden_state for a single image as [1, 197, 768]."""
    image = Image.open(img_path)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        return model(**inputs).last_hidden_state.cpu()


def _key_from_jpeg(file_path: str) -> str:
    return os.path.basename(file_path).replace("_0.jpeg", "")


def extract_keyframe_embeddings(
    df: pd.DataFrame, keyframe_dir: str, processor, model, device
) -> pd.DataFrame:
    """Populate df['keyframe_embeddings'] from pre-extracted _0.jpeg keyframes."""
    image_files = glob.glob(os.path.join(keyframe_dir, "*_0.jpeg"))
    non_match = []

    with tqdm(total=len(image_files), desc=f"Keyframes ({os.path.basename(keyframe_dir)})") as pbar:
        for img_path in image_files:
            key_prefix = _key_from_jpeg(img_path)
            matches = df[df["KEY"].str.startswith(key_prefix)].index
            if matches.empty:
                non_match.append(img_path)
            else:
                df.at[matches[0], "keyframe_embeddings"] = embed_image(
                    img_path, processor, model, device
                )
            pbar.update(1)

    if non_match:
        print(f"[WARN] {len(non_match)} keyframes had no matching row: {non_match[:5]}")
    return df


# ---------------------------------------------------------------------------
# Visual embeddings — fallback (full-video frame averaging)
# ---------------------------------------------------------------------------

def embed_video_frames(video_path: str, processor, model, device) -> torch.Tensor | None:
    """Average the last_hidden_state over all frames of a video.

    Used for videos where Katna could not extract a keyframe.
    Returns a tensor of shape [1, 197, 768], or None if the video is unreadable.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open {video_path}")
        return None

    embeddings = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            embeddings.append(model(**inputs).last_hidden_state.cpu())
    cap.release()

    if not embeddings:
        return None
    mean = torch.from_numpy(np.mean([e.numpy() for e in embeddings], axis=0))
    return mean  # [1, 197, 768]


def _key_from_mp4(video_path: str) -> str:
    return os.path.basename(video_path).replace(".mp4", "")


def extract_fallback_video_embeddings(
    df: pd.DataFrame, video_dir: str, processor, model, device
) -> pd.DataFrame:
    """Populate df['keyframe_embeddings'] for fallback videos via frame averaging."""
    for fname in tqdm(os.listdir(video_dir), desc=f"Fallback videos ({os.path.basename(video_dir)})"):
        if not fname.endswith(".mp4"):
            continue
        video_path = os.path.join(video_dir, fname)
        key_prefix = _key_from_mp4(video_path)
        matches = df[df["KEY"].str.startswith(key_prefix)].index

        if matches.empty:
            print(f"[WARN] No matching row for fallback video: {fname}")
            continue

        embedding = embed_video_frames(video_path, processor, model, device)
        if embedding is not None:
            df.at[matches[0], "keyframe_embeddings"] = embedding
        else:
            print(f"[WARN] No frames extracted from {fname}")
    return df


# ---------------------------------------------------------------------------
# Audio embeddings
# ---------------------------------------------------------------------------

def load_audio_models(device):
    processor = AutoProcessor.from_pretrained("facebook/data2vec-audio-base-960h")
    model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base-960h").to(device)
    model.eval()
    return processor, model


def embed_audio(file_path: str, processor, model, device) -> torch.Tensor:
    """Return last_hidden_state for an audio file as [1, T, 768]."""
    audio, sr = librosa.load(file_path, sr=16000)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        return model(**inputs).last_hidden_state.cpu()


def extract_audio_embeddings(
    df: pd.DataFrame, audio_dir: str, processor, model, device
) -> pd.DataFrame:
    """Populate df['audio_embeddings'] from .wav files in audio_dir."""
    for fname in tqdm(os.listdir(audio_dir), desc=f"Audio ({os.path.basename(audio_dir)})"):
        file_prefix = os.path.splitext(fname)[0]
        matches = df[df["KEY"].str.startswith(file_prefix)].index.tolist()
        if not matches:
            print(f"[WARN] No matching row for audio: {file_prefix}")
            continue
        df.at[matches[0], "audio_embeddings"] = embed_audio(
            os.path.join(audio_dir, fname), processor, model, device
        )
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    p = cfg["paths"]

    # --- Load and fix metadata ---
    df = pd.read_csv(p["csv_file"])
    # Known typo in the original dataset
    df.at[2596, "KEY"] = "1_S11E03_067_u"
    df["text_embeddings"] = None
    df["audio_embeddings"] = None
    df["keyframe_embeddings"] = None

    # --- Text ---
    print("\n=== Text embeddings ===")
    tokenizer, text_model = load_text_models(device)
    df = extract_text_embeddings(df, tokenizer, text_model, device)
    del tokenizer, text_model
    torch.cuda.empty_cache()

    # --- Visual: keyframes (context then utterance) ---
    print("\n=== Visual embeddings (keyframes) ===")
    img_proc, vis_model = load_vision_models(device)
    df = extract_keyframe_embeddings(df, p["context_keyframes"], img_proc, vis_model, device)
    df = extract_keyframe_embeddings(df, p["utterance_keyframes"], img_proc, vis_model, device)

    # --- Visual: fallback videos (no keyframe) ---
    print("\n=== Visual embeddings (fallback videos) ===")
    df = extract_fallback_video_embeddings(df, p["context_videos_fallback"], img_proc, vis_model, device)
    df = extract_fallback_video_embeddings(df, p["utterance_videos_fallback"], img_proc, vis_model, device)
    del img_proc, vis_model
    torch.cuda.empty_cache()

    n_missing_vis = df["keyframe_embeddings"].isna().sum()
    print(f"Rows still missing visual embedding: {n_missing_vis}")

    # --- Audio ---
    print("\n=== Audio embeddings ===")
    aud_proc, aud_model = load_audio_models(device)
    df = extract_audio_embeddings(df, p["context_audios"], aud_proc, aud_model, device)
    df = extract_audio_embeddings(df, p["utterance_audios"], aud_proc, aud_model, device)
    del aud_proc, aud_model
    torch.cuda.empty_cache()

    # --- Save ---
    os.makedirs(os.path.dirname(p["features_pkl"]), exist_ok=True)
    with open(p["features_pkl"], "wb") as f:
        pickle.dump(df, f)
    print(f"\nSaved features to {p['features_pkl']}")


if __name__ == "__main__":
    main()
