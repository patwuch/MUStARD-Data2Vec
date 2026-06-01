import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def apply_attention(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce a variable-length sequence [1, seq_len, 768] to [1, 768].

    Uses a random (untrained) linear projection to compute soft attention
    weights over the sequence dimension — preserving the behavior of the
    original notebook code. Text embeddings are pre-pooled to [1, 768]
    during feature extraction so they bypass this function.
    """
    attention_layer = torch.nn.Linear(tensor.shape[-1], 1)
    attention_scores = attention_layer(tensor)          # [1, seq_len, 1]
    attention_weights = F.softmax(attention_scores, dim=1)  # [1, seq_len, 1]
    return (tensor * attention_weights).sum(dim=-2)     # [1, 768]


class ContentDataset(Dataset):
    """Loads multimodal features, speaker info, and sarcasm labels.

    Adapted from MUStARD++'s ContentDataset. Each sample provides
    utterance and context embeddings for text, audio, and video modalities.

    Args:
        mapping: DataFrame with columns SCENE, SAR, SPEAKER.
        dataset: Dict mapping SCENE → dict of modality tensors.
        speaker_list: Sorted list of all speaker names (for one-hot encoding).
    """

    def __init__(self, mapping, dataset, speaker_list):
        self.mapping = mapping.reset_index(drop=True)
        self.dataset = dataset
        self.speakers_mapping = speaker_list

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        scene = self.mapping.loc[idx, "SCENE"]
        data = self.dataset[scene]
        label = int(self.mapping.loc[idx, "SAR"])
        spkr = np.eye(len(self.speakers_mapping))[
            self.speakers_mapping.index(self.mapping.loc[idx, "SPEAKER"])
        ]

        # Text is stored as [1, 768] CLS-token vectors — squeeze to [768]
        uText = data["uText"].squeeze()
        cText = data["cText"].squeeze()

        # Audio/video are variable-length sequences — apply attention pooling
        uAudio = apply_attention(data["uAudio"]).squeeze()
        cAudio = apply_attention(data["cAudio"]).squeeze()
        uVideo = apply_attention(data["uVideo"]).squeeze()
        cVideo = apply_attention(data["cVideo"]).squeeze()

        return uText, cText, uAudio, cAudio, uVideo, cVideo, spkr, label


def build_scene_dataset(df) -> dict:
    """Convert a DataFrame of utterance rows into the scene-keyed dict
    expected by ContentDataset."""
    return {
        row["SCENE"]: {
            "uText": row["uText"],
            "cText": row["cText"],
            "uAudio": row["uAudio"],
            "cAudio": row["cAudio"],
            "uVideo": row["uVideo"],
            "cVideo": row["cVideo"],
        }
        for _, row in df.iterrows()
    }
