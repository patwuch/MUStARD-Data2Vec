"""Training utilities: seeding, forward dispatch, training loop, evaluation.

The eval(call) string-dispatch from the original notebooks is replaced by
forward_model(), which resolves modality tensors by name and calls the model
directly.
"""

import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score

from src.model import SarcasmClassifier

# Maps the sorted modality string used internally to the modality names
# that appear in the per-batch tensor dict.
MODALITY_POSITIONS = {
    "VTA": ("Video", "Text", "Audio"),
    "VT": ("Video", "Text"),
    "VA": ("Video", "Audio"),
    "TA": ("Text", "Audio"),
    "V": ("Video",),
    "T": ("Text",),
    "A": ("Audio",),
}


def seed(seed_val: int = 42) -> None:
    np.random.seed(seed_val)
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    np.random.seed(42)
    random.seed(42)


def _normalise_modes(mode: str) -> str:
    """Sort modality letters so 'TAV' and 'VTA' both resolve to 'VTA'."""
    return "".join(reversed(sorted(mode.upper())))


def forward_model(
    mod: nn.Module,
    uText: torch.Tensor,
    cText: torch.Tensor,
    uAudio: torch.Tensor,
    cAudio: torch.Tensor,
    uVideo: torch.Tensor,
    cVideo: torch.Tensor,
    speaker: torch.Tensor,
    input_modes: str,
    use_context: bool,
    use_speaker: bool,
) -> torch.Tensor:
    """Call mod.forward() with the correct modality tensors.

    Replaces the eval(call) string-dispatch pattern from the original code.
    Modality positions A/B/C are assigned in MODALITY_POSITIONS order.
    """
    pos_labels = ("A", "B", "C")
    modalities = MODALITY_POSITIONS[input_modes]

    u_tensors = {"Text": uText, "Audio": uAudio, "Video": uVideo}
    c_tensors = {"Text": cText, "Audio": cAudio, "Video": cVideo}

    kwargs = {}
    for pos, modality in zip(pos_labels, modalities):
        kwargs[f"u{pos}"] = u_tensors[modality]
        if use_context:
            kwargs[f"c{pos}"] = c_tensors[modality]
    if use_speaker:
        kwargs["speaker_embedding"] = speaker

    return mod(**kwargs)


def build_model(
    mode: str,
    use_speaker: bool,
    use_context: bool,
    n_speaker: int,
    shared_dim: int,
    proj_dim: int,
    dropout: float,
    num_classes: int = 2,
    input_dim: int = 768,
) -> Tuple[SarcasmClassifier, str]:
    """Instantiate a SarcasmClassifier for the given ablation configuration.

    Returns (model, normalised_mode_string).
    """
    input_modes = _normalise_modes(mode)
    n_modalities = len(input_modes)
    mod = SarcasmClassifier(
        n_modalities=n_modalities,
        use_context=use_context,
        use_speaker=use_speaker,
        n_speaker=n_speaker if use_speaker else 0,
        input_dim=input_dim,
        shared_dim=shared_dim,
        proj_dim=proj_dim,
        dropout=dropout,
        num_classes=num_classes,
    )
    return mod, input_modes


def evaluation(
    loader,
    mod: nn.Module,
    input_modes: str,
    use_context: bool,
    use_speaker: bool,
    device: torch.device,
    report: bool = False,
    return_preds: bool = False,
):
    """Evaluate model on a DataLoader.

    Returns (macro_f1, mean_loss) unless return_preds=True, in which case
    returns (true_labels, predicted_labels).
    """
    criterion = nn.CrossEntropyLoss().to(device)
    pred, true, total_loss = [], [], []

    with torch.no_grad():
        seed()
        for batch in loader:
            uText = batch[0].float().to(device)
            cText = batch[1].float().to(device)
            uAudio = batch[2].float().to(device)
            cAudio = batch[3].float().to(device)
            uVideo = batch[4].float().to(device)
            cVideo = batch[5].float().to(device)
            speaker = batch[6].float().to(device)
            y_true = batch[7].long().to(device)

            output = torch.softmax(
                forward_model(
                    mod, uText, cText, uAudio, cAudio, uVideo, cVideo,
                    speaker, input_modes, use_context, use_speaker,
                ),
                dim=1,
            )
            total_loss.append(criterion(output, y_true))
            pred.extend(output.detach().cpu().tolist())
            true.extend(y_true.tolist())

    if return_preds:
        return true, np.argmax(pred, axis=1)
    if report:
        print(classification_report(true, np.argmax(pred, axis=1), digits=3))
    return f1_score(true, np.argmax(pred, axis=1), average="macro"), sum(total_loss) / len(total_loss)


def training(
    mod: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    valid_loader,
    input_modes: str,
    use_context: bool,
    use_speaker: bool,
    device: torch.device,
    fold: int = 0,
    max_epochs: int = 500,
    patience: int = 5,
    report: bool = False,
):
    """Train mod with early stopping on validation macro-F1.

    Returns ((true_labels, pred_labels), best_epoch).
    """
    print("-" * 100)
    print(f"fold={fold}  max_epochs={max_epochs}  patience={patience}")

    train_losses = [0.0]
    max_f1 = 0.0
    patience_flag = True
    best_epoch = 0
    best_model = mod  # initialised so it's always defined

    for epoch in range(1, max_epochs + 1):
        total_loss = []
        seed()

        for batch_data in train_loader:
            uText = batch_data[0].float().to(device)
            cText = batch_data[1].float().to(device)
            uAudio = batch_data[2].float().to(device)
            cAudio = batch_data[3].float().to(device)
            uVideo = batch_data[4].float().to(device)
            cVideo = batch_data[5].float().to(device)
            speaker = batch_data[6].float().to(device)
            y_true = batch_data[7].long().to(device)

            output = forward_model(
                mod, uText, cText, uAudio, cAudio, uVideo, cVideo,
                speaker, input_modes, use_context, use_speaker,
            )
            loss = criterion(output, y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.append(loss.detach().item())

        with torch.no_grad():
            valid_f1, valid_loss = evaluation(
                valid_loader, mod, input_modes, use_context, use_speaker,
                device, report=report,
            )

        epoch_train_loss = sum(total_loss) / len(total_loss)
        train_losses.append(epoch_train_loss)

        if valid_f1 > max_f1:
            max_f1 = valid_f1
            best_model = mod
            best_epoch = epoch
            print(
                f"Epoch:{epoch:4d} | Train Loss: {epoch_train_loss:.3f} "
                f"| Valid Loss: {valid_loss.detach().item():7.3f} "
                f"| Valid F1: {valid_f1:7.3f}"
            )

        # Early stopping: if train loss plateaued, finish after `patience` more epochs
        if abs(train_losses[-2] - train_losses[-1]) < 1e-4:
            if patience_flag:
                # Shorten remaining epochs to patience window
                remaining = max_epochs - epoch
                if remaining > patience:
                    max_epochs = epoch + patience
                patience_flag = False
        else:
            patience_flag = True

    return (
        evaluation(
            valid_loader, best_model, input_modes, use_context, use_speaker,
            device, report=report, return_preds=True,
        ),
        best_epoch,
    )
