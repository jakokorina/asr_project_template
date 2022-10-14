import logging
from typing import Any
import torch
from collections import defaultdict

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: list[dict]) -> dict[str, Any]:
    """
    Collate and pad fields in dataset items
    """
    result_batch: dict[str, Any] = defaultdict(list)
    for item in dataset_items:
        result_batch["text"].append(item["text"])
        result_batch["text_encoded_length"].append(item["text_encoded"].shape[-1])
        result_batch["spectrogram_length"].append(item["spectrogram"].shape[-1])
        result_batch["audio_path"].append(item["audio_path"])

    batch_text = torch.zeros(len(dataset_items), max(result_batch["text_encoded_length"]))
    batch_spec = torch.zeros(len(dataset_items), dataset_items[0]["spectrogram"].shape[1],
                             max(result_batch["spectrogram_length"]))

    for i, item in enumerate(dataset_items):
        batch_text[i, :result_batch["text_encoded_length"][i]] = item["text_encoded"]
        batch_spec[i, :, :result_batch["spectrogram_length"][i]] = item["spectrogram"]

    result_batch["text_encoded_length"] = torch.tensor(result_batch["text_encoded_length"], dtype=torch.int)
    result_batch["spectrogram_length"] = torch.tensor(result_batch["spectrogram_length"], dtype=torch.int)
    result_batch["text_encoded"] = batch_text
    result_batch["spectrogram"] = batch_spec

    return result_batch
