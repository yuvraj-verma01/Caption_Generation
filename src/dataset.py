from pathlib import Path
from typing import Optional, Tuple

import ast
import pickle

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


class ArtEmisCaptionDataset(Dataset):
    """
    Returns:
      - image: 3 x H x W float tensor
      - caption_in:  max_len-1 LongTensor (input tokens)
      - caption_out: max_len-1 LongTensor (target tokens)
      - length:      original (non-padded) caption length
    """
    def __init__(
        self,
        csv_path: str,
        vocab_path: str,
        transform=None,
    ):
        self.csv_path = Path(csv_path)
        self.vocab_path = Path(vocab_path)
        self.transform = transform

        # Load dataframe
        self.df = pd.read_csv(self.csv_path)

        # Load vocab
        with open(self.vocab_path, "rb") as f:
            vocab = pickle.load(f)

        self.stoi = vocab["stoi"]
        self.itos = vocab["itos"]
        self.max_len = vocab["max_len"]
        self.special_tokens = vocab["special_tokens"]

        # Pre-cache the things we need as lists (faster than hitting df every time)
        self.img_paths = self.df["img_resized_path"].tolist()

        # tokens_encoded column is stored as a string representation of a list → parse
        raw_enc = self.df["tokens_encoded"].tolist()
        self.encoded_caps = [
            ast.literal_eval(x) if isinstance(x, str) else x for x in raw_enc
        ]

        # If tokens_len not stored, compute from encoded (non-pad tokens)
        if "tokens_len" in self.df.columns:
            self.lengths = self.df["tokens_len"].tolist()
        else:
            pad_id = self.special_tokens["<pad>"]
            self.lengths = [
                sum(1 for t in seq if t != pad_id) for seq in self.encoded_caps
            ]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        # ---- image ----
        img_path = Path(self.img_paths[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # ---- caption (already [<start> ... <end> ... <pad>]) ----
        enc = self.encoded_caps[idx]

        # Ensure correct length (just in case)
        if len(enc) != self.max_len:
            # pad / cut
            pad_id = self.special_tokens["<pad>"]
            if len(enc) < self.max_len:
                enc = enc + [pad_id] * (self.max_len - len(enc))
            else:
                enc = enc[: self.max_len]

        enc = torch.tensor(enc, dtype=torch.long)

        # Typical seq2seq setup:
        #   caption_in  = [<start>, w1, w2, ..., w_{n-1}]
        #   caption_out = [w1, w2, ..., w_{n-1}, <end>]
        caption_in = enc[:-1]   # drop last token
        caption_out = enc[1:]   # drop first token

        length = self.lengths[idx]

        return img, caption_in, caption_out, length

