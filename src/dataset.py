from pathlib import Path
from typing import Tuple

import ast
import pickle

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


class ArtEmisCaptionDataset(Dataset):
   
    def __init__(
        self,
        csv_path: str,
        vocab_path: str,
        transform=None,
    ):
        self.csv_path = Path(csv_path)
        self.vocab_path = Path(vocab_path)
        self.transform = transform


        df = pd.read_csv(self.csv_path)

        if "img_resized_path" not in df.columns:
            raise ValueError(f"'img_resized_path' column not found in {self.csv_path}")

        if "tokens_encoded" not in df.columns:
            raise ValueError(f"'tokens_encoded' column not found in {self.csv_path}")

       
        df = df[df["img_resized_path"].notna()].copy()
        df = df[df["tokens_encoded"].notna()].copy()

        
        df["img_resized_path"] = df["img_resized_path"].astype(str)

  
        df = df.reset_index(drop=True)
        self.df = df

        
        with open(self.vocab_path, "rb") as f:
            vocab = pickle.load(f)

        self.stoi = vocab["stoi"]
        self.itos = vocab["itos"]
        self.max_len = vocab["max_len"]
        self.special_tokens = vocab["special_tokens"]

        
        self.img_paths = self.df["img_resized_path"].tolist()

        
        raw_enc = self.df["tokens_encoded"].tolist()
        self.encoded_caps = [
            ast.literal_eval(x) if isinstance(x, str) else x for x in raw_enc
        ]

       
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
       
        img_path = Path(self.img_paths[idx]) 
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        
        enc = self.encoded_caps[idx]

        
        if len(enc) != self.max_len:
            pad_id = self.special_tokens["<pad>"]
            if len(enc) < self.max_len:
                enc = enc + [pad_id] * (self.max_len - len(enc))
            else:
                enc = enc[: self.max_len]

        enc = torch.tensor(enc, dtype=torch.long)

        
        caption_in = enc[:-1]   
        caption_out = enc[1:]   
        length = self.lengths[idx]

        return img, caption_in, caption_out, length
