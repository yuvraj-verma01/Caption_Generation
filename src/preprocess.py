import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from collections import Counter

RAW_CSV = Path("C:\\Users\\yuvra\\OneDrive\\Desktop\\Monsoon 2025\\iml_assignments\\A3\\ArtEmisProject\\data\\artemis_dataset_release_v0.csv")   # adjust if needed
OUT_DIR = Path("C:\\Users\\yuvra\\OneDrive\\Desktop\\Monsoon 2025\\iml_assignments\\A3\\ArtEmisProject\\data")
OUT_DIR.mkdir(exist_ok=True, parents=True)

MIN_LEN = 5              # min tokens
LONG_PRC = 95            # percentile for max length
MIN_FREQ = 5             # min freq for vocab
SPLIT_LOADS = (0.85, 0.05, 0.10)  # train, val, test ratios
RANDOM_SEED = 2021

SPECIAL_TOKENS = {
    "<pad>": 0,
    "<unk>": 1,
    "<start>": 2,
    "<end>": 3,
}

def simple_tokenize(text: str):
    # very simple: lowercase + split on spaces
    return text.lower().strip().split()

def make_splits(df):
    """Split by unique (art_style, painting)."""
    rng = np.random.default_rng(RANDOM_SEED)
    artworks = df[["art_style", "painting"]].drop_duplicates().reset_index(drop=True)
    perm = rng.permutation(len(artworks))
    artworks = artworks.iloc[perm].reset_index(drop=True)

    n = len(artworks)
    n_train = int(SPLIT_LOADS[0] * n)
    n_val   = int(SPLIT_LOADS[1] * n)
    # rest go to test
    train_keys = artworks.iloc[:n_train]
    val_keys   = artworks.iloc[n_train:n_train+n_val]
    test_keys  = artworks.iloc[n_train+n_val:]

    def mark_split(keys_df, name):
        m = df.merge(keys_df, on=["art_style", "painting"], how="inner").index
        df.loc[m, "split"] = name

    df["split"] = "test"  # default
    mark_split(train_keys, "train")
    mark_split(val_keys, "val")
    # remaining are already "test"
    return df

def build_vocab(train_tokens):
    counter = Counter()
    for toks in train_tokens:
        counter.update(toks)

    # start with special tokens
    itos = list(SPECIAL_TOKENS.keys())
    for word, freq in counter.items():
        if freq >= MIN_FREQ:
            itos.append(word)

    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos

def encode(tokens, stoi, max_len):
    ids = [stoi.get("<start>")]
    for t in tokens:
        ids.append(stoi.get(t, stoi["<unk>"]))
    ids.append(stoi.get("<end>"))

    if len(ids) < max_len:
        ids = ids + [stoi["<pad>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

def main():
    df = pd.read_csv(RAW_CSV)
    print("Loaded", len(df), "rows")

    # 1) Split
    df = make_splits(df)
    print(df["split"].value_counts())

    # 2) Tokenize
    df["tokens"] = df["utterance"].astype(str).apply(simple_tokenize)
    df["tokens_len"] = df["tokens"].apply(len)

    # 3) Drop too short / too long based on TRAIN
    too_short = df["tokens_len"] < MIN_LEN
    long_threshold = np.percentile(df[df["split"] == "train"]["tokens_len"], LONG_PRC)
    too_long = df["tokens_len"] > long_threshold

    print("Dropping", int(too_short.sum()), "too-short captions")
    print("Long length threshold:", long_threshold,
          "| Dropping", int(too_long.sum()), "too-long captions")

    df = df[~(too_short | too_long)].reset_index(drop=True)

    # 4) Build vocab using train split
    train_tokens = df[df["split"] == "train"]["tokens"].tolist()
    stoi, itos = build_vocab(train_tokens)
    print("Vocab size:", len(itos))

    # 5) Encode to fixed length
    max_len = int(long_threshold) + 2 + 1   # rough: tokens + <start>/<end>
    df["tokens_encoded"] = df["tokens"].apply(lambda toks: encode(toks, stoi, max_len))

    # 6) Save
    out_csv = OUT_DIR / "artemis_preprocessed.csv"
    df.to_csv(out_csv, index=False)
    print("Saved preprocessed CSV to", out_csv)

    vocab_path = OUT_DIR / "vocab_simple.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump(
            {"stoi": stoi, "itos": itos, "max_len": max_len, "special_tokens": SPECIAL_TOKENS},
            f,
        )
    print("Saved vocab to", vocab_path)

if __name__ == "__main__":
    main()
