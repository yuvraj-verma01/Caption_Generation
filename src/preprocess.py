import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from collections import Counter

RAW_CSV = Path("C:\\Users\\yuvra\\OneDrive\\Desktop\\Monsoon 2025\\iml_assignments\\A3\\ArtEmisProject\\data\\artemis_dataset_release_v0.csv")
OUT_DIR = Path("C:\\Users\\yuvra\\OneDrive\\Desktop\\Monsoon 2025\\iml_assignments\\A3\\ArtEmisProject\\data\\artemis_preprocessed_1")
OUT_DIR.mkdir(exist_ok=True, parents=True)

MIN_LEN = 5
LONG_PRC = 95
MIN_FREQ = 5
SPLIT_RATIOS = (0.80, 0.10, 0.10)  
RANDOM_SEED = 2021

SPECIAL_TOKENS = {
    "<pad>": 0,
    "<unk>": 1,
    "<start>": 2,
    "<end>": 3,
}


def simple_tokenize(text: str):
    return text.lower().strip().split()


def make_splits_by_artwork(df):
    
    print("\n=== Splitting by Unique Artworks ===")

    unique_artworks = df[["art_style", "painting"]].drop_duplicates().reset_index(drop=True)
    print(f"Total unique artworks: {len(unique_artworks)}")


    rng = np.random.default_rng(RANDOM_SEED)
    shuffled_idx = rng.permutation(len(unique_artworks))
    unique_artworks = unique_artworks.iloc[shuffled_idx].reset_index(drop=True)

    n_total = len(unique_artworks)
    n_train = int(SPLIT_RATIOS[0] * n_total)
    n_val = int(SPLIT_RATIOS[1] * n_total)

    unique_artworks["split"] = "test" 
    unique_artworks.loc[:n_train - 1, "split"] = "train"
    unique_artworks.loc[n_train:n_train + n_val - 1, "split"] = "val"

    print("Artwork distribution:")
    print(f"  Train: {(unique_artworks['split'] == 'train').sum()} artworks")
    print(f"  Val:   {(unique_artworks['split'] == 'val').sum()} artworks")
    print(f"  Test:  {(unique_artworks['split'] == 'test').sum()} artworks")

    if "split" in df.columns:
        df = df.drop(columns=["split"])

    df = df.merge(
        unique_artworks[["art_style", "painting", "split"]],
        on=["art_style", "painting"],
        how="left",
    )


    print("\n=== Verification ===")
    train_artworks = set(df[df["split"] == "train"]["painting"].unique())
    val_artworks = set(df[df["split"] == "val"]["painting"].unique())
    test_artworks = set(df[df["split"] == "test"]["painting"].unique())

    overlap_train_val = len(train_artworks & val_artworks)
    overlap_train_test = len(train_artworks & test_artworks)
    overlap_val_test = len(val_artworks & test_artworks)

    print("Artwork overlap:")
    print(f"  Train & Val:  {overlap_train_val} (should be 0) {'good' if overlap_train_val == 0 else 'ERROR!'}")
    print(f"  Train & Test: {overlap_train_test} (should be 0) {'good' if overlap_train_test == 0 else 'ERROR!'}")
    print(f"  Val & Test:   {overlap_val_test} (should be 0) {'good' if overlap_val_test == 0 else 'ERROR!'}")

    print("\nCaption distribution:")
    print(df["split"].value_counts())

    return df


def build_vocab(train_tokens):
    counter = Counter()
    for toks in train_tokens:
        counter.update(toks)

    itos = list(SPECIAL_TOKENS.keys())
    for word, freq in counter.items():
        if freq >= MIN_FREQ:
            itos.append(word)

    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos


def encode(tokens, stoi, max_len):
    ids = [stoi["<start>"]]
    for t in tokens:
        ids.append(stoi.get(t, stoi["<unk>"]))
    ids.append(stoi["<end>"])

    if len(ids) < max_len:
        ids = ids + [stoi["<pad>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids


def main():

    print(f"Loading data from {RAW_CSV}")
    df = pd.read_csv(RAW_CSV)
    print(f"Loaded {len(df):,} rows")
    print(f"Columns: {df.columns.tolist()}")

    df = make_splits_by_artwork(df)

    print("\n=== Tokenization ===")
    df["tokens"] = df["utterance"].astype(str).apply(simple_tokenize)
    df["tokens_len"] = df["tokens"].apply(len)


    train_lengths = df[df["split"] == "train"]["tokens_len"]
    long_threshold = np.percentile(train_lengths, LONG_PRC)

    too_short = df["tokens_len"] < MIN_LEN
    too_long = df["tokens_len"] > long_threshold

    print(f"Dropping {too_short.sum():,} too-short captions (< {MIN_LEN} tokens)")
    print(f"Long threshold: {long_threshold:.0f} tokens (95th percentile)")
    print(f"Dropping {too_long.sum():,} too-long captions")

    df = df[~(too_short | too_long)].reset_index(drop=True)
    print(f"Remaining: {len(df):,} captions")

 
    print("\n=== Building Vocabulary ===")
    train_tokens = df[df["split"] == "train"]["tokens"].tolist()
    stoi, itos = build_vocab(train_tokens)
    print(f"Vocabulary size: {len(itos):,} (min_freq={MIN_FREQ})")


    max_len = int(long_threshold) + 3  
    print(f"Max sequence length: {max_len}")

    df["tokens_encoded"] = df["tokens"].apply(lambda toks: encode(toks, stoi, max_len))

    out_csv = OUT_DIR / "artemis_preprocessed_1.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved preprocessed CSV to {out_csv}")


    vocab_path = OUT_DIR / "vocab_1.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump(
            {
                "stoi": stoi,
                "itos": itos,
                "max_len": max_len,
                "special_tokens": SPECIAL_TOKENS,
            },
            f,
        )
    print(f"Saved vocabulary to {vocab_path}")


    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        split_path = OUT_DIR / f"{split_name}_by_artwork.csv"
        split_df.to_csv(split_path, index=False)
        print(
            f"Saved {split_name}: {len(split_df):,} captions, "
            f"{split_df['painting'].nunique()} unique artworks"
        )


    print("\n=== FINAL VERIFICATION ===")
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]


    train_paintings = set(train_df["painting"])
    val_paintings = set(val_df["painting"])
    test_paintings = set(test_df["painting"])

    print(
        f"Unique paintings: "
        f"Train={len(train_paintings)}, "
        f"Val={len(val_paintings)}, "
        f"Test={len(test_paintings)}"
    )
    print(f"Train & Val: {len(train_paintings & val_paintings)} (MUST BE 0)")
    print(f"Train & Test: {len(train_paintings & test_paintings)} (MUST BE 0)")
    print(f"Val & Test: {len(val_paintings & test_paintings)} (MUST BE 0)")


    if "img_resized_path" in df.columns:
        train_imgs = set(train_df["img_resized_path"])
        val_imgs = set(val_df["img_resized_path"])
        test_imgs = set(test_df["img_resized_path"])

        print(
            f"\nImage paths: "
            f"Train={len(train_imgs)}, "
            f"Val={len(val_imgs)}, "
            f"Test={len(test_imgs)}"
        )
        print(f"Train & Val: {len(train_imgs & val_imgs)} (MUST BE 0)")
        print(f"Train & Test: {len(train_imgs & test_imgs)} (MUST BE 0)")
        print(f"Val & Test: {len(val_imgs & test_imgs)} (MUST BE 0)")

    print("\nALL DONE! No data leakage if all overlaps are 0.")


if __name__ == "__main__":
    main()
