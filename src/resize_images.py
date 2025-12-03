from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pandas as pd

# -------------------------------------------------------------
# FULL ABSOLUTE PATHS — EXACTLY WHAT YOU ASKED FOR
# -------------------------------------------------------------

CSV_IN = Path("C:\\Users\\Home\\Desktop\\IML\\Caption_Generation\\artemis_preprocessed_with_paths.csv")

# Your raw images folder (contains Impressionism/, Romanticism/, etc.)
RAW_IMG_ROOT = Path("D:\wikiart")

# Output folder for resized 128×128 images
OUT_IMG_ROOT = Path("D:\\IML_Resized")
OUT_IMG_ROOT.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = (128, 128)

# -------------------------------------------------------------
# SCRIPT
# -------------------------------------------------------------

def main():
    df = pd.read_csv(CSV_IN)
    print("Loaded", len(df), "rows from", CSV_IN)

    resized_paths = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        style = row["art_style"]
        painting = row["painting"]

        # raw/Impressionism/painting.jpg
        in_path = RAW_IMG_ROOT / style / f"{painting}.jpg"

        # resized/wikiart_128/Impressionism/painting.jpg
        out_dir = OUT_IMG_ROOT / style
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{painting}.jpg"

        if not out_path.exists():
            try:
                img = Image.open(in_path).convert("RGB")
                img = img.resize(TARGET_SIZE)
                img.save(out_path, "JPEG")
            except Exception as e:
                print("Skipping", in_path, "-->", e)
                out_path = ""  # mark missing

        resized_paths.append(str(out_path))

    df["img_resized_path"] = resized_paths
    df = df[df["img_resized_path"] != ""].reset_index(drop=True)

    CSV_OUT = Path("D:\\IML_CSV\\artemis_preprocessed_with_paths2.csv")
    df.to_csv(CSV_OUT, index=False)

    print("Saved →", CSV_OUT)
    print("Example resized path:", df["img_resized_path"].iloc[0])

if __name__ == "__main__":
    main()
