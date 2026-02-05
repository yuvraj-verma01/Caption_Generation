import argparse
from pathlib import Path
import pickle

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from transformers import AutoProcessor, BlipForConditionalGeneration

from models_cnn_lstm import CNNLSTMCaptioner



PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
BLIP_DIR = PROJECT_ROOT / "blip_finetuned_artemis" / "final_model_epoch2"

VOCAB_PATH = DATA_DIR / "vocab_simple.pkl"


CNN_MODEL_FILES = {
    "TFIDF_SVD_CNNLSTM": "tfidf_svd_cnnlstm_best_new_old.pt",
    "W2V_CNNLSTM_OLD":   "w2v_cnnlstm_best_old.pt",
    "GLOVE_CNNLSTM_OLD": "glove_cnnlstm_best_old.pt",
}




image_transform_cnn = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])



def load_vocab(vocab_path: Path):
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    return vocab




def generate_caption_cnn(
    model: CNNLSTMCaptioner,
    image_tensor: torch.Tensor,
    stoi,
    itos,
    max_len: int,
    device: torch.device,
) -> str:
    model.eval()

    start_idx = stoi["<start>"]
    end_idx = stoi["<end>"]

    image_tensor = image_tensor.to(device)          # [1, 3, 128, 128]
    caption_in = torch.tensor([[start_idx]], dtype=torch.long, device=device)

    decoded_tokens = []

    for _ in range(max_len - 1):
        with torch.no_grad():
            logits = model(image_tensor, caption_in)    # [1, T, vocab_size]

        next_token_logits = logits[0, -1]               # [vocab_size]
        next_token_id = torch.argmax(next_token_logits).item()

        if next_token_id == end_idx:
            break

        if next_token_id < len(itos):
            decoded_tokens.append(itos[next_token_id])
        else:
            decoded_tokens.append("<unk>")

        next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        caption_in = torch.cat([caption_in, next_token_tensor], dim=1)

    return " ".join(decoded_tokens)




def load_cnn_model(
    ckpt_path: Path,
    vocab,
    device: torch.device,
) -> tuple[CNNLSTMCaptioner, list[str]]:
    """
    Returns (model, itos_ckpt).
    Uses checkpoint weights to infer embedding_dim and hidden_dim.
    """
    stoi = vocab["stoi"]
    itos = vocab["itos"]
    special_tokens = vocab["special_tokens"]

    print(f"\nLoading CNN-LSTM checkpoint from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)

    # infer sizes from checkpoint
    emb_weight = state_dict["embedding.weight"]
    vocab_size_ckpt, embed_dim_ckpt = emb_weight.shape

    img_to_h_weight = state_dict["img_to_h.weight"]
    hidden_dim_ckpt = img_to_h_weight.shape[0]

    print(f"  -> ckpt vocab_size: {vocab_size_ckpt}")
    print(f"  -> ckpt embed_dim : {embed_dim_ckpt}")
    print(f"  -> ckpt hidden_dim: {hidden_dim_ckpt}")

    if len(itos) < vocab_size_ckpt:
        raise ValueError(
            f"Loaded vocab size ({len(itos)}) is smaller than checkpoint vocab size ({vocab_size_ckpt}). "
            "You need the original vocab used for training."
        )

    itos_ckpt = itos[:vocab_size_ckpt]

    pad_idx = special_tokens["<pad>"]
    if pad_idx >= vocab_size_ckpt:
        raise ValueError(
            f"pad_idx={pad_idx} is outside checkpoint vocab range [0, {vocab_size_ckpt-1}]"
        )

    embedding_layer = nn.Embedding(
        num_embeddings=vocab_size_ckpt,
        embedding_dim=embed_dim_ckpt,
        padding_idx=pad_idx,
    )

    model = CNNLSTMCaptioner(
        embedding_layer=embedding_layer,
        hidden_dim=hidden_dim_ckpt,
        vocab_size=vocab_size_ckpt,
        pad_idx=pad_idx,
        img_feat_dim=256,  
    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, itos_ckpt




def load_blip_model(model_dir: Path, device: torch.device):
    print("\nLoading BLIP model from:", model_dir)
    processor = AutoProcessor.from_pretrained(model_dir)
    model = BlipForConditionalGeneration.from_pretrained(model_dir).to(device)
    model.eval()
    return processor, model


def generate_caption_blip(
    processor,
    model,
    img: Image.Image,
    device: torch.device,
    max_new_tokens: int = 30,
) -> str:
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return caption




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Folder containing test images (jpg/jpeg/png).",
    )
    args = parser.parse_args()

    folder_path = Path(args.folder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    
    exts = {".jpg", ".jpeg", ".png"}
    image_paths = sorted([p for p in folder_path.iterdir() if p.suffix.lower() in exts])
    if not image_paths:
        print("No images found in folder:", folder_path)
        return

    print(f"Found {len(image_paths)} images.")

    
    vocab = load_vocab(VOCAB_PATH)
    stoi = vocab["stoi"]
    max_len = vocab["max_len"]
    print("Loaded vocab with size:", len(vocab["itos"]))
    print("Max caption length:", max_len)


    cnn_models = {}
    for name, fname in CNN_MODEL_FILES.items():
        ckpt_path = MODELS_DIR / fname
        model, itos_ckpt = load_cnn_model(ckpt_path, vocab, device)
        cnn_models[name] = (model, itos_ckpt)


    processor, blip_model = load_blip_model(BLIP_DIR, device)


    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")

            print("\n" + "=" * 70)
            print("Image:", img_path.name)
            print("Path :", img_path)

            
            img_tensor_cnn = image_transform_cnn(img).unsqueeze(0)  # [1,3,128,128]

            for name, (model, itos_ckpt) in cnn_models.items():
                caption_cnn = generate_caption_cnn(
                    model=model,
                    image_tensor=img_tensor_cnn,
                    stoi=stoi,
                    itos=itos_ckpt,
                    max_len=max_len,
                    device=device,
                )
                print(f"[{name}]  {caption_cnn}")

            # BLIP model
            caption_blip = generate_caption_blip(
                processor=processor,
                model=blip_model,
                img=img,
                device=device,
                max_new_tokens=30,
            )
            print(f"[BLIP_TRANSFORMER]  {caption_blip}")

        except Exception as e:
            print(f"\nError processing {img_path}: {e}")


if __name__ == "__main__":
    main()
