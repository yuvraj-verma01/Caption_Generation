import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNNEncoder(nn.Module):
    """
    Small CNN encoder that takes 3x128x128 images
    and outputs a single feature vector per image.
    """
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            # [B, 3, 128, 128] -> [B, 32, 64, 64]
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # [B, 32, 64, 64] -> [B, 64, 32, 32]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # [B, 64, 32, 32] -> [B, 128, 16, 16]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # [B, 128, 16, 16] -> [B, 256, 8, 8]
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Final linear layer to get a fixed-size feature
        self.fc = nn.Linear(256, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, 3, 128, 128]
        returns: [B, out_dim]
        """
        x = self.conv(images)          # [B, 256, 8, 8]
        x = x.mean(dim=[2, 3])         # global average pool -> [B, 256]
        x = self.fc(x)                 # [B, out_dim]
        x = F.relu(x)
        return x


class CNNLSTMCaptioner(nn.Module):
    """
    Image captioning model:
      - CNN encoder -> image feature
      - LSTM decoder over caption tokens

    We pass in an *embedding layer* so you can swap
    TF-IDF, Word2Vec, GloVe, etc. freely.
    """
    def __init__(
        self,
        embedding_layer: nn.Embedding,
        hidden_dim: int,
        vocab_size: int,
        pad_idx: int,
        img_feat_dim: int = 256,
    ):
        super().__init__()

        self.embedding = embedding_layer          # frozen embedding from outside
        self.embed_dim = embedding_layer.embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

        # Image encoder
        self.cnn_encoder = SimpleCNNEncoder(out_dim=img_feat_dim)

        # Map image feature -> initial hidden/cell state of LSTM
        self.img_to_h = nn.Linear(img_feat_dim, hidden_dim)
        self.img_to_c = nn.Linear(img_feat_dim, hidden_dim)

        # LSTM decoder
        self.lstm = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,    # (B, T, D)
        )

        # Output layer: hidden state -> vocabulary logits
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images, captions_in):
        """
        images:      [B, 3, 128, 128]
        captions_in: [B, T]  (token IDs, without the final <end>)

        returns:
          logits: [B, T, vocab_size]
        """
        B = images.size(0)

        # 1) Encode images
        img_feat = self.cnn_encoder(images)  # [B, img_feat_dim]

        # 2) Initialize LSTM hidden & cell from image features
        h0 = torch.tanh(self.img_to_h(img_feat))  # [B, hidden_dim]
        c0 = torch.tanh(self.img_to_c(img_feat))  # [B, hidden_dim]

        # LSTM expects (num_layers, B, hidden_dim)
        h0 = h0.unsqueeze(0)   # [1, B, hidden_dim]
        c0 = c0.unsqueeze(0)   # [1, B, hidden_dim]

        # 3) Embed input captions
        # captions_in: [B, T] -> [B, T, embed_dim]
        embedded = self.embedding(captions_in)

        # 4) Run LSTM
        outputs, _ = self.lstm(embedded, (h0, c0))   # outputs: [B, T, hidden_dim]

        # 5) Project to vocabulary
        logits = self.fc_out(outputs)   # [B, T, vocab_size]
        return logits
