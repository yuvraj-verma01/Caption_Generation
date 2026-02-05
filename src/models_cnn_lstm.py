import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNNEncoder(nn.Module):

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),


            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),


            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(256, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.conv(images)          
        x = x.mean(dim=[2, 3])         
        x = self.fc(x)                 
        x = F.relu(x)
        return x


class CNNLSTMCaptioner(nn.Module):
   
    def __init__(
        self,
        embedding_layer: nn.Embedding,
        hidden_dim: int,
        vocab_size: int,
        pad_idx: int,
        img_feat_dim: int = 256,
    ):
        super().__init__()

        self.embedding = embedding_layer         
        self.embed_dim = embedding_layer.embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

  
        self.cnn_encoder = SimpleCNNEncoder(out_dim=img_feat_dim)

        
        self.img_to_h = nn.Linear(img_feat_dim, hidden_dim)
        self.img_to_c = nn.Linear(img_feat_dim, hidden_dim)


        self.lstm = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,    
        )

        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images, captions_in):

        B = images.size(0)

        img_feat = self.cnn_encoder(images)  

    
        h0 = torch.tanh(self.img_to_h(img_feat))  
        c0 = torch.tanh(self.img_to_c(img_feat))  

      
        h0 = h0.unsqueeze(0)   
        c0 = c0.unsqueeze(0)   

        
        embedded = self.embedding(captions_in)

        outputs, _ = self.lstm(embedded, (h0, c0))   


        logits = self.fc_out(outputs)  
        return logits
