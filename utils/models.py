import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module):
    def __init__(self, embed_dim, hidden_size, output_size, num_layers=3, vocab_size=128256, use_rnn=True):
        super(Baseline, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.use_rnn = use_rnn
        if use_rnn:
            self.rnn = nn.RNN(embed_dim, hidden_size, num_layers, batch_first=True)
        else:
            self.rnn = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        if self.use_rnn:
            h = torch.zeros(self.num_layers, x.size(0), self.embed_dim).to(x.device)
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.embed_dim).to(x.device)
            c0 = torch.zeros_like(h0)
            h = (h0, c0)
        assert not torch.isnan(x).any()
        embedded = self.embedding(x)
        assert not torch.isnan(embedded).any()
        rnn_out, _ = self.rnn(embedded, h)
        if lengths is not None:
            last_hidden = rnn_out[torch.arange(rnn_out.size(0)), lengths - 1]
        else:
            last_hidden = rnn_out[:, -1, :] 
        logits = self.fc(last_hidden)
        assert not torch.isnan(logits).any()
        return logits


class SmallTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, output_size, max_len=512, vocab_size=128256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.fc = nn.Linear(embed_dim, output_size)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(positions)
        x = x.permute(1, 0, 2)
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        x = x.permute(1, 0, 2)
        x = x + self.ffn(x)
        logits = x[:, 0, :]
        return self.fc(logits)
    

class TextCNN(nn.Module):
    def __init__(self, embed_dim, output_size, vocab_size=128256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, 100, kernel_size=k) for k in [3, 4, 5]
        ])
        self.fc = nn.Linear(300, output_size)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        features = [torch.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        return self.fc(torch.cat(features, dim=1))
