"""
LSTM-based Rumour Classifier
Non-transformer baseline for model-agnostic validation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMRumourClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, num_classes=3, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, input_ids, attention_mask=None):
        # input_ids: [batch_size, seq_len]
        embedded = self.embedding(input_ids)  # [batch, seq_len, embed_dim]
        
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use final hidden states from both directions
        # hidden shape: [num_layers * 2, batch, hidden_dim]
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # [batch, hidden_dim*2]
        
        hidden = self.dropout(hidden)
        logits = self.classifier(hidden)
        
        return logits