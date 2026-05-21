"""
TextCNN-based Rumour Classifier
Non-transformer baseline for model-agnostic validation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_filters=100, filter_sizes=[3, 4, 5], num_classes=3, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(len(filter_sizes) * num_filters, num_classes)
    
    def forward(self, input_ids, attention_mask=None):
        # input_ids: [batch_size, seq_len]
        embedded = self.embedding(input_ids).unsqueeze(1)  # [batch, 1, seq_len, embed_dim]
        
        # Apply each convolution
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded)).squeeze(3)  # [batch, num_filters, seq_len - fs + 1]
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # [batch, num_filters]
            conv_outputs.append(pooled)
        
        # Concatenate all pooled features
        concat = torch.cat(conv_outputs, dim=1)  # [batch, num_filters * len(filter_sizes)]
        concat = self.dropout(concat)
        logits = self.classifier(concat)
        
        return logits