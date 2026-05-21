"""
Save CALM-RV model in HuggingFace format for robustness evaluation
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

# Path to your existing CALM-RV checkpoint (from LOEO training)
# Note: The training script needs to save checkpoints first

print("This script requires a saved CALM-RV checkpoint.")
print("Run training with model saving enabled first.")