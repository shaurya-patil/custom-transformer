import torch
from config import TransformerConfig
from transformer import Transformer
from inference import translate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

config = TransformerConfig()
# config.n_layers = 2
# config.d_model = 64
# config.n_heads = 4
# config.d_ff = 256
# config.max_seq_len = 64

model = Transformer(config).to(device)
try:
    model.load_state_dict(torch.load('model.pth', map_location=device))
    print("Model loaded.")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)

text = "<afr> Tom got home late."
print(f"Input: {text}")
translation = translate(text, model, config, device)
print(f"Translation: {translation}")

text2 = "<afr> Hello world"
print(f"Input: {text2}")
translation2 = translate(text2, model, config, device)
print(f"Translation: {translation2}")
