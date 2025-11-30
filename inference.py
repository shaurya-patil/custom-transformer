import torch
import numpy as np
from config import TransformerConfig
from transformer import Transformer
from utils import tokenize, detokenize, create_padding_mask, create_look_ahead_mask

def translate(sentence, model, config, device, max_len=40):
    model.eval()
    
    enc_toks = [b + 4 for b in tokenize(sentence)]
    enc_toks = [1] + enc_toks + [2]
    
    pad_len = config.max_seq_len - len(enc_toks)
    if pad_len > 0:
        enc_toks = enc_toks + [0] * pad_len
    else:
        enc_toks = enc_toks[:config.max_seq_len]
        
    encoder_input = torch.tensor([enc_toks], dtype=torch.long).to(device)
    
    enc_padding_mask = create_padding_mask(encoder_input).to(device)
    
    with torch.no_grad():
        enc_output = model.encoder(encoder_input, enc_padding_mask)
    
    decoder_input = torch.tensor([[1]], dtype=torch.long).to(device)
    
    for i in range(max_len):
        look_ahead_mask = create_look_ahead_mask(decoder_input.size(1)).to(device)
        dec_target_padding_mask = create_padding_mask(decoder_input).to(device)
        combined_mask = torch.max(dec_target_padding_mask, look_ahead_mask)
        
        with torch.no_grad():
            dec_output = model.decoder(decoder_input, enc_output, combined_mask, enc_padding_mask)
            predictions = model.final_layer(dec_output)
        
        predictions = predictions[:, -1, :]
        predicted_id = torch.argmax(predictions, dim=-1).item()
        
        if predicted_id == 2:
            break
            
        decoder_input = torch.cat([decoder_input, torch.tensor([[predicted_id]], device=device)], dim=-1)
        
        if predicted_id == 0:
            break
            
    result_toks = decoder_input[0, 1:].tolist()
    
    result_bytes = []
    for t in result_toks:
        if t >= 4:
            result_bytes.append(t - 4)
            
    return detokenize(result_bytes)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        print("Loading model from model.pth...")
        config = TransformerConfig()
        # Ensure config matches training
        # config.n_layers = 2
        # config.d_model = 64
        # config.n_heads = 4
        # config.d_ff = 256
        # config.max_seq_len = 64
        
        model = Transformer(config).to(device)
        model.load_state_dict(torch.load('model.pth', map_location=device))
        print("Model loaded.")
        
        print("Enter an English sentence to translate (e.g. '<afr> Hello world'):")
        while True:
            text = input("> ")
            if text.lower() in ['q', 'quit', 'exit']:
                break
            
            translation = translate(text, model, config, device)
            print(f"Translation: {translation}")
            
    except FileNotFoundError:
        print("Error: model.pth not found. Please run train.py first.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
