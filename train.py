import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from config import TransformerConfig
from transformer import Transformer
from utils import tokenize, create_padding_mask, create_look_ahead_mask

def load_data(data_dir, limit_files=None):
    pairs = []
    files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    if limit_files:
        files = files[:limit_files]
        
    for filename in files:
        path = os.path.join(data_dir, filename)
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    if len(parts) == 3:
                        eng, tar, code = parts
                        src = f"<{code}> {eng}"
                        pairs.append((src, tar))
                    else:
                        eng, tar = parts[:2]
                        pairs.append((eng, tar))
    return pairs

def prepare_batch(pairs, max_seq_len, device):
    inputs = []
    targets = []
    
    for src, tar in pairs:
        src_toks = [b + 4 for b in tokenize(src)]
        tar_toks = [b + 4 for b in tokenize(tar)]
        
        src_toks = [1] + src_toks + [2]
        tar_toks = [1] + tar_toks + [2]
        
        if len(src_toks) < max_seq_len:
            src_toks = src_toks + [0] * (max_seq_len - len(src_toks))
        else:
            src_toks = src_toks[:max_seq_len]
            
        if len(tar_toks) < max_seq_len:
            tar_toks = tar_toks + [0] * (max_seq_len - len(tar_toks))
        else:
            tar_toks = tar_toks[:max_seq_len]
            
        inputs.append(src_toks)
        targets.append(tar_toks)
        
    return torch.tensor(inputs, dtype=torch.long).to(device), torch.tensor(targets, dtype=torch.long).to(device)

def train(config, data_dir, epochs=1, batch_size=8, limit_files=5, patience=3, min_loss=0.005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading data...")
    pairs = load_data(data_dir, limit_files=limit_files)
    print(f"Loaded {len(pairs)} pairs.")
    
    # Split data
    random.shuffle(pairs)
    val_split = int(len(pairs) * 0.1)
    train_pairs = pairs[val_split:]
    val_pairs = pairs[:val_split]
    print(f"Training samples: {len(train_pairs)}, Validation samples: {len(val_pairs)}")
    
    model = Transformer(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    model.train()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        np.random.shuffle(train_pairs)
        total_loss = 0
        start_time = time.time()
        model_step = 0
        
        for i in range(0, len(train_pairs), batch_size):
            batch_pairs = train_pairs[i:i+batch_size]
            if not batch_pairs:
                break
                
            inp_data, tar_data = prepare_batch(batch_pairs, config.max_seq_len, device)
            
            tar_inp = tar_data[:, :-1]
            tar_real = tar_data[:, 1:]
            
            enc_padding_mask = create_padding_mask(inp_data).to(device)
            dec_padding_mask = create_padding_mask(inp_data).to(device)
            look_ahead_mask = create_look_ahead_mask(tar_inp.size(1)).to(device)
            dec_target_padding_mask = create_padding_mask(tar_inp).to(device)
            combined_mask = torch.max(dec_target_padding_mask, look_ahead_mask)
            
            optimizer.zero_grad()
            
            predictions = model(inp_data, tar_inp, 
                                enc_padding_mask=enc_padding_mask, 
                                look_ahead_mask=combined_mask, 
                                dec_padding_mask=dec_padding_mask)
            
            # Flatten for loss
            loss = criterion(predictions.view(-1, config.vocab_size), tar_real.reshape(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            model_step += 1
            
            if model_step % 10 == 0:
                print(f"Epoch {epoch+1}, Step {model_step}, Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / (len(train_pairs) / batch_size)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}. Time: {time.time() - start_time:.2f}s")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i in range(0, len(val_pairs), batch_size):
                batch_pairs = val_pairs[i:i+batch_size]
                if not batch_pairs: break
                
                inp_data, tar_data = prepare_batch(batch_pairs, config.max_seq_len, device)
                tar_inp = tar_data[:, :-1]
                tar_real = tar_data[:, 1:]
                
                enc_padding_mask = create_padding_mask(inp_data).to(device)
                dec_padding_mask = create_padding_mask(inp_data).to(device)
                look_ahead_mask = create_look_ahead_mask(tar_inp.size(1)).to(device)
                dec_target_padding_mask = create_padding_mask(tar_inp).to(device)
                combined_mask = torch.max(dec_target_padding_mask, look_ahead_mask)
                
                predictions = model(inp_data, tar_inp, 
                                    enc_padding_mask=enc_padding_mask, 
                                    look_ahead_mask=combined_mask, 
                                    dec_padding_mask=dec_padding_mask)
                
                loss = criterion(predictions.view(-1, config.vocab_size), tar_real.reshape(-1))
                val_loss += loss.item()
                
        avg_val_loss = val_loss / (len(val_pairs) / batch_size) if len(val_pairs) > 0 else 0
        print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print("Saving model...")
            torch.save(model.state_dict(), 'model.pth')
            print("Model saved to model.pth")
            
            if best_val_loss < min_loss:
                print(f"Validation loss {best_val_loss:.4f} reached target {min_loss}. Stopping.")
                break
        else:
            patience_counter += 1
            print(f"Early Stopping Counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
        
        model.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--limit_files', type=int, default=5)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--min_loss', type=float, default=0.005)
    args = parser.parse_args()
    
    config = TransformerConfig()
    data_dir = os.path.join(os.path.dirname(__file__), 'cleaned_data')
    train(config, data_dir, epochs=args.epochs, batch_size=args.batch_size, limit_files=args.limit_files, patience=args.patience, min_loss=args.min_loss)
