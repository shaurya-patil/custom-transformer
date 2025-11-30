import torch

def tokenize(text):
    """
    Converts a string to a list of UTF-8 byte integers.
    """
    return list(text.encode('utf-8'))

def detokenize(tokens):
    """
    Converts a list of UTF-8 byte integers back to a string.
    """
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()
    return bytes(tokens).decode('utf-8', errors='replace')

def create_padding_mask(seq):
    """
    seq: [batch_size, seq_len]
    Returns: [batch_size, 1, 1, seq_len]
    """
    # seq is 0 at padding positions
    mask = (seq == 0).float()
    return mask.unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    """
    Returns: [size, size]
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask # (size, size)
