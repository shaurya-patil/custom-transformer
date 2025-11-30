import dataclasses

@dataclasses.dataclass
class TransformerConfig:
    vocab_size: int = 260  # 256 bytes + 4 specials (PAD, BOS, EOS, UNK)
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.1
    eps: float = 1e-6  # LayerNorm epsilon
