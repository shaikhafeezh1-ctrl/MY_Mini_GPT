# MY_Mini_GPT
Mini GPT (No Tokenizer)  :A lightweight GPT‑style language model built from scratch using PyTorch and transformers — without relying on tokenizers. Instead, it uses a simple character‑level vocabulary to encode and decode text, making the internals of attention and sequence modeling more transparent.
# Features
Character‑level encoding (no external tokenizers)
Transformer blocks with:
Self‑Attention
Multi‑Head Attention
Positional Embeddings
Feed‑Forward layers + LayerNorm
# Tech Stack
PyTorch for model building & training

nn.TransformerEncoderLayer for stacking transformer blocks

CrossEntropyLoss + Adam for optimization
