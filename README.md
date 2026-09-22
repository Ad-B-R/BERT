# BERT from Scratch - SNLI

A BERT-base–sized Transformer encoder (Devlin et al., 2019) implemented from scratch in PyTorch and trained from random initialization (no pre-training) as a 3-way NLI classifier on SNLI. Every component (embeddings, multi-head attention, LayerNorm, pre-norm encoder blocks, classifier head) from scratch; only the `bert-base-uncased` tokenizer is borrowed.

## Architecture

| Component | Setting |
|---|---|
| Input | `[CLS] premise [SEP] hypothesis [SEP]`, padded/truncated to 128 tokens |
| Embedding | Token + learned positional + segment, followed by LayerNorm |
| Encoder | 12 pre-norm blocks, `d_model=768`, 12 heads, `d_ff=3072`, ReLU FFN (~108.6M params) |
| Attention mask | Padding tokens masked with `-inf` before softmax |
| Head | Final LayerNorm → `[CLS]` token → linear classifier (3 classes) |
| Init | Xavier-uniform for linear layers, N(0, 0.02) for embeddings |

## Training

- **Data:** first 50K labelled SNLI training pairs; full validation and test splits
- **Optimizer:** AdamW, lr `2e-5`, weight decay `0.01`
- **Schedule:** linear warmup over 1 epoch, then constant lr
- **Epochs / batch size:** 3 / 16
- **Regularisation:** dropout 0.1, gradient clipping at norm 1.0
- **Logging:** Weights & Biases (`bert-spam`)

All hyperparameters live in `config.json`.

## Structure

```text
BERT/
├── model.py      # Embeddings, attention, LayerNorm, FFN, encoder, BERT head
├── data.py       # SNLI loading, tokenization and loaders
├── run.py        # Model assembly from config + training/eval loop
└── config.json   # Transformer and training settings
```

## Usage

```bash
pip install torch datasets transformers wandb
python run.py
```

SNLI and the tokenizer download automatically from Hugging Face.

## Results

| Metric | Value |
|---|---|
| Val accuracy | TBD |
| Test accuracy | TBD |

For reference: majority class ≈ 34%, hypothesis-only baseline ≈ 67%, fine-tuned pre-trained BERT-base ≈ 90%.