# Transformer vs RNN: Attention Mechanism Comparison

A hands-on comparison between Transformer (attention-based) and RNN models for Pinyin-to-Chinese character conversion, demonstrating the effectiveness of attention mechanisms.

## Overview

This repository is a modified version of lsdefine's Transformer implementation focuses on comparing two approaches for sequence-to-sequence learning:
- **Transformer** (`pinyin_main.py`): Attention-based model
- **RNN** (`pinyin_rnn.py`): Traditional recurrent approach

Both models solve the same task: converting Pinyin romanization to Chinese characters.

## Task: Pinyin → Chinese Characters

**Input**: `ji zhi hu die zai yang guang xia fei wu`  
**Output**: `几只蝴蝶在阳光下飞舞`

This task demonstrates attention's ability to handle long-range dependencies and parallel processing.

## Repository structure
project/
├── 📊 Core Models
│   ├── transformer.py         # Complete Transformer architecture
│   └── rnn_s2s.py             # GRU-based encoder-decoder baseline
│
├── 🎯 Main Experiments (Focus)
│   ├── pinyin_main.py         # Transformer: Pinyin→Chinese conversion
│   └── pinyin_rnn.py          # RNN: Same task for comparison
│
├── 🔧 dump/
│   └── en2de_main.py          # English-German translation demo
│
├── 🛠️ Utilities
│   ├── dataloader.py          # Data preprocessing & batch generation
│   └── ljqpy.py               # File I/O, text processing utilities
│
└── 📁 data/                   # Training datasets (create manually)
   ├── pinyin.corpus.examples.txt
   ├── pinyin_word.txt
   └── models/                # Saved model weights

## Running the Comparison

### 1. Prepare Data
Prepare the data: `data/pinyin.corpus.examples.txt` with format:
```
ji zhi hu die	几只蝴蝶
zai yang guang xia	在阳光下
fei wu	飞舞
```

### 2. Train Transformer Model
```bash
python pinyin_main.py train
```

**Progressive Training Strategy:**
- Layer 1: Basic attention mechanism
- Layer 2: Deeper representation learning  
- Layer 3: Complex pattern recognition

### 3. Train RNN Model
```bash
python pinyin_rnn.py train
```

**Standard Training:** All layers trained simultaneously.

### 4. Interactive Testing
```bash
# Test Transformer
python pinyin_main.py test

# Test RNN  
python pinyin_rnn.py test
```

## Key Architectural Differences

### Transformer (`pinyin_main.py`)

**Core Components:**
```python
# Multi-head attention for parallel processing
s2s = Transformer(itokens, otokens, len_limit=500, 
                  d_model=256, n_head=4, layers=3)

# Progressive training
s2s.compile(opt1, active_layers=1)  # Train layer 1
s2s.compile(opt2, active_layers=2)  # Add layer 2
s2s.compile(opt3, active_layers=3)  # Add layer 3
```

**Attention Mechanism:**
- **Self-attention**: Each position attends to all positions
- **Cross-attention**: Decoder attends to encoder outputs
- **Parallel processing**: All positions computed simultaneously

### RNN (`pinyin_rnn.py`)

**Core Components:**
```python
# Sequential GRU-based processing
s2s = RNNSeq2Seq(itokens, otokens, 128, layers=3)

# Standard training
s2s.model.fit([X,Y], Y[:,1:], batch_size=32, epochs=5)
```

**Sequential Processing:**
- **Hidden states**: Information flows through time steps
- **Memory bottleneck**: Fixed-size hidden state
- **Sequential dependency**: Must process tokens in order

## Attention vs Sequential Processing

### Attention Advantages

1. **Global Context**: Each output position can attend to any input position
2. **Parallel Training**: All positions processed simultaneously
3. **Long-range Dependencies**: Direct connections across sequence
4. **Interpretability**: Attention weights show model focus

### RNN Advantages

1. **Memory Efficiency**: Constant memory regardless of sequence length
2. **Simplicity**: Straightforward sequential processing
3. **Proven Architecture**: Well-established for sequence tasks

## Detailed Code Analysis

### Transformer Decoding
```python
# Fast parallel decoding with beam search
print(s2s.decode_sequence_fast(quest.split()))
rets = s2s.beam_search(quest.split())
```

### RNN Decoding  
```python
# Sequential step-by-step decoding
print(s2s.decode_sequence(quest.split()))
```

## Training Configuration Comparison

### Transformer Configuration
```python
d_model = 256
s2s = Transformer(itokens, otokens, len_limit=500, 
                  d_model=d_model, d_inner_hid=1024,
                  n_head=4, layers=3, dropout=0.1)
```

### RNN Configuration
```python
latent_dim = 128
s2s = RNNSeq2Seq(itokens, otokens, latent_dim, layers=3)
```

## Expected Performance Differences

### Transformer Benefits
- **Better long sequences**: Attention handles long Pinyin phrases better
- **Faster training**: Parallel processing speeds up training
- **Higher accuracy**: Global context improves character selection

### RNN Characteristics
- **Shorter sequences**: Works well for short Pinyin phrases
- **Memory efficient**: Lower memory usage during inference
- **Gradient issues**: May struggle with very long sequences

## Key Learning Points

1. **Attention Mechanism**: How attention enables global context understanding
2. **Parallel vs Sequential**: Trade-offs between processing approaches  
3. **Progressive Training**: Strategy for training deep attention models
4. **Sequence-to-Sequence**: Both architectures handle variable-length inputs/outputs

## Running Experiments

```bash
# Compare training speed
time python pinyin_main.py train
time python pinyin_rnn.py train

# Test on same examples
echo "ni hao shi jie" | python pinyin_main.py test
echo "ni hao shi jie" | python pinyin_rnn.py test
```

## Requirements

- TensorFlow 2.x
- Data: `data/pinyin.corpus.examples.txt`
- Models saved to: `models/pinyin.model.weights.h5`

This comparison provides hands-on experience with the fundamental differences between attention-based and recurrent architectures in sequence-to-sequence learning.

## Acknowledgements

This implementation builds upon several excellent open-source projects:

- **Original Paper**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- **PyTorch Reference**: [jadore801120/attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch) - Data preprocessing and model structure insights
- **Keras Implementation**: [lsdefine/attention-is-all-you-need-keras](https://github.com/lsdefine/attention-is-all-you-need-keras) - Keras-specific implementation patterns

## Citation

```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

## License

MIT License
