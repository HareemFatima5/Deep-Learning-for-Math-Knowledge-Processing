# Deep Learning for Mathematical Knowledge Processing

## Project Description
This project implements a multimodal deep learning system for solving mathematical word problems by generating executable equations from natural language problem statements. The model combines text understanding using T5 transformers with structural analysis using Graph Neural Networks (GNNs) to accurately interpret and solve mathematical problems.

## Architecture Overview
The system follows a pipeline approach:

1. **Text Processing**: Problem statements are encoded using T5-small transformer
2. **Structural Analysis**: Dependency graphs are built using spaCy and processed with GNN
3. **Multimodal Fusion**: Text and graph embeddings are fused using cross-attention
4. **Equation Generation**: A transformer decoder generates equations in prefix notation
5. **Evaluation**: Multiple metrics assess prediction quality

### Model Architecture
![Project Architecture]()


## Dataset
- **Source**: Combined ParaMAWPS + ASDiv datasets
- **Original Size**: 17,555 mathematical word problems
- **Filtered Size**: 17,158 problems (answers ≤ 10,000)
- **Final Size**: 16,928 valid problems after prefix conversion
- **Split**: 70% train (11,849), 15% validation (2,539), 15% test (2,540)

## Key Components

### 1. Data Preprocessing
- Converts equations to prefix notation (e.g., "56*9" becomes "* 56 9")
- Builds vocabulary from problem text and equations
- Creates train/validation/test splits

### 2. Graph Construction
- Uses spaCy for dependency parsing of problem statements
- Builds graph where nodes are tokens and edges represent syntactic relationships
- Processes graphs using GINConv layers in PyTorch Geometric

### 3. Text Encoding
- Uses T5-small pretrained model from Hugging Face
- Generates 512-dimensional sentence embeddings via mean pooling

### 4. Multimodal Fusion Model
- Projects T5 (512-dim) and GNN (128-dim) embeddings to 256-dim
- Uses multi-head cross-attention for fusion
- Includes layer normalization and dropout

### 5. Transformer Decoder
- 4-layer transformer decoder with 8 attention heads
- Positional encoding for sequence modeling
- Generates equations token by token

### 6. Inference Strategies
- **Greedy Decoding**: Always selects highest probability token
- **Beam Search**: Explores top 5 sequences, selects best overall

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install torch-geometric
pip install optuna pandas numpy matplotlib scikit-learn tqdm transformers spacy nltk
python -m spacy download en_core_web_sm


## Results
| Metric | Score |
|--------|-------|
| Exact Match Accuracy | 76.65% |
| Numerical Accuracy | 80.79% |
| Token-Level Precision | 0.8908 |
| Token-Level Recall | 0.8910 |
| Token-Level F1 Score | 0.8909 |
| BLEU Score | 0.6194 |
