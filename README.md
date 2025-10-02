# Tokens-impact-for-encoder-decoder-architectures

# Token-Level Explainability Framework for Transformer Architectures

A comprehensive framework for interpreting and understanding how individual tokens influence predictions in Transformer-based models through both internal mechanism analysis and external attribution methods.

## Overview

The quality of natural language generation and understanding models depends strongly on both the training process and the underlying architecture. Before the emergence of the Transformer, it was difficult to obtain high-quality generative output or strong performance on tasks such as language modeling and question answering. This limitation was largely due to the models' inability to effectively capture and interpret the relationships between words or tokens across long contexts.

The Transformer architecture changed this landscape by introducing attention mechanisms that explicitly model token-to-token interactions. This breakthrough not only improved performance but also opened the door to studying how the quality and structure of the input text affect model outputs.

### Key Questions This Project Addresses

- Which tokens matter most for a given prediction?
- How do attention layers and heads process and redistribute information?
- To what extent can we interpret the model's reasoning from the inside versus the outside?
- Where do attribution methods align (or fail to align) with internal mechanisms?

## Project Goals

This project explores and interprets the influence of input tokens on Transformer-based models through two complementary perspectives:

1. **Internal Analysis**: Probing the model itself, with a focus on attention weights, layer activations, and head ablations
2. **External Analysis**: Applying attribution methods to quantify how input tokens contribute to outputs from a model-agnostic point of view

## Tasks & Models

### Supported Tasks
- **Masked Language Modeling (MLM)**: Predicting masked tokens in context
- **Text Classification**: Categorizing input text into predefined classes

### Supported Architectures
- **Encoder-only models**: BERT (and variants like DistilBERT)
- **Decoder-only models**: GPT-2, T5

## Methodology

### Approach 1: Internal Mechanism Analysis

Understanding the model from the inside by examining its core computational components.

#### 1.1 Attention Weight Inspection
- Visualize attention matrices per-head and per-layer
- Identify which tokens receive the most attention
- Analyze alignment between attention patterns and input importance

#### 1.2 Attention Rollout / Flow
Based on Abnar & Zuidema (2020), this technique:
- Aggregates attention connections across layers
- Traces information propagation through the entire model
- Provides a global picture of token influence on final outputs

#### 1.3 Head/Layer Ablation
- Systematically remove specific attention heads or layers
- Measure impact on model performance
- Identify critical components for predictions

### Approach 2: Attribution Methods

Understanding the model from an external, model-agnostic perspective.

#### 2.1 Gradient × Input
- Computes gradient of output with respect to input embeddings
- Multiplies gradients by input embeddings
- Highlights token-level contributions to predictions

#### 2.2 Integrated Gradients (IG)
- Interpolates from baseline (zero embeddings) to actual input
- Integrates gradients along the interpolation path
- Provides more stable and less noisy attributions

#### 2.3 SHAP (Shapley Additive Explanations)
- Game-theoretic approach to feature attribution
- Computes fair contribution of each token
- Model-agnostic and theoretically grounded

#### 2.4 Layer-wise Relevance Propagation (LRP)
- Backpropagates relevance scores from output to inputs
- Uses conservation rules to distribute relevance
- Maintains consistency across layers

## Key Features

- **Token-level granularity**: Understand influence at the finest level
- **Interactive visualizations**: 
  - Heatmaps showing input-output token relationships
  - Network graphs with nodes (tokens) and edges (influence strength)
  - Layer-wise analysis views
- **Comparative analysis**: Compare internal mechanisms with external attributions
- **Multi-prompt support**: Analyze how token influence changes across input variations
- **Keyword highlighting**: Track impact of specific words or phrases

## Tech Stack

### Core Libraries
- **Transformers**: HuggingFace Transformers (model loading and inference)
- **PyTorch**: Deep learning framework
- **Captum**: Attribution methods (Integrated Gradients, SHAP, LRP)

### Visualization
- **Matplotlib**: Static plots and heatmaps
- **Plotly**: Interactive visualizations
- **D3.js**: Advanced network graphs (optional)

### UI/Frontend (Optional)
- **Streamlit**: Rapid prototyping and interactive dashboards
- **Gradio**: Simple web interfaces for model interaction

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/token-explainability.git
cd token-explainability

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```
transformers>=4.30.0
torch>=2.0.0
captum>=0.6.0
matplotlib>=3.7.0
plotly>=5.14.0
streamlit>=1.22.0  # optional
gradio>=3.35.0     # optional
numpy>=1.24.0
pandas>=2.0.0
```

## Usage

### Basic Example: Analyzing Token Influence

```python
from token_explainability import TokenExplainer
from transformers import AutoTokenizer, AutoModel

# Load model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Initialize explainer
explainer = TokenExplainer(model, tokenizer)

# Analyze input text
text = "The cat sat on the mat."
results = explainer.analyze(
    text,
    methods=["attention", "integrated_gradients", "shap"],
    visualize=True
)

# Display results
explainer.show_heatmap(results)
explainer.show_attention_flow(results)
```

### Interactive UI

```bash
# Launch Streamlit app
streamlit run app.py

# Or launch Gradio interface
python gradio_app.py
```

## Project Structure

```
token-explainability/
├── src/
│   ├── models/              # Model wrappers and utilities
│   ├── attribution/         # Attribution method implementations
│   ├── attention/           # Attention analysis tools
│   ├── visualization/       # Plotting and UI components
│   └── utils/               # Helper functions
├── notebooks/               # Jupyter notebooks for experiments
├── tests/                   # Unit tests
├── examples/                # Usage examples
├── app.py                   # Streamlit application
├── gradio_app.py           # Gradio interface
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Expected Outcomes

1. **Interactive Visualizations**: Explore token influence through intuitive interfaces
2. **Model Interpretability**: Understand how Transformers process and weight input tokens
3. **Attribution Comparison**: See where internal and external methods agree or diverge
4. **Practical Insights**: 
   - Guide prompt engineering and input optimization
   - Identify model biases and failure modes
   - Improve trustworthiness of NLP systems

## Research Applications

- **Model Debugging**: Identify why models make specific predictions
- **Bias Detection**: Uncover tokens that disproportionately influence outputs
- **Prompt Engineering**: Design better inputs based on token importance
- **Architecture Analysis**: Compare different Transformer variants
- **Explainable AI**: Build more transparent and trustworthy NLP systems

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## References

- Abnar, S., & Zuidema, W. (2020). Quantifying attention flow in transformers. ACL.
- Sundararajan, M., et al. (2017). Axiomatic attribution for deep networks. ICML.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NeurIPS.
- Bach, S., et al. (2015). On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation. PLoS ONE.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{token_explainability_2025,
  title={Token-Level Explainability Framework for Transformer Architectures},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/token-explainability}
}
```



- HuggingFace team for the Transformers library
- PyTorch team for the deep learning framework
- Captum team for attribution method implementations
