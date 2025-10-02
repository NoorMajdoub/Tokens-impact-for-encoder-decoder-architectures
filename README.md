# Tokens-impact-for-encoder-decoder-architectures

## Project Goals

This project explores and interprets the influence of input tokens on Transformer-based models through aApplying attribution methods to quantify how input tokens contribute to outputs from a model point of view

### Key Questions I wanted to understand

- Which tokens matter most for a given prediction?
- How do attention layers and heads process and redistribute information?
- To what extent can we interpret the model's reasoning from the inside versus the outside?
- WHY SOME WORDS THAT SHOULD NOT CONTRIBUTE MUCH HAVE SO MUCH IMPORTANCE?

## Tasks & Models

### Supported Tasks
- **Masked Language Modeling (MLM)**: Predicting masked tokens in context(for the decoder model , since it is best fit for generation)
- **Text Classification**: Categorizing input text into predefined classes(for the encoder model )

### Supported Architectures
- **Encoder-only models**: BERT 
- **Decoder-only models**: GPT-2 (I wanna test of another one cause didnt really do much progress here)

## Methodology



### Attribution Methods

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

## About the report
- I am trying to rewrite my tech report and comments in the notebooks in a way that make more sense cause my og ones were english/arabic /gibberish 
