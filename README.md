# Language Models Are Injective - Interactive Demonstration

This repository contains interactive marimo notebooks demonstrating the groundbreaking paper:

**"LANGUAGE MODELS ARE INJECTIVE AND HENCE INVERTIBLE"** (ICLR 2026)
[Paper Link](https://arxiv.org/abs/2510.15511)

## Key Findings

### The Core Discovery
- **Injectivity**: Decoder-only Transformers are almost-surely injective
- Different inputs → Different hidden representations (almost always)
- Hidden states preserve ALL input information
- The paper proves this with theoretical guarantees and experimental validation

### The SIPIT Algorithm
- **First algorithm** with provable linear-time guarantees for exact input reconstruction
- Sequential Inverse Prompt via ITerative updates
- Recovers exact input in T|V| steps worst case
- In practice, explores <0.22% of vocabulary on average!

### Billion-Scale Evidence
- **5+ billion** pairwise comparisons between distinct prompts
- **343 billion** exhaustive local collision searches  
- **0 collisions** detected across all experiments
- Minimum distances: 100-10,000× larger than collision threshold

## Files

### Core Implementation
- **`language_models_are_injective_demo_models.py`**: Simplified Transformer model and injectivity utilities
- **`language_models_are_injective_demo.py`**: Main interactive demo notebook
- **`language_models_are_injective_demo_gemma.py`**: Gemma-4 enhanced version

### How to Run

```bash
# Install dependencies
pip install marimo torch numpy transformers accelerate

# Run the demo
marimo run language_models_are_injective_demo.py

# Or with Gemma-4 (if available)
marimo run language_models_are_injective_demo_gemma.py
```

## Interactive Features

### 1. Injectivity Explorer
- Test whether different inputs produce different hidden states
- Compare distances between representations
- Visualize the injectivity phenomenon

### 2. SIPIT Algorithm Visualization
- Understand how exact input recovery works
- See the sequential reconstruction process
- Learn about the theoretical guarantees

### 3. Evidence Explorer
- Explore billion-scale experiments
- See collision search results
- Understand the statistical significance

## Critical Implications

### Privacy & Ethics
> "Any system storing, caching, or transmitting hidden states is effectively handling verbatim user text."

- **Storing hidden states = Storing user prompts!**
- **GDPR compliance** requires treating internal representations as personal data
- **Privacy policies** need updates to reflect this reality

### Theoretical Impact
- LLMs are not "lossy" as previously assumed
- Hidden states are lossless encodings of inputs
- Foundation for provable interpretability

## Requirements

- Python 3.8+
- PyTorch 2.0+
- marimo
- transformers (for Gemma-4 version)

## License

This repository is for educational and research purposes.

## Citation

```bibtex
@article{nikolaou2025language,
  title={Language Models Are Injective and Hence Invertible},
  author={Nikolaou, Giorgos and Mencattini, Tommaso and Crisostomi, Donato and Santilli, Andrea and Panagakis, Yannis and Rodol{\`a}, Emanuele},
  journal={arXiv preprint arXiv:2510.15511},
  year={2025}
}
```
