# Language Models Are Injective - Interactive Demonstration

This repository contains interactive marimo notebooks demonstrating the groundbreaking paper:

**"LANGUAGE MODELS ARE INJECTIVE AND HENCE INVERTIBLE"** (ICLR 2026)
[Paper Link](https://alphaxiv.org/abs/2510.15511)

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

- **`language_models_are_injective_demo.py`**: Main interactive demo notebook

### How to Run

```bash
# Install dependencies
uv sync

# Run the demo
uv run marimo edit language_models_are_injective_demo.py
```

### Privacy & Ethics
>
> "Any system storing, caching, or transmitting hidden states is effectively handling verbatim user text."

- **Storing hidden states = Storing user prompts!**
- **GDPR compliance** requires treating internal representations as personal data
- **Privacy policies** need updates to reflect this reality

### Theoretical Impact

- LLMs are not "lossy" as previously assumed
- Hidden states are lossless encodings of inputs
- Foundation for provable interpretability

## License

This repository is for educational purposes.
