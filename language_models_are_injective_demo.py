# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "marimo>=0.23.2",
#   "numpy==2.4.4",
#   "pillow==12.2.0",
#   "torch==2.11.0",
#   "torchvision",
#   "accelerate",
#   "pyarrow",
#   "transformers==5.6.2",
#   "pandas==3.0.2",
#   "altair==6.1.0",
#   "bitsandbytes>=0.46.1",
# ]
# ///

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""
    # Are Language Models Injective? Can We Invert Them?

    This interactive notebook explores the core idea behind the paper **“Language Models are Injective and Hence Invertible.”** It demonstrates how token-level hidden representations can be mapped back toward vocabulary embeddings using cosine similarity.

    Rather than treating hidden states as opaque intermediate activations, this notebook treats them as structured representations that may preserve enough information to support inversion or near-inversion at the token level.

    > Paper: [Language Models are Injective and Hence Invertible](https://arxiv.org/abs/2510.15511)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Motivation

    A common intuition is that language models gradually “compress away” the input as representations pass through the network. The central claim of the paper challenges that view: under the stated conditions, decoder-only transformers are almost surely injective with respect to the relevant representations.

    This notebook does **not** implement the full SIPIT algorithm. Instead, it provides an interpretable approximation: for a selected hidden layer, each token representation is matched against the model’s input embedding table to find the nearest tokens by cosine similarity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SIPIT in Brief

    The paper introduces **SIPIT** (Sequential Inversion via Prefix-Indexed Transformer dynamics) as a constructive inversion procedure.

    At a high level:
    1. The transformer is causal, so the hidden state at position \(t\) depends only on the prefix up to \(t\).
    2. The inversion procedure searches for the token whose induced hidden state best matches the target representation at that position.
    3. This process is repeated sequentially to recover the input.

    This notebook uses a much simpler baseline: nearest-neighbor retrieval from the embedding table using cosine similarity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    prompt = mo.ui.text(
        value="The weather today is pleasant and bright.",
        label="Input text",
        placeholder="Enter a prompt to analyze",
        full_width=True,
    )

    selected_layer = mo.ui.slider(
        start=0,
        stop=35,
        value=10,
        step=1,
        label="Hidden layer",
    )

    form = (
        mo.md(
            """
    ## Experiment Setup

    {text_input}

    {selected_layer}
    """
        )
        .batch(
            text_input=prompt,
            selected_layer=selected_layer,
        )
        .form(show_clear_button=True, bordered=False)
    )
    return (form,)


@app.cell(hide_code=True)
def _(mo, model_id):
    mo.md(rf"""
    ## Model

    This demo uses [`{model_id}`](https://huggingface.co/{model_id}) and compares hidden-state vectors against the model’s input embedding table. The selected layer controls which internal representation is used for reconstruction.
    """)
    return


@app.cell
def _(
    layer_idx,
    mean_cosine_similarity,
    mo,
    original_tokens,
    reconstructed_text,
    token_recovery,
):
    results_md = mo.vstack(
        [
            mo.md("## Results"),
            mo.md(f"**Selected layer:** `{layer_idx}`"),
            mo.md("**Original sequence**"),
            mo.md(f"`{original_tokens}`"),
            mo.md("**Reconstructed sequence**"),
            mo.md(f"`{reconstructed_text}`"),
            mo.md(
                f"**Mean cosine similarity to nearest embedding:** `{mean_cosine_similarity:.4f}`"
            ),
            mo.md(f"**Exact token recovery rate:** `{token_recovery * 100:.2f}%`"),
        ]
    )
    return (results_md,)


@app.cell
def _(form):
    form
    return


@app.cell
def _(results_md):
    results_md
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpretation

    This reconstruction is a nearest-neighbor approximation in embedding space, not a proof of exact inversion by itself. Strong recovery at early or intermediate layers suggests that token identity remains highly recoverable from hidden states, while degradation across layers can reveal where representations become more contextual or abstract.

    The exact inversion result in the paper is stronger: it concerns injectivity of the model mapping and motivates dedicated inversion procedures such as SIPIT rather than simple nearest-neighbor lookup.
    """)
    return


@app.cell(hide_code=True)
def _(chart, mo):
    mo.ui.altair_chart(
        chart.properties(width=600, height=400),
        label="Top-10 nearest tokens for each hidden-state position",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why This Matters

    If hidden states retain enough information to reconstruct the original input, then they should not be treated as harmless intermediate artifacts. This has implications for interpretability, model auditing, and privacy.

    In practical terms, systems that store or transmit hidden states may be storing information that is much closer to the original prompt than many practitioners assume.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Notes and Limitations

    - This notebook uses embedding-space nearest neighbors, not the full inversion algorithm from the paper.
    - Reconstruction quality depends heavily on the selected layer.
    - Cosine similarity is a useful proxy for token alignment, but it is not equivalent to exact invertibility.
    - Chat templates, tokenizer behavior, and special tokens can affect the recovered sequence.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reference

    **Paper:** *Language Models are Injective and Hence Invertible*
    **Conference:** ICLR 2026
    **arXiv:** [2510.15511](https://arxiv.org/abs/2510.15511)
    """)
    return


@app.cell(column=1)
def _():
    import os
    import sys
    from transformers import AutoModelForCausalLM, AutoProcessor

    return AutoModelForCausalLM, AutoProcessor, os, sys


@app.cell
def _(os, sys):
    import marimo as mo
    import pandas as pd
    import altair as alt
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    return F, alt, mo, pd, torch


@app.cell
def _():
    model_id = "google/gemma-4-E2B-it"
    return (model_id,)


@app.cell
def _(AutoModelForCausalLM, AutoProcessor, model_id):
    processor = AutoProcessor.from_pretrained(
        model_id,
        dtype="auto",
        device_map="cpu",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype="auto",
        device_map="cpu",
    )
    return model, processor


@app.cell
def _(F, form, model, processor, torch):
    if form.value is None:
        text_content = "The weather today is pleasant and bright."
        layer_idx = 10
    else:
        text_content, layer_idx = form.value.values()

    model.eval()

    inputs = processor(text=text_content, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    layer_hidden = hidden_states[layer_idx]  # (batch, seq_len, hidden_dim)

    embeddings = model.get_input_embeddings().weight
    embeddings_norm = F.normalize(embeddings, dim=-1)

    token_hidden = layer_hidden[0]  # (seq_len, hidden_dim)
    token_hidden_norm = F.normalize(token_hidden, dim=-1)

    all_similarities = token_hidden_norm @ embeddings_norm.T  # (seq_len, vocab_size)
    closest_ids = all_similarities.argmax(dim=-1)
    reconstructed_text = processor.decode(
        closest_ids, skip_special_tokens=False
    ).strip()

    top10 = all_similarities.topk(10, dim=-1)

    rec_embeds = embeddings[closest_ids].to(token_hidden.device)
    mean_cosine_similarity = (
        F.cosine_similarity(rec_embeds, token_hidden, dim=-1).mean().item()
    )

    original_ids = inputs["input_ids"][0]
    compare_len = min(len(original_ids), len(closest_ids))
    token_recovery = (
        (closest_ids[:compare_len] == original_ids[:compare_len]).float().mean().item()
        if compare_len > 0
        else 0.0
    )

    original_tokens = processor.decode(original_ids, skip_special_tokens=False)
    return (
        layer_idx,
        mean_cosine_similarity,
        original_tokens,
        reconstructed_text,
        text_content,
        token_recovery,
        top10,
    )


@app.cell
def _(alt, pd, processor, text_content, top10):
    original_words = text_content.split()
    seq_len = top10.values.shape[0]

    _data = []
    for _i in range(seq_len):
        for j in range(10):
            _data.append(
                {
                    "Position": _i,
                    "Original Text": (
                        text_content.split(" ")[_i]
                        if _i < len(text_content.split(" "))
                        else ""
                    ),
                    "Token Text": processor.decode(top10.indices[_i, j].item()).strip(),
                    "Score": top10.values[_i, j].item(),
                }
            )
    df = pd.DataFrame(_data).sort_values("Position", ascending=True)
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x="Score",
            y="Original Text",
            color="Score",
            tooltip=["Position", "Token Text", "Score"],
        )
    )
    return (chart,)


if __name__ == "__main__":
    app.run()
