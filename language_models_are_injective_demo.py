import marimo

__generated_with = "0.23.1"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(f"""
    # 🤖 Are Language Models Injective? Can We Invert Them?

    This interactive notebook demonstrates the groundbreaking discovery from the paper:

    **"LANGUAGE MODELS ARE INJECTIVE AND HENCE INVERTIBLE"** (ICLR 2026)

    ---

    ## **Key Insight**: Decoder-only Transformers are almost-surely injective, meaning they preserve input information in their hidden representations!

    > **Paper**: [LANGUAGE MODELS ARE INJECTIVE AND HENCE INVERTIBLE](https://arxiv.org/abs/2510.15511)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 🔍 The Core Question

    Before this paper, the prevailing belief was that LLMs are "lossy" - that information gets lost as text passes through the network.

    **This paper proves**: For any two distinct input sequences, they almost surely map to different last-token hidden states!

    This is called **injectivity** - a fundamental mathematical property with profound implications.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    prompt1 = mo.ui.text(
        value="The weather today is",
        label="Text Input",
        placeholder="Enter a text prompt",
        full_width=True,
    )
    selected_layer = mo.ui.slider(
        start=0,
        stop=35,
        value=10,
        step=1,
        label="Extract from hidden layer:",
    )
    # Create a form with multiple elements
    form = (
        mo.md("""
        ## Inputs

        {text_input}

        {selected_layer}
    """)
        .batch(
            text_input=prompt1,
            selected_layer=selected_layer,
        )
        .form(show_clear_button=True, bordered=False)
    )
    return (form,)


@app.cell
def _(form):
    form
    return


@app.cell
def _(mo, reconstructed_text):
    mo.vstack(
        [
            mo.vstack(
                [
                    mo.md("## Results"),
                    mo.md("Reconstructed Text:"),
                    mo.md(f"`{reconstructed_text}`"),
                ],
            ),
            mo.md("""
    **What does this tell us?**

    - ✅ **Distance > 0.01**: Different inputs → Different representations (**injective** behavior)
    - ⚠️ **Distance ≈ 0**: Different inputs → Same representation (**collision** - never observed!)

    > **From the paper**: In billion-scale experiments on real LLMs, minimum distances were typically **0.1 to 10+**, far exceeding any reasonable collision threshold.
        """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 🧪 The SIPIT Algorithm - Exact Input Recovery

    The paper introduces **SIPIT** (Sequential Inverse Prompt via ITerative updates): an algorithm that provably recovers the exact input from hidden states in linear time!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "How SIPIT Works": mo.md("""
        **SIPIT exploits the causal nature of Transformers:**

        1. **Causal Decoding**: The hidden state at position *t* only depends on positions 1, 2, ..., *t*.

        2. **One-Step Inversion**: For position *t*, define *F(v; π)* = predicted hidden state if token *v* followed prefix *π*.

        3. **Sequential Reconstruction**: 
           - Start with empty prefix
           - At each position, find token *v* such that *F(v; π)* matches target hidden state
           - Append *v* to prefix and continue

        4. **Gradient-Guided Search**: Uses gradient signals to prioritize likely candidates.

        ---

        **Theoretical Guarantee**: Recovers exact input in at most *T|V|* steps with probability 1.

        **Experimental Performance**: Explores <0.22% of vocabulary on average!
        """),
            "Why SIPIT Matters": mo.md("""
        - ✅ **First algorithm** with exact reconstruction guarantees
        - ✅ **100% exact recovery** accuracy
        - ✅ **Training-free** and efficient
        - ✅ Enables new auditing and debugging approaches
        - ✅ **Proves** hidden states aren't "lossy" - they preserve all information
        """),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## 🌐 Part 5: Why This Matters - The Big Picture

    ### 🧠 **Theoretical Impact**

    > "This paper established injectivity as a structural consequence of the architecture and standard training procedures."

    - LLMs are not "lossy" as previously assumed
    - Hidden states are **lossless encodings** of inputs
    - Foundation for **provable interpretability**

    ### 🔍 **Practical Impact**

    - **SIPIT** enables exact input recovery from hidden states
    - New **auditing and debugging tools** for LLMs
    - Linear-time guarantees for inversion

    ### 🛡️ **Privacy & Ethics - CRITICAL FINDING**

    > "Any system storing, caching, or transmitting hidden states is effectively handling verbatim user text."

    - **Storing hidden states = Storing user prompts!**
    - **GDPR compliance** requires treating internal representations as personal data
    - **Privacy policies** need to be updated to reflect this reality
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    ## 🎓 Key Takeaway

    **Language models are injective → Information is preserved → Hidden states are reversible.**

    This changes how we think about LLMs: from "black boxes that forget" to "lossless encoders that we can invert."

    The SIPIT algorithm demonstrates that this isn't just theory — it's **practical and works at scale**!
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    ## 📚 References

    **Paper**: [LANGUAGE MODELS ARE INJECTIVE AND HENCE INVERTIBLE](https://arxiv.org/abs/2510.15511)
    **Authors**: Giorgos Nikolaou, Tommaso Mencattini, Donato Crisostomi, Andrea Santilli, Yannis Panagakis, Emanuele Rodolà
    **Conference**: ICLR 2026
    **Paper ID**: 2510.15511
    """)
    return


@app.cell(column=1)
def _():
    import sys
    import os
    from transformers import AutoProcessor, AutoModelForCausalLM

    return AutoModelForCausalLM, AutoProcessor, os, sys


@app.cell
def _(os, sys):
    import marimo as mo
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from typing import List, Tuple

    # Add current directory to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    return F, mo, torch


@app.cell
def _(AutoModelForCausalLM, AutoProcessor, torch):
    processor = AutoProcessor.from_pretrained(
        "google/gemma-4-E2B-it",
        dtype=torch.float16,
        device_map="cpu",
    )
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E2B-it",
        dtype=torch.float16,
        device_map="cpu",
    )
    model.eval()
    return model, processor


@app.cell
def _(model, processor, torch):
    @torch.no_grad()
    def generate_text(text_content: str):
        # Prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text_content},
        ]

        # Process input
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        )
        inputs = processor(text=text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        # Generate output
        outputs = model(**inputs, max_new_tokens=1024, output_hidden_states=True)
        return outputs.hidden_states

    return


@app.cell
def _(F, form, model, processor, torch):
    text_content, layer_idx = form.value.values()

    # Process input
    inputs = processor(text=text_content, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    # Generate output
    with torch.no_grad():
        outputs = model(**inputs, max_new_tokens=1024, output_hidden_states=True)
        hidden_states = outputs.hidden_states

    layer_hidden = hidden_states[layer_idx]

    reconstructed_tokens = []
    similarities_all = []
    top10_tokens_all = []
    top10_scores_all = []

    for i, hidden_vector in enumerate(layer_hidden[0]):
        hidden_norm = F.normalize(hidden_vector.unsqueeze(0), dim=-1)
        embeddings = model.get_input_embeddings().weight
        embeddings_norm = F.normalize(embeddings, dim=-1)

        # cosine similarity
        similarities = (hidden_norm @ embeddings_norm.T).squeeze()
        top10_ids = similarities.topk(10).indices
        top10_tokens = [processor.decode([id.item()]) for id in top10_ids]
        top10_scores = similarities[top10_ids].tolist()

        similarities_all.append(similarities)
        top10_tokens_all.append(top10_tokens)
        top10_scores_all.append(top10_scores)

        closest_token_id = torch.argmax(similarities).item()
        closest_token = processor.decode([closest_token_id]).strip()
        reconstructed_tokens.append(closest_token)

    reconstructed_text = " ".join(reconstructed_tokens)
    return (reconstructed_text,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
