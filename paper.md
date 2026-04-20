The following report provides a detailed analysis of the paper "LANGUAGE MODELS ARE INJECTIVE AND HENCE INVERTIBLE," published as a conference paper at ICLR 2026.

### 1. Authors and Institution(s)

The research was conducted by a collaborative team from several institutions:
*   **Giorgos Nikolaou** (EPFL; Archimedes/Athena RC, Greece)
*   **Tommaso Mencattini** (EPFL; Sapienza University of Rome)
*   **Donato Crisostomi** (Sapienza University of Rome)
*   **Andrea Santilli** (Sapienza University of Rome)
*   **Yannis Panagakis** (University of Athens; Archimedes/Athena RC, Greece)
*   **Emanuele Rodol`a** (Sapienza University of Rome; Paradigma)

### 2. How This Work Fits into the Broader Research Landscape

The work addresses a fundamental question regarding the nature of information preservation within large language models (LLMs). The prevailing assumption in the research community has been that Transformer architectures, characterized by non-linear activations, normalization layers, and many-to-one attention mechanisms, inherently discard information during the mapping from input text to internal representations. This assumption has fueled concerns about the transparency, robustness, and safe deployment of LLMs, as it implies that exact recovery of an input from its latent representation is fundamentally impossible.

Existing theoretical analyses on Transformer expressivity often demonstrate their capacity for universal approximation or Turing completeness, implying they can represent highly complex, potentially many-to-one functions. Some prior work has explored properties such as surjectivity or injectivity with respect to the entire hidden-state matrix at initialization. However, these works have not definitively addressed the injectivity of the map from discrete input sequences to the last-token representation—a quantity crucial for next-token prediction and operational use—nor have they typically guaranteed such properties are preserved during training.

Concurrently, research into inverse problems in language modeling has focused on reconstructing prompts from various outputs or internal signals. These approaches typically involve inferring prompts from generated continuations, training auxiliary inverters to map probability vectors or encoder logits to text, or employing gradient-based methods for approximate prompt discovery. While these contributions highlight potential privacy risks and demonstrate approximate reconstruction capabilities, they often rely on trained inverters, are computationally intensive, lack formal guarantees of exactness, and do not specifically target the direct inversion of hidden states in decoder-only LLMs.

This paper challenges the conventional "lossy" intuition by providing a rigorous mathematical proof that decoder-only Transformers are almost-surely injective from discrete input sequences to their continuous last-token representations. This property is shown to be established at initialization and preserved throughout training. Furthermore, the paper introduces SIPIT, an algorithm that leverages this injectivity to achieve provable, exact, and efficient reconstruction of input text from hidden states. By doing so, the work moves beyond mere empirical observation or approximate reconstruction, offering a foundational theoretical understanding and a practical, guaranteed method for information recovery, thereby distinguishing itself from prior approximate and trained-inverter approaches in the field.

### 3. Key Objectives and Motivation

The primary objectives and motivations for this research are multi-faceted:

1.  **Challenge the "Lossy" Paradigm**: The central motivation is to rigorously investigate and challenge the widely held belief that Transformer architectures, due to their intricate non-linearities, normalization, and attention mechanisms, inherently discard information, making exact input recovery from internal representations impossible. The authors aim to demonstrate that this intuition, despite its prevalence, may be misleading.

2.  **Establish Mathematical Proof of Injectivity**: A key objective is to provide a formal mathematical proof that decoder-only Transformer language models, when viewed as functions mapping discrete input sequences to their continuous last-token representations, are almost-surely injective. This objective includes demonstrating that this property is not merely an asymptotic idealization but a structural consequence of the architecture, established at initialization and, critically, preserved during the process of gradient-based training.

3.  **Provide Empirical Validation**: Complementing the theoretical proofs, the research seeks to empirically confirm the injectivity property through extensive, large-scale collision tests across multiple state-of-the-art language models. This involves performing billions of pairwise comparisons to observe whether distinct inputs ever map to identical hidden representations in practice.

4.  **Operationalize Injectivity for Practical Use**: Beyond theoretical demonstration, a significant objective is to translate the proven injectivity into a practical, efficient algorithm for exact input reconstruction. The goal is to develop an algorithm, named SIPIT, that can provably and efficiently reconstruct the exact input text from observed hidden activations, providing linear-time guarantees and demonstrating full invertibility in real-world scenarios.

5.  **Re-evaluate Transparency, Interpretability, and Safety**: The overarching motivation is to establish injectivity as a fundamental and exploitable property of language models. This has direct implications for LLM transparency (by confirming that internal states faithfully encode inputs), interpretability (by providing a solid foundation for causal and probing analyses, as information is not inherently lost), and safe deployment (by highlighting the direct relationship between hidden states and sensitive user data).

In essence, the paper aims to provide a definitive answer to whether LLM internal representations faithfully preserve input information, moving this understanding from intuition and approximation to rigorous proof and exact, practical application.

### 4. Methodology and Approach

The methodology employed in this research integrates theoretical proofs based on real analysis with extensive empirical validation and the development of a novel algorithm.

**4.1. Theoretical Approach for Injectivity**

The theoretical framework considers decoder-only Transformer language models as functions that map discrete input sequences to continuous last-token representations. The argument for almost-sure injectivity is constructed in three main steps:

1.  **Real-Analyticity of Transformer Components**: The authors prove that all standard components of a decoder-only Transformer, including embeddings, positional encodings, LayerNorm (provided the epsilon parameter is strictly positive), causal attention mechanisms, Multi-Layer Perceptrons (MLPs) with real-analytic activation functions (e.g., tanh, GELU, SiLU), and residual connections, are real-analytic functions in their parameters. Real-analytic functions are infinitely differentiable and locally expressible as a convergent power series. This property ensures a high degree of smoothness and predictability in their behavior. The proof relies on showing that fundamental operations (addition, multiplication, composition, quotient with non-zero denominator) preserve real-analyticity.

2.  **Almost-Sure Injectivity at Initialization**: Building on real-analyticity, the authors invoke the "zero-set theorem," which states that if a non-trivial real-analytic function maps to zero, its zero set has Lebesgue measure zero. They define a discrepancy function h(θ) = ||r(s;θ) - r(s';θ)||^2 for two distinct input sequences s and s'. Since h(θ) is real-analytic, to prove its zero set has measure zero, it suffices to show that h(θ) is not identically zero (i.e., there exists at least one parameter setting θ* where r(s;θ*) != r(s';θ*)).
    *   **Witness Construction**: The authors explicitly construct such a θ* for two exhaustive cases:
        *   If 's' and 's'' differ in length or their last token, they devise a parameter setting (e.g., zeroing out most network components and setting specific unique embeddings) that forces the last-token representations to be distinct.
        *   If 's' and 's'' have the same length and last token but differ at an earlier position, they construct a network configuration where all blocks after the first are identity mappings, and the first block's attention mechanism is engineered to selectively highlight the first differing token, thereby forcing distinct final representations. This construction requires minimum embedding dimensions (d ≥ 4 and dη ≥ 1) and at least one attention head per block, conditions typically met by modern LLMs.

3.  **Preservation of Injectivity During Training**: The core argument is that gradient descent (GD) updates (including SGD and mini-batch GD with standard learning rates) preserve the absolute continuity of the parameter distribution.
    *   **Jacobian Analysis**: The GD update map ϕ(θ) = θ - η∇L(θ) (where L is the cross-entropy loss) is real-analytic. Its Jacobian determinant, det(Dϕ(θ)), is also real-analytic. The authors demonstrate that det(Dϕ(θ)) is not identically zero by explicitly calculating the Hessian of the loss function at the zero-parameter initialization (θ* = 0). They show that this Hessian has non-zero eigenvalues, ensuring det(Dϕ(θ*)) != 0.
    *   **Change of Variables**: Since det(Dϕ(θ)) is not identically zero, its zero set (where GD might "collapse" regions) has Lebesgue measure zero. On the complement of this set, the Inverse Function Theorem applies, meaning ϕ is a local C1-diffeomorphism. Using a countable cover of these local diffeomorphism charts and the change-of-variables formula for Lebesgue measure, they prove that if a parameter distribution is absolutely continuous with respect to Lebesgue measure, its pushforward under ϕ remains absolutely continuous. This iterative argument extends to any finite number of GD steps, ensuring injectivity is preserved.

**4.2. Algorithmic Approach (SIPIT)**

The SIPIT (Sequential Inverse Prompt via ITerative updates) algorithm operationalizes the proven injectivity for exact input reconstruction:

1.  **Leveraging Causality**: SIPIT exploits the causal nature of decoder-only Transformers, where the hidden state at position *t* depends only on the prefix *⟨s1, ..., s(t-1)⟩* and the current token *s_t*. This localizes the inversion problem.

2.  **One-Step Inversion**: For a given position *t* and a reconstructed prefix *π* = *⟨s1, ..., s(t-1)⟩*, the algorithm defines a "one-step map" *F(v; π, t)* that predicts the hidden state at position *t* if the next token were *v*. Based on the theoretical injectivity, this map *F* is almost-surely injective with a positive separation margin (∆π,t).

3.  **Sequential Reconstruction Loop**:
    *   The algorithm iterates token-by-token from *t=1* to *T* (the sequence length).
    *   At each step *t*, it has the currently reconstructed prefix *π* and the observed target hidden state *bht* (at layer ℓ).
    *   It then cycles through candidate tokens from the vocabulary *V*. For each candidate *v*, it computes *F(v; π, t)* and uses a "local verifier" to check if *bht* lies within an ε-ball around *F(v; π, t)*.
    *   Due to the almost-sure injectivity and positive margin, with a sufficiently small ε (specifically ε < ∆π,t/2), only the true token *s_t* will satisfy this condition.
    *   Once the unique matching token is found, it is appended to the reconstructed prefix, and the process moves to the next position *t+1*.

4.  **Candidate Search Policy**: To improve practical efficiency, SIPIT employs a gradient-guided policy rather than a pure brute-force search. This policy uses gradient signals (e.g., from an L2 loss between a continuous proxy embedding and the target *bht*) to iteratively refine a proxy embedding, then ranks vocabulary tokens by their proximity to this proxy, prioritizing the most likely candidates. While this heuristic speeds up the search, the worst-case guarantee remains *T|V|* iterations (one full pass through the vocabulary per position).

**4.3. Empirical Validation**

Extensive experiments were conducted to confirm the theoretical findings and validate SIPIT:

1.  **Collision Search**:
    *   **Large-scale dataset**: 100,000 prompts were uniformly sampled from a mixture of four datasets (Wikipedia, C4, The Pile, Python GitHub code).
    *   **Models**: Six state-of-the-art decoder-only LLMs (GPT-2 family, Gemma-3 family, Llama-3.1-8B, Mistral-7B-v0.1, Phi-4-mini-instruct, TinyStories-33M) were tested across all layers.
    *   **Measurements**: Billions of pairwise ℓ2 distance comparisons between last-token hidden states were performed. A collision was defined by PyTorch's `torch.allclose` function (tolerances 10^-5 and 10^-8).
    *   **Stress-testing**: An "exhaustive collision test" focused on 10 prompts with the smallest initial ℓ2 distances. For each, every vocabulary token was appended, and all pairwise distances of the resulting continuations were checked (over 343 billion pairs per model).
    *   **Quantization**: Experiments with FP4 and INT8 quantized models (Llama-3.1-8B, Mistral-7B-v0.1, Phi-4-mini-instruct, Phi-4 14B, Llama-3.1-70B) assessed the impact of reduced precision.
    *   **Controlled Next-Token Prediction**: Prompts were constructed to yield identical next-token predictions (e.g., varying delimiters in translation/math tasks) to examine whether semantic equivalence induced representational collisions.
    *   **Qualitative Analysis**: Closest pairs of representations were manually inspected to understand their semantic and syntactic similarities.

2.  **SIPIT Performance Evaluation**:
    *   **Accuracy and Efficiency**: SIPIT was tested on GPT-2 Small with 100 prompts (20 tokens each), comparing its exact token-level recovery accuracy and runtime against `HARDPROMPTS` (an approximate method) and a brute-force version of SIPIT.
    *   **Vocabulary Scaling**: Tested on Mistral-7B-v0.1 (32K vocabulary) and Llama-3.1-8B (128K vocabulary) using FP4-quantized versions, measuring inversion time and the percentage of vocabulary explored to confirm linear scaling.
    *   **Robustness to Data Distribution**: Evaluated SIPIT on in-distribution (WebText), out-of-distribution (Wikipedia), and out-of-distribution random token sequences.
    *   **Effect of Layer Depth**: Inversion times were measured across different layers for prompts of varying lengths (20-200 tokens).

3.  **Relation to Anisotropy and Intrinsic Dimension (Exploratory)**: An exploratory study on GPT-2 Small correlated the average anisotropy and intrinsic dimension of last-token representations with injectivity margins, aiming to understand the geometric properties of the representation space in relation to injectivity.

### 5. Main Findings and Results

The research yielded a comprehensive set of theoretical and empirical findings supporting the injectivity of language models and the efficacy of the SIPIT algorithm.

**5.1. Theoretical Findings (Injectivity Proofs)**

*   **Almost-Sure Injectivity**: Decoder-only Transformer language models, conceptualized as maps from discrete input sequences to continuous last-token representations, are proven to be almost-surely injective. This means that, with probability one, any two distinct input prompts will map to distinct last-token hidden states.
*   **Initialization Property**: This injectivity property is established at the point of random parameter initialization, provided the parameters are drawn from a distribution with a density (e.g., Gaussian, uniform, Xavier/Glorot). The set of parameter values that would lead to collisions constitutes a set of Lebesgue measure zero.
*   **Preservation During Training**: The research demonstrates that this injectivity is preserved throughout the training process. Gradient-based optimization procedures (SGD, mini-batch, or full-batch GD with step sizes in (0,1)) do not cause parameters to converge to or cross the measure-zero collision set. Thus, the model remains almost-surely injective after any finite number of training steps.
*   **Foundation in Real-Analyticity**: The proofs rely on showing that all standard components of Transformer architectures (embeddings, LayerNorm with ε>0, causal attention, MLPs with common activation functions like GELU and SiLU, and residual connections) are real-analytic functions of their parameters. This smoothness is crucial for applying the zero-set theorem to conclude that collision sets have measure zero.
*   **Global Distinctness**: The injectivity extends to finite sets of prompts, guaranteeing that the representations of any finite collection of distinct prompts will almost surely be distinct.

**5.2. Empirical Findings (Collision Tests)**

*   **Absence of Observed Collisions**: Across billions of pairwise comparisons (approximately 5 billion for general prompts and 343 billion for exhaustive local searches) on various state-of-the-art LLMs (GPT-2, Gemma-3, Llama-3.1-8B, Mistral-7B-v0.1, Phi-4-mini-instruct, TinyStories-33M), no collisions were detected. The numerical criterion for collision (PyTorch's `torch.allclose` with tolerances 10^-5/10^-8) was never met by distinct prompts.
*   **Significant Separation Margins**: The minimum pairwise ℓ2 distances between last-token hidden states were consistently orders of magnitude above the defined collision threshold (10^-6), typically ranging from 10^-3 to 10^1 and sometimes higher, indicating clear separation in the representation space.
*   **Increasing Separation with Depth and Scale**: For models like GPT-2 and Gemma-3, the minimum pairwise distances generally increased with model depth and size, suggesting that deeper layers and larger models further refine and separate distinct input representations.
*   **Robustness to Sequence Length**: While pairwise distances exhibit rapid growth for short sequence lengths, they stabilize for longer sequences. The minimum distance consistently remained far from zero across all tested sequence lengths, implying that adding tokens does not erode representational separability.
*   **Quantization Does Not Introduce Collisions**: Experiments with FP4 and INT8 quantized versions of several models (Llama-3.1-8B, Mistral-7B-v0.1, Phi-4-mini-instruct, Phi-4 14B, Llama-3.1-70B) confirmed that quantization did not introduce any collisions. In many cases, quantization actually increased the minimum distance between representations, preserving the integrity of the representation space.
*   **Semantic Equivalence Does Not Imply Representational Collapse**: Prompts crafted to produce the exact same next-token prediction (e.g., different delimiters in translation/arithmetic tasks, or identical suffixes following random prefixes) still generated clearly distinct last-token embeddings, further confirming the robustness of injectivity.
*   **Qualitative Similarity for Closest Pairs**: The prompts exhibiting the smallest ℓ2 distances between their last-token representations were consistently found to be near-duplicates (e.g., Python code snippets differing only by trailing newline characters or minor formatting), rather than semantically disparate inputs.

**5.3. Algorithmic Findings (SIPIT Performance)**

*   **Exact and Efficient Recovery**: SIPIT successfully recovered the exact input sequence with 100% token-level accuracy across all tested scenarios. Its gradient-guided policy demonstrated high efficiency, requiring significantly less time than brute-force search and outperforming an approximate prompt discovery method (`HARDPROMPTS`) which completely failed to recover inputs.
*   **Provable Linear-Time Guarantees**: The algorithm is proven to recover the true input sequence in at most *T|V|* steps (where *T* is sequence length, *|V|* is vocabulary size) in the worst case, reflecting its linear complexity with respect to these parameters.
*   **High Search Efficiency in Practice**: Despite the worst-case bound, the gradient-guided policy was highly efficient, exploring less than 0.22% of the total vocabulary on average to find the correct token at each position.
*   **Robustness to Quantization and Noise**: SIPIT maintained perfect accuracy on FP4-quantized models and is theoretically robust to perturbations in hidden states, as long as the noise magnitude is less than half the separation margin (∆π,t/2).
*   **Effect of Layer Depth**: Inversion times showed only a mild increase with deeper layers, indicating that SIPIT is efficient across various depths.
*   **Robustness to Data Distribution**: SIPIT achieved 100% accuracy on in-distribution, out-of-distribution (natural language), and out-of-distribution (random token sequences) data. Random token sequences were, unexpectedly, faster to invert, possibly due to their less clustered hidden representations providing clearer gradient signals.

**5.4. Exploratory Findings (Anisotropy and Intrinsic Dimension)**

*   **Correlation with Injectivity Margin**: An exploratory analysis suggested that layers with higher anisotropy (more structured representations) exhibited larger injectivity margins (Pearson correlation 0.72). Conversely, layers with lower intrinsic dimensionality (more compressed representations) also showed larger margins (Pearson correlation -0.60). These observations are consistent with the view that information is preserved while being transformed into increasingly structured and well-separated representations, rather than being lost through compression.

### 6. Significance and Potential Impact

This research establishes injectivity as a fundamental and exploitable property of decoder-only Transformer language models, with significant implications across theoretical understanding, practical application, and ethical considerations.

**6.1. Scientific Impact and Fundamental Understanding of LLMs:**

*   **Reconciliation of Competing Views**: The work fundamentally shifts the understanding of information flow in Transformers. It rigorously demonstrates that despite their complex non-linear components, these models are almost-surely lossless in mapping discrete inputs to continuous last-token representations. This challenges the prevalent intuition of Transformers as inherently "lossy" systems, advocating for a view where information is preserved end-to-end when viewed as maps on sequence space.
*   **Foundation for Mechanistic Interpretability**: By proving that last-token states faithfully encode the full input information, the research provides a sound theoretical foundation for mechanistic interpretability and probing analyses. Researchers can be confident that if their interpretability tools fail, it is not because the necessary information is absent from the internal representations, but rather due to limitations of the tools themselves. This enables more robust causal and attribution studies.
*   **Structural Property, Not Idealization**: The findings establish injectivity as a structural consequence of the architecture and standard training procedures, rather than an asymptotic or idealized property. This understanding is relevant for models currently in deployment and under active development.

**6.2. Practical Impact and Algorithmic Advancements:**

*   **Exact Input Recovery (SIPIT)**: The introduction of SIPIT, the first algorithm providing provable linear-time guarantees for exact input reconstruction from hidden states, transforms injectivity from a theoretical curiosity into an operational tool. Unlike prior approximate, black-box, or trained-inverter methods, SIPIT is training-free, efficient, and offers exactness guarantees.
*   **Auditing and Debugging Tool**: SIPIT can serve as a powerful tool for auditing and debugging LLMs. It allows researchers and developers to precisely reconstruct the original prompts that generated specific internal activations, which can be invaluable for understanding model behavior, identifying biases, or diagnosing errors.

**6.3. Ethical, Legal, and Societal Impact (Privacy and Data Protection):**

*   **Re-evaluation of Data Privacy in LLMs**: The most direct and immediate practical implication lies in data privacy. The paper unequivocally demonstrates that hidden states are not abstract mathematical representations devoid of personal information, but rather lossless encodings of the user's exact input. This finding directly challenges regulatory perspectives, such as that of the Hamburg Data Protection Commissioner, which suggested that LLM parameters or "extractable data records" might not constitute personal data.
*   **Accountability for Internal Representations**: The work highlights that any system storing, caching, or transmitting hidden states is effectively handling verbatim user text. Consequently, the legal and ethical obligations (e.g., under GDPR for personal data, data deletion requests, or consent requirements) that apply to raw prompts should extend to these internal representations. This necessitates a re-evaluation of data protection policies and practices in LLM deployment, prompting regulatory bodies and industry stakeholders to adapt to this new understanding.
*   **Transparency and User Trust**: By establishing the recoverability of inputs, the research contributes to greater transparency about what information LLMs retain internally. This transparency can foster greater trust among users, while simultaneously placing a higher burden of responsibility on developers and deployers to manage internal data securely.

**6.4. Future Research Directions:**

The work opens several avenues for future research, including:
*   Extending the injectivity analysis to multimodal architectures (e.g., vision, music Transformers).
*   Investigating the robustness of invertibility under more severe noise or aggressive quantization schemes beyond those tested.
*   Developing formal bridges between these technical insights and the evolving regulatory frameworks for AI to ensure safer and more responsible LLM deployment.