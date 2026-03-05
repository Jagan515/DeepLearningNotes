
1. **Input text → tokens**  
   Split text → assign each piece an integer ID from vocabulary (no math yet, just lookup table).

2. **Token ID → embedding vector**  
   xᵢ = EmbeddingMatrix[token_id]  
   → simple table lookup: turns integer into d-dimensional vector (d = 512 / 4096 etc.)

3. **Add positional encoding** (sine/cosine or RoPE)  
   xᵢ ← xᵢ + PE(position_i)  
   → adds fixed/rotary numbers so model knows word order (vector addition).

4. **Input to layer 1: x = embeddings + positions** (shape: sequence_length × d)

5. **Layer Normalization (pre-norm style – most common today)**  
   x_norm = LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + ε) × γ + β  
   → makes numbers stable & centered (small vector ops per position).

6. **Compute Query, Key, Value projections**  
   Q = x_norm × W_Q + b_Q  
   K = x_norm × W_K + b_K  
   V = x_norm × W_V + b_V  
   → three separate matrix multiplications (biggest compute step here).

7. **Split into multiple heads** (h heads)  
   Q, K, V each split into h parts → each head has d_head = d / h dimension.

8. **Compute raw attention scores for each head**  
   scores = Q_head × K_headᵀ  
   → matrix multiply: every position dot-products with every earlier position.

9. **Scale the scores**  
   scores ← scores / √d_head  
   → prevents numbers from growing too big (simple division).

10. **Apply causal mask** (very important for decoder-only)  
    scores[future positions] = -∞ (or very large negative number)  
    → forces model to ignore future tokens (element-wise).

11. **Softmax to get attention weights**  
    attn_weights = softmax(scores, dim=last) = exp(scores) / sum(exp(scores))  
    → turns raw scores into probabilities that sum to 1 per row.

12. **Weighted sum: apply attention**  
    head_output = attn_weights × V_head  
    → matrix multiply: mixes values using the weights.

13. **Concatenate all heads**  
    multihead_out = concat(head1, head2, …, head_h)

14. **Final output projection after multi-head**  
    attn_out = multihead_out × W_O + b_O  
    → one more matrix multiply to combine heads.

15. **First residual connection + norm**  
    x ← x + attn_out  
    x ← LayerNorm(x)  
    → adds original input back + normalize again (vector add + norm).

16. **Feed-forward network (position-wise MLP)**  
    hidden = x × W1 + b1  
    hidden = activation(hidden)   (usually GELU or SwiGLU: hidden × sigmoid(…) or similar)  
    ffn_out = hidden × W2 + b2  
    → two big matrix multiplies + non-linear activation.

17. **Second residual connection + norm**  
    x ← x + ffn_out  
    x ← LayerNorm(x)  
    → again: add back + normalize.

18. **Repeat steps 5–17 for every layer** (12 to 100+ times)  
    → x gets passed to next layer as new input.

19. **After final layer: take output vector of last position**  
    last_hidden = x[-1, :]   (or during training, all positions)

20. **Unembedding / output projection**  
    logits = last_hidden × W_unembed + b_unembed  
    → matrix multiply: projects back to vocabulary size (usually W_unembed = transpose of embedding matrix).

21. **Optional: tie embedding weights** (many models do this)  
    logits = last_hidden × EmbeddingMatrixᵀ  
    → re-uses same weights as input embedding (saves memory).

22. **Softmax to probabilities** (during generation / evaluation)  
    probs = softmax(logits)  
    → turns scores into % chance for each possible next word.

23. **Next token selection** (during generation)  
    next_token = argmax(probs)   or sample(probs)   or top-k/top-p sampling  
    → picks integer ID of next word (greedy, random, etc.).

24. **Append & repeat autoregressively**  
    Add predicted token to input sequence → go back to step 2 (but smartly reuse previous computations via KV cache).

