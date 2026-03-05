### 🧠 Attention (in Deep Learning)

![Image](https://d3lkc3n5th01x7.cloudfront.net/wp-content/uploads/2023/11/08031612/Attention-mechanism-27190-01-1.png)

![Image](https://lena-voita.github.io/resources/lectures/seq2seq/attention/attn_for_steps/2-min.png)

![Image](https://miro.medium.com/1%2AvrSX_Ku3EmGPyqF_E-2_Vg.png)

![Image](https://sebastianraschka.com/images/blog/2023/self-attention-from-scratch/summary.png)

**Attention** is a mechanism that lets a model **focus on the most important parts of the input** when making a prediction.

Instead of treating all input equally, the model asks:

> *“Which parts matter most **right now**?”*

---

## 🚫 The problem Attention solves

In early sequence models (like basic RNN encoder–decoders), the entire input was squeezed into **one vector**.
For long sequences, important details were lost.

Attention fixes this by letting the model **look back at the input** every time it predicts something.

---

## ✅ Core idea (simple)

When generating an output:

1. Compare the current state with **all input states**
2. Assign **weights** (importance scores)
3. Take a **weighted sum** of the input
4. Use that focused information to decide the output

---

## 🔁 How Attention works (step by step)

Assume:

* Encoder outputs: (h_1, h_2, \dots, h_n)
* Current decoder state: (s_t)

### 1️⃣ Score (relevance)

Compute how relevant each input is:
[
\text{score}(s_t, h_i)
]

### 2️⃣ Weights (softmax)

Convert scores into probabilities:
[
\alpha_i = \text{softmax(score)}
]

### 3️⃣ Context vector

Combine inputs using weights:
[
c_t = \sum_i \alpha_i h_i
]

### 4️⃣ Prediction

Use (c_t) + decoder state to generate output.

---

## 🧠 Intuition with an example

**Input:**

> “The cat sitting on the mat is sleeping”

**When predicting “sleeping”**, the model focuses more on:

* *cat*
* *is*

and less on:

* *on*
* *the*

That selective focus = **attention**.

---

## 🧩 Types of Attention

### 1️⃣ Bahdanau Attention (Additive)

* Uses a small neural network to compute scores
* Flexible, good for long sequences

### 2️⃣ Luong Attention (Multiplicative)

* Uses dot product
* Faster, simpler

### 3️⃣ Self-Attention

* Each word attends to **other words in the same sentence**
* Core idea behind **Transformers**

---

## ⭐ Why Attention is important

✔ Handles long sequences
✔ Improves accuracy
✔ Creates interpretable alignments
✔ Foundation of modern models (BERT, GPT, etc.)

---

## 🗣️ One-line interview answer

> Attention is a mechanism that dynamically weights input features so a model can focus on the most relevant information when making predictions.

---
