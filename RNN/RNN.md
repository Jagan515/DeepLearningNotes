# 📘 RECURRENT NEURAL NETWORK (RNN) — COMPLETE NOTES

*(ONLY Simple / Vanilla RNN)*

![Image](https://upload.wikimedia.org/wikipedia/commons/b/b5/Recurrent_neural_network_unfold.svg)

![Image](https://www.researchgate.net/publication/346853926/figure/fig2/AS%3A1007113883119616%401617126069530/A-simple-RNN-with-a-single-hidden-layer-At-each-time-step-output-is-produced-through.ppm)

![Image](https://d2l.ai/_images/rnn.svg)

![Image](https://discuss.pytorch.org/uploads/default/original/2X/e/e7496a33d835f085d800ee17c0ade05895a89551.png)

---

## 1️⃣ What is RNN?

A **Recurrent Neural Network (RNN)** is a neural network designed to handle **sequential data** by **remembering past information**.

🔑 RNN has a **memory** → called **hidden state**.

---

## 2️⃣ Why RNN?

ANN & CNN assume:

* Inputs are **independent**
* No order matters

But in real life:

* Language
* Time series
* Speech
* Stock prices

👉 **Order matters**

RNN solves this by passing information **from previous time step to next**.

---

## 3️⃣ Sequential Data Examples

| Data        | Sequence       |
| ----------- | -------------- |
| Text        | word by word   |
| Speech      | audio frames   |
| Time series | day by day     |
| Video       | frame by frame |

---

## 4️⃣ RNN Architecture (Core Idea)

At time step `t`:
[
h_t = f(W_x x_t + W_h h_{t-1} + b)
]
[
y_t = W_y h_t
]

Where:

* `x_t` → input at time `t`
* `h_t` → hidden state (memory)
* `W_x` → input weights
* `W_h` → recurrent weights
* `b` → bias

---

## 5️⃣ Key Components of RNN

### 🔹 Input (`xₜ`)

* Current element of sequence

### 🔹 Hidden State (`hₜ`)

* Memory of past
* Passed to next time step

### 🔹 Output (`yₜ`)

* Prediction at time `t`

---

## 6️⃣ Weight Sharing (VERY IMPORTANT)

In RNN:

* **Same weights** are used at **all time steps**

This allows:

* Learning long sequences
* Fewer parameters

---

## 7️⃣ Forward Propagation in RNN

For each time step:

1. Multiply input with weights
2. Add previous hidden state
3. Add bias
4. Apply activation

➡️ This repeats for every time step

---

## 8️⃣ Activation Functions in RNN

### Hidden State Activation

* **Tanh** (most common)
* Sometimes ReLU

Why Tanh?

* Output range (-1,1)
* Keeps values stable

---

### Output Activation (depends on task)

| Task                       | Activation |
| -------------------------- | ---------- |
| Binary classification      | Sigmoid    |
| Multi-class classification | Softmax    |
| Regression                 | Linear     |

---

## 9️⃣ Types of RNN Outputs (IMPORTANT)

### 1️⃣ One-to-One

* ANN-like
* Not sequence-based

---

### 2️⃣ One-to-Many

Example:

* Image → Caption

---

### 3️⃣ Many-to-One

Example:

* Sentiment analysis

---

### 4️⃣ Many-to-Many

Example:

* Language translation

---

## 🔟 Loss Functions in RNN

| Task                       | Loss Function            |
| -------------------------- | ------------------------ |
| Binary classification      | Binary Crossentropy      |
| Multi-class classification | Categorical Crossentropy |
| Regression                 | MSE                      |

Loss can be:

* Calculated at **each time step**
* Or only at **final output**

---

## 1️⃣1️⃣ Backpropagation Through Time (BPTT)

RNN uses **BPTT** instead of normal backpropagation.

### Steps:

1. Unroll RNN through time
2. Compute loss
3. Backpropagate errors backward in time
4. Update shared weights

---

## 1️⃣2️⃣ Vanishing Gradient Problem 🚨

### What happens?

* Gradients become very small
* Early time steps stop learning

### Why?

* Repeated multiplication of small numbers

📌 **This is the biggest limitation of simple RNN**

---

## 1️⃣3️⃣ Exploding Gradient Problem

* Gradients become very large
* Model becomes unstable

### Solution:

* Gradient clipping

---

## 1️⃣4️⃣ Why Simple RNN Struggles

| Problem              | Reason                 |
| -------------------- | ---------------------- |
| Long-term dependency | Vanishing gradient     |
| Slow training        | Sequential computation |
| Memory loss          | Short-term memory only |

👉 This is **why LSTM/GRU exist**, but **we stop here** as requested.

---

## 1️⃣5️⃣ RNN Parameters

### Weights:

* Input-to-hidden
* Hidden-to-hidden
* Hidden-to-output

### Bias:

* Shifts activation
* Helps learning

---

## 1️⃣6️⃣ Example: Sentiment Analysis (Many-to-One)

```text
Input: "I love this movie"
↓
Word embeddings → RNN
↓
Final hidden state
↓
Sigmoid → Positive / Negative
```

---

## 1️⃣7️⃣ Example RNN Architecture (Keras)

```python
SimpleRNN(64, activation='tanh')
Dense(1, activation='sigmoid')
```

---

## 1️⃣8️⃣ When to Use Simple RNN?

✅ Use when:

* Short sequences
* Simple temporal patterns
* Learning basic sequence behavior

❌ Avoid when:

* Long sequences
* Long-term dependencies

---

## 1️⃣9️⃣ ANN vs CNN vs RNN (Quick Table)

| Feature        | ANN     | CNN   | RNN      |
| -------------- | ------- | ----- | -------- |
| Data type      | Tabular | Image | Sequence |
| Memory         | ❌       | ❌     | ✅        |
| Weight sharing | ❌       | ✅     | ✅        |
| Temporal info  | ❌       | ❌     | ✅        |

---

## 2️⃣0️⃣ FINAL MEMORY TABLE 🧠

| Term               | Meaning                  |
| ------------------ | ------------------------ |
| Hidden state       | Memory                   |
| Weight sharing     | Same weights across time |
| BPTT               | Learning method          |
| Vanishing gradient | Main RNN problem         |
| Tanh               | Default activation       |

---
