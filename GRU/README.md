# 📘 GATED RECURRENT UNIT (GRU) — COMPLETE NOTES

![Image](https://towardsdatascience.com/wp-content/uploads/2022/02/1LfH52lSd1mq-UrWuejzO8g.png)

![Image](https://miro.medium.com/v2/resize%3Afit%3A1400/1%2ApLy26OilW-imHRcZqUih_Q.png)

![Image](https://www.researchgate.net/publication/379792888/figure/fig2/AS%3A11431281238989947%401714141220159/The-reset-gate-update-gate-and-hidden-status-of-the-GRU.tif)

![Image](https://www.researchgate.net/publication/342801257/figure/fig2/AS%3A911226158731264%401594264654127/GRU-cell-architecture-GRU-has-two-gates-reset-gate-and-update-gate-These-are-useful-in.ppm)

---

## 1️⃣ What is GRU?

**GRU (Gated Recurrent Unit)** is a type of recurrent neural network designed to:

* handle **long-term dependencies**
* solve the **vanishing gradient problem**
* be **simpler and faster than LSTM**

🔑 GRU combines **memory and hidden state into one**.

---

## 2️⃣ Why GRU was Introduced?

LSTM works well but:

* Too many parameters
* Slower training
* Complex architecture

👉 **GRU simplifies LSTM** while keeping performance similar.

---

## 3️⃣ GRU Architecture (High Level)

GRU has:

* **Hidden State (`hₜ`)** → acts as memory
* **Two Gates**

  1. Update Gate
  2. Reset Gate

❌ No separate cell state
❌ No output gate

---

## 4️⃣ Key Components of GRU

### 🔹 Hidden State (`hₜ`)

* Stores both long-term & short-term memory
* Passed to next time step

---

## 5️⃣ GRU Gates (MOST IMPORTANT 🔥)

Gates use **sigmoid activation** → values between 0 and 1.

---

### 1️⃣ Update Gate (`zₜ`)

📌 **Purpose:** How much past memory to keep?

[
z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)
]

* `zₜ ≈ 1` → keep old memory
* `zₜ ≈ 0` → replace with new memory

---

### 2️⃣ Reset Gate (`rₜ`)

📌 **Purpose:** How much past info to forget?

[
r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)
]

* Small value → forget past
* Large value → use past

---

## 6️⃣ Candidate Hidden State

[
\tilde{h}*t = \tanh(W_h \cdot [r_t * h*{t-1}, x_t] + b_h)
]

✔ Reset gate controls how much past is used
✔ Tanh keeps values stable

---

## 7️⃣ Final Hidden State Update

[
h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t
]

✔ Smooth blending of old & new memory
✔ Key to long-term dependency learning

---

## 8️⃣ Forward Propagation in GRU

At each time step:

1. Compute update gate
2. Compute reset gate
3. Compute candidate memory
4. Update hidden state

---

## 9️⃣ Why GRU Solves Vanishing Gradient

* Update gate allows **direct memory flow**
* Gradients don’t vanish easily
* Similar benefit as LSTM but simpler

---

## 🔟 Weight Sharing in GRU

* Same weights used at **all time steps**
* Separate weights for:

  * Update gate
  * Reset gate
  * Candidate state

---

## 1️⃣1️⃣ Activation Functions in GRU

### Inside GRU Cell

* Sigmoid → gates
* Tanh → candidate hidden state

### Output Layer

| Task                       | Activation |
| -------------------------- | ---------- |
| Binary classification      | Sigmoid    |
| Multi-class classification | Softmax    |
| Regression                 | Linear     |

---

## 1️⃣2️⃣ Loss Functions in GRU

| Task                       | Loss                     |
| -------------------------- | ------------------------ |
| Binary classification      | Binary Crossentropy      |
| Multi-class classification | Categorical Crossentropy |
| Regression                 | MSE                      |

---

## 1️⃣3️⃣ Backpropagation in GRU

Uses **Backpropagation Through Time (BPTT)**.

✔ Fewer gates → fewer gradients
✔ More stable learning than simple RNN

---

## 1️⃣4️⃣ GRU Parameters

| Model | Parameters |
| ----- | ---------- |
| RNN   | Least      |
| GRU   | Medium     |
| LSTM  | Most       |

---

## 1️⃣5️⃣ Example GRU Architecture (Keras)

```python
GRU(128, return_sequences=False)
Dense(1, activation='sigmoid')
```

---

## 1️⃣6️⃣ return_sequences in GRU

| Value | Meaning               |
| ----- | --------------------- |
| False | Output last time step |
| True  | Output all time steps |

Used for stacked GRUs or sequence outputs.

---

## 1️⃣7️⃣ Overfitting in GRU

### Causes

* High model capacity
* Small dataset

### Solutions

* Dropout
* Early stopping
* Data augmentation (for sequences)

---

## 1️⃣8️⃣ Dropout in GRU

```python
GRU(64, dropout=0.2, recurrent_dropout=0.2)
```

---

## 1️⃣9️⃣ When to Use GRU?

✅ Use when:

* Long sequences
* Limited computation
* Faster training needed
* Similar performance to LSTM acceptable

❌ Avoid when:

* Very complex long-term dependencies

---

## 2️⃣0️⃣ GRU vs LSTM (Quick Comparison)

| Feature        | GRU    | LSTM   |
| -------------- | ------ | ------ |
| Gates          | 2      | 3      |
| Cell state     | ❌      | ✅      |
| Parameters     | Fewer  | More   |
| Training speed | Faster | Slower |

---

## 🧠 FINAL MEMORY TABLE (EXAM GOLD)

| Term         | Meaning                   |
| ------------ | ------------------------- |
| Update gate  | Controls memory retention |
| Reset gate   | Controls forgetting       |
| Hidden state | Memory + output           |
| Sigmoid      | Gate control              |
| Tanh         | Candidate memory          |

---
