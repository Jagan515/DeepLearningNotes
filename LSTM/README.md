# 📘 LONG SHORT-TERM MEMORY (LSTM) — COMPLETE NOTES

![Image](https://pluralsight2.imgix.net/guides/8a8ac7c1-8bac-4e89-ace8-9e28813ab635_3.JPG)

![Image](https://www.researchgate.net/publication/346242875/figure/fig3/AS%3A11431281416043914%401746051476274/The-unfolded-chain-structure-of-LSTM-in-time-sequence.tif)

![Image](https://www.researchgate.net/publication/344554659/figure/fig2/AS%3A944635081932807%401602229962427/An-LSTM-cell-structure-showing-the-Input-Forget-and-Output-gates.png)

![Image](https://www.researchgate.net/publication/348356800/figure/fig2/AS%3A978406866952192%401610281783624/LSTM-architecture-1-Forget-gate-2-Input-gate-3-Output-gate-According-to-Fig2-is.png)

---

## 1️⃣ What is LSTM?

**LSTM (Long Short-Term Memory)** is a special type of **Recurrent Neural Network** designed to **remember long-term dependencies** and solve the **vanishing gradient problem** of simple RNNs.

🔑 Core idea:

> LSTM has a **memory cell** that decides **what to remember, what to forget, and what to output**.

---

## 2️⃣ Why LSTM was Created?

### Problems with Simple RNN

* Vanishing gradient
* Cannot learn long-term dependencies
* Memory fades quickly

### LSTM Solution

* Explicit **memory cell**
* **Gates** to control information flow
* Stable gradient flow

---

## 3️⃣ LSTM Architecture (Big Picture)

Each LSTM cell contains:

* **Cell State (`Cₜ`)** → long-term memory
* **Hidden State (`hₜ`)** → short-term memory
* **Three Gates**

  1. Forget Gate
  2. Input Gate
  3. Output Gate

---

## 4️⃣ Key Components of LSTM

### 🔹 Cell State (`Cₜ`)

* Acts like a **conveyor belt**
* Carries long-term information
* Changes very little → avoids vanishing gradient

---

### 🔹 Hidden State (`hₜ`)

* Output of the LSTM cell
* Passed to next time step and output layer

---

## 5️⃣ LSTM Gates (MOST IMPORTANT 🔥)

Gates are **neural networks** with **sigmoid activation** that output values between **0 and 1**.

---

### 1️⃣ Forget Gate

📌 **Purpose:** What information to forget?

[
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
]

* `0` → forget completely
* `1` → keep completely

---

### 2️⃣ Input Gate

📌 **Purpose:** What new information to store?

#### Step 1: Decide *what* to update

[
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
]

#### Step 2: Create candidate values

[
\tilde{C}*t = \tanh(W_c \cdot [h*{t-1}, x_t] + b_c)
]

---

### 3️⃣ Cell State Update

[
C_t = f_t * C_{t-1} + i_t * \tilde{C}_t
]

✔ Old memory × forget
✔ New memory × input

---

### 4️⃣ Output Gate

📌 **Purpose:** What to output?

[
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
]
[
h_t = o_t * \tanh(C_t)
]

---

## 6️⃣ Why Sigmoid & Tanh in LSTM?

| Function | Reason                        |
| -------- | ----------------------------- |
| Sigmoid  | Acts like a switch (0–1)      |
| Tanh     | Keeps values between −1 and 1 |

---

## 7️⃣ Forward Propagation in LSTM

For each time step:

1. Forget old memory
2. Select new memory
3. Update cell state
4. Generate output

This repeats for every sequence element.

---

## 8️⃣ Weight Sharing in LSTM

* Same weights used at **all time steps**
* Separate weights for:

  * Forget gate
  * Input gate
  * Cell candidate
  * Output gate

---

## 9️⃣ Activation Functions in LSTM

### Inside LSTM Cell

* Sigmoid → gates
* Tanh → memory & output shaping

### Output Layer (depends on task)

| Task                       | Activation |
| -------------------------- | ---------- |
| Binary classification      | Sigmoid    |
| Multi-class classification | Softmax    |
| Regression                 | Linear     |

---

## 🔟 Loss Functions in LSTM

| Task                       | Loss                     |
| -------------------------- | ------------------------ |
| Binary classification      | Binary Crossentropy      |
| Multi-class classification | Categorical Crossentropy |
| Regression                 | MSE                      |

Loss may be:

* Calculated at final step
* Or at each time step

---

## 1️⃣1️⃣ Backpropagation in LSTM

Uses **Backpropagation Through Time (BPTT)**.

Why LSTM works better:

* Cell state allows **gradient flow without shrinking**
* Gates control gradient explosion

---

## 1️⃣2️⃣ Vanishing Gradient — How LSTM Fixes It

### In RNN:

[
h_t = \tanh(W h_{t-1})
]

### In LSTM:

[
C_t = f_t * C_{t-1}
]

✔ Multiplication by values close to **1**
✔ Gradient does not vanish

---

## 1️⃣3️⃣ Types of LSTM Output Usage

### Many-to-One

* Sentiment analysis

### Many-to-Many

* Machine translation
* Time series prediction

---

## 1️⃣4️⃣ LSTM Parameters

Each LSTM unit has:

* Input weights
* Recurrent weights
* Biases

📌 More parameters than RNN → more power

---

## 1️⃣5️⃣ Example LSTM Architecture (Keras)

```python
LSTM(128, return_sequences=False)
Dense(1, activation='sigmoid')
```

---

## 1️⃣6️⃣ return_sequences Parameter

| Value | Meaning               |
| ----- | --------------------- |
| False | Output last time step |
| True  | Output all time steps |

Used for stacked LSTMs.

---

## 1️⃣7️⃣ Overfitting in LSTM

### Causes

* Too many parameters
* Small dataset

### Solutions

* Dropout
* More data
* Early stopping

---

## 1️⃣8️⃣ Dropout in LSTM

* Dropout → input connections
* Recurrent dropout → recurrent connections

```python
LSTM(64, dropout=0.2, recurrent_dropout=0.2)
```

---

## 1️⃣9️⃣ When to Use LSTM?

✅ Use when:

* Long sequences
* Long-term dependency exists
* Language modeling
* Time series forecasting

❌ Avoid when:

* Very small data
* Simple patterns

---

## 2️⃣0️⃣ RNN vs LSTM (Quick Memory)

| Feature            | RNN | LSTM   |
| ------------------ | --- | ------ |
| Long-term memory   | ❌   | ✅      |
| Gates              | ❌   | ✅      |
| Vanishing gradient | ❌   | Solved |
| Parameters         | Few | More   |

---

## 🧠 FINAL MEMORY TABLE (EXAM GOLD)

| Term         | Meaning          |
| ------------ | ---------------- |
| Cell state   | Long-term memory |
| Hidden state | Output memory    |
| Forget gate  | Deletes info     |
| Input gate   | Adds info        |
| Output gate  | Controls output  |
| Sigmoid      | Gate control     |
| Tanh         | Memory scaling   |

---


