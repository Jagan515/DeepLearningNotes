

## 1. Simple RNN (Vanilla RNN)

The most basic form of recurrence. It processes input at each time step and passes information to the next step.

* **How it works:** It takes the current input  and the previous hidden state , combines them, and passes them through a `tanh` activation function to produce the current hidden state .
* **The Problem:** It suffers from the **Vanishing Gradient Problem**. As the sequence gets longer, the network "forgets" information from the very beginning because the gradients used for training shrink to near zero.
* **Key Component:** A single `tanh` layer that regulates the information flow.

## 2. LSTM (Long Short-Term Memory)

Designed specifically to solve the vanishing gradient problem by using a "Cell State" () that acts like a long-term memory conveyor belt.

* **How it works:** It uses three "gates" to control information:
1. **Forget Gate ():** Decides what information from the previous state to throw away.
2. **Input Gate ( and `tanh`):** Decides which new information to store in the cell state.
3. **Output Gate ():** Decides what part of the cell state to output as the hidden state .


* **Notes:** The horizontal line at the top represents the **Cell State**, which allows information to flow through the entire sequence with only minor linear interactions.

## 3. GRU (Gated Recurrent Unit)

A newer, streamlined version of the LSTM that is computationally more efficient.

* **How it works:** It merges the cell state and hidden state into one. It uses only two gates:
1. **Reset Gate ():** Determines how much of the past information to forget.
2. **Update Gate ():** Decides how much of the past state to keep and how much of the new candidate state to add.


* **Notes:** GRUs have fewer parameters than LSTMs, making them faster to train while often providing similar performance, especially on smaller datasets.

---

### Comparison Summary

| Feature | Simple RNN | LSTM | GRU |
| --- | --- | --- | --- |
| **Complexity** | Low | High | Medium |
| **Memory** | Short-term only | Long & Short-term | Long & Short-term |
| **Gates** | None | 3 (Forget, Input, Output) | 2 (Reset, Update) |
| **Performance** | Poor on long sequences | Excellent on long sequences | Excellent / Efficient |









----------------------------------------------------------------------------------------------------------
# Recurrent Neural Networks (RNN Family)

These models are designed for **sequential data**:

* Text
* Time series
* Speech
* Sensor data

Examples:

```
Sentence → words in order
Stock price → time steps
Audio → signal over time
```

---

## 1. Simple RNN (Foundation)

### What is an RNN?

An RNN processes input **one time step at a time** and keeps a **hidden state (memory)**.

```
x₁ → h₁ → h₂ → h₃ → ...
```

### Core equation

```
hₜ = tanh(Wx·xₜ + Wh·hₜ₋₁ + b)
```

### Problem

 Cannot remember long-term dependencies
 Suffers from **vanishing gradient**

---

## 2. LSTM (Long Short-Term Memory)

LSTM was created to **fix RNN’s memory problem**.

---

### Key Idea

LSTM introduces a **cell state (Cₜ)** that can carry information over long sequences.

```
Cₜ₋₁ → Cₜ → Cₜ₊₁
```

---

### Gates in LSTM

#### 1. Forget Gate

Decides what to remove

```
fₜ = σ(Wf · [hₜ₋₁, xₜ])
```

#### 2. Input Gate

Decides what to add

```
iₜ = σ(Wi · [hₜ₋₁, xₜ])
```

#### 3. Candidate Memory

```
Ĉₜ = tanh(Wc · [hₜ₋₁, xₜ])
```

#### 4. Cell State Update

```
Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ Ĉₜ
```

#### 5. Output Gate

```
oₜ = σ(Wo · [hₜ₋₁, xₜ])
hₜ = oₜ ⊙ tanh(Cₜ)
```

---

### Why LSTM Works

* Explicit memory cell
* Controlled information flow
* Solves vanishing gradient

---

## 3. GRU (Gated Recurrent Unit)

GRU is a **simplified LSTM**.

### Key Difference

❌ No separate cell state
✅ Fewer gates → faster training

---

### Gates in GRU

#### 1. Update Gate

Controls memory retention

```
zₜ = σ(Wz · [hₜ₋₁, xₜ])
```

#### 2. Reset Gate

Controls past information usage

```
rₜ = σ(Wr · [hₜ₋₁, xₜ])
```

#### 3. Candidate Hidden State

```
ĥₜ = tanh(W · [rₜ ⊙ hₜ₋₁, xₜ])
```

#### 4. Final Hidden State

```
hₜ = (1 − zₜ) ⊙ hₜ₋₁ + zₜ ⊙ ĥₜ
```

---

### Why GRU Is Popular

* Faster than LSTM
* Fewer parameters
* Similar performance in many tasks

---

## 4. Bidirectional RNN / LSTM / GRU

### Problem with normal RNNs

They only see **past context**.

Example:

```
"I live in New ___"
```

You need **future words** to predict correctly.

---

### Solution: Bidirectional RNN

Processes sequence in **both directions**:

```
Forward:   x₁ → x₂ → x₃
Backward:  x₃ → x₂ → x₁
```

Outputs are combined:

```
hₜ = [→hₜ ; ←hₜ]
```

---

### Can Be Applied To:

* Bidirectional RNN
* Bidirectional LSTM
* Bidirectional GRU

---

### TensorFlow Example

```python
layers.Bidirectional(layers.LSTM(64))
layers.Bidirectional(layers.GRU(64))
```

---

## End-to-End Comparison

| Model   | Memory         | Gates | Speed  | Use Case               |
| ------- | -------------- | ----- | ------ | ---------------------- |
| RNN     | Short          | None  | Fast   | Very simple sequences  |
| LSTM    | Long           | 3     | Slower | Long-term dependencies |
| GRU     | Long           | 2     | Faster | Efficient alternative  |
| Bi-LSTM | Long + context | 3×2   | Slower | NLP, speech            |
| Bi-GRU  | Long + context | 2×2   | Medium | NLP, time series       |

---

## When to Use What

### Use RNN when:

* Sequence is very short
* Educational purpose

### Use LSTM when:

* Long-term dependencies matter
* Accuracy is critical

### Use GRU when:

* Speed & efficiency matter
* Dataset is medium

### Use Bidirectional models when:

* Entire sequence is available
* Context from both sides is important

---

## Interview One-Liners

* **RNN**: Sequential model with short memory
* **LSTM**: RNN with long-term memory using gates
* **GRU**: Simplified LSTM with fewer parameters
* **Bidirectional**: Learns from past and future context

---

## Mental Model (Easy)

* RNN → short memory
* LSTM → notebook memory
* GRU → smart sticky note
* Bidirectional → reads sentence forward + backward

---



