
# 📘 SEQUENCE-TO-SEQUENCE (Seq2Seq) — COMPLETE NOTES

![Image](https://miro.medium.com/v2/resize%3Afit%3A1400/1%2A1JcHGUU7rFgtXC_mydUA_Q.jpeg)

![Image](https://d2l.ai/_images/seq2seq.svg)

![Image](https://miro.medium.com/0%2A7fZwj5ebQyfsRlht.png)

---

## 1️⃣ What is Seq2Seq?

**Sequence-to-Sequence (Seq2Seq)** is a neural network architecture that **maps an input sequence to an output sequence**, where **both sequences can have different lengths**.

### Simple definition (exam-ready)

> Seq2Seq converts a **variable-length input sequence** into a **variable-length output sequence** using an **Encoder–Decoder architecture**.

---

## 2️⃣ Why Seq2Seq is Needed

Traditional models:

* Assume fixed-size input & output
* Cannot handle variable-length sequences well

Seq2Seq solves:

* Different input/output lengths
* Order-dependent data

---

## 3️⃣ Where Seq2Seq is Used

| Application         | Input → Output         |
| ------------------- | ---------------------- |
| Machine Translation | English → French       |
| Chatbots            | Question → Answer      |
| Text Summarization  | Long text → Short text |
| Speech Recognition  | Audio → Text           |
| Text Generation     | Prompt → Sentence      |

---

## 4️⃣ Core Architecture of Seq2Seq

Seq2Seq has **two main components**:

1. **Encoder**
2. **Decoder**

---

## 5️⃣ Encoder (Input Processing)

### Role of Encoder

* Reads the **entire input sequence**
* Converts it into a **fixed-length representation**

### Output of Encoder

* Final **hidden state**
* Called the **context vector**

📌 The encoder **does not generate output text**.

---

## 6️⃣ Context Vector (Key Concept)

* A **single vector** that represents the **meaning of the entire input sequence**
* Passed from encoder → decoder
* Acts as a **bridge** between them

⚠️ Limitation:

* Fixed size → information loss for long sequences

---

## 7️⃣ Decoder (Output Generation)

### Role of Decoder

* Takes the **context vector**
* Generates output **step by step**
* Uses previous output to predict the next token

Output continues until:

* End-of-sequence token is generated

---

## 8️⃣ Training vs Inference (Very Important)

### 🔹 Training (Teacher Forcing)

* Decoder receives the **actual target word** as next input
* Faster convergence
* More stable learning

### 🔹 Inference (Prediction Time)

* Decoder uses **its own previous prediction**
* Errors may accumulate

---

## 9️⃣ Loss Calculation in Seq2Seq

* Loss is calculated at **each time step**
* Total loss = sum (or average) of step losses
* Common loss:

  * **Categorical Cross-Entropy**

---

## 🔟 Types of Seq2Seq Mapping

| Type         | Description                | Example            |
| ------------ | -------------------------- | ------------------ |
| One-to-One   | Fixed input → fixed output | Classification     |
| One-to-Many  | Single input → sequence    | Image captioning   |
| Many-to-One  | Sequence → single output   | Sentiment analysis |
| Many-to-Many | Sequence → sequence        | Translation        |

---

## 1️⃣1️⃣ Seq2Seq with RNN / LSTM / GRU

* Encoder and decoder can be:

  * RNN
  * LSTM
  * GRU

📌 LSTM/GRU preferred due to:

* Better long-term memory
* Reduced vanishing gradient

---

## 1️⃣2️⃣ Weight Sharing in Seq2Seq

* Encoder weights shared across time steps
* Decoder weights shared across time steps
* Encoder and decoder weights are **different**

---

## 1️⃣3️⃣ Advantages of Seq2Seq

✔ Handles variable-length sequences
✔ Captures order information
✔ Flexible architecture
✔ Widely applicable in NLP

---

## 1️⃣4️⃣ Limitations of Basic Seq2Seq

| Problem                | Reason                |
| ---------------------- | --------------------- |
| Long sequences         | Single context vector |
| Information bottleneck | Fixed-size memory     |
| Poor long sentences    | Memory compression    |
| Slow inference         | Sequential decoding   |

➡️ These led to **Attention mechanisms**

---

## 1️⃣5️⃣ Seq2Seq vs Traditional Models

| Feature           | Traditional NN | Seq2Seq   |
| ----------------- | -------------- | --------- |
| Input length      | Fixed          | Variable  |
| Output length     | Fixed          | Variable  |
| Memory            | ❌              | ✅         |
| Sequence handling | Poor           | Excellent |

---

## 1️⃣6️⃣ Key Terminologies (Exam Gold)

| Term            | Meaning                   |
| --------------- | ------------------------- |
| Encoder         | Reads input sequence      |
| Decoder         | Generates output sequence |
| Context vector  | Encoded meaning           |
| Teacher forcing | Training strategy         |
| EOS token       | End of output             |

---

## 1️⃣7️⃣ One-Line Interview Answers

* **Seq2Seq definition:**
  *Seq2Seq is an encoder-decoder architecture that maps variable-length input sequences to variable-length output sequences.*

* **Main limitation:**
  *Fixed-size context vector causes information loss.*

* **Why attention was introduced:**
  *To remove the context vector bottleneck.*

---

## 🧠 FINAL MEMORY SUMMARY

| Concept         | Purpose          |
| --------------- | ---------------- |
| Encoder         | Encode input     |
| Decoder         | Generate output  |
| Context vector  | Store meaning    |
| Teacher forcing | Faster training  |
| Seq2Seq         | Sequence mapping |

---

\
