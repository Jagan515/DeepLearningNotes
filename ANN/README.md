# 📘 ARTIFICIAL NEURAL NETWORK (ANN) — COMPLETE NOTES

---

## 1️⃣ What is ANN?

An **Artificial Neural Network (ANN)** is a computational model inspired by the **human brain**, made of **neurons**, **weights**, **bias**, and **activation functions** that learn patterns from data.

---

## 2️⃣ Basic Structure of ANN

### Components

* **Input Layer** → takes input features
* **Hidden Layer(s)** → performs computation
* **Output Layer** → produces final result

### Neuron Equation

[
z = \sum (w_i x_i) + b
]
[
a = f(z)
]

Where:

* `x` → input
* `w` → weight
* `b` → bias
* `f` → activation function
* `a` → neuron output

---

## 3️⃣ Weights and Bias (Very Important)

### 🔹 Weight

* Controls **importance** of an input
* Learned during training
* Large weight → strong influence

### 🔹 Bias

* Shifts the activation function
* Allows model to learn even if inputs are zero
* Improves flexibility

👉 **Without bias, ANN is weak**

---

## 4️⃣ Forward Propagation (Multiplication happens here)

Steps:

1. Multiply input with weights → `x × w`
2. Add bias → `xw + b`
3. Apply activation function → output

This is called **forward propagation**.

---

## 5️⃣ Activation Functions (CORE CONCEPT)

### Why activation functions?

* Introduce **non-linearity**
* Without activation → ANN becomes linear (useless)

---

## 6️⃣ Types of Activation Functions

### 1️⃣ Sigmoid

[
\sigma(x) = \frac{1}{1+e^{-x}}
]

📌 Range: (0,1)

✅ Use when:

* Binary classification (output layer)

❌ Problems:

* Vanishing gradient
* Slow learning

---

### 2️⃣ Tanh

[
\tanh(x)
]

📌 Range: (-1,1)

✅ Better than sigmoid
❌ Still vanishing gradient

---

### 3️⃣ ReLU (MOST IMPORTANT)

[
f(x) = \max(0, x)
]

📌 Range: (0, ∞)

✅ Use when:

* Hidden layers (default choice)
* Faster training

❌ Problem:

* Dying ReLU (neurons output 0 forever)

---

### 4️⃣ Leaky ReLU

[
f(x) = \max(0.01x, x)
]

✅ Solves dying ReLU problem

---

### 5️⃣ Softmax

[
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum e^{x_j}}
]

📌 Range: (0,1), sum = 1

✅ Use when:

* Multi-class classification
* Output layer

---

## 7️⃣ Which Activation Function to Use?

| Layer                        | Activation        |
| ---------------------------- | ----------------- |
| Hidden layer                 | ReLU / Leaky ReLU |
| Binary classification output | Sigmoid           |
| Multi-class output           | Softmax           |
| Regression output            | Linear            |

---

## 8️⃣ Loss Functions (VERY IMPORTANT)

### What is loss?

Loss measures **how wrong** the model is.

---

### 1️⃣ Mean Squared Error (MSE)

[
\frac{1}{n}\sum(y - \hat{y})^2
]

✅ Use when:

* Regression problems

---

### 2️⃣ Binary Cross Entropy

[
-(y\log \hat{y} + (1-y)\log(1-\hat{y}))
]

✅ Use when:

* Binary classification
* Output activation = Sigmoid

---

### 3️⃣ Categorical Cross Entropy

[
-\sum y \log(\hat{y})
]

✅ Use when:

* Multi-class classification
* Output activation = Softmax

---

### 🔑 Loss Function Selection Rule

| Problem                    | Output Activation | Loss                     |
| -------------------------- | ----------------- | ------------------------ |
| Regression                 | Linear            | MSE                      |
| Binary classification      | Sigmoid           | Binary Crossentropy      |
| Multi-class classification | Softmax           | Categorical Crossentropy |

---

## 9️⃣ Backpropagation (Learning Process)

Steps:

1. Calculate loss
2. Compute gradient of loss wrt weights
3. Update weights:
   [
   w = w - \eta \frac{\partial L}{\partial w}
   ]

Where:

* `η` = learning rate

---

## 🔟 Learning Rate

* Controls step size
* Too large → unstable
* Too small → slow learning

---

## 1️⃣1️⃣ Overfitting & Underfitting

### Overfitting

* Model memorizes data
* Poor test accuracy

### Underfitting

* Model too simple
* Poor training accuracy

---

## 1️⃣2️⃣ Regularization Techniques (Advanced)

### 1️⃣ L1 Regularization

* Makes weights sparse

### 2️⃣ L2 Regularization

* Penalizes large weights
* Most common

---

### 3️⃣ Dropout

* Randomly disables neurons
* Prevents overfitting

---

## 1️⃣3️⃣ Weight Initialization

| Method            | Use            |
| ----------------- | -------------- |
| Random            | Basic          |
| Xavier            | Sigmoid / Tanh |
| He Initialization | ReLU           |

---

## 1️⃣4️⃣ Batch, Epoch, Iteration

* **Batch** → subset of data
* **Epoch** → full dataset pass
* **Iteration** → one batch pass

---

## 1️⃣5️⃣ ANN Classification Judgment

### Binary Classification

* Output neuron = 1
* Sigmoid output > 0.5 → class 1

### Multi-Class

* Highest Softmax value → predicted class

---

## 1️⃣6️⃣ ANN Architecture Examples

### Binary Classification

```python
Dense(64, activation='relu')
Dense(1, activation='sigmoid')
```

### Multi-class Classification

```python
Dense(64, activation='relu')
Dense(10, activation='softmax')
```

---

## 1️⃣7️⃣ When to Use ANN?

✅ Use ANN when:

* Data is tabular
* Relationships are non-linear
* Feature engineering is hard

❌ Avoid ANN when:

* Very small dataset
* Simple linear relationships

---

## 🔥 FINAL MEMORY TABLE (VERY IMPORTANT)

| Concept         | Purpose            |
| --------------- | ------------------ |
| Weight          | Feature importance |
| Bias            | Shifts output      |
| Activation      | Non-linearity      |
| Loss            | Error measurement  |
| Optimizer       | Weight update      |
| Backpropagation | Learning           |
| Regularization  | Avoid overfitting  |

---

