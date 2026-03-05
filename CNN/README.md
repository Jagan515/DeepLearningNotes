# 📘 CONVOLUTIONAL NEURAL NETWORK (CNN) — COMPLETE NOTES

![Image](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/10/90650dnn2.webp)

![Image](https://www.researchgate.net/publication/374143931/figure/fig4/AS%3A11431281190918268%401695521918070/Architecture-of-CNN-with-convolutional-pooling-fully-connected-layer-and-output.png)

![Image](https://miro.medium.com/0%2Ae-SMFTzO8r7skkpc)

![Image](https://towardsdatascience.com/wp-content/uploads/2019/10/1jIv2CLxdXsxvx60Urc11Tw.png)

---

## 1️⃣ What is CNN?

A **Convolutional Neural Network (CNN)** is a special type of neural network designed **specifically for image and spatial data**.

🔑 CNN automatically learns:

* edges
* textures
* shapes
* objects

👉 Unlike ANN, **CNN preserves spatial structure** of images.

---

## 2️⃣ Why CNN over ANN for Images?

### Problem with ANN

* Images are high-dimensional
* Flattening loses spatial information
* Too many parameters

### CNN solves this by:

* Using **local connections**
* **Weight sharing**
* **Feature extraction**

---

## 3️⃣ CNN Architecture (High Level)

Typical flow:

```
Input Image
 → Convolution
 → Activation
 → Pooling
 → Convolution
 → Pooling
 → Flatten
 → Fully Connected
 → Output
```

---

## 4️⃣ Input Image Representation

Image shape:

```
(H, W, C)
```

Examples:

* Grayscale → `(28, 28, 1)`
* RGB → `(224, 224, 3)`

---

## 5️⃣ Convolution Layer (MOST IMPORTANT)

### What is Convolution?

* A **filter/kernel** slides over the image
* Performs **element-wise multiplication**
* Produces a **feature map**

### Mathematical view:

[
\text{Feature Map} = \sum (Image \times Kernel) + Bias
]

🔁 **Multiplication happens here**

---

## 6️⃣ Filters / Kernels

### Filter

* Small matrix (e.g., 3×3, 5×5)
* Learns features like:

  * edges
  * corners
  * textures

Each filter has:

* its own **weights**
* its own **bias**

More filters → more features learned

---

## 7️⃣ Stride

### What is stride?

* Step size of filter movement

| Stride | Effect                 |
| ------ | ---------------------- |
| 1      | Detailed features      |
| 2      | Smaller output, faster |

---

## 8️⃣ Padding

### Why padding?

* Prevent image shrinking
* Preserve border information

| Padding | Meaning                  |
| ------- | ------------------------ |
| Valid   | No padding               |
| Same    | Output size = input size |

---

## 9️⃣ Output Size Formula (IMPORTANT)

[
\frac{(W - F + 2P)}{S} + 1
]

Where:

* `W` = input size
* `F` = filter size
* `P` = padding
* `S` = stride

---

## 🔟 Activation Functions in CNN

### Why activation?

* Introduce non-linearity

### Common Activations

#### 1️⃣ ReLU (DEFAULT)

[
f(x) = \max(0, x)
]

✅ Used after every convolution
❌ Can cause dying neurons

---

#### 2️⃣ Leaky ReLU

[
f(x) = \max(0.01x, x)
]

✅ Fixes dying ReLU

---

#### 3️⃣ Softmax (Output Layer)

* Used for multi-class classification

---

#### 4️⃣ Sigmoid (Output Layer)

* Used for binary classification

---

## 1️⃣1️⃣ Pooling Layer

### Purpose

* Reduce spatial size
* Reduce computation
* Prevent overfitting

---

### Types of Pooling

#### 1️⃣ Max Pooling (MOST COMMON)

* Takes maximum value

#### 2️⃣ Average Pooling

* Takes average

Typical:

```python
MaxPooling2D(pool_size=(2,2))
```

---

## 1️⃣2️⃣ Feature Maps

* Output of convolution layers
* Each feature map = one learned pattern
* Deeper layers → complex features

---

## 1️⃣3️⃣ Flatten Layer

* Converts 2D feature maps → 1D vector
* Connects CNN to ANN

```python
Flatten()
```

---

## 1️⃣4️⃣ Fully Connected (Dense) Layer

* Works like ANN
* Performs final decision making

---

## 1️⃣5️⃣ CNN Weights & Bias

### Weights

* Stored in filters
* Shared across image
* Fewer parameters than ANN

### Bias

* One bias per filter
* Shifts activation

---

## 1️⃣6️⃣ Loss Functions in CNN

| Task                  | Output Activation | Loss                     |
| --------------------- | ----------------- | ------------------------ |
| Binary classification | Sigmoid           | Binary Crossentropy      |
| Multi-class           | Softmax           | Categorical Crossentropy |
| Regression            | Linear            | MSE                      |

---

## 1️⃣7️⃣ Backpropagation in CNN

CNN learns by:

1. Forward propagation
2. Loss calculation
3. Backpropagation
4. Updating:

   * filter weights
   * bias

Uses **gradient descent**

---

## 1️⃣8️⃣ Overfitting in CNN

### Signs

* High training accuracy
* Low test accuracy

---

### Solutions

* Data augmentation
* Dropout
* Regularization
* More data

---

## 1️⃣9️⃣ Dropout in CNN

* Randomly disables neurons
* Usually applied to **dense layers**

```python
Dropout(0.5)
```

---

## 2️⃣0️⃣ Batch Normalization (Advanced)

### Purpose

* Faster convergence
* Stable training

Placed:

```text
Conv → BatchNorm → ReLU
```

---

## 2️⃣1️⃣ Data Augmentation

Artificially increase dataset:

* rotation
* flipping
* zoom
* shifting

Helps reduce overfitting.

---

## 2️⃣2️⃣ CNN Classification Decision

### Binary

```text
output > 0.5 → class 1
```

### Multi-class

```text
class = argmax(softmax output)
```

---

## 2️⃣3️⃣ Typical CNN Architecture Example

```python
Conv2D(32, (3,3), activation='relu')
MaxPooling2D(2,2)

Conv2D(64, (3,3), activation='relu')
MaxPooling2D(2,2)

Flatten()
Dense(128, activation='relu')
Dense(10, activation='softmax')
```

---

## 2️⃣4️⃣ When to Use CNN?

✅ Use CNN when:

* Image data
* Spatial relationships matter
* Pattern recognition required

❌ Avoid CNN when:

* Tabular data
* Very small datasets

---

## 🔥 ANN vs CNN (Quick Comparison)

| Feature      | ANN     | CNN       |
| ------------ | ------- | --------- |
| Input        | 1D      | 2D/3D     |
| Spatial info | Lost    | Preserved |
| Parameters   | High    | Low       |
| Best for     | Tabular | Images    |

---

## 🧠 FINAL MEMORY TABLE

| Component   | Purpose            |
| ----------- | ------------------ |
| Convolution | Feature extraction |
| Filter      | Pattern detector   |
| Pooling     | Downsampling       |
| ReLU        | Non-linearity      |
| Flatten     | CNN → ANN          |
| Softmax     | Class probability  |
| Loss        | Error measurement  |

---




#  CNN IMPORTANT TERMS (IN DETAIL)

---

## Kernel / Filter

### What is a Kernel?

A **kernel (or filter)** is a **small matrix** used to scan the input image and extract features.

 Common sizes:

* `3 × 3`
* `5 × 5`
* `7 × 7`

Example kernel:

```
1  0 -1
1  0 -1
1  0 -1
```

 Detects **vertical edges**

---

###  Why do we use kernels?

* To detect **features** like:

  * edges
  * textures
  * shapes
* Different kernels learn different features automatically

---

###  How do we use kernels?

* Kernel slides over the image
* Element-wise multiplication + sum
* Produces a **feature map**

 One kernel → one feature map
 Multiple kernels → multiple feature maps

---

##  Stride

###  What is Stride?

Stride is the **number of pixels the kernel moves** each time.

📌 Example:

* Stride = 1 → move 1 pixel
* Stride = 2 → skip 1 pixel

---

### 🔹 Why do we use stride?

* Controls **output size**
* Reduces **computation**
* Larger stride → smaller output

---

### 🔹 How do we use stride?

* Set stride value in convolution layer

📌 Example:

* Input: `7 × 7`
* Kernel: `3 × 3`
* Stride = 1 → Output: `5 × 5`
* Stride = 2 → Output: `3 × 3`

---

## 3️⃣ Padding

### 🔹 What is Padding?

Padding means **adding extra pixels (usually zeros)** around the image border.

📌 Types:

* **Valid padding** → no padding
* **Same padding** → output same size as input

---

### 🔹 Why do we use padding?

* Prevents loss of edge information
* Controls feature map size
* Allows deep CNNs without shrinking too fast

---

### 🔹 How do we use padding?

* Add zeros around image

📌 Example:

* Input: `5 × 5`
* Kernel: `3 × 3`
* Padding = 1 → Output remains `5 × 5`

---

## 4️⃣ Feature Map

### 🔹 What is a Feature Map?

* Output produced after applying a kernel
* Highlights where a feature appears

---

### 🔹 Why feature maps matter?

* Show **where** features are located
* Deeper layers → complex features

---

### 🔹 How are they created?

* Convolution + activation (ReLU)

---

## 5️⃣ Activation Function (ReLU)

### 🔹 What is ReLU?

[
ReLU(x) = \max(0, x)
]

---

### 🔹 Why do we use ReLU?

* Adds **non-linearity**
* Speeds up training
* Avoids vanishing gradients

---

### 🔹 How do we use it?

* Applied **after convolution**
* Replaces negative values with zero

---

## 6️⃣ Pooling

### 🔹 What is Pooling?

Pooling **down-samples** feature maps.

---

### 🔹 Types of Pooling

#### 🔸 Max Pooling (Most Common)

```
2  4
1  3  →  4
```

#### 🔸 Average Pooling

```
(2+4+1+3)/4 = 2.5
```

---

### 🔹 Why do we use pooling?

* Reduces spatial size
* Lowers computation
* Prevents overfitting
* Makes CNN robust to small shifts

---

### 🔹 How do we use pooling?

* Common size: `2 × 2`
* Stride = 2

📌 Output size reduced by half

---

## 7️⃣ Flattening

### 🔹 What is Flattening?

* Converts 2D feature maps into **1D vector**

---

### 🔹 Why do we flatten?

* Fully connected layers accept only vectors

---

### 🔹 How do we use it?

* Applied after final pooling layer

---

## 8️⃣ Fully Connected (Dense) Layer

### 🔹 What is it?

* Every neuron connects to every neuron

---

### 🔹 Why do we use it?

* Makes final decisions
* Combines extracted features

---

### 🔹 How do we use it?

* Placed near output
* Often followed by dropout

---

## 9️⃣ Dropout

### 🔹 What is Dropout?

* Randomly disables neurons during training

---

### 🔹 Why do we use it?

* Prevents overfitting
* Forces model to generalize

---

### 🔹 How do we use it?

* Typical values: `0.3 – 0.5`

---

## 🔟 Output Layer

### 🔹 Activation based on task

| Task                  | Activation |
| --------------------- | ---------- |
| Binary classification | Sigmoid    |
| Multi-class           | Softmax    |
| Regression            | Linear     |

---

## 🔢 Output Size Formula (IMPORTANT)

[
\text{Output} = \frac{N - F + 2P}{S} + 1
]

Where:

* `N` = input size
* `F` = filter size
* `P` = padding
* `S` = stride

---

## 🧠 Final One-Line Purpose Summary

> **CNN uses kernels to extract features, stride & padding to control size, pooling to reduce complexity, and dense layers to classify.**

---

