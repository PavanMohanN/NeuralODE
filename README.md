![image](https://github.com/user-attachments/assets/79d6a5f9-30c3-4d35-a5bb-a167de5ae883)


# **Catastrophic Forgetting Simulation with Neural Networks and Neural ODEs**

This project demonstrates the phenomenon of **catastrophic forgetting** in neural networks and compares it with **Neural ODEs (Ordinary Differential Equations)** as an alternative to mitigate catastrophic forgetting. The goal is to simulate how these models behave when trained sequentially on two tasks, evaluate their performance before and after learning a new task, and compare their abilities to retain knowledge of the previous task.

## **Project Overview**

### **Catastrophic Forgetting:**
- **Catastrophic forgetting** occurs when a model, after learning a new task, forgets what it learned from previous tasks.
- In this project, we simulate two tasks: **Task 1** and **Task 2**. After training on **Task 1**, the model is exposed to **Task 2**, and we evaluate how well it retains the knowledge of **Task 1**.

### **Methods Compared:**
- **Neural Network (NN)**: A simple feedforward neural network trained using backpropagation.
- **Neural ODE**: A model based on continuous dynamics, represented by differential equations, which is expected to handle catastrophic forgetting better.

---

## **Dataset Creation**

We generate two synthetic datasets for the tasks using the `sklearn.datasets.make_classification` function. Each task is created using specific parameters to simulate different difficulties.

### **Task 1: Linearly Separable Dataset**
- The first task is generated with a linearly separable dataset. This means the data points from different classes can be separated by a straight line (or hyperplane).
- Parameters for Task 1:
  - **n_samples = 1000**: 1000 data points.
  - **n_features = 2**: Two features for classification.
  - **n_classes = 2**: Two distinct classes.
  - **n_informative = 2**: Both features are informative for classification.
  - **class_sep = 2**: The classes are well-separated in the feature space.

### **Task 2: Non-Linearly Separable Dataset (with noise)**
- The second task is created by adding noise to the original Task 1 dataset. This noise makes the data points less predictable and non-linearly separable.
- Parameters for Task 2:
  - **n_samples = 1000**: Same number of data points as Task 1.
  - **n_features = 2**: Same number of features.
  - **n_classes = 2**: Same two classes.
  - **noise = np.random.randn(n_samples, n_features) * 2**: Gaussian noise is added to make the task more difficult and non-linear.

### **Task 1 and Task 2 Overview**
1. **Task 1**: Simple, linearly separable dataset.
2. **Task 2**: More complex dataset with added noise, making it harder for the model to learn.

---

## **Model Definitions**

We define two models to compare their abilities to learn the tasks and retain knowledge:

### **1. Simple Neural Network (NN)**

The simple feedforward neural network consists of:
- **Input Layer**: Accepts 2 features (from Task 1 and Task 2 datasets).
- **Hidden Layer**: A fully connected layer with 64 units and ReLU activation.
- **Output Layer**: A fully connected layer that outputs two values (for binary classification).

```python
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

### **2. Neural ODE (Neural Ordinary Differential Equation)**

The Neural ODE is defined similarly to the Neural Network, but its forward pass is based on solving differential equations using the `odeint` solver from the `torchdiffeq` library. The key difference is that Neural ODEs model the network as a continuous dynamical system.

```python
class NeuralODE(nn.Module):
    def __init__(self):
        super(NeuralODE, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, t, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

- **ODE Solver**: `odeint` is used to integrate the system over a period, effectively applying continuous learning to the model's parameters.

---

## **Training Procedure**

We train both models sequentially on **Task 1** and **Task 2**. The steps are as follows:

1. **Train on Task 1**:
   - The models are first trained on **Task 1** (linearly separable data).
   - We use **CrossEntropyLoss** as the loss function and **Adam optimizer** for optimization.

2. **Evaluate on Task 1 (before Task 2)**:
   - After training on Task 1, we evaluate the models on **Task 1** to get a baseline accuracy.

3. **Train on Task 2**:
   - The models are then trained on **Task 2** (non-linearly separable data with noise).
   - The same training procedure is followed (Adam optimizer, CrossEntropyLoss).

4. **Evaluate on Task 1 and Task 2 (after Task 2)**:
   - After training on Task 2, we evaluate the models on both **Task 1** and **Task 2** to observe:
     - **Performance on Task 1**: Did the model forget what it learned from Task 1 (catastrophic forgetting)?
     - **Performance on Task 2**: How well did the model learn the new task?

---

## **Evaluation**

### **Metrics**
- **Accuracy**: The accuracy is calculated for both tasks before and after training on Task 2. We use the formula:
  
![Accuracy Equation](https://latex.codecogs.com/svg.latex?\textcolor{darkblue}{\text{Accuracy}=\frac{\text{Number%20of%20Correct%20Predictions}}{\text{Total%20Number%20of%20Predictions}}})




### **Results Analysis**
1. **Before Task 2**:
   - Both models are trained on Task 1 and evaluated for accuracy.
   - **Expected result**: Models should perform well on Task 1.

2. **After Task 2**:
   - Both models are trained on Task 2 and evaluated again on Task 1 and Task 2.
   - **Catastrophic Forgetting**: If the accuracy on Task 1 drops significantly after training on Task 2, catastrophic forgetting has occurred.

### **Key Insights from Evaluation**:
- **Neural Network**: Shows a large drop in Task 1 accuracy after learning Task 2, indicating significant **catastrophic forgetting**.
- **Neural ODE**: Demonstrates **less catastrophic forgetting**, with a smaller drop in Task 1 accuracy after learning Task 2, suggesting it may be more resistant to catastrophic forgetting.

---

## **Conclusion**

- **Neural ODE**: Shows better **retention** of knowledge across tasks, with only a small drop in performance on Task 1 after learning Task 2. This suggests that Neural ODEs handle catastrophic forgetting better than traditional neural networks.
- **Simple Neural Network (NN)**: Struggles with catastrophic forgetting. The performance on Task 1 drops significantly after learning Task 2.

### **Next Steps**
- **Continual Learning**: Explore techniques like **Elastic Weight Consolidation (EWC)** or **Progressive Networks** to further improve the retention in neural networks.
- **Other Architectures**: Experiment with other types of **Neural ODEs** or hybrid architectures to reduce catastrophic forgetting.

---

## **How to Run**

1. **Install Dependencies**:
   - Install the required libraries using `pip`:
     ```bash
     pip install torch torchdiffeq sklearn matplotlib
     ```

2. **Run the Script**:
   - Save the script as `catastrophic_forgetting_simulation.py` and run it:
     ```bash
     python catastrophic_forgetting_simulation.py
     ```

---

## **References**
- [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)
- [TorchDiffeq: A library for solving differential equations](https://github.com/rtqichen/torchdiffeq)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
  
---


`Created in Dec 2024`

`@author: Pavan Mohan Neelamraju`

`Affiliation: Indian Institute of Technology Madras`

**Email**: npavanmohan3@gmail.com

**Personal Website 🔴🔵**: [[pavanmohan.netlify.app](https://pavanmohan.netlify.app/)]


---
