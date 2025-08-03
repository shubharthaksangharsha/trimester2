# Computer Vision Assignment 3: Deep Learning for Perception Tasks

## 🎯 Overview

This repository contains **Computer Vision Assignment 3** focusing on deep learning for perception tasks, specifically Fashion-MNIST classification. The project includes both the core assignment implementation and an interactive Flask web application for model visualization and testing.

**Author:** Shubharthak Sangharasha  
**Course:** Computer Vision  
**Institution:** [Your University]  
**Academic Year:** 2024-2025

---

## 📁 Project Structure

```
assignment3/
├── 📊 Assignment_3_Notebook.ipynb          # Main assignment notebook
├── 📁 data/                                # Fashion-MNIST dataset
│   └── FashionMNIST/                       # Downloaded dataset files
├── 📁 models/                              # Trained model files
│   ├── q15_base_network.pth               # Q1.5 Base MLP
│   ├── q15_wider_network.pth              # Q1.5 Wider MLP
│   ├── q15_deeper_network.pth             # Q1.5 Deeper MLP
│   ├── q17_simplecnn.pth                  # Q1.7 Simple CNN
│   ├── q17_fullcnn.pth                    # Q1.7 Full CNN
│   ├── q17_mlpbase.pth                    # Q1.7 MLP Base
│   ├── q2_relu_xavier.pth                 # Q2 ReLU (Xavier)
│   ├── q2_tanh_xavier.pth                 # Q2 Tanh (Xavier)
│   ├── q2_sigmoid_xavier.pth              # Q2 Sigmoid (Xavier)
│   ├── q2_sigmoid_default.pth             # Q2 Sigmoid (Default)
│   └── comprehensive_model_summary.txt    # Model performance summary
└── 📁 app/                                 # Flask web application
    ├── app.py                             # Main Flask application
    ├── templates/                         # HTML templates
    ├── static/                            # CSS, JS, and assets
    └── requirements.txt                   # Python dependencies
```

---

## 🧠 Assignment Details

### **Question 1: Neural Network Training and Analysis (40 marks)**

#### **Q1.1: Dataset Exploration (5 marks)**
- **Objective:** Explore the Fashion-MNIST dataset
- **Implementation:** Extract and display 3 sample images with their labels
- **Key Learning:** Understanding dataset structure and image preprocessing

#### **Q1.2: Learning Rate Analysis (10 marks)**
- **Objective:** Train neural networks with different learning rates
- **Implementation:** 
  - Train `NeuralNetwork` (MLP) with learning rates: 0.001, 0.01, 0.1
  - Track training/test loss and accuracy over 10 epochs
  - Plot learning curves to analyze convergence
- **Key Findings:** Optimal learning rate identification and convergence patterns

#### **Q1.3: Convergence Analysis (10 marks)**
- **Objective:** Extended training to study convergence behavior
- **Implementation:**
  - Extend training to 50 epochs for the same learning rates
  - Implement `check_convergence()` function
  - Track best accuracy and convergence epoch
  - Analyze early stopping criteria
- **Key Findings:** Convergence patterns and optimal training duration

#### **Q1.5: Architecture Comparison (10 marks)**
- **Objective:** Compare different MLP architectures
- **Implementation:**
  - `BaseNetwork`: 2 hidden layers (512 units each)
  - `WiderNetwork`: 2 hidden layers (1024 units each)
  - `DeeperNetwork`: 4 hidden layers (512 units each)
  - Train all architectures for 30 epochs
  - Compare parameter count vs. accuracy
- **Key Findings:** Depth vs. width trade-offs in neural networks

#### **Q1.6: Gradient Analysis (5 marks)**
- **Objective:** Analyze gradient behavior during training
- **Implementation:**
  - Implement `compute_gradient_norm()` function
  - Sample gradient norms every 50 batches over 50 epochs
  - Plot gradient norm curves and analyze vanishing/exploding gradients
- **Key Findings:** Gradient flow analysis and training stability

#### **Q1.7: CNN vs MLP Comparison (10 marks)**
- **Objective:** Compare Convolutional Neural Networks with Multi-Layer Perceptrons
- **Implementation:**
  - `SimpleCNN`: Basic convolutional architecture
  - `ConvolutionalNetwork` (Full CNN): Advanced CNN with multiple conv layers
  - `BaseNetwork`: MLP baseline
  - Train all models for 30 epochs
  - Compare accuracy and parameter efficiency
- **Key Findings:** CNN superiority for image classification tasks

### **Question 2: Activation Functions and Initialization (20 marks)**

#### **Q2: Activation Function Analysis (20 marks)**
- **Objective:** Study the impact of activation functions and initialization methods
- **Implementation:**
  - **Activation Functions:** ReLU, Tanh, Sigmoid
  - **Initialization Methods:** Xavier Uniform, Default PyTorch
  - Train models with different combinations
  - Analyze performance differences
- **Key Findings:** 
  - ReLU consistently outperforms Tanh and Sigmoid
  - Xavier initialization significantly improves Sigmoid performance
  - Proper initialization is crucial for saturating activations

---

## 🌐 Interactive Web Application

### **Features**

#### **🎨 Beautiful 3D Visualizations**
- **Background Wallpaper:** Animated 3D neural networks using Three.js
- **Hero Visualizer:** Single neural network in unique green color
- **Non-interactive Background:** Multiple animating networks for aesthetic appeal

#### **📊 Model Dashboard**
- **Statistics Overview:** Model count, best accuracy, experiments, max parameters
- **Performance Charts:** Interactive bar charts showing model comparisons
- **Key Insights:** Analysis of CNN vs MLP, activation functions, initialization

#### **🔍 Model Explorer**
- **Organized Tabs:** Q1.5 Architecture, Q1.7 CNN vs MLP, Q2 Activation Functions
- **Model Cards:** Individual cards with accuracy, parameters, and architecture details
- **View Model Details:** Sliding drawer with comprehensive model information

#### **🎯 Interactive Demo**
- **Image Upload:** Drag & drop or click to upload images
- **Drawing Interface:** Canvas-based drawing with optional label input
- **Random Samples:** Load random Fashion-MNIST samples
- **Multi-Model Prediction:** Compare predictions across all models
- **Prediction Modes:**
  - Selected Models: Choose specific models to compare
  - Quick Predict: Use best performing model
  - Compare All: Test all models simultaneously

#### **🌙 Dark/Light Mode**
- **Theme Toggle:** Beautiful animated theme switcher
- **Responsive Design:** Works on all screen sizes
- **Modern UI:** Bootstrap 5 with custom styling

---

## 🚀 Quick Start

### **Prerequisites**
```bash
# Python 3.8+ required
python --version

# Install required packages
pip install torch torchvision flask pillow numpy matplotlib seaborn
```

### **Running the Assignment Notebook**
```bash
# Navigate to assignment directory
cd assignment3

# Start Jupyter notebook
jupyter notebook

# Open Assignment_3_Notebook.ipynb
```

### **Running the Web Application**
```bash
# Navigate to app directory
cd app

# Install Flask dependencies
pip install -r requirements.txt

# Start the Flask server
python app.py

# Open browser to http://localhost:5000
```

### **Model Training (if needed)**
```bash
# Run the notebook cells to train models
# Models will be saved to models/ directory
# Pre-trained models are already included
```

---

## 📈 Model Performance Summary

| Model | Architecture | Parameters | Accuracy | Experiment |
|-------|-------------|------------|----------|------------|
| q15_base_network | MLP (2×512) | 669,706 | 86.59% | Q1.5 Base |
| q15_wider_network | MLP (2×1024) | 1,863,690 | 86.95% | Q1.5 Wider |
| q15_deeper_network | MLP (4×512) | 1,195,018 | 87.64% | Q1.5 Deeper |
| q17_simplecnn | Simple CNN | 846,922 | 88.23% | Q1.7 CNN |
| q17_fullcnn | Full CNN | 1,847,306 | 89.31% | Q1.7 CNN |
| q17_mlpbase | MLP Base | 669,706 | 86.59% | Q1.7 MLP |
| q2_relu_xavier | ReLU (Xavier) | 669,706 | 87.12% | Q2 ReLU |
| q2_tanh_xavier | Tanh (Xavier) | 669,706 | 85.89% | Q2 Tanh |
| q2_sigmoid_xavier | Sigmoid (Xavier) | 669,706 | 84.67% | Q2 Sigmoid |
| q2_sigmoid_default | Sigmoid (Default) | 669,706 | 72.34% | Q2 Sigmoid |

---

## 🔧 Technical Implementation

### **Neural Network Architectures**

#### **MLP Architectures**
```python
# Base Network (Q1.5)
class BaseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 10)
        )

# Wider Network (Q1.5)
class WiderNetwork(nn.Module):
    # Same as Base but with 1024 units per layer

# Deeper Network (Q1.5)
class DeeperNetwork(nn.Module):
    # 4 hidden layers with 512 units each
```

#### **CNN Architectures**
```python
# Simple CNN (Q1.7)
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

# Full CNN (Q1.7)
class ConvolutionalNetwork(nn.Module):
    # More complex CNN with multiple conv layers
```

#### **Activation Function Network (Q2)**
```python
class NetworkWithActivation(nn.Module):
    def __init__(self, activation_func, use_xavier=True):
        super().__init__()
        self.activation = activation_func
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            activation_func(),
            nn.Linear(512, 512),
            activation_func(),
            nn.Linear(512, 10)
        )
        if use_xavier:
            self._xavier_init(self.linear_stack)
```

### **Training Configuration**
- **Optimizer:** Stochastic Gradient Descent (SGD)
- **Loss Function:** Cross-Entropy Loss
- **Batch Size:** 64
- **Learning Rate:** 0.01 (default), varies for experiments
- **Epochs:** 10-50 depending on experiment
- **Dataset:** Fashion-MNIST (60K train, 10K test)

---

## 🎨 Web Application Features

### **Frontend Technologies**
- **HTML5:** Semantic markup and structure
- **CSS3:** Modern styling with gradients and animations
- **Bootstrap 5:** Responsive grid system and components
- **JavaScript (ES6+):** Interactive functionality
- **Three.js:** 3D neural network visualizations
- **Chart.js:** Performance charts and graphs
- **GSAP:** Smooth animations and transitions

### **Backend Technologies**
- **Flask:** Python web framework
- **PyTorch:** Deep learning framework
- **PIL (Pillow):** Image processing
- **NumPy:** Numerical computations
- **Jinja2:** Template engine

### **Key Features**
1. **Real-time Model Loading:** Dynamic loading of trained models
2. **Interactive Predictions:** Live model inference on uploaded/drawn images
3. **Performance Analytics:** Comprehensive model comparison charts
4. **Responsive Design:** Works on desktop, tablet, and mobile
5. **Dark/Light Theme:** User preference with smooth transitions
6. **Error Handling:** Robust error handling with user-friendly messages

---

## 📊 Results and Analysis

### **Key Findings**

#### **Q1.5: Architecture Comparison**
- **Deeper networks** achieve slightly better accuracy (87.64%) but with diminishing returns
- **Wider networks** show marginal improvement over base architecture
- **Parameter efficiency** decreases with increased model complexity

#### **Q1.7: CNN vs MLP**
- **CNNs significantly outperform MLPs** with fewer parameters
- **Full CNN** achieves best accuracy (89.31%) with 1.8M parameters
- **Simple CNN** provides good balance of accuracy (88.23%) and efficiency
- **Parameter sharing** in CNNs leads to better generalization

#### **Q2: Activation Functions**
- **ReLU** consistently achieves best performance (87.12%)
- **Tanh** performs moderately well (85.89%)
- **Sigmoid** suffers from vanishing gradients (84.67% with Xavier, 72.34% with default)
- **Xavier initialization** is crucial for sigmoid activation

### **Performance Insights**
1. **CNNs are superior** for image classification tasks
2. **ReLU activation** is optimal for deep networks
3. **Proper initialization** significantly impacts training success
4. **Depth vs. width** trade-offs depend on task complexity

---

## 🛠️ Development and Deployment

### **Local Development**
```bash
# Clone repository
git clone [repository-url]
cd assignment3

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r app/requirements.txt

# Run development server
cd app
python app.py
```

### **Production Deployment**
```bash
# Using Gunicorn (recommended)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Using Docker
docker build -t cv-assignment3 .
docker run -p 5000:5000 cv-assignment3
```

### **Environment Variables**
```bash
# Optional: Set Flask environment
export FLASK_ENV=development
export FLASK_DEBUG=1
```

---

## 📝 Assignment Submission

### **Files to Submit**
1. **Assignment_3_Notebook.ipynb** - Complete notebook with all questions
2. **models/** - All trained model files (.pth)
3. **data/** - Fashion-MNIST dataset
4. **app/** - Flask web application (bonus)
5. **README.md** - This documentation

### **Grading Criteria**
- **Q1.1-Q1.3:** Learning rate and convergence analysis (25 marks)
- **Q1.5:** Architecture comparison (10 marks)
- **Q1.6:** Gradient analysis (5 marks)
- **Q1.7:** CNN vs MLP comparison (10 marks)
- **Q2:** Activation function analysis (20 marks)
- **Code Quality:** Clean, well-documented code (10 marks)
- **Analysis:** Comprehensive insights and conclusions (20 marks)

---

## 🤝 Contributing

This is an academic assignment, but suggestions for improvements are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is for educational purposes. The Fashion-MNIST dataset is available under the MIT License.

---

## 👨‍💻 Author

**Shubharthak Sangharasha**
- **Portfolio:** [devshubh.me](https://devshubh.me)
- **GitHub:** [github.com/shubharthaksangharsha](https://github.com/shubharthaksangharsha)
- **LinkedIn:** [linkedin.com/in/shubharthaksangharsha](https://linkedin.com/in/shubharthaksangharsha)

---

## 🙏 Acknowledgments

- **Fashion-MNIST Dataset:** Created by Zalando Research
- **PyTorch:** Deep learning framework
- **Three.js:** 3D graphics library
- **Bootstrap:** CSS framework
- **Chart.js:** Charting library

---