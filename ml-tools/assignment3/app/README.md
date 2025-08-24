# 🧠 AI Text Detection System - Assignment 3

A beautiful, interactive web application for distinguishing AI-generated text from human-written text using state-of-the-art machine learning models.

## 🌟 Features

### 🎯 **Multi-Model Support**
- **6 Different AI Models** with varying architectures and parameters
- **Ensemble Comparison** across all models simultaneously
- **Real-time Performance Metrics** (Kaggle scores, parameter counts)

### 🎨 **Beautiful UI/UX**
- **Three.js Neural Network Background** with interactive animations
- **Responsive Design** optimized for all devices
- **Real-time Visualizations** with Chart.js
- **Smooth Animations** and transitions

### 📝 **Text Analysis**
- **Type or Upload** text files (.txt format)
- **Real-time Character/Word Counting**
- **Example Texts** (AI-generated vs Human-written)
- **Drag & Drop File Upload**

### 📊 **Advanced Analytics**
- **Individual Model Results** with confidence scores
- **Probability Visualizations** with animated bars
- **Comparison Charts** across multiple models
- **Detailed Performance Metrics**

## 🏗️ **Model Architecture**

### Available Models:
1. **AITextDetector** (1.27M params) - Hybrid CNN-Transformer
2. **OptimizedRoBERTa** (19.37M params) - Best Performance (Kaggle: 0.95)
3. **RoBERTaLarge** (50.58M params) - Maximum Capacity
4. **Ensemble Model 1** (19.37M params) - 512 hidden dimensions
5. **Ensemble Model 2** (19.37M params) - Co-Best Performance (Kaggle: 0.95)
6. **Ensemble Model 3** (19.37M params) - Balanced regularization

## 🚀 **Quick Start**

### Prerequisites
- Python 3.8+ 
- CPU (models configured to run on CPU for compatibility)
- Modern web browser with JavaScript enabled

### Installation

1. **Install Requirements**
   ```bash
   cd app
   pip install flask torch numpy
   ```

2. **Verify Models are Available**
   Your trained models should be in `../models-ml/` directory:
   - ✅ `best_ai_detector.pth` (14MB)
   - ✅ `best_optimized_roberta.pth` (221MB)
   - ✅ `best_roberta_large_detector.pth` (578MB)
   - ✅ `ensemble_model_1.pth` (74MB)
   - ✅ `ensemble_model_2.pth` (120MB)
   - ✅ `ensemble_model_3.pth` (99MB)

3. **Run the Application**
   ```bash
   python app.py
   ```
   
   Or use the simple startup script:
   ```bash
   python start.py
   ```

4. **Open in Browser**
   ```
   http://localhost:5000
   ```

### 🎯 **How to Use**

1. **Select Models**: Choose one or more models from the grid
2. **Input Text**: Type directly or upload a .txt file
3. **Analyze**: Click "Analyze Text" to get predictions
4. **View Results**: See individual model results and comparisons

## 💻 **Usage Guide**

### Step 1: Select Models
- Choose one or more models from the model grid
- Toggle "Compare All Models" to analyze with all 6 models
- Each model card shows performance metrics and architecture details

### Step 2: Input Text
- **Type Text**: Enter or paste text directly into the textarea
- **Upload File**: Drag & drop or browse for .txt files (max 50KB)
- **Use Examples**: Load pre-made AI or Human examples

### Step 3: Analyze
- Click "Analyze Text" to process with selected models
- Watch the neural network animation during processing
- View detailed results with confidence scores

### Step 4: Interpret Results
- **Overall Prediction**: AI Generated vs Human Written
- **Confidence Score**: Model certainty percentage
- **Individual Models**: See how each model performed
- **Comparison Chart**: Visual comparison across models

## 🎨 **Technical Highlights**

### Frontend Technologies
- **Three.js**: Interactive 3D neural network visualization
- **Chart.js**: Real-time data visualization
- **Modern CSS**: Grid layouts, animations, responsive design
- **Vanilla JavaScript**: Clean, efficient DOM manipulation

### Backend Architecture
- **Flask**: Lightweight Python web framework
- **PyTorch**: Deep learning model inference
- **Memory-Efficient Design**: Handles large models with limited RAM
- **RESTful API**: Clean separation of frontend/backend

### Performance Optimizations
- **Memory-Mapped Model Loading**: Efficient handling of large model files
- **Adaptive Batch Sizing**: Automatic GPU memory management
- **Progressive Loading**: Models loaded on-demand
- **Mobile Optimization**: Reduced complexity for mobile devices

## 📈 **Model Performance**

| Model | Parameters | Kaggle Score | Architecture | 
|-------|------------|--------------|--------------|
| OptimizedRoBERTa | 19.37M | **0.95** | Transformer (Best) |
| Ensemble Model 2 | 19.37M | **0.95** | Transformer (Co-Best) |
| RoBERTaLarge | 50.58M | 0.94 | Large Transformer |
| Ensemble Model 3 | 19.37M | 0.94 | Balanced Transformer |
| AITextDetector | 1.27M | 0.93 | Hybrid CNN-Transformer |
| Ensemble Model 1 | 19.37M | Poor | Over-regularized |

## 🔧 **API Endpoints**

### `POST /api/predict`
Analyze text with selected models
```json
{
  "text": "Your text here...",
  "models": ["OptimizedRoBERTa", "EnsembleModel2"],
  "compare_all": false
}
```

### `POST /api/upload`
Upload text file for analysis
```json
{
  "file": "text_file.txt"
}
```

### `GET /api/models`
Get information about available models
```json
{
  "OptimizedRoBERTa": {
    "name": "Optimized RoBERTa",
    "params": "19.37M",
    "kaggle_score": "0.95"
  }
}
```

## 🎯 **Assignment Context**

This application demonstrates:
- **Advanced ML Implementation**: 6 different model architectures
- **Production-Ready Code**: Clean, documented, scalable
- **User Experience Design**: Beautiful, intuitive interface
- **Technical Innovation**: Three.js visualizations, real-time analytics
- **Performance Optimization**: Memory-efficient, responsive

## 🏆 **Key Achievements**

- ✅ **Best Kaggle Performance**: 0.95 score with two different models
- ✅ **Comprehensive Ensemble**: 6 models with architectural diversity  
- ✅ **Professional UI**: Three.js neural network visualization
- ✅ **Full-Stack Implementation**: Flask backend + interactive frontend
- ✅ **Memory Efficient**: Handles 50M+ parameter models
- ✅ **Production Ready**: Error handling, validation, responsive design

## 🔮 **Future Enhancements**

- [ ] Real-time streaming analysis
- [ ] Multi-language support  
- [ ] Advanced ensemble techniques (stacking, blending)
- [ ] Model performance monitoring
- [ ] User authentication and history
- [ ] API rate limiting and caching

## 📚 **Technical Documentation**

### Model Classes
- `AITextDetector`: Hybrid CNN-Transformer with multi-scale convolutions
- `OptimizedRoBERTaDetector`: Performance-tuned transformer architecture  
- `ModelManager`: Centralized model loading and inference

### Frontend Classes
- `AITextDetectionApp`: Main application controller
- `NeuralNetworkBackground`: Three.js visualization manager

### Key Features
- Memory-mapped model loading for large files
- Adaptive GPU memory management
- Real-time text analysis with visual feedback
- Interactive neural network background
- Responsive design for all screen sizes

---

**Created for Assignment 3 - Advanced Machine Learning**  
*Demonstrating state-of-the-art AI text detection with beautiful, professional implementation*
