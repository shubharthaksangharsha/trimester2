#!/usr/bin/env python3
"""
Computer Vision Assignment 3 - Interactive Model Visualization App
A Flask web application with Three.js visualization for exploring trained neural networks
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import base64
from PIL import Image
from io import BytesIO
from flask import Flask, render_template, request, jsonify, send_from_directory
from torchvision import datasets, transforms
import logging

# Add parent directory to path to access models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'cv_assignment_3_2025'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
models = {}
model_info = {}
test_dataset = None
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Model Architecture Definitions
class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class WiderNetwork(nn.Module):
    def __init__(self):
        super(WiderNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class DeeperNetwork(nn.Module):
    def __init__(self):
        super(DeeperNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class NetworkWithActivation(nn.Module):
    def __init__(self, activation_func, use_xavier=True):
        super(NetworkWithActivation, self).__init__()
        self.flatten = nn.Flatten()
        
        self.linear1 = nn.Linear(28*28, 512)
        self.activation1 = activation_func()
        self.linear2 = nn.Linear(512, 512) 
        self.activation2 = activation_func()
        self.linear3 = nn.Linear(512, 10)
        
        if use_xavier:
            self.apply(self._xavier_init)
    
    def _xavier_init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.activation1(self.linear1(x))
        x = self.activation2(self.linear2(x))
        logits = self.linear3(x)
        return logits

def load_models():
    """Load all trained models"""
    global models, model_info
    
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    
    # Model architecture mapping
    model_classes = {
        'BaseNetwork': BaseNetwork,
        'WiderNetwork': WiderNetwork,
        'DeeperNetwork': DeeperNetwork,
        'ConvolutionalNetwork': ConvolutionalNetwork,
        'SimpleCNN': SimpleCNN,
        'NetworkWithActivation': NetworkWithActivation
    }
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    
    for model_file in model_files:
        try:
            model_path = os.path.join(models_dir, model_file)
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            model_name = model_file.replace('.pth', '')
            arch_name = checkpoint['model_architecture']
            
            # Create model instance
            if arch_name == 'NetworkWithActivation':
                # Determine activation function from saved data
                activation_name = checkpoint.get('activation_function', 'ReLU')
                activation_map = {
                    'ReLU': nn.ReLU,
                    'Tanh': nn.Tanh,
                    'Sigmoid': nn.Sigmoid
                }
                use_xavier = checkpoint.get('initialization_method', 'Xavier') == 'Xavier'
                model = NetworkWithActivation(activation_map[activation_name], use_xavier)
            else:
                model = model_classes[arch_name]()
            
            # Load model weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            models[model_name] = model
            model_info[model_name] = {
                'architecture': arch_name,
                'description': checkpoint.get('description', ''),
                'experiment': checkpoint.get('experiment', ''),
                'parameters': checkpoint.get('total_parameters', 0),
                'accuracy': checkpoint.get('final_test_accuracy', checkpoint.get('final_accuracy', 0.0)),
                'activation': checkpoint.get('activation_function', 'ReLU'),
                'initialization': checkpoint.get('initialization_method', 'Default'),
                'architecture_type': checkpoint.get('architecture_type', 'MLP')
            }
            
            logger.info(f"Loaded model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_file}: {e}")
    
    logger.info(f"Successfully loaded {len(models)} models")

def load_test_data():
    """Load Fashion-MNIST test dataset"""
    global test_dataset
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    try:
        test_dataset = datasets.FashionMNIST(
            root=data_dir,
            train=False,
            download=False,
            transform=transform
        )
        logger.info(f"Loaded test dataset with {len(test_dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to load test dataset: {e}")

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html', 
                         models=model_info, 
                         class_names=class_names)

@app.route('/model/<model_name>')
def model_detail(model_name):
    """Detailed view for a specific model"""
    if model_name not in models:
        return "Model not found", 404
    
    return render_template('model_detail.html', 
                         model_name=model_name,
                         model_info=model_info[model_name],
                         models=model_info,
                         class_names=class_names)

@app.route('/api/models')
def api_models():
    """API endpoint to get all model information"""
    return jsonify(model_info)

@app.route('/api/predict/<model_name>', methods=['POST'])
def api_predict(model_name):
    """API endpoint for model prediction"""
    if model_name not in models:
        return jsonify({'error': 'Model not found'}), 404
    
    try:
        # Get image data from request
        if 'image' in request.files:
            # Handle file upload
            image_file = request.files['image']
            image = Image.open(image_file.stream).convert('L')
            image = image.resize((28, 28))
        elif 'image_data' in request.json:
            # Handle base64 image data
            image_data = request.json['image_data']
            image_data = base64.b64decode(image_data.split(',')[1])
            image = Image.open(BytesIO(image_data)).convert('L')
            image = image.resize((28, 28))
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Preprocess image
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        
        # Make prediction
        model = models[model_name]
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Convert probabilities to list for JSON serialization
        prob_list = probabilities[0].tolist()
        
        return jsonify({
            'predicted_class': predicted_class,
            'predicted_label': class_names[predicted_class],
            'confidence': confidence,
            'probabilities': prob_list,
            'class_names': class_names
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sample/<int:index>')
def api_sample(index):
    """Get a sample from the test dataset"""
    if test_dataset is None or index >= len(test_dataset):
        return jsonify({'error': 'Invalid index'}), 400
    
    try:
        image, label = test_dataset[index]
        
        # Convert tensor to PIL Image
        image_pil = transforms.ToPILImage()(image.squeeze())
        
        # Convert to base64
        buffer = BytesIO()
        image_pil.save(buffer, format='PNG')
        image_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            'image': f'data:image/png;base64,{image_b64}',
            'label': int(label),
            'class_name': class_names[label]
        })
        
    except Exception as e:
        logger.error(f"Sample error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare', methods=['POST'])
def api_compare():
    """Compare multiple models on the same input"""
    try:
        data = request.json
        model_names = data.get('models', [])
        image_data = data.get('image_data')
        
        if not model_names or not image_data:
            return jsonify({'error': 'Models and image data required'}), 400
        
        # Process image
        image_data = base64.b64decode(image_data.split(',')[1])
        image = Image.open(BytesIO(image_data)).convert('L')
        image = image.resize((28, 28))
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        
        results = {}
        for model_name in model_names:
            if model_name in models:
                model = models[model_name]
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
                
                results[model_name] = {
                    'predicted_class': predicted_class,
                    'predicted_label': class_names[predicted_class],
                    'confidence': confidence,
                    'probabilities': probabilities[0].tolist()
                }
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.route('/favicon.ico')
def favicon():
    """Serve favicon"""
    return send_from_directory('static', 'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':
    print("🚀 Loading Computer Vision Assignment 3 Web App...")
    
    # Load models and data
    load_models()
    load_test_data()
    
    print(f"✅ Loaded {len(models)} models")
    print("🌐 Starting Flask server...")
    print("📱 Open http://localhost:5000 in your browser")
    
    app.run(debug=True, host='0.0.0.0', port=5000)