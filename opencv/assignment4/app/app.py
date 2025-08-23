#!/usr/bin/env python3
"""
🌸 Flower Classification Web Application 🌸
Advanced AI-Powered Flower Recognition System
Created for Computer Vision 2025 Assignment
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import base64
import io
from datetime import datetime
import random
import glob

app = Flask(__name__)
app.config['SECRET_KEY'] = 'flower_classification_2025_secret'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

FLOWER_CLASSES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

MODEL_CONFIGS = {
    'resnet18_baseline': {
        'name': 'ResNet18 Baseline',
        'description': 'Original transfer learning model',
        'accuracy': 91.05,
        'gflops': 1.82614,
        'efficiency': 49.86,
        'parameters': '11.3M',
        'file': 'flower_classification_final.pth',
        'color': '#FF6B6B',
        'icon': '🌹'
    },
    'efficientnet_fine_tuned': {
        'name': 'EfficientNet Fine-tuned ⭐',
        'description': 'Best performing model with 94.14% accuracy',
        'accuracy': 94.14,
        'gflops': 0.40165,
        'efficiency': 234.49,
        'parameters': '5.0M',
        'file': 'flower_classification_FINE_TUNED.pth',
        'color': '#4ECDC4',
        'icon': '🏆'
    },
    'efficientnet_tta': {
        'name': 'EfficientNet + TTA',
        'description': 'Test Time Augmentation enhanced model',
        'accuracy': 93.98,
        'gflops': 0.40165,
        'efficiency': 233.99,
        'parameters': '5.0M',
        'file': 'flower_classification_TTA_ENHANCED.pth',
        'color': '#45B7D1',
        'icon': '🚀'
    },
    'efficientnet_ultimate': {
        'name': 'Ultimate Optimized',
        'description': 'Final optimized model with all enhancements',
        'accuracy': 93.98,
        'gflops': 0.40165,
        'efficiency': 233.99,
        'parameters': '5.0M',
        'file': 'flower_classification_ULTIMATE_OPTIMIZED.pth',
        'color': '#96CEB4',
        'icon': '💎'
    }
}

loaded_models = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model classes (copy from your existing code)
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_out = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv(x_out)
        return self.sigmoid(x_out)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class EfficientNetFlowerClassifier(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(EfficientNetFlowerClassifier, self).__init__()
        
        try:
            from torchvision.models import efficientnet_b0
            self.backbone = efficientnet_b0(pretrained=pretrained)
        except ImportError:
            from torchvision.models import mobilenet_v3_small
            self.backbone = mobilenet_v3_small(pretrained=pretrained)
        
        if hasattr(self.backbone, 'classifier'):
            if isinstance(self.backbone.classifier, nn.Sequential):
                in_features = self.backbone.classifier[-1].in_features
                self.backbone.classifier = nn.Identity()
            else:
                in_features = self.backbone.classifier.in_features
                self.backbone.classifier = nn.Identity()
        else:
            in_features = self.backbone.features[-1].out_channels if hasattr(self.backbone, 'features') else 1280
            if hasattr(self.backbone, 'classifier'):
                self.backbone.classifier = nn.Identity()
        
        self.attention = CBAM(in_features)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        if hasattr(self.backbone, 'features'):
            features = self.backbone.features(x)
        else:
            features = x
            for name, module in self.backbone.named_children():
                if name != 'classifier':
                    features = module(features)
        
        features = self.attention(features)
        features = self.global_pool(features)
        features = torch.flatten(features, 1)
        outputs = self.classifier(features)
        return outputs

class FlowerClassificationModel(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(FlowerClassificationModel, self).__init__()
        
        self.backbone = models.resnet18(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        for layer in self.backbone.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        return self.backbone(x)

def load_model(model_key):
    if model_key in loaded_models:
        return loaded_models[model_key]
    
    config = MODEL_CONFIGS[model_key]
    model_path = os.path.join('..', 'models', config['file'])
    
    try:
        if 'resnet18' in model_key.lower():
            model = FlowerClassificationModel(num_classes=5, pretrained=False)
        else:
            model = EfficientNetFlowerClassifier(num_classes=5, pretrained=False)
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        loaded_models[model_key] = model
        print(f"✅ Successfully loaded {config['name']}")
        return model
        
    except Exception as e:
        print(f"❌ Error loading {config['name']}: {str(e)}")
        return None

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor.to(device)

def predict_image(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = outputs.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence, probabilities[0].cpu().numpy()

# Flower browser functionality
def get_flower_images():
    """Get sample images from each flower category"""
    flower_images = {}
    flowers_path = '../flowers'
    
    for flower_class in FLOWER_CLASSES:
        class_path = os.path.join(flowers_path, flower_class)
        if os.path.exists(class_path):
            # Get all jpg images from the class folder
            image_files = glob.glob(os.path.join(class_path, '*.jpg'))
            # Randomly select up to 10 images for display
            selected_images = random.sample(image_files, min(10, len(image_files)))
            flower_images[flower_class] = [
                {
                    'filename': os.path.basename(img),
                    'path': os.path.relpath(img, '.').replace('\\', '/'),
                    'class': flower_class
                }
                for img in selected_images
            ]
    
    return flower_images

@app.route('/flowers/<class_name>/<filename>')
def serve_flower_image(class_name, filename):
    """Serve flower images from the dataset"""
    if class_name not in FLOWER_CLASSES:
        return "Invalid flower class", 404
    
    flowers_path = os.path.join('..', 'flowers', class_name)
    return send_from_directory(flowers_path, filename)

@app.route('/api/flower-images')
def get_flower_images_api():
    """API endpoint to get flower images for browsing"""
    try:
        images = get_flower_images()
        return jsonify(images)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict-flower-image', methods=['POST'])
def predict_flower_image():
    """Predict using a selected flower image from the dataset"""
    try:
        data = request.get_json()
        model_key = data.get('model', 'efficientnet_fine_tuned')
        flower_class = data.get('class')
        filename = data.get('filename')
        
        if not flower_class or not filename:
            return jsonify({'error': 'Missing class or filename'}), 400
        
        model = load_model(model_key)
        if model is None:
            return jsonify({'error': 'Failed to load model'}), 500
        
        # Load image from flowers directory
        image_path = os.path.join('..', 'flowers', flower_class, filename)
        if not os.path.exists(image_path):
            return jsonify({'error': 'Image not found'}), 404
        
        image = Image.open(image_path)
        image_tensor = preprocess_image(image)
        predicted_class, confidence, all_probabilities = predict_image(model, image_tensor)
        
        results = {
            'predicted_class': FLOWER_CLASSES[predicted_class],
            'confidence': float(confidence),
            'true_class': flower_class,
            'correct_prediction': predicted_class == FLOWER_CLASSES.index(flower_class),
            'all_predictions': [
                {
                    'class': FLOWER_CLASSES[i],
                    'probability': float(prob)
                }
                for i, prob in enumerate(all_probabilities)
            ],
            'model_used': MODEL_CONFIGS[model_key]['name'],
            'model_accuracy': MODEL_CONFIGS[model_key]['accuracy'],
            'image_info': {
                'filename': filename,
                'true_class': flower_class,
                'image_url': f'/flowers/{flower_class}/{filename}'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare-all-models', methods=['POST'])
def compare_all_models():
    """Compare predictions across all available models"""
    try:
        comparison_results = {}
        
        # Get image data (same logic as predict endpoint)
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            image = Image.open(file.stream)
            
        elif 'canvas_data' in request.form:
            canvas_data = request.form['canvas_data']
            image_data = canvas_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
        elif 'flower_image_path' in request.form:
            flower_image_path = request.form['flower_image_path']
            full_path = os.path.join('..', 'flowers', flower_image_path)
            if not os.path.exists(full_path):
                return jsonify({'error': 'Flower image not found'}), 404
            image = Image.open(full_path)
            
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Preprocess image once
        image_tensor = preprocess_image(image)
        
        # Test with all available models
        for model_key, config in MODEL_CONFIGS.items():
            try:
                model = load_model(model_key)
                if model is not None:
                    predicted_class, confidence, all_probabilities = predict_image(model, image_tensor)
                    
                    comparison_results[model_key] = {
                        'model_name': config['name'],
                        'predicted_class': FLOWER_CLASSES[predicted_class],
                        'confidence': float(confidence),
                        'accuracy': config['accuracy'],
                        'efficiency': config['efficiency'],
                        'gflops': config['gflops'],
                        'parameters': config['parameters'],
                        'all_predictions': [
                            {
                                'class': FLOWER_CLASSES[i],
                                'probability': float(prob)
                            }
                            for i, prob in enumerate(all_probabilities)
                        ],
                        'icon': config['icon'],
                        'color': config['color']
                    }
                else:
                    comparison_results[model_key] = {
                        'model_name': config['name'],
                        'error': 'Failed to load model',
                        'accuracy': config['accuracy'],
                        'efficiency': config['efficiency'],
                        'icon': config['icon'],
                        'color': config['color']
                    }
            except Exception as e:
                comparison_results[model_key] = {
                    'model_name': config['name'],
                    'error': str(e),
                    'accuracy': config['accuracy'],
                    'efficiency': config['efficiency'],
                    'icon': config['icon'],
                    'color': config['color']
                }
        
        # Calculate consensus and statistics
        valid_predictions = [result for result in comparison_results.values() if 'predicted_class' in result]
        
        if valid_predictions:
            # Find most common prediction (consensus)
            class_counts = {}
            for result in valid_predictions:
                pred_class = result['predicted_class']
                class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
            
            consensus_class = max(class_counts, key=class_counts.get)
            consensus_count = class_counts[consensus_class]
            consensus_percentage = (consensus_count / len(valid_predictions)) * 100
            
            # Calculate average confidence for consensus class
            consensus_confidences = [
                result['confidence'] for result in valid_predictions 
                if result['predicted_class'] == consensus_class
            ]
            avg_consensus_confidence = sum(consensus_confidences) / len(consensus_confidences)
            
            summary = {
                'total_models': len(MODEL_CONFIGS),
                'successful_predictions': len(valid_predictions),
                'consensus_class': consensus_class,
                'consensus_percentage': consensus_percentage,
                'consensus_confidence': avg_consensus_confidence,
                'agreement_level': 'High' if consensus_percentage >= 75 else 'Medium' if consensus_percentage >= 50 else 'Low'
            }
        else:
            summary = {
                'total_models': len(MODEL_CONFIGS),
                'successful_predictions': 0,
                'consensus_class': None,
                'consensus_percentage': 0,
                'agreement_level': 'None'
            }
        
        return jsonify({
            'comparison_results': comparison_results,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html', 
                         models=MODEL_CONFIGS, 
                         flower_classes=FLOWER_CLASSES)

@app.route('/api/models')
def get_models():
    return jsonify(MODEL_CONFIGS)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        model_key = request.form.get('model', 'efficientnet_fine_tuned')
        
        model = load_model(model_key)
        if model is None:
            return jsonify({'error': 'Failed to load model'}), 500
        
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            image = Image.open(file.stream)
            
        elif 'canvas_data' in request.form:
            canvas_data = request.form['canvas_data']
            image_data = canvas_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
        elif 'flower_image_path' in request.form:
            flower_image_path = request.form['flower_image_path']
            full_path = os.path.join('..', 'flowers', flower_image_path)
            if not os.path.exists(full_path):
                return jsonify({'error': 'Flower image not found'}), 404
            image = Image.open(full_path)
            
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        image_tensor = preprocess_image(image)
        predicted_class, confidence, all_probabilities = predict_image(model, image_tensor)
        
        results = {
            'predicted_class': FLOWER_CLASSES[predicted_class],
            'confidence': float(confidence),
            'all_predictions': [
                {
                    'class': FLOWER_CLASSES[i],
                    'probability': float(prob)
                }
                for i, prob in enumerate(all_probabilities)
            ],
            'model_used': MODEL_CONFIGS[model_key]['name'],
            'model_accuracy': MODEL_CONFIGS[model_key]['accuracy'],
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Production configuration
if os.environ.get('FLASK_ENV') == 'production':
    app.config.from_object('production_config.ProductionConfig')
    print("🏭 Running in production mode")

if __name__ == '__main__':
    print("🌸 Starting Flower Classification Web Application 🌸")
    print("Loading models...")
    
    # Only load the best model in production
    load_model('efficientnet_fine_tuned')
    
    print("🚀 Application ready!")
    
    # Use different settings for production vs development
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    app.run(debug=debug_mode, host='0.0.0.0', port=5003)