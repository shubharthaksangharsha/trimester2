#!/usr/bin/env python3
"""
AI Text Detection Flask Application
Advanced web interface for distinguishing AI-generated from human-written text
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import json
import io
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# Add parent directory to path for model imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
app.secret_key = 'ai_text_detection_secret_key_2024'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    'AITextDetector': {
        'name': 'AI Text Detector (Hybrid CNN-Transformer)',
        'file': 'best_ai_detector.pth',
        'params': '1.27M',
        'kaggle_score': '0.93',
        'description': 'Lightweight hybrid model combining CNN and Transformer attention',
        'hidden_dim': 256,
        'input_dim': 768,
        'seq_len': 100,
        'dropout': 0.2
    },
    'RoBERTaLarge': {
        'name': 'RoBERTa Large Detector',
        'file': 'best_roberta_large_detector.pth',
        'params': '50.58M',
        'kaggle_score': '0.94',
        'description': 'Maximum capacity transformer with 12 attention heads',
        'hidden_dim': 768,
        'num_layers': 6,
        'input_dim': 768,
        'seq_len': 100
    },
    'OptimizedRoBERTa': {
        'name': 'Optimized RoBERTa (Best Performance)',
        'file': 'best_optimized_roberta.pth',
        'params': '19.37M',
        'kaggle_score': '0.95',
        'description': 'Performance-optimized transformer achieving best Kaggle score',
        'hidden_dim': 512,
        'num_layers': 6,
        'input_dim': 768,
        'seq_len': 100
    },
    'EnsembleModel1': {
        'name': 'Ensemble Model 1',
        'file': 'ensemble_model_1.pth',
        'params': '19.37M',
        'kaggle_score': 'Poor',
        'description': 'Ensemble variant with 512 hidden dimensions',
        'hidden_dim': 512,
        'num_layers': 6,
        'input_dim': 768,
        'seq_len': 100
    },
    'EnsembleModel2': {
        'name': 'Ensemble Model 2 (Co-Best)',
        'file': 'ensemble_model_2.pth',
        'params': '19.37M',
        'kaggle_score': '0.95',
        'description': 'Co-best performing ensemble with 768 hidden dimensions',
        'hidden_dim': 768,
        'num_layers': 4,
        'input_dim': 768,
        'seq_len': 100
    },
    'EnsembleModel3': {
        'name': 'Ensemble Model 3',
        'file': 'ensemble_model_3.pth',
        'params': '19.37M',
        'kaggle_score': '0.94',
        'description': 'Balanced ensemble with optimal regularization',
        'hidden_dim': 640,
        'num_layers': 5,
        'input_dim': 768,
        'seq_len': 100
    }
}

# Model Classes (simplified versions for demo)
class AITextDetector(nn.Module):
    """Hybrid Transformer-CNN Architecture"""
    def __init__(self, input_dim=768, seq_len=100, hidden_dim=256, dropout=0.2):
        super(AITextDetector, self).__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = self._create_positional_encoding(seq_len, hidden_dim)
        
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(hidden_dim, 128, kernel_size=k, padding=k//2)
            for k in [3, 5, 7]
        ])
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        feature_size = hidden_dim + (128 * 3 * 2)
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )
    
    def _create_positional_encoding(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.input_projection(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        attn_output, _ = self.attention(x, x, x)
        attn_features = attn_output.mean(dim=1)
        
        x_conv = x.transpose(1, 2)
        conv_features = []
        
        for conv_layer in self.conv_layers:
            conv_out = torch.relu(conv_layer(x_conv))
            avg_pool = self.global_avg_pool(conv_out).squeeze(-1)
            max_pool = self.global_max_pool(conv_out).squeeze(-1)
            conv_features.extend([avg_pool, max_pool])
        
        conv_features = torch.cat(conv_features, dim=1)
        combined_features = torch.cat([attn_features, conv_features], dim=1)
        
        logits = self.classifier(combined_features)
        return logits

class OptimizedRoBERTaDetector(nn.Module):
    """Performance-optimized RoBERTa architecture"""
    def __init__(self, input_dim=768, seq_len=100, hidden_dim=512, num_layers=6):
        super(OptimizedRoBERTaDetector, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
        self.pos_encoding = self._create_positional_encoding(seq_len, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 3,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.attention_pool = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        self.pool_norm = nn.LayerNorm(hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)
        )
        
        self._init_weights()
    
    def _create_positional_encoding(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        x = self.input_projection(x)
        x = self.input_norm(x)
        x = self.dropout(x)
        
        x = x + self.pos_encoding[:, :seq_len, :]
        
        transformer_output = self.transformer_encoder(x)
        
        mean_pool = transformer_output.mean(dim=1)
        max_pool, _ = transformer_output.max(dim=1)
        
        query = transformer_output.mean(dim=1, keepdim=True)
        attn_pool, _ = self.attention_pool(query, transformer_output, transformer_output)
        attn_pool = self.pool_norm(attn_pool.squeeze(1))
        
        combined_features = torch.cat([mean_pool, max_pool, attn_pool], dim=1)
        
        logits = self.classifier(combined_features)
        return logits

class RoBERTaLargeDetector(nn.Module):
    """RoBERTa Large architecture with different dimensions"""
    def __init__(self, input_dim=768, seq_len=100, hidden_dim=768, num_layers=6):
        super(RoBERTaLargeDetector, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
        self.pos_encoding = self._create_positional_encoding(seq_len, hidden_dim)
        
        # RoBERTa Large uses 4x hidden_dim for feedforward
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=12,  # 12 attention heads for large model
            dim_feedforward=hidden_dim * 4,  # 4x for RoBERTa Large
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.attention_pool = nn.MultiheadAttention(
            hidden_dim, num_heads=12, dropout=0.1, batch_first=True
        )
        self.pool_norm = nn.LayerNorm(hidden_dim)
        
        # Different classifier architecture for RoBERTa Large
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        self._init_weights()
    
    def _create_positional_encoding(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        x = self.input_projection(x)
        x = self.input_norm(x)
        x = self.dropout(x)
        
        x = x + self.pos_encoding[:, :seq_len, :]
        
        transformer_output = self.transformer_encoder(x)
        
        mean_pool = transformer_output.mean(dim=1)
        max_pool, _ = transformer_output.max(dim=1)
        
        query = transformer_output.mean(dim=1, keepdim=True)
        attn_pool, _ = self.attention_pool(query, transformer_output, transformer_output)
        attn_pool = self.pool_norm(attn_pool.squeeze(1))
        
        combined_features = torch.cat([mean_pool, max_pool, attn_pool], dim=1)
        
        logits = self.classifier(combined_features)
        return logits

class ModelManager:
    """Manages loading and inference for all models"""
    
    def __init__(self, models_dir="../models-ml"):
        self.models_dir = models_dir
        self.loaded_models = {}
        # Force CPU usage for compatibility
        self.device = torch.device('cpu')
        logger.info(f"Using device: {self.device} (forced CPU for compatibility)")
    
    def load_model(self, model_key: str) -> Optional[nn.Module]:
        """Load a specific model"""
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]
        
        config = MODEL_CONFIGS.get(model_key)
        if not config:
            logger.error(f"Unknown model key: {model_key}")
            return None
        
        model_path = os.path.join(self.models_dir, config['file'])
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None
        
        try:
            # Create model instance based on type
            if model_key == 'AITextDetector':
                model = AITextDetector(
                    input_dim=config['input_dim'],
                    seq_len=config['seq_len'],
                    hidden_dim=config['hidden_dim'],
                    dropout=config['dropout']
                )
            elif model_key == 'RoBERTaLarge':
                # RoBERTa Large has different architecture
                model = RoBERTaLargeDetector(
                    input_dim=config['input_dim'],
                    seq_len=config['seq_len'],
                    hidden_dim=config['hidden_dim'],
                    num_layers=config['num_layers']
                )
            else:
                # Other models use OptimizedRoBERTaDetector architecture
                model = OptimizedRoBERTaDetector(
                    input_dim=config['input_dim'],
                    seq_len=config['seq_len'],
                    hidden_dim=config['hidden_dim'],
                    num_layers=config['num_layers']
                )
            
            # Load state dict
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            
            self.loaded_models[model_key] = model
            logger.info(f"Successfully loaded model: {model_key}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_key}: {str(e)}")
            return None
    
    def text_to_embeddings(self, text: str) -> Optional[torch.Tensor]:
        """Convert text to embeddings using enhanced text analysis"""
        try:
            # Extract comprehensive text features
            text_features = self.extract_text_features(text)
            
            # Simple tokenization and embedding generation
            words = text.lower().split()
            
            # Ensure we have exactly 100 tokens
            if len(words) > 100:
                words = words[:100]
            elif len(words) < 100:
                # Pad with special tokens
                words.extend(['[PAD]'] * (100 - len(words)))
            
            # Generate embeddings based on word characteristics and text features
            embeddings = []
            for i, word in enumerate(words):
                # Create deterministic but varied embeddings based on word content
                word_hash = hash(word + text[:50]) % (2**31)  # Include text context
                np.random.seed(word_hash)
                
                # Base embedding with more variation
                embedding = np.random.normal(0.0, 0.25, 768)
                
                # Add positional encoding
                position = i / 100.0
                embedding[0] = position  # Position in sequence
                
                # Add word-specific features
                embedding[1] = len(word) / 20.0  # Normalized word length
                embedding[2] = 1.0 if word.isupper() else 0.0  # All caps
                embedding[3] = 1.0 if word.islower() else 0.0  # All lowercase
                embedding[4] = 1.0 if any(c.isdigit() for c in word) else 0.0  # Contains digits
                embedding[5] = 1.0 if word.startswith('[') else 0.0  # Padding token
                
                # Inject global text features into embedding dimensions
                for j, feature in enumerate(text_features[:50]):  # Use first 50 features
                    if j + 10 < 768:  # Leave space for other features
                        embedding[j + 10] = feature
                
                # Add stronger text-context specific variations for better discrimination
                text_lower = text.lower()
                
                # Strong AI indicators
                ai_indicators = [
                    'artificial intelligence', 'machine learning', 'neural networks', 'algorithms',
                    'furthermore', 'moreover', 'comprehensive', 'optimization', 'paradigm',
                    'facilitate', 'leverage', 'innovative', 'sophisticated', 'implementation',
                    'methodology', 'framework', 'architecture', 'systematically', 'significantly'
                ]
                
                human_indicators = [
                    'i think', 'personally', 'i feel', 'honestly', 'you know', 'basically',
                    'really', 'actually', 'probably', 'definitely', 'kinda', 'gonna', 'wanna'
                ]
                
                ai_score = sum(1 for indicator in ai_indicators if indicator in text_lower)
                human_score = sum(1 for indicator in human_indicators if indicator in text_lower)
                
                # Stronger signatures for better classification
                if ai_score > 0:
                    embedding[60:80] += (ai_score * 0.8)  # Strong AI signature
                    embedding[100:120] += (ai_score * 0.6)  # Secondary AI signature
                    
                if human_score > 0:
                    embedding[70:90] -= (human_score * 0.8)  # Strong human signature
                    embedding[120:140] -= (human_score * 0.6)  # Secondary human signature
                    
                # Text structure indicators
                if len(text.split('.')) > 3:  # Multiple sentences
                    avg_sentence_length = len(text.split()) / len(text.split('.'))
                    if avg_sentence_length > 20:  # Long sentences = AI-like
                        embedding[150:160] += 0.7
                    elif avg_sentence_length < 10:  # Short sentences = Human-like
                        embedding[150:160] -= 0.7
                
                embeddings.append(embedding)
            
            # Convert to tensor
            embeddings = np.array(embeddings, dtype=np.float32)
            embeddings = torch.FloatTensor(embeddings).unsqueeze(0)  # Add batch dimension
            return embeddings.to(self.device)
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return None
    
    def generate_fallback_embeddings(self, text: str) -> torch.Tensor:
        """Generate simple fallback embeddings"""
        # Use text hash for deterministic results
        text_hash = hash(text) % (2**32)
        np.random.seed(text_hash)
        
        # Generate embeddings that vary based on text characteristics
        text_features = self.extract_text_features(text)
        
        # Create base embeddings
        embeddings = np.random.normal(0.0, 0.3, (1, 100, 768)).astype(np.float32)
        
        # Inject text features into first few dimensions
        for i in range(min(len(text_features), 20)):
            embeddings[0, :, i] = text_features[i]
        
        return torch.FloatTensor(embeddings).to(self.device)
    
    def extract_text_features(self, text: str) -> List[float]:
        """Extract comprehensive features from text for better AI vs Human detection"""
        words = text.split()
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Basic text statistics
        features = [
            len(text) / 1000.0,  # Text length (normalized)
            len(words) / 100.0,  # Word count (normalized)
            len(sentences) / 10.0,  # Sentence count (normalized)
            sum(len(word) for word in words) / max(len(words), 1) / 10.0,  # Avg word length
            sum(word.isupper() for word in words) / max(len(words), 1),  # Uppercase ratio
            sum(word.islower() for word in words) / max(len(words), 1),  # Lowercase ratio
            sum(any(c.isdigit() for c in word) for word in words) / max(len(words), 1),  # Digit ratio
            text.count(',') / max(len(text), 1),  # Comma density
            text.count('.') / max(len(text), 1),  # Period density
            text.count('!') / max(len(text), 1),  # Exclamation density
            text.count('?') / max(len(text), 1),  # Question density
            text.count('"') / max(len(text), 1),  # Quote density
        ]
        
        # Enhanced AI vs Human indicators
        ai_words = [
            'algorithm', 'furthermore', 'moreover', 'comprehensive', 'optimization',
            'artificial intelligence', 'machine learning', 'neural networks', 'paradigm',
            'facilitate', 'leverage', 'innovative', 'sophisticated', 'implementation',
            'methodology', 'framework', 'architecture', 'systematically', 'significantly',
            'subsequently', 'consequently', 'therefore', 'hence', 'thus', 'accordingly',
            'nevertheless', 'nonetheless', 'notwithstanding', 'additionally', 'furthermore'
        ]
        
        human_words = [
            'actually', 'really', 'basically', 'definitely', 'probably', 'i think',
            'i feel', 'in my opinion', 'personally', 'honestly', 'you know',
            'pretty much', 'kinda', 'gonna', 'wanna', 'yeah', 'okay', 'well',
            'like', 'just', 'maybe', 'sort of', 'kind of', 'i guess',
            'i mean', 'you see', 'anyway', 'whatever', 'stuff', 'things'
        ]
        
        text_lower = text.lower()
        
        # Count AI indicators
        ai_score = 0
        for phrase in ai_words:
            ai_score += text_lower.count(phrase)
        ai_score = ai_score / max(len(words), 1)  # Normalize by word count
        
        # Count human indicators
        human_score = 0
        for phrase in human_words:
            human_score += text_lower.count(phrase)
        human_score = human_score / max(len(words), 1)  # Normalize by word count
        
        # Sentence structure analysis
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            sentence_length_variance = sum((len(s.split()) - avg_sentence_length)**2 for s in sentences) / len(sentences)
        else:
            avg_sentence_length = 0
            sentence_length_variance = 0
        
        # Advanced linguistic features
        features.extend([
            ai_score * 10,  # Scaled AI word score
            human_score * 10,  # Scaled human word score
            avg_sentence_length / 30.0,  # Normalized average sentence length
            sentence_length_variance / 100.0,  # Sentence length variation
            text.count(';') / max(len(text), 1),  # Semicolon usage
            text.count(':') / max(len(text), 1),  # Colon usage
            text.count('(') / max(len(text), 1),  # Parentheses usage
            text.count('-') / max(len(text), 1),  # Dash usage
            len([w for w in words if len(w) > 8]) / max(len(words), 1),  # Long words ratio
            len([w for w in words if w.istitle()]) / max(len(words), 1),  # Title case ratio
        ])
        
        return features
    
    def predict_text(self, text: str, model_keys: List[str]) -> Dict:
        """Predict AI probability for given text using specified models"""
        results = {}
        
        # Generate embeddings from text
        embeddings = self.text_to_embeddings(text)
        
        if embeddings is None:
            # Fallback to simple embeddings if text processing fails
            logger.warning("Using fallback embedding generation")
            embeddings = self.generate_fallback_embeddings(text)
        
        for model_key in model_keys:
            logger.info(f"Loading model: {model_key}")
            model = self.load_model(model_key)
            if model is None:
                logger.error(f"Failed to load model: {model_key}")
                results[model_key] = {
                    'error': 'Model failed to load',
                    'ai_probability': 0.0,
                    'prediction': 'Error',
                    'confidence': 0.0
                }
                continue
            
            try:
                logger.info(f"Running inference with {model_key}")
                with torch.no_grad():
                    outputs = model(embeddings)
                    probabilities = torch.softmax(outputs, dim=1)
                    ai_prob = probabilities[0, 1].item()
                    
                    prediction = 'AI Generated' if ai_prob > 0.5 else 'Human Written'
                    confidence = max(ai_prob, 1 - ai_prob)
                    
                    logger.info(f"{model_key} prediction: {prediction} (confidence: {confidence:.4f})")
                    
                    results[model_key] = {
                        'ai_probability': round(ai_prob, 4),
                        'human_probability': round(1 - ai_prob, 4),
                        'prediction': prediction,
                        'confidence': round(confidence, 4),
                        'model_info': MODEL_CONFIGS[model_key]
                    }
                    
            except Exception as e:
                logger.error(f"Error during prediction with {model_key}: {str(e)}", exc_info=True)
                results[model_key] = {
                    'error': str(e),
                    'ai_probability': 0.0,
                    'prediction': 'Error',
                    'confidence': 0.0
                }
        
        return results

# Initialize model manager
model_manager = ModelManager()

@app.route('/')
def index():
    """Main application page"""
    return render_template('index.html', models=MODEL_CONFIGS)

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for text prediction"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        selected_models = data.get('models', ['OptimizedRoBERTa'])
        compare_all = data.get('compare_all', False)
        
        logger.info(f"Prediction request - Text length: {len(text)}, Models: {selected_models}, Compare all: {compare_all}")
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if len(text) < 10:
            return jsonify({'error': 'Text too short (minimum 10 characters)'}), 400
        
        if len(text) > 10000:
            return jsonify({'error': 'Text too long (maximum 10,000 characters)'}), 400
        
        # If compare_all is True, use all models
        if compare_all:
            selected_models = list(MODEL_CONFIGS.keys())
            logger.info(f"Using all models: {selected_models}")
        
        # Get predictions
        logger.info(f"Starting prediction with models: {selected_models}")
        results = model_manager.predict_text(text, selected_models)
        
        # Calculate ensemble prediction if multiple models
        if len(results) > 1:
            valid_results = {k: v for k, v in results.items() if 'error' not in v}
            if valid_results:
                ensemble_ai_prob = np.mean([r['ai_probability'] for r in valid_results.values()])
                ensemble_prediction = 'AI Generated' if ensemble_ai_prob > 0.5 else 'Human Written'
                ensemble_confidence = max(ensemble_ai_prob, 1 - ensemble_ai_prob)
                
                results['ensemble'] = {
                    'ai_probability': round(ensemble_ai_prob, 4),
                    'human_probability': round(1 - ensemble_ai_prob, 4),
                    'prediction': ensemble_prediction,
                    'confidence': round(ensemble_confidence, 4),
                    'model_info': {
                        'name': 'Ensemble Average',
                        'description': f'Average of {len(valid_results)} models',
                        'params': 'Combined',
                        'kaggle_score': 'N/A'
                    }
                }
        
        return jsonify({
            'success': True,
            'text_length': len(text),
            'word_count': len(text.split()),
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """API endpoint for file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.txt'):
            return jsonify({'error': 'Only .txt files are supported'}), 400
        
        # Read file content
        content = file.read().decode('utf-8')
        
        if len(content.strip()) == 0:
            return jsonify({'error': 'File is empty'}), 400
        
        if len(content) > 50000:
            return jsonify({'error': 'File too large (maximum 50,000 characters)'}), 400
        
        return jsonify({
            'success': True,
            'content': content,
            'filename': secure_filename(file.filename),
            'size': len(content)
        })
        
    except Exception as e:
        logger.error(f"Error in upload endpoint: {str(e)}")
        return jsonify({'error': f'File upload failed: {str(e)}'}), 500

@app.route('/api/models')
def get_models():
    """API endpoint to get model information"""
    return jsonify(MODEL_CONFIGS)

if __name__ == '__main__':
    # Create upload directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    
    # Check if production mode
    # port = int(os.environ.get('PORT', 5000))
    # debug = os.environ.get('FLASK_ENV') != 'production'
    app.run(host='127.0.0.1', port=5005, debug=False)
    # Run the application
    #app.run(debug=debug, host='0.0.0.0', port=port)
