// AI Text Detection App - Main JavaScript
class AITextDetectionApp {
    constructor() {
        this.selectedModels = new Set(['OptimizedRoBERTa']);
        this.isAnalyzing = false;
        this.currentResults = null;
        this.comparisonChart = null;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.updateCharCount();
        this.loadExampleTexts();
    }
    
    setupEventListeners() {
        // Model selection
        this.setupModelSelection();
        
        // Text input
        this.setupTextInput();
        
        // File upload
        this.setupFileUpload();
        
        // Tabs
        this.setupTabs();
        
        // Analysis
        this.setupAnalysis();
        
        // Examples
        this.setupExamples();
    }
    
    setupModelSelection() {
        const checkboxes = document.querySelectorAll('input[name="selected_models"]');
        const selectAllBtn = document.getElementById('selectAll');
        const clearAllBtn = document.getElementById('clearAll');
        const compareAllToggle = document.getElementById('compareAll');
        
        // Individual checkbox changes
        checkboxes.forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                if (e.target.checked) {
                    this.selectedModels.add(e.target.value);
                } else {
                    this.selectedModels.delete(e.target.value);
                }
                this.updateModelCards();
                this.validateSelection();
            });
        });
        
        // Select all button
        selectAllBtn.addEventListener('click', () => {
            checkboxes.forEach(checkbox => {
                checkbox.checked = true;
                this.selectedModels.add(checkbox.value);
            });
            this.updateModelCards();
            this.validateSelection();
        });
        
        // Clear all button
        clearAllBtn.addEventListener('click', () => {
            checkboxes.forEach(checkbox => {
                checkbox.checked = false;
                this.selectedModels.delete(checkbox.value);
            });
            this.updateModelCards();
            this.validateSelection();
        });
        
        // Compare all toggle
        compareAllToggle.addEventListener('change', (e) => {
            if (e.target.checked) {
                selectAllBtn.click();
            }
        });
        
        // Model card clicks
        document.querySelectorAll('.model-card').forEach(card => {
            card.addEventListener('click', (e) => {
                if (e.target.type !== 'checkbox') {
                    const checkbox = card.querySelector('input[type="checkbox"]');
                    checkbox.click();
                }
            });
        });
    }
    
    updateModelCards() {
        document.querySelectorAll('.model-card').forEach(card => {
            const checkbox = card.querySelector('input[type="checkbox"]');
            if (checkbox.checked) {
                card.classList.add('selected');
            } else {
                card.classList.remove('selected');
            }
        });
    }
    
    validateSelection() {
        const analyzeBtn = document.getElementById('analyzeBtn');
        if (this.selectedModels.size === 0) {
            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Please select at least one model';
        } else {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="fas fa-search"></i> <span>Analyze Text</span> <div class="btn-spinner" style="display: none;"><i class="fas fa-spinner fa-spin"></i></div>';
        }
        
        // Log current selection for debugging
        console.log('Selected models:', Array.from(this.selectedModels));
    }
    
    setupTextInput() {
        const textInput = document.getElementById('textInput');
        const charCount = document.getElementById('charCount');
        const wordCount = document.getElementById('wordCount');
        
        textInput.addEventListener('input', () => {
            this.updateCharCount();
        });
        
        textInput.addEventListener('paste', () => {
            setTimeout(() => this.updateCharCount(), 10);
        });
    }
    
    updateCharCount() {
        const textInput = document.getElementById('textInput');
        const charCount = document.getElementById('charCount');
        const wordCount = document.getElementById('wordCount');
        
        const text = textInput.value;
        const chars = text.length;
        const words = text.trim() ? text.trim().split(/\\s+/).length : 0;
        
        charCount.textContent = chars.toLocaleString();
        wordCount.textContent = words.toLocaleString();
        
        // Color coding
        if (chars < 10) {
            charCount.style.color = '#ef4444';
        } else if (chars > 10000) {
            charCount.style.color = '#f59e0b';
        } else {
            charCount.style.color = '#22c55e';
        }
    }
    
    setupFileUpload() {
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.getElementById('uploadArea');
        const fileInfo = document.getElementById('fileInfo');
        const removeFileBtn = document.getElementById('removeFile');
        
        // File input change
        fileInput.addEventListener('change', (e) => {
            this.handleFileSelect(e.target.files[0]);
        });
        
        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            this.handleFileSelect(e.dataTransfer.files[0]);
        });
        
        // Remove file
        removeFileBtn.addEventListener('click', () => {
            this.clearFile();
        });
    }
    
    async handleFileSelect(file) {
        if (!file) return;
        
        if (!file.name.toLowerCase().endsWith('.txt')) {
            this.showError('Please select a .txt file');
            return;
        }
        
        if (file.size > 50 * 1024) { // 50KB
            this.showError('File size must be less than 50KB');
            return;
        }
        
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                // Switch to text tab and populate content
                this.switchTab('text');
                document.getElementById('textInput').value = result.content;
                this.updateCharCount();
                
                // Show file info
                this.showFileInfo(result.filename, result.size);
                
                this.showSuccess(`File "${result.filename}" loaded successfully`);
            } else {
                this.showError(result.error || 'File upload failed');
            }
        } catch (error) {
            this.showError('Failed to upload file: ' + error.message);
        }
    }
    
    showFileInfo(filename, size) {
        const fileInfo = document.getElementById('fileInfo');
        const filenameSpan = fileInfo.querySelector('.filename');
        const filesizeSpan = fileInfo.querySelector('.filesize');
        
        filenameSpan.textContent = filename;
        filesizeSpan.textContent = this.formatFileSize(size);
        fileInfo.style.display = 'flex';
    }
    
    clearFile() {
        const fileInfo = document.getElementById('fileInfo');
        const fileInput = document.getElementById('fileInput');
        
        fileInput.value = '';
        fileInfo.style.display = 'none';
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }
    
    setupTabs() {
        const tabBtns = document.querySelectorAll('.tab-btn');
        const tabContents = document.querySelectorAll('.tab-content');
        
        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const tabName = btn.dataset.tab;
                this.switchTab(tabName);
            });
        });
    }
    
    switchTab(tabName) {
        // Update buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
        
        // Update content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabName}-tab`).classList.add('active');
    }
    
    setupAnalysis() {
        const analyzeBtn = document.getElementById('analyzeBtn');
        
        analyzeBtn.addEventListener('click', async () => {
            if (this.isAnalyzing) return;
            
            const text = document.getElementById('textInput').value.trim();
            if (!text) {
                this.showError('Please enter some text to analyze');
                return;
            }
            
            if (text.length < 10) {
                this.showError('Text must be at least 10 characters long');
                return;
            }
            
                    if (this.selectedModels.size === 0) {
            this.showError('Please select at least one model');
            return;
        }
        
        console.log('Starting analysis with models:', Array.from(this.selectedModels));
        await this.analyzeText(text);
        });
    }
    
    async analyzeText(text) {
        this.isAnalyzing = true;
        this.showLoading(true);
        
        // Add neural network effect
        if (window.neuralBackground) {
            window.neuralBackground.addTemporaryEffect('analyze');
        }
        
        const analyzeBtn = document.getElementById('analyzeBtn');
        const spinner = analyzeBtn.querySelector('.btn-spinner');
        
        analyzeBtn.disabled = true;
        if (spinner) {
            spinner.style.display = 'inline-block';
        }
        
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text: text,
                    models: Array.from(this.selectedModels),
                    compare_all: document.getElementById('compareAll').checked
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.currentResults = result;
                this.displayResults(result);
                this.scrollToResults();
            } else {
                this.showError(result.error || 'Analysis failed');
            }
        } catch (error) {
            this.showError('Failed to analyze text: ' + error.message);
        } finally {
            this.isAnalyzing = false;
            this.showLoading(false);
            analyzeBtn.disabled = false;
            if (spinner) {
                spinner.style.display = 'none';
            }
        }
    }
    
    displayResults(data) {
        const resultsSection = document.getElementById('resultsSection');
        const resultSummary = document.getElementById('resultSummary');
        const modelResults = document.getElementById('modelResults');
        const chartContainer = document.getElementById('chartContainer');
        
        resultsSection.style.display = 'block';
        resultsSection.classList.add('fade-in');
        
        // Display summary
        this.displaySummary(data, resultSummary);
        
        // Display individual model results
        this.displayModelResults(data, modelResults);
        
        // Display comparison chart if multiple models
        if (Object.keys(data.results).length > 1) {
            chartContainer.style.display = 'block';
            this.createComparisonChart(data);
        } else {
            chartContainer.style.display = 'none';
        }
    }
    
    displaySummary(data, container) {
        const results = data.results;
        const models = Object.keys(results);
        
        let summary;
        if (models.length === 1) {
            const modelResult = results[models[0]];
            summary = {
                prediction: modelResult.prediction,
                confidence: modelResult.confidence,
                ai_probability: modelResult.ai_probability
            };
        } else {
            // Use ensemble result if available, otherwise average
            if (results.ensemble) {
                summary = {
                    prediction: results.ensemble.prediction,
                    confidence: results.ensemble.confidence,
                    ai_probability: results.ensemble.ai_probability
                };
            } else {
                const validResults = Object.values(results).filter(r => !r.error);
                const avgAiProb = validResults.reduce((sum, r) => sum + r.ai_probability, 0) / validResults.length;
                summary = {
                    prediction: avgAiProb > 0.5 ? 'AI Generated' : 'Human Written',
                    confidence: Math.max(avgAiProb, 1 - avgAiProb),
                    ai_probability: avgAiProb
                };
            }
        }
        
        const predictionClass = summary.prediction === 'AI Generated' ? 'ai' : 'human';
        const confidencePercent = (summary.confidence * 100).toFixed(1);
        const aiPercent = (summary.ai_probability * 100).toFixed(1);
        const humanPercent = ((1 - summary.ai_probability) * 100).toFixed(1);
        
        container.innerHTML = `
            <div class="prediction-result ${predictionClass}">
                ${summary.prediction}
            </div>
            <div class="confidence-score">
                Confidence: ${confidencePercent}%
            </div>
            <div class="prediction-details">
                <div class="detail-item">
                    <span class="detail-value">${data.text_length}</span>
                    <span class="detail-label">Characters</span>
                </div>
                <div class="detail-item">
                    <span class="detail-value">${data.word_count}</span>
                    <span class="detail-label">Words</span>
                </div>
                <div class="detail-item">
                    <span class="detail-value">${aiPercent}%</span>
                    <span class="detail-label">AI Probability</span>
                </div>
                <div class="detail-item">
                    <span class="detail-value">${humanPercent}%</span>
                    <span class="detail-label">Human Probability</span>
                </div>
            </div>
        `;
    }
    
    displayModelResults(data, container) {
        const results = data.results;
        let html = '';
        
        Object.entries(results).forEach(([modelKey, result]) => {
            if (result.error) {
                html += this.createErrorResultHTML(modelKey, result);
            } else {
                html += this.createModelResultHTML(modelKey, result);
            }
        });
        
        container.innerHTML = html;
        
        // Animate probability bars
        setTimeout(() => {
            this.animateProbabilityBars();
        }, 100);
    }
    
    createModelResultHTML(modelKey, result) {
        const predictionClass = result.prediction === 'AI Generated' ? 'ai' : 'human';
        const aiPercent = (result.ai_probability * 100).toFixed(1);
        const humanPercent = (result.human_probability * 100).toFixed(1);
        const confidencePercent = (result.confidence * 100).toFixed(1);
        
        return `
            <div class="model-result">
                <div class="model-result-header">
                    <div class="model-name">${result.model_info.name}</div>
                    <div class="prediction-badge ${predictionClass}">
                        ${result.prediction} (${confidencePercent}%)
                    </div>
                </div>
                <div class="probability-bars">
                    <div class="probability-bar">
                        <span class="probability-label">AI Generated</span>
                        <div class="probability-visual">
                            <div class="probability-fill ai" data-width="${aiPercent}"></div>
                        </div>
                        <span class="probability-value">${aiPercent}%</span>
                    </div>
                    <div class="probability-bar">
                        <span class="probability-label">Human Written</span>
                        <div class="probability-visual">
                            <div class="probability-fill human" data-width="${humanPercent}"></div>
                        </div>
                        <span class="probability-value">${humanPercent}%</span>
                    </div>
                </div>
                <div class="model-info">
                    <small><strong>Parameters:</strong> ${result.model_info.params} | <strong>Kaggle Score:</strong> ${result.model_info.kaggle_score}</small>
                    <br>
                    <small>${result.model_info.description}</small>
                </div>
            </div>
        `;
    }
    
    createErrorResultHTML(modelKey, result) {
        return `
            <div class="model-result error">
                <div class="model-result-header">
                    <div class="model-name">${modelKey}</div>
                    <div class="prediction-badge error">Error</div>
                </div>
                <div class="error-message">
                    <i class="fas fa-exclamation-triangle"></i>
                    ${result.error}
                </div>
            </div>
        `;
    }
    
    animateProbabilityBars() {
        document.querySelectorAll('.probability-fill').forEach(bar => {
            const width = bar.dataset.width;
            bar.style.width = '0%';
            setTimeout(() => {
                bar.style.width = width + '%';
            }, 100);
        });
    }
    
    createComparisonChart(data) {
        const ctx = document.getElementById('comparisonChart');
        
        if (this.comparisonChart) {
            this.comparisonChart.destroy();
        }
        
        const results = data.results;
        const validResults = Object.entries(results).filter(([_, result]) => !result.error);
        
        const labels = validResults.map(([key, result]) => result.model_info.name);
        const aiProbabilities = validResults.map(([_, result]) => result.ai_probability * 100);
        const humanProbabilities = validResults.map(([_, result]) => result.human_probability * 100);
        
        this.comparisonChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'AI Probability (%)',
                        data: aiProbabilities,
                        backgroundColor: 'rgba(239, 68, 68, 0.7)',
                        borderColor: 'rgba(239, 68, 68, 1)',
                        borderWidth: 2
                    },
                    {
                        label: 'Human Probability (%)',
                        data: humanProbabilities,
                        backgroundColor: 'rgba(34, 197, 94, 0.7)',
                        borderColor: 'rgba(34, 197, 94, 1)',
                        borderWidth: 2
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            color: '#94a3b8'
                        },
                        grid: {
                            color: 'rgba(148, 163, 184, 0.2)'
                        }
                    },
                    x: {
                        ticks: {
                            color: '#94a3b8'
                        },
                        grid: {
                            color: 'rgba(148, 163, 184, 0.2)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            color: '#94a3b8'
                        }
                    }
                }
            }
        });
    }
    
    setupExamples() {
        document.querySelectorAll('.example-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const exampleType = btn.dataset.example;
                this.loadExample(exampleType);
            });
        });
    }
    
    loadExample(type) {
        const textInput = document.getElementById('textInput');
        
        if (type === 'ai') {
            textInput.value = this.examples.ai;
        } else {
            textInput.value = this.examples.human;
        }
        
        this.updateCharCount();
        this.switchTab('text');
    }
    
    loadExampleTexts() {
        this.examples = {
            ai: `The rapid advancement of artificial intelligence has fundamentally transformed the landscape of modern technology and society. Machine learning algorithms now permeate virtually every aspect of our digital lives, from personalized content recommendations to sophisticated fraud detection systems. These neural networks possess the remarkable ability to process vast datasets and identify complex patterns that would be impossible for humans to discern manually. The integration of AI into healthcare has yielded particularly promising results, enabling more accurate diagnoses and personalized treatment plans. Furthermore, autonomous systems continue to evolve, promising revolutionary changes in transportation, manufacturing, and service industries.`,
            
            human: `I've been thinking a lot lately about how technology has changed our daily lives. My grandmother used to tell me stories about when she was young - no smartphones, no internet, just face-to-face conversations and handwritten letters. Sometimes I wonder if we've gained convenience but lost something important along the way. Don't get me wrong, I love being able to video call my friends across the world or instantly look up any information I need. But there's something to be said for the slower pace of life, for being fully present in a moment without the constant buzz of notifications. I try to put my phone down during meals now, just to see what it feels like to eat without scrolling through social media.`
        };
    }
    
    scrollToResults() {
        const resultsSection = document.getElementById('resultsSection');
        resultsSection.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }
    
    showLoading(show) {
        const overlay = document.getElementById('loadingOverlay');
        if (show) {
            overlay.style.display = 'flex';
            // Add pulse effect to neural background
            if (window.neuralBackground) {
                window.neuralBackground.addTemporaryEffect('pulse');
            }
        } else {
            overlay.style.display = 'none';
        }
    }
    
    showError(message) {
        this.showNotification(message, 'error');
    }
    
    showSuccess(message) {
        this.showNotification(message, 'success');
    }
    
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <i class="fas ${type === 'error' ? 'fa-exclamation-triangle' : type === 'success' ? 'fa-check-circle' : 'fa-info-circle'}"></i>
            <span>${message}</span>
            <button class="notification-close">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        // Style the notification
        Object.assign(notification.style, {
            position: 'fixed',
            top: '20px',
            right: '20px',
            padding: '1rem 1.5rem',
            borderRadius: '0.5rem',
            color: 'white',
            zIndex: '10000',
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            minWidth: '300px',
            maxWidth: '500px',
            boxShadow: '0 10px 25px rgba(0,0,0,0.2)',
            transform: 'translateX(100%)',
            transition: 'transform 0.3s ease'
        });
        
        if (type === 'error') {
            notification.style.background = 'linear-gradient(135deg, #ef4444, #dc2626)';
        } else if (type === 'success') {
            notification.style.background = 'linear-gradient(135deg, #22c55e, #16a34a)';
        } else {
            notification.style.background = 'linear-gradient(135deg, #3b82f6, #2563eb)';
        }
        
        // Add to document
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 10);
        
        // Close functionality
        const closeBtn = notification.querySelector('.notification-close');
        closeBtn.addEventListener('click', () => {
            this.removeNotification(notification);
        });
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (document.body.contains(notification)) {
                this.removeNotification(notification);
            }
        }, 5000);
    }
    
    removeNotification(notification) {
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
            if (document.body.contains(notification)) {
                document.body.removeChild(notification);
            }
        }, 300);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.aiApp = new AITextDetectionApp();
});

// Export for debugging
window.AITextDetectionApp = AITextDetectionApp;
