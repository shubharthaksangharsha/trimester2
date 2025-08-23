// 🌸 Flower Classification App - Main JavaScript
class FlowerClassificationApp {
    constructor() {
        this.selectedModel = 'efficientnet_fine_tuned';
        this.currentTab = 'upload';
        this.uploadedFile = null;
        this.selectedFlowerImage = null;
        this.flowerImages = {};
        
        this.initializeApp();
        this.setupEventListeners();
        this.init3DBackground();
    }

    initializeApp() {
        // Initialize model selection
        this.selectModel(this.selectedModel);
        
        // Initialize flower emojis mapping
        this.flowerEmojis = {
            'daisy': '🌼',
            'dandelion': '🌻',
            'rose': '🌹',
            'sunflower': '🌻',
            'tulip': '🌷'
        };
    }

    setupEventListeners() {
        // Model selection
        document.querySelectorAll('.model-card').forEach(card => {
            card.addEventListener('click', (e) => {
                const modelKey = e.currentTarget.dataset.model;
                this.selectModel(modelKey);
            });
        });

        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const tabName = e.currentTarget.dataset.tab;
                this.switchTab(tabName);
            });
        });

        // File upload
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        
        fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        
        // Remove uploaded image
        document.getElementById('remove-image').addEventListener('click', this.removeUploadedImage.bind(this));

        // Predict button
        document.getElementById('predict-btn').addEventListener('click', this.predict.bind(this));

        // Canvas controls
        document.getElementById('clear-canvas').addEventListener('click', () => {
            if (window.drawingCanvas) {
                window.drawingCanvas.clear();
            }
        });

        // Brush size control
        const brushSize = document.getElementById('brush-size');
        const brushSizeValue = document.getElementById('brush-size-value');
        
        brushSize.addEventListener('input', (e) => {
            const size = e.target.value;
            brushSizeValue.textContent = `${size}px`;
            if (window.drawingCanvas) {
                window.drawingCanvas.setBrushSize(size);
            }
        });

        // Flower browsing controls
        const refreshFlowers = document.getElementById('refresh-flowers');
        if (refreshFlowers) {
            refreshFlowers.addEventListener('click', this.loadFlowerImages.bind(this));
        }

        const clearSelection = document.getElementById('clear-selection');
        if (clearSelection) {
            clearSelection.addEventListener('click', this.clearFlowerSelection.bind(this));
        }


    }

    selectModel(modelKey) {
        // Remove previous selection
        document.querySelectorAll('.model-card').forEach(card => {
            card.classList.remove('selected');
        });

        // Add selection to new model
        const selectedCard = document.querySelector(`[data-model="${modelKey}"]`);
        if (selectedCard) {
            selectedCard.classList.add('selected');
            this.selectedModel = modelKey;
        }
    }

    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabName}-tab`).classList.add('active');

        this.currentTab = tabName;

        // Initialize canvas if switching to draw tab
        if (tabName === 'draw' && !window.drawingCanvas) {
            // Initialize canvas in next tick to ensure DOM is ready
            setTimeout(() => {
                window.initializeCanvas();
            }, 100);
        }

        // Load flower images if switching to browse tab
        if (tabName === 'browse') {
            this.loadFlowerImages();
        }
    }

    // File upload handlers
    handleDragOver(e) {
        e.preventDefault();
        e.currentTarget.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.handleFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.handleFile(file);
        }
    }

    handleFile(file) {
        if (!file.type.startsWith('image/')) {
            this.showError('Please select an image file.');
            return;
        }

        this.uploadedFile = file;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            const previewImage = document.getElementById('preview-image');
            previewImage.src = e.target.result;
            
            document.getElementById('upload-area').querySelector('.upload-content').style.display = 'none';
            document.getElementById('upload-preview').style.display = 'block';
        };
        reader.readAsDataURL(file);
    }

    removeUploadedImage() {
        this.uploadedFile = null;
        document.getElementById('upload-area').querySelector('.upload-content').style.display = 'block';
        document.getElementById('upload-preview').style.display = 'none';
        document.getElementById('file-input').value = '';
    }

    async predict() {
        try {
            this.showLoading(true);

            // Check if compare all models is enabled
            const compareAllModels = document.getElementById('compare-all-models').checked;

            const formData = new FormData();
            if (!compareAllModels) {
                formData.append('model', this.selectedModel);
            }

            if (this.currentTab === 'upload') {
                if (!this.uploadedFile) {
                    this.showError('Please upload an image first.');
                    return;
                }
                formData.append('file', this.uploadedFile);
            } else if (this.currentTab === 'browse') {
                if (!this.selectedFlowerImage) {
                    this.showError('Please select a flower image first.');
                    return;
                }
                formData.append('flower_image_path', this.selectedFlowerImage.path);
            } else {
                // Get canvas data
                const canvas = document.getElementById('drawing-canvas');
                const canvasData = canvas.toDataURL('image/png');
                
                if (this.isCanvasEmpty(canvas)) {
                    this.showError('Please draw a flower first.');
                    return;
                }
                
                formData.append('canvas_data', canvasData);
            }

            // Choose endpoint based on comparison mode
            const endpoint = compareAllModels ? '/api/compare-all-models' : '/api/predict';
            
            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                if (compareAllModels) {
                    this.displayComparisonResults(result);
                } else {
                    this.displayResults(result);
                }
            } else {
                this.showError(result.error || 'Prediction failed');
            }

        } catch (error) {
            console.error('Prediction error:', error);
            this.showError('Network error occurred');
        } finally {
            this.showLoading(false);
        }
    }

    isCanvasEmpty(canvas) {
        const context = canvas.getContext('2d');
        const pixelBuffer = new Uint32Array(
            context.getImageData(0, 0, canvas.width, canvas.height).data.buffer
        );
        return !pixelBuffer.some(color => color !== 0);
    }

    displayResults(result) {
        // Hide comparison results and show regular results
        const comparisonSection = document.getElementById('comparison-results');
        const resultsSection = document.getElementById('results-section');
        
        comparisonSection.style.display = 'none';
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });

        // Update flower emoji and name
        const flowerEmoji = document.getElementById('flower-emoji');
        const predictedClass = document.getElementById('predicted-class');
        
        flowerEmoji.textContent = this.flowerEmojis[result.predicted_class] || '🌸';
        predictedClass.textContent = result.predicted_class;

        // Update confidence circle
        const confidence = Math.round(result.confidence * 100);
        const confidenceValue = document.getElementById('confidence-value');
        const confidenceCircle = document.getElementById('confidence-circle');
        
        confidenceValue.textContent = `${confidence}%`;
        
        // Animate confidence circle
        const circumference = 2 * Math.PI * 45;
        const offset = circumference - (confidence / 100) * circumference;
        confidenceCircle.style.strokeDashoffset = offset;

        // Update all predictions list
        const predictionsList = document.getElementById('predictions-list');
        predictionsList.innerHTML = '';
        
        result.all_predictions
            .sort((a, b) => b.probability - a.probability)
            .forEach((pred, index) => {
                const probability = Math.round(pred.probability * 100);
                const isTop = index === 0;
                
                const item = document.createElement('div');
                item.className = 'prediction-item';
                item.innerHTML = `
                    <span class="flower-name">${this.flowerEmojis[pred.class] || '🌸'} ${pred.class}</span>
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <div class="probability-bar">
                            <div class="probability-fill" style="width: ${probability}%"></div>
                        </div>
                        <span class="probability">${probability}%</span>
                    </div>
                `;
                predictionsList.appendChild(item);
            });

        // Update model info
        document.getElementById('model-used').textContent = result.model_used;
        document.getElementById('model-accuracy').textContent = `${result.model_accuracy.toFixed(2)}%`;

        // Add animation class
        resultsSection.classList.add('fadeInUp');
    }

    displayComparisonResults(data) {
        // Hide regular results and show comparison results
        const resultsSection = document.getElementById('results-section');
        const comparisonSection = document.getElementById('comparison-results');
        
        if (!comparisonSection) {
            console.error('Comparison results element not found!');
            return;
        }
        
        // Hide regular results section
        resultsSection.style.display = 'none';
        
        // Show comparison results section
        comparisonSection.style.display = 'block';
        
        const { comparison_results, summary } = data;

        // Display consensus summary
        this.displayConsensusSummary(summary);

        // Display model comparison grid
        this.displayModelComparisonGrid(comparison_results, summary);
        
        // Scroll to comparison results
        comparisonSection.scrollIntoView({ behavior: 'smooth' });
    }

    displayConsensusSummary(summary) {
        const consensusSummary = document.getElementById('consensus-summary');
        
        if (!consensusSummary) {
            console.error('Consensus summary element not found!');
            return;
        }
        
        if (summary.consensus_class) {
            const emoji = this.flowerEmojis[summary.consensus_class] || '🌸';
            
            consensusSummary.innerHTML = `
                <div class="consensus-title">
                    <i class="fas fa-vote-yea"></i>
                    Model Consensus
                </div>
                <div class="consensus-class">
                    ${emoji} ${summary.consensus_class}
                </div>
                <p>Agreement: ${summary.consensus_percentage.toFixed(1)}% of models agree</p>
                <div class="agreement-level agreement-${summary.agreement_level.toLowerCase()}">
                    ${summary.agreement_level} Consensus
                </div>
                <div class="consensus-stats">
                    <div class="consensus-stat">
                        <div class="consensus-stat-label">Models Tested</div>
                        <div class="consensus-stat-value">${summary.total_models}</div>
                    </div>
                    <div class="consensus-stat">
                        <div class="consensus-stat-label">Successful Predictions</div>
                        <div class="consensus-stat-value">${summary.successful_predictions}</div>
                    </div>
                    <div class="consensus-stat">
                        <div class="consensus-stat-label">Consensus Confidence</div>
                        <div class="consensus-stat-value">${summary.consensus_confidence ? (summary.consensus_confidence * 100).toFixed(1) : 'N/A'}%</div>
                    </div>
                    <div class="consensus-stat">
                        <div class="consensus-stat-label">Agreement Level</div>
                        <div class="consensus-stat-value">${summary.agreement_level}</div>
                    </div>
                </div>
            `;
        } else {
            consensusSummary.innerHTML = `
                <div class="consensus-title">
                    <i class="fas fa-exclamation-triangle"></i>
                    No Consensus
                </div>
                <p>Unable to generate consensus - no successful predictions</p>
            `;
        }
    }

    displayModelComparisonGrid(comparison_results, summary) {
        const grid = document.getElementById('model-comparison-grid');
        
        if (!grid) {
            console.error('Model comparison grid element not found!');
            return;
        }
        
        grid.innerHTML = '';

        Object.entries(comparison_results).forEach(([modelKey, result]) => {
            const isConsensus = summary.consensus_class && 
                              result.predicted_class === summary.consensus_class;
            const hasError = 'error' in result;

            const card = document.createElement('div');
            card.className = `comparison-model-card ${isConsensus ? 'consensus' : ''} ${hasError ? 'error' : ''}`;
            card.style.setProperty('--model-color', result.color);

            if (hasError) {
                card.innerHTML = `
                    <div class="comparison-model-header">
                        <div class="comparison-model-icon">${result.icon}</div>
                        <div class="comparison-model-name">${result.model_name}</div>
                    </div>
                    <div class="comparison-error">
                        Error: ${result.error}
                    </div>
                    <div class="comparison-model-stats">
                        <div class="comparison-stat">
                            <div class="comparison-stat-label">Accuracy</div>
                            <div class="comparison-stat-value">${result.accuracy.toFixed(2)}%</div>
                        </div>
                        <div class="comparison-stat">
                            <div class="comparison-stat-label">Efficiency</div>
                            <div class="comparison-stat-value">${result.efficiency.toFixed(0)}</div>
                        </div>
                    </div>
                `;
            } else {
                const emoji = this.flowerEmojis[result.predicted_class] || '🌸';
                
                card.innerHTML = `
                    <div class="comparison-model-header">
                        <div class="comparison-model-icon">${result.icon}</div>
                        <div class="comparison-model-name">${result.model_name}</div>
                        ${isConsensus ? '<div class="consensus-badge">Consensus</div>' : ''}
                    </div>
                    <div class="comparison-prediction">
                        <div class="comparison-flower-name">${emoji} ${result.predicted_class}</div>
                        <div class="comparison-confidence">${(result.confidence * 100).toFixed(1)}% Confidence</div>
                    </div>
                    <div class="comparison-model-stats">
                        <div class="comparison-stat">
                            <div class="comparison-stat-label">Accuracy</div>
                            <div class="comparison-stat-value">${result.accuracy.toFixed(2)}%</div>
                        </div>
                        <div class="comparison-stat">
                            <div class="comparison-stat-label">Efficiency</div>
                            <div class="comparison-stat-value">${result.efficiency.toFixed(0)}</div>
                        </div>
                        <div class="comparison-stat">
                            <div class="comparison-stat-label">GFLOPs</div>
                            <div class="comparison-stat-value">${result.gflops.toFixed(3)}</div>
                        </div>
                        <div class="comparison-stat">
                            <div class="comparison-stat-label">Parameters</div>
                            <div class="comparison-stat-value">${result.parameters}</div>
                        </div>
                    </div>
                `;
            }

            grid.appendChild(card);
        });
    }

    showLoading(show) {
        const loadingOverlay = document.getElementById('loading-overlay');
        if (show) {
            loadingOverlay.classList.add('active');
        } else {
            loadingOverlay.classList.remove('active');
        }
    }

    showError(message) {
        alert(message); // You can replace this with a nice toast notification
    }

    // Flower browsing functionality
    async loadFlowerImages() {
        try {
            const categoriesContainer = document.getElementById('flower-categories');
            categoriesContainer.innerHTML = `
                <div class="loading-flowers">
                    <div class="spinner-small"></div>
                    <p>Loading flower images...</p>
                </div>
            `;

            const response = await fetch('/api/flower-images');
            if (!response.ok) {
                throw new Error('Failed to load flower images');
            }

            const flowerImages = await response.json();
            this.flowerImages = flowerImages;
            this.displayFlowerImages(flowerImages);

        } catch (error) {
            console.error('Error loading flower images:', error);
            const categoriesContainer = document.getElementById('flower-categories');
            categoriesContainer.innerHTML = `
                <div class="error-message">
                    <p>Failed to load flower images. Please try again.</p>
                </div>
            `;
        }
    }

    displayFlowerImages(flowerImages) {
        const categoriesContainer = document.getElementById('flower-categories');
        categoriesContainer.innerHTML = '';

        Object.entries(flowerImages).forEach(([flowerClass, images]) => {
            const categoryDiv = document.createElement('div');
            categoryDiv.className = 'flower-category';
            
            const emoji = this.flowerEmojis[flowerClass] || '🌸';
            
            categoryDiv.innerHTML = `
                <div class="category-header">
                    <span>${emoji} ${flowerClass}</span>
                    <span class="category-count">${images.length} images</span>
                </div>
                <div class="flower-grid" id="grid-${flowerClass}">
                    ${images.map(image => `
                        <div class="flower-item" data-class="${image.class}" data-filename="${image.filename}" data-path="${image.path}">
                            <img src="/flowers/${image.class}/${image.filename}" 
                                 alt="${image.class}" 
                                 loading="lazy"
                                 onerror="this.parentElement.style.display='none'">
                        </div>
                    `).join('')}
                </div>
            `;
            
            categoriesContainer.appendChild(categoryDiv);
        });

        // Add click handlers to flower items
        document.querySelectorAll('.flower-item').forEach(item => {
            item.addEventListener('click', (e) => {
                this.selectFlowerImage(e.currentTarget);
            });
        });
    }

    selectFlowerImage(itemElement) {
        // Remove previous selection
        document.querySelectorAll('.flower-item').forEach(item => {
            item.classList.remove('selected');
        });

        // Add selection to clicked item
        itemElement.classList.add('selected');

        // Store selected image data
        this.selectedFlowerImage = {
            class: itemElement.dataset.class,
            filename: itemElement.dataset.filename,
            path: itemElement.dataset.path
        };

        // Show preview
        this.showFlowerPreview();
    }

    showFlowerPreview() {
        const previewContainer = document.getElementById('selected-flower-preview');
        const selectedImage = document.getElementById('selected-flower-image');
        const flowerName = document.getElementById('selected-flower-name');
        const flowerClass = document.getElementById('selected-flower-class');
        const flowerFile = document.getElementById('selected-flower-file');

        if (this.selectedFlowerImage) {
            selectedImage.src = `/flowers/${this.selectedFlowerImage.class}/${this.selectedFlowerImage.filename}`;
            flowerName.textContent = this.selectedFlowerImage.class;
            flowerClass.querySelector('span').textContent = this.selectedFlowerImage.class;
            flowerFile.querySelector('span').textContent = this.selectedFlowerImage.filename;
            
            previewContainer.style.display = 'block';
            previewContainer.scrollIntoView({ behavior: 'smooth' });
        }
    }

    clearFlowerSelection() {
        // Clear selection
        document.querySelectorAll('.flower-item').forEach(item => {
            item.classList.remove('selected');
        });

        this.selectedFlowerImage = null;
        document.getElementById('selected-flower-preview').style.display = 'none';
    }



    init3DBackground() {
        // Simple particle system with Three.js
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ alpha: true });
        
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setClearColor(0x000000, 0);
        
        const container = document.getElementById('three-container');
        container.appendChild(renderer.domElement);

        // Create floating particles
        const particles = new THREE.BufferGeometry();
        const particleCount = 100;
        const positions = new Float32Array(particleCount * 3);

        for (let i = 0; i < particleCount * 3; i++) {
            positions[i] = (Math.random() - 0.5) * 20;
        }

        particles.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        const material = new THREE.PointsMaterial({
            color: 0x4ECDC4,
            size: 0.1,
            transparent: true,
            opacity: 0.6
        });

        const particleSystem = new THREE.Points(particles, material);
        scene.add(particleSystem);

        camera.position.z = 5;

        // Animation loop
        function animate() {
            requestAnimationFrame(animate);

            particleSystem.rotation.x += 0.001;
            particleSystem.rotation.y += 0.002;

            renderer.render(scene, camera);
        }

        animate();

        // Handle window resize
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.flowerApp = new FlowerClassificationApp();
});