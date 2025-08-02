/**
 * Computer Vision Assignment 3 - Main JavaScript
 * Common functionality and utilities for the web application
 */

// Global application state
const CVApp = {
    models: {},
    currentImage: null,
    isLoading: false,
    
    // Initialize the application
    init() {
        this.setupEventListeners();
        this.loadModels();
        this.initializeAnimations();
        console.log('🚀 Computer Vision Assignment 3 App Initialized');
    },
    
    // Setup global event listeners
    setupEventListeners() {
        // Handle loading states
        document.addEventListener('htmx:beforeRequest', () => this.showLoading(true));
        document.addEventListener('htmx:afterRequest', () => this.showLoading(false));
        
        // Handle navigation
        document.addEventListener('DOMContentLoaded', () => {
            this.setupNavigation();
            this.setupScrollEffects();
        });
        
        // Handle window resize
        window.addEventListener('resize', this.debounce(() => {
            this.handleResize();
        }, 250));
    },
    
    // Load model information
    async loadModels() {
        try {
            const response = await fetch('/api/models');
            this.models = await response.json();
            console.log(`✅ Loaded ${Object.keys(this.models).length} models`);
        } catch (error) {
            console.error('❌ Failed to load models:', error);
            this.showToast('Failed to load model information', 'error');
        }
    },
    
    // Setup navigation highlighting
    setupNavigation() {
        const currentPath = window.location.pathname;
        const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
        
        navLinks.forEach(link => {
            if (link.getAttribute('href') === currentPath) {
                link.classList.add('active');
            }
        });
    },
    
    // Setup scroll effects
    setupScrollEffects() {
        // Parallax effect for hero section
        const hero = document.querySelector('.hero-section');
        if (hero) {
            window.addEventListener('scroll', () => {
                const scrolled = window.pageYOffset;
                const rate = scrolled * -0.5;
                hero.style.transform = `translateY(${rate}px)`;
            });
        }
        
        // Fade in animation for cards
        this.observeElements('.card', 'fade-in');
        this.observeElements('.stats-card', 'slide-in');
    },
    
    // Initialize animations
    initializeAnimations() {
        // GSAP animations if available
        if (typeof gsap !== 'undefined' && typeof ScrollTrigger !== 'undefined') {
            gsap.registerPlugin(ScrollTrigger);
            
            // Animate stats cards
            gsap.from('.stats-card', {
                duration: 0.8,
                y: 50,
                opacity: 0,
                stagger: 0.2,
                ease: 'back.out(1.7)',
                scrollTrigger: {
                    trigger: '.stats-card',
                    start: 'top 80%'
                }
            });
            
            // Animate model cards
            gsap.from('.model-card', {
                duration: 0.6,
                y: 30,
                opacity: 0,
                stagger: 0.1,
                ease: 'power2.out',
                scrollTrigger: {
                    trigger: '.model-card',
                    start: 'top 85%'
                }
            });
        } else {
            // Fallback CSS animations
            this.setupFallbackAnimations();
        }
    },
    
    // Fallback animations without GSAP
    setupFallbackAnimations() {
        const cards = document.querySelectorAll('.stats-card, .model-card');
        cards.forEach((card, index) => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(30px)';
            card.style.transition = 'all 0.6s ease';
            
            setTimeout(() => {
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, index * 100);
        });
    },
    
    // Observe elements for animations
    observeElements(selector, animationClass) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add(animationClass);
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.1 });
        
        document.querySelectorAll(selector).forEach(el => {
            observer.observe(el);
        });
    },
    
    // Handle window resize
    handleResize() {
        // Trigger custom resize event for Three.js scenes
        window.dispatchEvent(new CustomEvent('cvapp:resize'));
    },
    
    // Show/hide loading overlay
    showLoading(show) {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.style.display = show ? 'flex' : 'none';
        }
        this.isLoading = show;
    },
    
    // Show toast notification
    showToast(message, type = 'success') {
        const toastId = type === 'error' ? 'errorToast' : 'successToast';
        const toastElement = document.getElementById(toastId);
        
        if (toastElement) {
            const bodyElement = toastElement.querySelector('.toast-body');
            if (bodyElement) {
                bodyElement.textContent = message;
            }
            
            const toast = new bootstrap.Toast(toastElement);
            toast.show();
        }
    },
    
    // Utility: Debounce function
    debounce(func, wait, immediate) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                timeout = null;
                if (!immediate) func(...args);
            };
            const callNow = immediate && !timeout;
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
            if (callNow) func(...args);
        };
    },
    
    // Utility: Throttle function
    throttle(func, limit) {
        let inThrottle;
        return function(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    },
    
    // Format number with commas
    formatNumber(num) {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    },
    
    // Format percentage
    formatPercentage(num, decimals = 2) {
        return `${(num * 100).toFixed(decimals)}%`;
    },
    
    // Get model color based on experiment
    getModelColor(modelName) {
        if (modelName.includes('q15')) return '#007bff'; // Blue
        if (modelName.includes('q17')) return '#28a745'; // Green
        if (modelName.includes('q2')) return '#ffc107';  // Yellow
        return '#6c757d'; // Gray
    },
    
    // Get architecture icon
    getArchitectureIcon(archType) {
        switch (archType.toLowerCase()) {
            case 'cnn': return 'fas fa-project-diagram';
            case 'mlp': return 'fas fa-cube';
            default: return 'fas fa-network-wired';
        }
    }
};

// Image handling utilities
const ImageUtils = {
    // Convert canvas to base64
    canvasToBase64(canvas) {
        return canvas.toDataURL('image/png');
    },
    
    // Resize image to 28x28
    resizeImage(imageElement, size = 28) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = size;
        canvas.height = size;
        
        // Draw image scaled to canvas
        ctx.drawImage(imageElement, 0, 0, size, size);
        
        return canvas;
    },
    
    // Convert file to base64
    fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    },
    
    // Load image from URL
    loadImage(url) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = url;
        });
    },
    
    // Preprocess image for model
    preprocessImage(imageData) {
        // This would typically involve normalization
        // For now, just return the image data
        return imageData;
    }
};

// Drawing utilities for canvas
const DrawingUtils = {
    // Setup drawing canvas
    setupCanvas(canvas, options = {}) {
        const ctx = canvas.getContext('2d');
        const defaults = {
            strokeStyle: 'black',
            lineWidth: 8,
            lineCap: 'round',
            lineJoin: 'round',
            fillStyle: 'white'
        };
        
        const settings = { ...defaults, ...options };
        
        // Apply settings
        Object.keys(settings).forEach(key => {
            ctx[key] = settings[key];
        });
        
        // Clear canvas
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        return ctx;
    },
    
    // Add drawing events to canvas
    addDrawingEvents(canvas, ctx) {
        let isDrawing = false;
        let lastX = 0;
        let lastY = 0;
        
        function getCoordinates(e) {
            const rect = canvas.getBoundingClientRect();
            return {
                x: e.clientX - rect.left,
                y: e.clientY - rect.top
            };
        }
        
        canvas.addEventListener('mousedown', (e) => {
            isDrawing = true;
            const coords = getCoordinates(e);
            [lastX, lastY] = [coords.x, coords.y];
            
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
        });
        
        canvas.addEventListener('mousemove', (e) => {
            if (!isDrawing) return;
            
            const coords = getCoordinates(e);
            ctx.lineTo(coords.x, coords.y);
            ctx.stroke();
            
            [lastX, lastY] = [coords.x, coords.y];
        });
        
        canvas.addEventListener('mouseup', () => {
            if (isDrawing) {
                isDrawing = false;
                ctx.closePath();
                
                // Trigger custom event
                canvas.dispatchEvent(new CustomEvent('drawingComplete', {
                    detail: { imageData: canvas.toDataURL() }
                }));
            }
        });
        
        canvas.addEventListener('mouseout', () => {
            if (isDrawing) {
                isDrawing = false;
                ctx.closePath();
            }
        });
        
        // Touch events for mobile
        canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousedown', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            canvas.dispatchEvent(mouseEvent);
        });
        
        canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousemove', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            canvas.dispatchEvent(mouseEvent);
        });
        
        canvas.addEventListener('touchend', (e) => {
            e.preventDefault();
            const mouseEvent = new MouseEvent('mouseup', {});
            canvas.dispatchEvent(mouseEvent);
        });
    },
    
    // Clear canvas
    clearCanvas(canvas, ctx) {
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
};

// API utilities
const APIUtils = {
    // Base fetch wrapper
    async fetchAPI(url, options = {}) {
        const defaults = {
            headers: {
                'Content-Type': 'application/json',
            },
        };
        
        const config = { ...defaults, ...options };
        
        try {
            CVApp.showLoading(true);
            const response = await fetch(url, config);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('API Error:', error);
            CVApp.showToast(`API Error: ${error.message}`, 'error');
            throw error;
        } finally {
            CVApp.showLoading(false);
        }
    },
    
    // Get model prediction
    async predict(modelName, imageData) {
        return this.fetchAPI(`/api/predict/${modelName}`, {
            method: 'POST',
            body: JSON.stringify({ image_data: imageData })
        });
    },
    
    // Compare multiple models
    async compare(modelNames, imageData) {
        return this.fetchAPI('/api/compare', {
            method: 'POST',
            body: JSON.stringify({
                models: modelNames,
                image_data: imageData
            })
        });
    },
    
    // Get random sample
    async getSample(index) {
        return this.fetchAPI(`/api/sample/${index}`);
    },
    
    // Get all models
    async getModels() {
        return this.fetchAPI('/api/models');
    }
};

// Chart utilities
const ChartUtils = {
    // Default chart options
    getDefaultOptions() {
        return {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: 'white',
                    bodyColor: 'white',
                    cornerRadius: 8,
                    padding: 12
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                }
            }
        };
    },
    
    // Create performance chart
    createPerformanceChart(ctx, data) {
        return new Chart(ctx, {
            type: 'bar',
            data: data,
            options: {
                ...this.getDefaultOptions(),
                scales: {
                    ...this.getDefaultOptions().scales,
                    y: {
                        ...this.getDefaultOptions().scales.y,
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    },
    
    // Create comparison chart
    createComparisonChart(ctx, data) {
        return new Chart(ctx, {
            type: 'scatter',
            data: data,
            options: {
                ...this.getDefaultOptions(),
                scales: {
                    ...this.getDefaultOptions().scales,
                    x: {
                        ...this.getDefaultOptions().scales.x,
                        title: {
                            display: true,
                            text: 'Parameters (Millions)'
                        }
                    },
                    y: {
                        ...this.getDefaultOptions().scales.y,
                        title: {
                            display: true,
                            text: 'Accuracy (%)'
                        }
                    }
                }
            }
        });
    }
};

// Smooth scrolling utility
function smoothScrollTo(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }
}

// Copy to clipboard utility
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        CVApp.showToast('Copied to clipboard!', 'success');
    } catch (err) {
        console.error('Failed to copy:', err);
        CVApp.showToast('Failed to copy to clipboard', 'error');
    }
}

// Download utility
function downloadData(data, filename, type = 'application/json') {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    CVApp.init();
});

// Export utilities for use in other modules
window.CVApp = CVApp;
window.ImageUtils = ImageUtils;
window.DrawingUtils = DrawingUtils;
window.APIUtils = APIUtils;
window.ChartUtils = ChartUtils;