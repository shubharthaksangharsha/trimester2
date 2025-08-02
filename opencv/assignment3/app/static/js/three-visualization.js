/**
 * Computer Vision Assignment 3 - Three.js Background Wallpaper
 * Beautiful 3D neural network visualization as a background
 */

class NeuralNetworkBackground {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.networks = [];
        this.animationFrameId = null;
        this.isInitialized = false;
        
        this.init();
    }
    
    init() {
        try {
            this.setupScene();
            this.setupCamera();
            this.setupRenderer();
            this.setupLighting();
            this.createMultipleNetworks();
            this.animate();
            this.isInitialized = true;
            
            // Handle window resize
            window.addEventListener('resize', this.onWindowResize.bind(this));
            
            console.log('✅ Three.js background initialized successfully');
        } catch (error) {
            console.error('❌ Failed to initialize Three.js background:', error);
            this.createFallbackBackground();
        }
    }
    
    setupScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a1a); // Dark blue background
        
        // Add fog for depth
        this.scene.fog = new THREE.Fog(0x0a0a1a, 10, 50);
    }
    
    setupCamera() {
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.camera.position.set(0, 0, 20);
    }
    
    setupRenderer() {
        this.renderer = new THREE.WebGLRenderer({ 
            antialias: true, 
            alpha: true,
            powerPreference: "high-performance"
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        
        // Position as background
        this.renderer.domElement.style.position = 'fixed';
        this.renderer.domElement.style.top = '0';
        this.renderer.domElement.style.left = '0';
        this.renderer.domElement.style.width = '100%';
        this.renderer.domElement.style.height = '100%';
        this.renderer.domElement.style.zIndex = '-1';
        this.renderer.domElement.style.pointerEvents = 'none';
        
        document.body.appendChild(this.renderer.domElement);
    }
    
    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        this.scene.add(ambientLight);
        
        // Directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 5);
        this.scene.add(directionalLight);
        
        // Point lights for ambiance
        const pointLight1 = new THREE.PointLight(0x667eea, 0.5, 100);
        pointLight1.position.set(10, 10, 10);
        this.scene.add(pointLight1);
        
        const pointLight2 = new THREE.PointLight(0x764ba2, 0.5, 100);
        pointLight2.position.set(-10, -10, 10);
        this.scene.add(pointLight2);
        
        // Additional colored lights
        const pointLight3 = new THREE.PointLight(0xff6b6b, 0.3, 80);
        pointLight3.position.set(15, -5, 15);
        this.scene.add(pointLight3);
        
        const pointLight4 = new THREE.PointLight(0x4ecdc4, 0.3, 80);
        pointLight4.position.set(-15, 5, -15);
        this.scene.add(pointLight4);
    }
    
    createMultipleNetworks() {
        // Create 3 different neural networks at different positions
        const networkConfigs = [
            {
                position: { x: -8, y: 0, z: -5 },
                scale: 0.8,
                color: 0x667eea,
                nodeCount: 20,
                rotationSpeed: 0.002
            },
            {
                position: { x: 8, y: 2, z: -8 },
                scale: 1.2,
                color: 0x764ba2,
                nodeCount: 25,
                rotationSpeed: 0.003
            },
            {
                position: { x: 0, y: -3, z: -10 },
                scale: 1.0,
                color: 0xff6b6b,
                nodeCount: 18,
                rotationSpeed: 0.0015
            }
        ];
        
        networkConfigs.forEach((config, index) => {
            const network = this.createNetwork(config, index);
            this.networks.push(network);
            this.scene.add(network);
        });
    }
    
    createNetwork(config, networkIndex) {
        const networkGroup = new THREE.Group();
        networkGroup.position.set(config.position.x, config.position.y, config.position.z);
        networkGroup.scale.setScalar(config.scale);
        networkGroup.userData = {
            isNetwork: true,
            rotationSpeed: config.rotationSpeed,
            networkIndex: networkIndex
        };
        
        // Create nodes in a network pattern
        const nodeMaterial = new THREE.MeshPhongMaterial({ 
            color: config.color, 
            shininess: 100,
            transparent: true,
            opacity: 0.8
        });
        
        const nodes = [];
        for (let i = 0; i < config.nodeCount; i++) {
            const geometry = new THREE.SphereGeometry(0.12, 16, 16);
            const node = new THREE.Mesh(geometry, nodeMaterial.clone());
            
            // Position nodes in a more complex network pattern
            const angle = (i / config.nodeCount) * Math.PI * 2;
            const radius = 2 + Math.sin(angle * 3) * 1.5;
            const height = Math.cos(angle * 2) * 1.5;
            const depth = Math.sin(angle * 1.5) * 1;
            
            node.position.set(
                Math.cos(angle) * radius,
                height,
                Math.sin(angle) * radius + depth
            );
            
            // Add animation properties
            node.userData.originalPosition = node.position.clone();
            node.userData.animationOffset = Math.random() * Math.PI * 2;
            node.userData.velocity = new THREE.Vector3(
                (Math.random() - 0.5) * 0.015,
                (Math.random() - 0.5) * 0.015,
                (Math.random() - 0.5) * 0.015
            );
            
            nodes.push(node);
            networkGroup.add(node);
        }
        
        // Create connections between nearby nodes
        const lineMaterial = new THREE.LineBasicMaterial({ 
            color: config.color, 
            transparent: true, 
            opacity: 0.2 
        });
        
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const distance = nodes[i].position.distanceTo(nodes[j].position);
                if (distance < 3) {
                    const geometry = new THREE.BufferGeometry().setFromPoints([
                        nodes[i].position,
                        nodes[j].position
                    ]);
                    const line = new THREE.Line(geometry, lineMaterial);
                    networkGroup.add(line);
                }
            }
        }
        
        return networkGroup;
    }
    
    animate() {
        this.animationFrameId = requestAnimationFrame(this.animate.bind(this));
        
        // Animate each network
        this.networks.forEach((network, networkIndex) => {
            // Rotate the entire network
            network.rotation.y += network.userData.rotationSpeed;
            network.rotation.x += network.userData.rotationSpeed * 0.3;
            
            // Animate individual nodes
            network.children.forEach((child, childIndex) => {
                if (child.type === 'Mesh') {
                    const time = Date.now() * 0.001;
                    
                    // Floating animation
                    child.position.y += Math.sin(time + child.userData.animationOffset) * 0.003;
                    child.position.x += Math.cos(time + child.userData.animationOffset) * 0.002;
                    
                    // Pulsing scale
                    const scale = 1 + Math.sin(time * 2 + child.userData.animationOffset) * 0.15;
                    child.scale.setScalar(scale);
                    
                    // Occasional color change
                    if (Math.random() < 0.0005) {
                        child.material.emissive.setHex(0x667eea);
                        child.material.emissiveIntensity = 0.4;
                        setTimeout(() => {
                            child.material.emissive.setHex(0x000000);
                            child.material.emissiveIntensity = 0;
                        }, 400);
                    }
                }
            });
        });
        
        // Slow rotation of the entire scene
        this.scene.rotation.y += 0.0003;
        this.scene.rotation.x += 0.0001;
        
        this.renderer.render(this.scene, this.camera);
    }
    
    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }
    
    createFallbackBackground() {
        // Create a simple CSS gradient background as fallback
        const style = document.createElement('style');
        style.textContent = `
            body::before {
                content: '';
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #ff6b6b 100%);
                z-index: -1;
                animation: gradientShift 15s ease-in-out infinite;
            }
            
            @keyframes gradientShift {
                0%, 100% { background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #ff6b6b 100%); }
                33% { background: linear-gradient(135deg, #764ba2 0%, #ff6b6b 50%, #667eea 100%); }
                66% { background: linear-gradient(135deg, #ff6b6b 0%, #667eea 50%, #764ba2 100%); }
            }
        `;
        document.head.appendChild(style);
        console.log('✅ Fallback CSS background created');
    }
    
    destroy() {
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
        }
        
        if (this.renderer && this.renderer.domElement) {
            document.body.removeChild(this.renderer.domElement);
            this.renderer.dispose();
        }
        
        window.removeEventListener('resize', this.onWindowResize.bind(this));
    }
}

// Hero Visualizer for the hero section
class HeroVisualizer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container ${containerId} not found`);
            return;
        }
        
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.network = null;
        this.animationFrameId = null;
        
        this.init();
    }
    
    init() {
        try {
            this.setupScene();
            this.setupCamera();
            this.setupRenderer();
            this.setupLighting();
            this.createHeroNetwork();
            this.animate();
            
            console.log('✅ Hero visualizer initialized successfully');
        } catch (error) {
            console.error('❌ Failed to initialize hero visualizer:', error);
            this.createFallbackHero();
        }
    }
    
    setupScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a2e); // Darker background for hero
    }
    
    setupCamera() {
        const rect = this.container.getBoundingClientRect();
        const aspect = rect.width / rect.height;
        this.camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
        this.camera.position.set(0, 0, 8);
    }
    
    setupRenderer() {
        const rect = this.container.getBoundingClientRect();
        this.renderer = new THREE.WebGLRenderer({ 
            antialias: true, 
            alpha: true,
            powerPreference: "high-performance"
        });
        this.renderer.setSize(rect.width, rect.height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        
        this.container.appendChild(this.renderer.domElement);
        
        // Handle resize
        const resizeObserver = new ResizeObserver(() => this.handleResize());
        resizeObserver.observe(this.container);
    }
    
    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.8);
        this.scene.add(ambientLight);
        
        // Directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
        directionalLight.position.set(5, 5, 5);
        this.scene.add(directionalLight);
        
        // Point light for hero network
        const pointLight = new THREE.PointLight(0x00ff88, 0.8, 20);
        pointLight.position.set(0, 0, 5);
        this.scene.add(pointLight);
    }
    
    createHeroNetwork() {
        // Create a single, beautiful neural network for the hero section
        this.network = new THREE.Group();
        this.network.userData = { isHeroNetwork: true };
        
        // Unique color for hero network - vibrant green
        const heroColor = 0x00ff88;
        const nodeMaterial = new THREE.MeshPhongMaterial({ 
            color: heroColor, 
            shininess: 100,
            transparent: true,
            opacity: 0.9
        });
        
        // Create a more complex network structure
        const layers = [8, 12, 10, 6]; // Input, hidden1, hidden2, output
        const layerSpacing = 2;
        const nodes = [];
        
        layers.forEach((nodeCount, layerIndex) => {
            const layerGroup = new THREE.Group();
            const layerNodes = [];
            
            const startY = -(nodeCount - 1) * 0.4 / 2;
            
            for (let i = 0; i < nodeCount; i++) {
                const geometry = new THREE.SphereGeometry(0.08, 16, 16);
                const node = new THREE.Mesh(geometry, nodeMaterial.clone());
                
                node.position.set(
                    layerIndex * layerSpacing - (layers.length - 1) * layerSpacing / 2,
                    startY + i * 0.4,
                    0
                );
                
                // Add animation properties
                node.userData.originalPosition = node.position.clone();
                node.userData.animationOffset = Math.random() * Math.PI * 2;
                node.userData.layerIndex = layerIndex;
                node.userData.nodeIndex = i;
                
                layerNodes.push(node);
                layerGroup.add(node);
                nodes.push(node);
            }
            
            this.network.add(layerGroup);
        });
        
        // Create connections between layers
        const lineMaterial = new THREE.LineBasicMaterial({ 
            color: heroColor, 
            transparent: true, 
            opacity: 0.4 
        });
        
        for (let layerIndex = 0; layerIndex < layers.length - 1; layerIndex++) {
            const currentLayerCount = layers[layerIndex];
            const nextLayerCount = layers[layerIndex + 1];
            
            for (let i = 0; i < currentLayerCount; i++) {
                for (let j = 0; j < nextLayerCount; j++) {
                    const fromNode = nodes.find(n => n.userData.layerIndex === layerIndex && n.userData.nodeIndex === i);
                    const toNode = nodes.find(n => n.userData.layerIndex === layerIndex + 1 && n.userData.nodeIndex === j);
                    
                    if (fromNode && toNode) {
                        const geometry = new THREE.BufferGeometry().setFromPoints([
                            fromNode.position,
                            toNode.position
                        ]);
                        const line = new THREE.Line(geometry, lineMaterial);
                        this.network.add(line);
                    }
                }
            }
        }
        
        this.scene.add(this.network);
    }
    
    animate() {
        this.animationFrameId = requestAnimationFrame(this.animate.bind(this));
        
        if (this.network) {
            // Rotate the entire network slowly
            this.network.rotation.y += 0.005;
            this.network.rotation.x += 0.002;
            
            // Animate individual nodes
            this.network.children.forEach((layerGroup, layerIndex) => {
                layerGroup.children.forEach((node, nodeIndex) => {
                    if (node.type === 'Mesh') {
                        const time = Date.now() * 0.001;
                        
                        // Floating animation
                        const floatOffset = Math.sin(time + node.userData.animationOffset) * 0.02;
                        node.position.y = node.userData.originalPosition.y + floatOffset;
                        
                        // Pulsing scale
                        const scale = 1 + Math.sin(time * 3 + node.userData.animationOffset) * 0.1;
                        node.scale.setScalar(scale);
                        
                        // Layer-based color variation
                        const hue = (0.3 + layerIndex * 0.1) % 1;
                        const color = new THREE.Color().setHSL(hue, 0.8, 0.6);
                        node.material.color.copy(color);
                        
                        // Occasional glow effect
                        if (Math.random() < 0.001) {
                            node.material.emissive.setHex(0x00ff88);
                            node.material.emissiveIntensity = 0.5;
                            setTimeout(() => {
                                node.material.emissive.setHex(0x000000);
                                node.material.emissiveIntensity = 0;
                            }, 300);
                        }
                    }
                });
            });
        }
        
        this.renderer.render(this.scene, this.camera);
    }
    
    handleResize() {
        const rect = this.container.getBoundingClientRect();
        this.camera.aspect = rect.width / rect.height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(rect.width, rect.height);
    }
    
    createFallbackHero() {
        // Create a simple CSS gradient for hero section
        this.container.style.background = 'linear-gradient(135deg, #00ff88 0%, #00cc6a 50%, #00994d 100%)';
        this.container.style.borderRadius = '10px';
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        
        // Add animated elements
        const style = document.createElement('style');
        style.textContent = `
            #${this.container.id}::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: radial-gradient(circle at 30% 30%, rgba(0, 255, 136, 0.3) 0%, transparent 50%);
                animation: pulse 3s ease-in-out infinite;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 0.3; transform: scale(1); }
                50% { opacity: 0.6; transform: scale(1.1); }
            }
        `;
        document.head.appendChild(style);
        
        console.log('✅ Fallback hero visualizer created');
    }
    
    destroy() {
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
        }
        
        if (this.renderer && this.renderer.domElement) {
            this.container.removeChild(this.renderer.domElement);
            this.renderer.dispose();
        }
    }
}

// Model Parameter Drawer Management
class ModelDrawerManager {
    constructor() {
        this.isOpen = false;
        this.currentModel = null;
        this.init();
    }
    
    init() {
        this.createDrawerHTML();
        this.setupEventListeners();
    }
    
    createDrawerHTML() {
        // Create drawer overlay
        const overlay = document.createElement('div');
        overlay.id = 'modelDrawerOverlay';
        overlay.className = 'model-drawer-overlay';
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            z-index: 9998;
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s ease;
        `;
        
        // Create drawer
        const drawer = document.createElement('div');
        drawer.id = 'modelDrawer';
        drawer.className = 'model-drawer';
        drawer.style.cssText = `
            position: fixed;
            top: 0;
            right: -400px;
            width: 400px;
            height: 100%;
            background: white;
            box-shadow: -2px 0 10px rgba(0, 0, 0, 0.1);
            z-index: 9999;
            transition: right 0.3s ease;
            overflow-y: auto;
        `;
        
        // Create drawer content
        drawer.innerHTML = `
            <div class="drawer-header" style="padding: 20px; border-bottom: 1px solid #eee; background: #f8f9fa;">
                <div class="d-flex justify-content-between align-items-center">
                    <h5 class="mb-0" id="drawerTitle">
                        <i class="fas fa-cube me-2"></i>Model Details
                    </h5>
                    <button id="closeDrawer" class="btn btn-sm btn-outline-secondary">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            </div>
            <div class="drawer-content" id="drawerContent" style="padding: 20px;">
                <div class="text-center text-muted">
                    <i class="fas fa-info-circle fa-3x mb-3"></i>
                    <p>Select a model to view its details</p>
                </div>
            </div>
        `;
        
        document.body.appendChild(overlay);
        document.body.appendChild(drawer);
    }
    
    setupEventListeners() {
        // Close drawer events
        document.getElementById('closeDrawer').addEventListener('click', () => this.close());
        document.getElementById('modelDrawerOverlay').addEventListener('click', () => this.close());
        
        // Handle view model parameter buttons
        document.addEventListener('click', (e) => {
            if (e.target.closest('.view-model-btn')) {
                e.preventDefault();
                const btn = e.target.closest('.view-model-btn');
                const modelName = btn.getAttribute('data-model-name');
                
                console.log('Button clicked for model:', modelName);
                console.log('Global modelData available:', !!window.modelData);
                console.log('Available models:', Object.keys(window.modelData || {}));
                
                try {
                    let modelInfo = null;
                    
                    // Get model info from global modelData
                    if (window.modelData && window.modelData[modelName]) {
                        modelInfo = window.modelData[modelName];
                        console.log('Found model info:', modelInfo);
                    }
                    
                    if (modelInfo) {
                        this.open(modelName, modelInfo);
                    } else {
                        console.error('Model info not found for:', modelName);
                        console.log('Available models:', Object.keys(window.modelData || {}));
                        
                        // Try to fetch model info from API as fallback
                        fetch(`/api/models`)
                            .then(response => response.json())
                            .then(data => {
                                if (data[modelName]) {
                                    this.open(modelName, data[modelName]);
                                } else {
                                    alert('Model information not available. Please try again.');
                                }
                            })
                            .catch(error => {
                                console.error('API fallback failed:', error);
                                alert('Model information not available. Please try again.');
                            });
                    }
                } catch (error) {
                    console.error('Error loading model info:', error);
                    alert('Unable to load model information. Please try again.');
                }
            }
        });
    }
    
    open(modelName, modelInfo) {
        this.currentModel = { name: modelName, info: modelInfo };
        
        // Update drawer content
        const content = this.generateModelContent(modelName, modelInfo);
        document.getElementById('drawerContent').innerHTML = content;
        document.getElementById('drawerTitle').innerHTML = `
            <i class="fas fa-cube me-2"></i>${modelInfo.description || modelName}
        `;
        
        // Show drawer
        document.getElementById('modelDrawerOverlay').style.opacity = '1';
        document.getElementById('modelDrawerOverlay').style.visibility = 'visible';
        document.getElementById('modelDrawer').style.right = '0';
        
        this.isOpen = true;
    }
    
    close() {
        document.getElementById('modelDrawerOverlay').style.opacity = '0';
        document.getElementById('modelDrawerOverlay').style.visibility = 'hidden';
        document.getElementById('modelDrawer').style.right = '-400px';
        
        this.isOpen = false;
        this.currentModel = null;
    }
    
    generateModelContent(modelName, modelInfo) {
        const accuracy = (modelInfo.accuracy * 100).toFixed(2);
        const parameters = this.formatNumber(modelInfo.parameters || 0);
        
        return `
            <div class="mb-4">
                <h6><i class="fas fa-info-circle me-2"></i>Architecture Details</h6>
                <div class="card">
                    <div class="card-body">
                        <p><strong>Type:</strong> ${modelInfo.architecture_type || 'Unknown'}</p>
                        <p><strong>Parameters:</strong> ${parameters}</p>
                        <p><strong>Accuracy:</strong> ${accuracy}%</p>
                        ${modelInfo.activation ? `<p><strong>Activation:</strong> ${modelInfo.activation}</p>` : ''}
                        ${modelInfo.initialization ? `<p><strong>Initialization:</strong> ${modelInfo.initialization}</p>` : ''}
                    </div>
                </div>
            </div>
            
            <div class="mb-4">
                <h6><i class="fas fa-chart-line me-2"></i>Performance Metrics</h6>
                <div class="card">
                    <div class="card-body">
                        <div class="mb-3">
                            <div class="d-flex justify-content-between">
                                <span>Test Accuracy:</span>
                                <strong>${accuracy}%</strong>
                            </div>
                            <div class="progress mt-1">
                                <div class="progress-bar" role="progressbar" style="width: ${accuracy}%"></div>
                            </div>
                        </div>
                        <p class="mb-1"><strong>Parameter Count:</strong> ${(modelInfo.parameters || 0).toLocaleString()}</p>
                        <p class="mb-0"><strong>Model Size:</strong> ${parameters} parameters</p>
                    </div>
                </div>
            </div>
            
            <div class="mb-4">
                <h6><i class="fas fa-cogs me-2"></i>Model Configuration</h6>
                <div class="card">
                    <div class="card-body">
                        <p><strong>Model Name:</strong> ${modelName}</p>
                        <p><strong>Description:</strong> ${modelInfo.description || 'No description available'}</p>
                        <p><strong>Category:</strong> ${this.getModelCategory(modelName)}</p>
                    </div>
                </div>
            </div>
        `;
    }
    
    formatNumber(num) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(2) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toLocaleString();
    }
    
    getModelCategory(modelName) {
        if (modelName.includes('q15')) return 'Q1.5 Architecture Models';
        if (modelName.includes('q17')) return 'Q1.7 CNN vs MLP';
        if (modelName.includes('q2')) return 'Q2 Activation Functions';
        return 'Other';
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Initialize Three.js background
    window.neuralBackground = new NeuralNetworkBackground();
    
    // Initialize hero visualizer
    window.heroVisualizer = new HeroVisualizer('heroVisualization');
    
    // Initialize model drawer
    window.modelDrawer = new ModelDrawerManager();
    
    console.log('🚀 Computer Vision Assignment 3 - Enhanced Background, Hero & Drawer Initialized');
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.neuralBackground) {
        window.neuralBackground.destroy();
    }
    if (window.heroVisualizer) {
        window.heroVisualizer.destroy();
    }
});

// Export for global use
window.NeuralNetworkBackground = NeuralNetworkBackground;
window.HeroVisualizer = HeroVisualizer;
window.ModelDrawerManager = ModelDrawerManager;