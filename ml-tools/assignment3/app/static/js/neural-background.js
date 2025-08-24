// Neural Network Background Visualization with Three.js
class NeuralNetworkBackground {
    constructor() {
        this.container = document.getElementById('neural-background');
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.nodes = [];
        this.connections = [];
        this.animationId = null;
        this.mouse = { x: 0, y: 0 };
        this.isAnimating = true;
        
        this.config = {
            nodeCount: 50,
            connectionDistance: 150,
            nodeSize: 2,
            connectionOpacity: 0.3,
            animationSpeed: 0.0005,
            mouseInfluence: 100,
            colors: {
                nodes: 0x667eea,
                connections: 0x764ba2,
                highlight: 0x4facfe
            }
        };
        
        this.init();
        this.setupEventListeners();
    }
    
    init() {
        this.createScene();
        this.createCamera();
        this.createRenderer();
        this.createNodes();
        this.createConnections();
        this.animate();
    }
    
    createScene() {
        this.scene = new THREE.Scene();
        this.scene.fog = new THREE.Fog(0x0f172a, 200, 1000);
    }
    
    createCamera() {
        this.camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        this.camera.position.z = 300;
    }
    
    createRenderer() {
        this.renderer = new THREE.WebGLRenderer({
            alpha: true,
            antialias: true
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setClearColor(0x0f172a, 0.1);
        this.container.appendChild(this.renderer.domElement);
    }
    
    createNodes() {
        const geometry = new THREE.SphereGeometry(this.config.nodeSize, 8, 8);
        const material = new THREE.MeshBasicMaterial({
            color: this.config.colors.nodes,
            transparent: true,
            opacity: 0.8
        });
        
        for (let i = 0; i < this.config.nodeCount; i++) {
            const node = new THREE.Mesh(geometry, material.clone());
            
            // Random position
            node.position.x = (Math.random() - 0.5) * 800;
            node.position.y = (Math.random() - 0.5) * 600;
            node.position.z = (Math.random() - 0.5) * 400;
            
            // Random velocity for animation
            node.velocity = {
                x: (Math.random() - 0.5) * 0.5,
                y: (Math.random() - 0.5) * 0.5,
                z: (Math.random() - 0.5) * 0.2
            };
            
            // Store original position for wave animation
            node.originalPosition = {
                x: node.position.x,
                y: node.position.y,
                z: node.position.z
            };
            
            this.nodes.push(node);
            this.scene.add(node);
        }
    }
    
    createConnections() {
        this.connectionGeometry = new THREE.BufferGeometry();
        this.connectionMaterial = new THREE.LineBasicMaterial({
            color: this.config.colors.connections,
            transparent: true,
            opacity: this.config.connectionOpacity,
            blending: THREE.AdditiveBlending
        });
        
        this.updateConnections();
        
        this.connectionLines = new THREE.LineSegments(
            this.connectionGeometry,
            this.connectionMaterial
        );
        this.scene.add(this.connectionLines);
    }
    
    updateConnections() {
        const positions = [];
        const colors = [];
        const connectionCount = [];
        
        // Reset connection count for each node
        this.nodes.forEach(node => {
            node.connectionCount = 0;
        });
        
        for (let i = 0; i < this.nodes.length; i++) {
            for (let j = i + 1; j < this.nodes.length; j++) {
                const distance = this.nodes[i].position.distanceTo(this.nodes[j].position);
                
                if (distance < this.config.connectionDistance) {
                    positions.push(
                        this.nodes[i].position.x,
                        this.nodes[i].position.y,
                        this.nodes[i].position.z
                    );
                    positions.push(
                        this.nodes[j].position.x,
                        this.nodes[j].position.y,
                        this.nodes[j].position.z
                    );
                    
                    // Color based on distance (closer = brighter)
                    const intensity = 1 - (distance / this.config.connectionDistance);
                    const color = new THREE.Color().setHSL(0.6, 0.8, intensity * 0.5);
                    
                    colors.push(color.r, color.g, color.b);
                    colors.push(color.r, color.g, color.b);
                    
                    this.nodes[i].connectionCount++;
                    this.nodes[j].connectionCount++;
                }
            }
        }
        
        this.connectionGeometry.setAttribute(
            'position',
            new THREE.Float32BufferAttribute(positions, 3)
        );
        this.connectionGeometry.setAttribute(
            'color',
            new THREE.Float32BufferAttribute(colors, 3)
        );
        
        // Update node colors based on connection count
        this.nodes.forEach(node => {
            const intensity = Math.min(node.connectionCount / 5, 1);
            const hue = 0.6 + (intensity * 0.2); // Blue to cyan
            node.material.color.setHSL(hue, 0.8, 0.5 + (intensity * 0.3));
            node.material.opacity = 0.5 + (intensity * 0.4);
        });
    }
    
    animate() {
        if (!this.isAnimating) return;
        
        this.animationId = requestAnimationFrame(() => this.animate());
        
        const time = Date.now() * this.config.animationSpeed;
        
        // Animate nodes
        this.nodes.forEach((node, index) => {
            // Gentle floating animation
            node.position.x = node.originalPosition.x + Math.sin(time + index * 0.1) * 20;
            node.position.y = node.originalPosition.y + Math.cos(time + index * 0.15) * 15;
            node.position.z = node.originalPosition.z + Math.sin(time + index * 0.05) * 10;
            
            // Mouse interaction
            const mouseInfluence = this.config.mouseInfluence;
            const mouseX = (this.mouse.x / window.innerWidth) * 2 - 1;
            const mouseY = -(this.mouse.y / window.innerHeight) * 2 + 1;
            
            const mouseVector = new THREE.Vector3(mouseX * 400, mouseY * 300, 0);
            const distance = node.position.distanceTo(mouseVector);
            
            if (distance < mouseInfluence) {
                const force = (mouseInfluence - distance) / mouseInfluence;
                const direction = new THREE.Vector3()
                    .subVectors(node.position, mouseVector)
                    .normalize()
                    .multiplyScalar(force * 20);
                
                node.position.add(direction);
                
                // Highlight nodes near mouse
                node.material.color.setHSL(0.3, 0.9, 0.7);
                node.material.opacity = 1;
            }
            
            // Boundary constraints
            const boundary = 400;
            if (Math.abs(node.position.x) > boundary) {
                node.position.x = node.originalPosition.x;
            }
            if (Math.abs(node.position.y) > boundary) {
                node.position.y = node.originalPosition.y;
            }
        });
        
        // Update connections every few frames for performance
        if (Date.now() % 5 === 0) {
            this.updateConnections();
        }
        
        // Camera animation
        this.camera.position.x = Math.sin(time * 0.1) * 50;
        this.camera.position.y = Math.cos(time * 0.15) * 30;
        this.camera.lookAt(this.scene.position);
        
        this.renderer.render(this.scene, this.camera);
    }
    
    setupEventListeners() {
        // Mouse movement
        document.addEventListener('mousemove', (event) => {
            this.mouse.x = event.clientX;
            this.mouse.y = event.clientY;
        });
        
        // Window resize
        window.addEventListener('resize', () => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
        });
        
        // Visibility change
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pause();
            } else {
                this.resume();
            }
        });
        
        // Performance optimization for mobile
        if (window.innerWidth < 768) {
            this.config.nodeCount = 30;
            this.config.connectionDistance = 120;
            this.optimizeForMobile();
        }
    }
    
    optimizeForMobile() {
        // Reduce quality for better performance on mobile
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
        
        // Simplify node geometry
        const simpleGeometry = new THREE.SphereGeometry(this.config.nodeSize, 6, 6);
        this.nodes.forEach(node => {
            node.geometry.dispose();
            node.geometry = simpleGeometry;
        });
    }
    
    pause() {
        this.isAnimating = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
    }
    
    resume() {
        this.isAnimating = true;
        this.animate();
    }
    
    addTemporaryEffect(type = 'pulse') {
        switch (type) {
            case 'pulse':
                this.pulseEffect();
                break;
            case 'wave':
                this.waveEffect();
                break;
            case 'analyze':
                this.analyzeEffect();
                break;
        }
    }
    
    pulseEffect() {
        const pulseNodes = this.nodes.slice(0, 10);
        pulseNodes.forEach((node, index) => {
            setTimeout(() => {
                const originalScale = node.scale.x;
                const targetScale = originalScale * 2;
                
                // Animate scale up and down
                const animateScale = (startTime) => {
                    const elapsed = Date.now() - startTime;
                    const duration = 500;
                    const progress = Math.min(elapsed / duration, 1);
                    
                    if (progress < 0.5) {
                        node.scale.setScalar(originalScale + (targetScale - originalScale) * (progress * 2));
                    } else {
                        node.scale.setScalar(targetScale - (targetScale - originalScale) * ((progress - 0.5) * 2));
                    }
                    
                    if (progress < 1) {
                        requestAnimationFrame(() => animateScale(startTime));
                    }
                };
                
                animateScale(Date.now());
            }, index * 100);
        });
    }
    
    waveEffect() {
        const waveSpeed = 0.01;
        const waveAmplitude = 30;
        const startTime = Date.now();
        
        const animateWave = () => {
            const elapsed = Date.now() - startTime;
            if (elapsed > 2000) return; // 2 second effect
            
            this.nodes.forEach((node, index) => {
                const wave = Math.sin((elapsed * waveSpeed) + (index * 0.1)) * waveAmplitude;
                node.position.y = node.originalPosition.y + wave;
            });
            
            requestAnimationFrame(animateWave);
        };
        
        animateWave();
    }
    
    analyzeEffect() {
        // Create data flow effect
        const dataNodes = [];
        const nodeCount = 20;
        
        for (let i = 0; i < nodeCount; i++) {
            const geometry = new THREE.SphereGeometry(1, 8, 8);
            const material = new THREE.MeshBasicMaterial({
                color: this.config.colors.highlight,
                transparent: true,
                opacity: 0.8
            });
            
            const dataNode = new THREE.Mesh(geometry, material);
            dataNode.position.set(
                -400 + (i * 40),
                Math.sin(i * 0.5) * 100,
                0
            );
            
            this.scene.add(dataNode);
            dataNodes.push(dataNode);
        }
        
        // Animate data flow
        const animateDataFlow = (startTime) => {
            const elapsed = Date.now() - startTime;
            const duration = 3000;
            const progress = elapsed / duration;
            
            if (progress >= 1) {
                // Clean up
                dataNodes.forEach(node => {
                    this.scene.remove(node);
                    node.geometry.dispose();
                    node.material.dispose();
                });
                return;
            }
            
            dataNodes.forEach((node, index) => {
                const delay = index * 0.05;
                const nodeProgress = Math.max(0, Math.min(1, (progress - delay) / (1 - delay)));
                
                if (nodeProgress > 0) {
                    node.position.x = -400 + (nodeProgress * 800);
                    node.position.y = Math.sin(elapsed * 0.01 + index * 0.5) * 100;
                    node.material.opacity = Math.sin(nodeProgress * Math.PI) * 0.8;
                }
            });
            
            requestAnimationFrame(() => animateDataFlow(startTime));
        };
        
        animateDataFlow(Date.now());
    }
    
    dispose() {
        this.pause();
        
        // Clean up Three.js objects
        this.nodes.forEach(node => {
            this.scene.remove(node);
            node.geometry.dispose();
            node.material.dispose();
        });
        
        this.scene.remove(this.connectionLines);
        this.connectionGeometry.dispose();
        this.connectionMaterial.dispose();
        
        if (this.renderer) {
            this.renderer.dispose();
            this.container.removeChild(this.renderer.domElement);
        }
    }
}

// Initialize neural network background when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.neuralBackground = new NeuralNetworkBackground();
});

// Export for use in other modules
window.NeuralNetworkBackground = NeuralNetworkBackground;
