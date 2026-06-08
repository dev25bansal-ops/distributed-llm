/**
 * Interactive 3D Network Topology Visualizer
 *
 * Features:
 * - Animated 3D graph showing GPU nodes and connections
 * - Real-time token flow visualization
 * - Zoom/pan/rotate controls
 * - Click-to-inspect node details
 * - Layer-to-GPU mapping visualization
 *
 * Uses Three.js for 3D rendering.
 *
 * Usage:
 *   <div id="networkTopology" style="height: 500px;"></div>
 *   <script type="module">
 *     import { initNetworkTopology } from './js/network-topology.js';
 *     initNetworkTopology();
 *   </script>
 */

// ── Configuration ──────────────────────────────────────────────────────

const CONFIG = {
    colors: {
        node: 0x00e676,
        nodeIdle: 0x555555,
        connection: 0x22c55e,
        connectionDim: 0x333333,
        flow: 0x00e676,
        coordinator: 0x06b6d4,
        background: 0x050505,
        grid: 0x1a1a1a,
        text: 0xededed,
    },
    nodeSize: 0.5,
    connectionWidth: 2,
    flowSpeed: 0.02,
    rotationSpeed: 0.001,
};

// ── State ──────────────────────────────────────────────────────────────

let scene, camera, renderer, controls;
let nodes = [], connections = [], flowParticles = [];
let animationId;
let isInitialized = false;

// ── Three.js Loading ───────────────────────────────────────────────────

async function loadThreeJS() {
    // Check if Three.js is already loaded
    if (window.THREE) return window.THREE;

    // Load Three.js dynamically with Subresource Integrity (SRI)
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js';
        script.integrity = 'sha384-CI3ELBVUz9XQO+97x6nwMDPosPR5XvsxW2ua7N1Xeygeh1IxtgqtCkGfQY9WWdHu';
        script.crossOrigin = 'anonymous';
        script.onload = () => {
            // Load OrbitControls
            const controlsScript = document.createElement('script');
            controlsScript.src = 'https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js';
            controlsScript.integrity = 'sha384-wagZhIFgY4hD+7awjQjR4e2E294y6J2HSnd8eTNc15ZubTeQeVRZwhQJ+W6hnBsf';
            controlsScript.crossOrigin = 'anonymous';
            controlsScript.onload = () => resolve(window.THREE);
            controlsScript.onerror = reject;
            document.head.appendChild(controlsScript);
        };
        script.onerror = reject;
        document.head.appendChild(script);
    });
}

// ── Cluster Data ───────────────────────────────────────────────────────

function generateClusterData() {
    return {
        coordinator: {
            id: 'coordinator',
            name: 'Coordinator',
            position: { x: 0, y: 0, z: 0 },
            status: 'online',
        },
        workers: [
            {
                id: 'worker-0',
                name: 'RTX 4090 #1',
                position: { x: -3, y: 1, z: -2 },
                gpu: { util: 0.72, mem: 0.65, temp: 68 },
                layers: [0, 7],
                status: 'online',
            },
            {
                id: 'worker-1',
                name: 'RTX 4090 #2',
                position: { x: 3, y: 1, z: -2 },
                gpu: { util: 0.85, mem: 0.78, temp: 74 },
                layers: [8, 15],
                status: 'online',
            },
            {
                id: 'worker-2',
                name: 'RTX 3090 #1',
                position: { x: -3, y: -1, z: 2 },
                gpu: { util: 0.61, mem: 0.52, temp: 62 },
                layers: [16, 23],
                status: 'online',
            },
            {
                id: 'worker-3',
                name: 'RTX 3090 #2',
                position: { x: 3, y: -1, z: 2 },
                gpu: { util: 0.93, mem: 0.88, temp: 81 },
                layers: [24, 31],
                status: 'online',
            },
        ],
        connections: [
            { from: 'coordinator', to: 'worker-0' },
            { from: 'coordinator', to: 'worker-1' },
            { from: 'coordinator', to: 'worker-2' },
            { from: 'coordinator', to: 'worker-3' },
            { from: 'worker-0', to: 'worker-1' },
            { from: 'worker-1', to: 'worker-2' },
            { from: 'worker-2', to: 'worker-3' },
        ],
    };
}

// ── Scene Setup ────────────────────────────────────────────────────────

function initScene(container, THREE) {
    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(CONFIG.colors.background);

    // Camera
    const aspect = container.clientWidth / container.clientHeight;
    camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 1000);
    camera.position.set(8, 6, 8);
    camera.lookAt(0, 0, 0);

    // Renderer
    renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
    });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    // Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.enablePan = true;
    controls.enableZoom = true;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.5;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambientLight);

    const pointLight = new THREE.PointLight(0xffffff, 1, 100);
    pointLight.position.set(10, 10, 10);
    scene.add(pointLight);

    // Grid
    const gridHelper = new THREE.GridHelper(20, 20, CONFIG.colors.grid, CONFIG.colors.grid);
    gridHelper.position.y = -2;
    scene.add(gridHelper);

    // Handle resize
    window.addEventListener('resize', () => {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    });
}

// ── Node Creation ──────────────────────────────────────────────────────

function createNode(nodeData, THREE) {
    const isCoordinator = nodeData.id === 'coordinator';
    const size = isCoordinator ? CONFIG.nodeSize * 1.5 : CONFIG.nodeSize;

    // Create sphere
    const geometry = new THREE.SphereGeometry(size, 32, 32);
    const material = new THREE.MeshPhongMaterial({
        color: isCoordinator ? CONFIG.colors.coordinator : CONFIG.colors.node,
        emissive: isCoordinator ? CONFIG.colors.coordinator : CONFIG.colors.node,
        emissiveIntensity: 0.3,
        transparent: true,
        opacity: 0.9,
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(nodeData.position.x, nodeData.position.y, nodeData.position.z);
    mesh.userData = { ...nodeData, isCoordinator };

    // Add glow effect
    const glowGeometry = new THREE.SphereGeometry(size * 1.3, 32, 32);
    const glowMaterial = new THREE.MeshBasicMaterial({
        color: isCoordinator ? CONFIG.colors.coordinator : CONFIG.colors.node,
        transparent: true,
        opacity: 0.1,
    });
    const glow = new THREE.Mesh(glowGeometry, glowMaterial);
    mesh.add(glow);

    // Add label
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = 256;
    canvas.height = 64;
    context.fillStyle = '#000000';
    context.fillRect(0, 0, 256, 64);
    context.fillStyle = '#ffffff';
    context.font = '24px Inter, sans-serif';
    context.textAlign = 'center';
    context.fillText(nodeData.name, 128, 40);

    const texture = new THREE.CanvasTexture(canvas);
    const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
    const sprite = new THREE.Sprite(spriteMaterial);
    sprite.position.y = size + 0.8;
    sprite.scale.set(2, 0.5, 1);
    mesh.add(sprite);

    scene.add(mesh);
    nodes.push(mesh);

    return mesh;
}

// ── Connection Creation ────────────────────────────────────────────────

function createConnection(from, to, THREE) {
    const points = [
        new THREE.Vector3(from.position.x, from.position.y, from.position.z),
        new THREE.Vector3(to.position.x, to.position.y, to.position.z),
    ];

    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({
        color: CONFIG.colors.connectionDim,
        transparent: true,
        opacity: 0.5,
    });

    const line = new THREE.Line(geometry, material);
    line.userData = { from: from.userData.id, to: to.userData.id };
    scene.add(line);
    connections.push(line);

    return line;
}

// ── Flow Particle Creation ─────────────────────────────────────────────

function createFlowParticle(from, to, THREE) {
    const geometry = new THREE.SphereGeometry(0.1, 16, 16);
    const material = new THREE.MeshBasicMaterial({
        color: CONFIG.colors.flow,
        transparent: true,
        opacity: 0.8,
    });

    const particle = new THREE.Mesh(geometry, material);
    particle.userData = {
        from: new THREE.Vector3(from.position.x, from.position.y, from.position.z),
        to: new THREE.Vector3(to.position.x, to.position.y, to.position.z),
        progress: Math.random(),
        speed: CONFIG.flowSpeed + Math.random() * 0.01,
    };

    scene.add(particle);
    flowParticles.push(particle);

    return particle;
}

// ── Animation ──────────────────────────────────────────────────────────

function animate() {
    animationId = requestAnimationFrame(animate);

    // Update controls
    controls.update();

    // Animate nodes (pulse effect)
    nodes.forEach(node => {
        const scale = 1 + Math.sin(Date.now() * 0.002) * 0.05;
        node.scale.set(scale, scale, scale);
    });

    // Animate flow particles
    flowParticles.forEach(particle => {
        particle.userData.progress += particle.userData.speed;
        if (particle.userData.progress > 1) particle.userData.progress = 0;

        const { from, to, progress } = particle.userData;
        particle.position.lerpVectors(from, to, progress);

        // Fade in/out at ends
        const opacity = progress < 0.1 ? progress * 10 : progress > 0.9 ? (1 - progress) * 10 : 1;
        particle.material.opacity = opacity * 0.8;
    });

    // Render
    renderer.render(scene, camera);
}

// ── UI Overlay ─────────────────────────────────────────────────────────

function createUIOverlay(container) {
    const overlay = document.createElement('div');
    overlay.className = 'topology-overlay';
    overlay.innerHTML = `
        <div class="topology-info">
            <h4>Network Topology</h4>
            <p>Interactive 3D visualization of DistLLM cluster</p>
            <div class="topology-controls">
                <div class="control-item">
                    <span class="control-key">🖱️ Drag</span>
                    <span class="control-desc">Rotate</span>
                </div>
                <div class="control-item">
                    <span class="control-key">🖱️ Scroll</span>
                    <span class="control-desc">Zoom</span>
                </div>
                <div class="control-item">
                    <span class="control-key">🖱️ Right-drag</span>
                    <span class="control-desc">Pan</span>
                </div>
            </div>
        </div>
        <div class="topology-legend">
            <div class="legend-item">
                <span class="legend-color" style="background: #06b6d4;"></span>
                <span>Coordinator</span>
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background: #00e676;"></span>
                <span>GPU Worker</span>
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background: #22c55e; opacity: 0.6;"></span>
                <span>Data Flow</span>
            </div>
        </div>
    `;
    container.appendChild(overlay);
}

// ── Initialization ─────────────────────────────────────────────────────

export async function initNetworkTopology() {
    const container = document.getElementById('networkTopology');
    if (!container) return;

    try {
        // Load Three.js
        const THREE = await loadThreeJS();

        // Initialize scene
        initScene(container, THREE);

        // Generate cluster data
        const cluster = generateClusterData();

        // Create nodes
        const coordinatorNode = createNode(cluster.coordinator, THREE);
        const workerNodes = cluster.workers.map(worker => createNode(worker, THREE));

        // Create connections
        cluster.connections.forEach(conn => {
            const from = conn.from === 'coordinator'
                ? coordinatorNode
                : workerNodes.find(n => n.userData.id === conn.from);
            const to = conn.to === 'coordinator'
                ? coordinatorNode
                : workerNodes.find(n => n.userData.id === conn.to);

            if (from && to) {
                createConnection(from, to, THREE);

                // Create flow particles for active connections
                if (from.userData.status === 'online' && to.userData.status === 'online') {
                    for (let i = 0; i < 3; i++) {
                        createFlowParticle(from, to, THREE);
                    }
                }
            }
        });

        // Create UI overlay
        createUIOverlay(container);

        // Start animation
        animate();
        isInitialized = true;

    } catch (e) {
        console.error('[NetworkTopology] Failed to initialize:', e);
        container.innerHTML = `
            <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: var(--dim);">
                <p>Failed to load 3D visualization. Please check your internet connection.</p>
            </div>
        `;
    }
}

// Cleanup function
export function cleanupNetworkTopology() {
    if (animationId) {
        cancelAnimationFrame(animationId);
    }
    if (renderer) {
        renderer.dispose();
    }
    nodes = [];
    connections = [];
    flowParticles = [];
    isInitialized = false;
}
