import * as THREE from 'three';
import { GPUComputationRenderer } from 'three/addons/misc/GPUComputationRenderer.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { 
    computeShader, 
    particleVertexShader, 
    particleFragmentShader, 
    trailVertexShader, 
    trailFragmentShader, 
    displayFragmentShader 
} from './shaders.js';

const SIM_WIDTH = 512; // 512 x 512 = 262,144 partículas; equilibrio entre detalle y rendimiento
const PARTICLES = SIM_WIDTH * SIM_WIDTH;
const WHALE_ROTATION_2D = Math.PI * 0.5;

let renderer;
let gpuCompute, particleVariable;
let particleMaterial, trailMaterial, displayMaterial;

// Escenas dedicadas para evitar oclusión de buffers
let particleScene, particleCamera;
let trailScene, trailCamera;
let displayScene, displayCamera;

let trailRenderTarget1, trailRenderTarget2;
let audio, analyser, audioData;
let audioContext, vocalBuffer, vocalSource;
let silentAnalysisGain;
let vocalAnalysisLoading = false;
let vocalAnalysisPlaying = false;
let audioLevel = 0;
let bassLevel = 0;
let pulseLevel = 0;
let beatPulse = 0;
let previousBassLevel = 0;
let selectedPoint = '1';
let whaleShapeTexture;
let whaleMode = false;
let presentationMode = false;
let simulationTime = 0;

let mouse = new THREE.Vector2(-10, -10);
let isMouseDown = false;

// Estado e interpolación de parámetros
const currentParams = {
    sensorDistance: 0.025,
    sensorAngle: 0.45,
    rotationAngle: 0.4,
    moveDistance: 0.0025,
    attractForce: 0.0
    ,adaptiveStrength: 0.65
    ,deposit: 0.08
    ,trailDecay: 0.82
    ,brightness: 2.5
    ,shapeStrength: 0.0
};

const targetParams = { ...currentParams };

const PRESETS = {
    '1': { sensorDistance: 0.015, sensorAngle: 0.25, rotationAngle: 0.2, moveDistance: 0.0015, attractForce: 0.0 },
    '2': { sensorDistance: 0.035, sensorAngle: 0.55, rotationAngle: 0.45, moveDistance: 0.003, attractForce: 0.2 },
    '3': { sensorDistance: 0.065, sensorAngle: 1.15, rotationAngle: 0.85, moveDistance: 0.006, attractForce: 0.8 },
    '4': { sensorDistance: 0.01,  sensorAngle: 1.57, rotationAngle: 1.2,  moveDistance: 0.004, attractForce: 2.0 }
};

const POINT_KEYS = '1234567890abcdefghijklmnopqrstuvwxyz';
const SENSOR_DISTANCES = [0.008, 0.014, 0.022, 0.035, 0.05, 0.075];
const SENSOR_ANGLES = [0.15, 0.3, 0.5, 0.75, 1.1, 1.45];
const ROTATION_ANGLES = [0.12, 0.22, 0.35, 0.5, 0.75, 1.1];
const MOVE_DISTANCES = [0.0012, 0.0018, 0.0025, 0.0035, 0.0045, 0.006];
const POINTS = Object.fromEntries(POINT_KEYS.split('').map((key, index) => {
    const row = Math.floor(index / 6);
    return [key, {
        sensorDistance: SENSOR_DISTANCES[index % 6],
        sensorAngle: SENSOR_ANGLES[row],
        rotationAngle: ROTATION_ANGLES[(index * 3 + row) % 6],
        moveDistance: MOVE_DISTANCES[(index + row) % 6],
        attractForce: (index % 5) * 0.2,
        adaptiveStrength: 0.35 + (row % 3) * 0.3,
        deposit: 0.045 + (index % 4) * 0.018,
        trailDecay: 0.74 + (index % 5) * 0.025,
        brightness: 1.8 + (row % 4) * 0.45
    }];
}));

init();
animate();

function init() {
    renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: "high-performance" });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    document.body.appendChild(renderer.domElement);

    const w = window.innerWidth;
    const h = window.innerHeight;
    const type = THREE.HalfFloatType;

    // 1. Escena de Partículas
    particleScene = new THREE.Scene();
    particleCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    // 2. Escena de Difusión (Trail Map)
    trailScene = new THREE.Scene();
    trailCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    // 3. Escena Final de Salida
    displayScene = new THREE.Scene();
    displayCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    const trailOptions = {
        type,
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        wrapS: THREE.RepeatWrapping,
        wrapT: THREE.RepeatWrapping
    };
    trailRenderTarget1 = new THREE.WebGLRenderTarget(w, h, trailOptions);
    trailRenderTarget2 = new THREE.WebGLRenderTarget(w, h, trailOptions);

    initGPGPU();
    initParticles();
    initTrailPass();
    initDisplayPass();
    initAudio();
    initWhaleShape();

    window.addEventListener('resize', onWindowResize);
    window.addEventListener('keydown', (e) => {
        const key = e.key.toLowerCase();
        if (key === 'h') {
            presentationMode = !presentationMode;
            document.body.classList.toggle('presentation-mode', presentationMode);
            return;
        }
        if (key === 'w') {
            whaleMode = !whaleMode;
            targetParams.shapeStrength = whaleMode ? 0.56 : 0.0;
            if (whaleMode) clearTrails();
            updateAudioStatus(whaleMode ? 'Contenedor ballena activo' : 'Forma libre activa');
            return;
        }
        if (key === 'g') {
            beatPulse = 1.0;
            updateAudioStatus(`Beat manual · Point ${selectedPoint.toUpperCase()}`);
            return;
        }

        function clearTrails() {
            renderer.setRenderTarget(trailRenderTarget1);
            renderer.clear();
            renderer.setRenderTarget(trailRenderTarget2);
            renderer.clear();
            renderer.setRenderTarget(null);
        }
        if (POINTS[key]) {
            selectedPoint = key;
            Object.assign(targetParams, POINTS[key]);
            updateAudioStatus(`Point ${key.toUpperCase()} activo`);
        }
    });
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerdown', (e) => { isMouseDown = true; onPointerMove(e); });
    window.addEventListener('pointerup', () => { isMouseDown = false; });
}

function initWhaleShape() {
    const fallback = new Float32Array([0.5, 0.5, 0, 1]);
    whaleShapeTexture = new THREE.DataTexture(fallback, 1, 1, THREE.RGBAFormat, THREE.FloatType);
    whaleShapeTexture.needsUpdate = true;
    particleVariable.material.uniforms.uShapePositions.value = whaleShapeTexture;

    const loader = new GLTFLoader();
    loader.load('./whale.glb', (gltf) => {
        const points = [];
        const bounds = new THREE.Box3();
        gltf.scene.updateMatrixWorld(true);
        gltf.scene.traverse((object) => {
            if (!object.isMesh || !object.geometry.attributes.position) return;
            const positions = object.geometry.attributes.position;
            const vertex = new THREE.Vector3();
            for (let i = 0; i < positions.count; i++) {
                vertex.fromBufferAttribute(positions, i).applyMatrix4(object.matrixWorld);
                points.push(vertex.clone());
                bounds.expandByPoint(vertex);
            }
        });

        if (!points.length) {
            updateAudioStatus('whale.glb no contiene geometría utilizable');
            return;
        }

        const size = bounds.getSize(new THREE.Vector3());
        const data = new Float32Array(PARTICLES * 4);
        for (let i = 0; i < PARTICLES; i++) {
            const point = points[i % points.length];
            const x = size.z > 0 ? (point.z - bounds.min.z) / size.z : 0.5;
            const y = size.y > 0 ? (point.y - bounds.min.y) / size.y : 0.5;
            const canvasX = 0.08 + x * 0.84;
            const canvasY = 0.08 + (1.0 - y) * 0.84;
            const centeredX = canvasX - 0.5;
            const centeredY = canvasY - 0.5;
            const rotatedX = centeredX * Math.cos(WHALE_ROTATION_2D) - centeredY * Math.sin(WHALE_ROTATION_2D);
            const rotatedY = centeredX * Math.sin(WHALE_ROTATION_2D) + centeredY * Math.cos(WHALE_ROTATION_2D);
            data[i * 4] = rotatedX + 0.5;
            data[i * 4 + 1] = rotatedY + 0.5;
            data[i * 4 + 3] = 1;
        }
        whaleShapeTexture.dispose();
        whaleShapeTexture = new THREE.DataTexture(data, SIM_WIDTH, SIM_WIDTH, THREE.RGBAFormat, THREE.FloatType);
        whaleShapeTexture.minFilter = THREE.NearestFilter;
        whaleShapeTexture.magFilter = THREE.NearestFilter;
        whaleShapeTexture.needsUpdate = true;
        particleVariable.material.uniforms.uShapePositions.value = whaleShapeTexture;
        updateAudioStatus('Ballena lista · pulsa W para activar');
    }, undefined, () => updateAudioStatus('No se pudo cargar whale.glb'));
}

function initAudio() {
    const defaultAudioUrl = './assets/Jhen%C3%A9%20Aiko%20-%20So%20Good%20feat.%20Kendrick%20Lamar%20%28Official%20Audio%29.wav';
    audio = new Audio(defaultAudioUrl);
    audio.loop = true;
    audio.preload = 'auto';
    audio.addEventListener('canplay', () => updateAudioStatus('Canción local lista · pulsa reproducir'));
    audio.addEventListener('error', () => updateAudioStatus('No se pudo cargar la canción local'));

    const button = document.getElementById('audioToggle');
    const beatButton = document.getElementById('beatTrigger');
    const fileInput = document.getElementById('audioFile');
    fileInput.addEventListener('change', () => {
        const file = fileInput.files[0];
        if (!file) return;
        audio.src = URL.createObjectURL(file);
        updateAudioStatus('Audio seleccionado · pulsa reproducir');
    });

    button.addEventListener('click', async () => {
        if (!analyser) {
            audioContext = new AudioContext();
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            audioData = new Uint8Array(analyser.frequencyBinCount);
            silentAnalysisGain = audioContext.createGain();
            silentAnalysisGain.gain.value = 0;
            analyser.connect(silentAnalysisGain);
            silentAnalysisGain.connect(audioContext.destination);
            await audioContext.resume();
        }

        if (audio.paused) {
            try {
                await startVocalAnalysis();
                await audio.play();
                button.textContent = 'Pausar canción';
                updateAudioStatus(`Point ${selectedPoint.toUpperCase()} · análisis vocal activo`);
            } catch (error) {
                updateAudioStatus('Selecciona el WAV o abre el proyecto con un servidor local');
            }
        } else {
            audio.pause();
            stopVocalAnalysis();
            button.textContent = 'Reproducir canción';
            updateAudioStatus(`Point ${selectedPoint.toUpperCase()} · análisis pausado`);
        }
    });

    beatButton.addEventListener('click', () => {
        beatPulse = 1.0;
        updateAudioStatus(`Beat manual · Point ${selectedPoint.toUpperCase()}`);
    });
}

async function startVocalAnalysis() {
    if (!vocalBuffer && !vocalAnalysisLoading) {
        vocalAnalysisLoading = true;
        try {
            const response = await fetch('./assets/music/Vocals.mp3');
            if (!response.ok) throw new Error(`No se pudo cargar Vocals.mp3 (${response.status})`);
            vocalBuffer = await audioContext.decodeAudioData(await response.arrayBuffer());
        } finally {
            vocalAnalysisLoading = false;
        }
    }
    if (!vocalBuffer || vocalAnalysisPlaying) return;

    vocalSource = audioContext.createBufferSource();
    vocalSource.buffer = vocalBuffer;
    vocalSource.loop = true;
    vocalSource.connect(analyser);
    vocalSource.onended = () => {
        vocalSource = null;
        vocalAnalysisPlaying = false;
    };
    vocalSource.start();
    vocalAnalysisPlaying = true;
}

function stopVocalAnalysis() {
    if (!vocalSource || !vocalAnalysisPlaying) return;
    vocalSource.stop();
    vocalSource.disconnect();
    vocalSource = null;
    vocalAnalysisPlaying = false;
}

function updateAudioStatus(text) {
    const status = document.getElementById('audioStatus');
    if (status) status.textContent = text;
}

function updateAudioLevels() {
    if (!analyser) {
        audioLevel *= 0.96;
        bassLevel *= 0.96;
        pulseLevel *= 0.9;
        return;
    }

    analyser.getByteFrequencyData(audioData);
    let total = 0;
    let bass = 0;
    for (let i = 0; i < audioData.length; i++) {
        total += audioData[i];
        if (i < 8) bass += audioData[i];
    }
    audioLevel += (total / audioData.length / 255 - audioLevel) * 0.14;
    bassLevel += (bass / 8 / 255 - bassLevel) * 0.17;
    const bassRise = Math.max(0, bassLevel - previousBassLevel);
    pulseLevel = Math.max(pulseLevel * 0.9, bassRise * 4.0);
    previousBassLevel = bassLevel;
}

function initGPGPU() {
    gpuCompute = new GPUComputationRenderer(SIM_WIDTH, SIM_WIDTH, renderer);

    const dtPositions = gpuCompute.createTexture();
    const posArray = dtPositions.image.data;

    for (let i = 0; i < posArray.length; i += 4) {
        posArray[i]     = Math.random();
        posArray[i + 1] = Math.random();
        posArray[i + 2] = Math.random() * Math.PI * 2;
        posArray[i + 3] = 1.0;
    }

    particleVariable = gpuCompute.addVariable("texturePosition", computeShader, dtPositions);
    gpuCompute.setVariableDependencies(particleVariable, [particleVariable]);
    Object.assign(particleVariable.material.uniforms, {
        uTrailMap: { value: trailRenderTarget1.texture },
        uSensorDist: { value: currentParams.sensorDistance },
        uSensorAngle: { value: currentParams.sensorAngle },
        uRotAngle: { value: currentParams.rotationAngle },
        uMoveDist: { value: currentParams.moveDistance },
        uMouse: { value: new THREE.Vector2(-10, -10) },
        uAttractForce: { value: 0.0 },
        uAdaptiveStrength: { value: 0.65 },
        uAudioLevel: { value: 0.0 },
        uMotionJitter: { value: 0.0 },
        uAudioBass: { value: 0.0 },
        uBeat: { value: 0.0 },
        uShapePositions: { value: null },
        uShapeStrength: { value: 0.0 },
        uTime: { value: 0.0 }
    });

    gpuCompute.init();
}

function initParticles() {
    const geometry = new THREE.BufferGeometry();
    const uvs = new Float32Array(PARTICLES * 2);

    let p = 0;
    for (let j = 0; j < SIM_WIDTH; j++) {
        for (let i = 0; i < SIM_WIDTH; i++) {
            uvs[p++] = i / SIM_WIDTH;
            uvs[p++] = j / SIM_WIDTH;
        }
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(PARTICLES * 3), 3));
    geometry.setAttribute('particleUv', new THREE.BufferAttribute(uvs, 2));

    particleMaterial = new THREE.ShaderMaterial({
        vertexShader: particleVertexShader,
        fragmentShader: particleFragmentShader,
        uniforms: {
            texturePosition: { value: null },
            uDeposit: { value: currentParams.deposit },
            uAudioLevel: { value: 0.0 },
            uBeat: { value: 0.0 }
        },
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        depthTest: false
    });

    const points = new THREE.Points(geometry, particleMaterial);
    particleScene.add(points);
}

function initTrailPass() {
    trailMaterial = new THREE.ShaderMaterial({
        vertexShader: trailVertexShader,
        fragmentShader: trailFragmentShader,
        uniforms: {
            uPrevTrail: { value: null },
            uTexelSize: { value: new THREE.Vector2(1 / window.innerWidth, 1 / window.innerHeight) },
            uDecay: { value: currentParams.trailDecay }
        },
        depthWrite: false,
        depthTest: false
    });

    const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), trailMaterial);
    trailScene.add(quad);
}

function initDisplayPass() {
    displayMaterial = new THREE.ShaderMaterial({
        vertexShader: trailVertexShader,
        fragmentShader: displayFragmentShader,
        uniforms: {
            uTrailMap: { value: null },
            uResolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) },
            uAudioLevel: { value: 0.0 },
            uAudioBass: { value: 0.0 },
            uBeat: { value: 0.0 },
            uBrightness: { value: currentParams.brightness },
            uPulse: { value: 0.0 }
        },
        depthWrite: false,
        depthTest: false
    });

    const displayQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), displayMaterial);
    displayScene.add(displayQuad);
}

function onPointerMove(e) {
    mouse.x = e.clientX / window.innerWidth;
    mouse.y = 1.0 - (e.clientY / window.innerHeight);
}

function onWindowResize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    renderer.setSize(w, h);
    trailRenderTarget1.setSize(w, h);
    trailRenderTarget2.setSize(w, h);
    trailMaterial.uniforms.uTexelSize.value.set(1 / w, 1 / h);
    displayMaterial.uniforms.uResolution.value.set(w, h);
}

function animate() {
    requestAnimationFrame(animate);

    // LERP para transiciones suaves de parámetros
    const lerp = 0.04;
    for (let k in currentParams) {
        currentParams[k] += (targetParams[k] - currentParams[k]) * lerp;
    }

    const uniforms = particleVariable.material.uniforms;
    updateAudioLevels();
    uniforms.uSensorDist.value  = currentParams.sensorDistance;
    uniforms.uSensorAngle.value = currentParams.sensorAngle;
    uniforms.uRotAngle.value    = currentParams.rotationAngle;
    uniforms.uMoveDist.value    = currentParams.moveDistance;
    uniforms.uAttractForce.value = isMouseDown
        ? Math.max(0.75, currentParams.attractForce * 2.5)
        : currentParams.attractForce;
    uniforms.uMouse.value.copy(mouse);
    uniforms.uAdaptiveStrength.value = currentParams.adaptiveStrength + bassLevel * 0.35;
    uniforms.uAudioLevel.value = audioLevel;
    uniforms.uMotionJitter.value = audioLevel * 0.08 + bassLevel * 0.04;
    uniforms.uAudioBass.value = bassLevel;
    uniforms.uShapeStrength.value = currentParams.shapeStrength;
    uniforms.uTime.value = simulationTime;
    particleMaterial.uniforms.uDeposit.value = currentParams.deposit * (1.0 + bassLevel * 0.25 + pulseLevel * 0.08);
    particleMaterial.uniforms.uAudioLevel.value = audioLevel;
    trailMaterial.uniforms.uDecay.value = Math.min(0.9, currentParams.trailDecay + audioLevel * 0.015 + bassLevel * 0.01);
    displayMaterial.uniforms.uAudioLevel.value = audioLevel;
    displayMaterial.uniforms.uAudioBass.value = bassLevel;
    displayMaterial.uniforms.uPulse.value = pulseLevel;
    beatPulse *= 0.985;
    uniforms.uBeat.value = beatPulse;
    particleMaterial.uniforms.uBeat.value = beatPulse;
    displayMaterial.uniforms.uBeat.value = beatPulse;
    displayMaterial.uniforms.uBrightness.value = currentParams.brightness + audioLevel * 0.04 + pulseLevel * 0.02;
    simulationTime += 0.016;

    // 1. Compute Pass: Calcular nuevas posiciones
    gpuCompute.compute();
    const posTexture = gpuCompute.getCurrentRenderTarget(particleVariable).texture;
    particleMaterial.uniforms.texturePosition.value = posTexture;

    // 2. Difusión: Renderizar trail map difuminado previo hacia RT2
    trailMaterial.uniforms.uPrevTrail.value = trailRenderTarget1.texture;
    renderer.setRenderTarget(trailRenderTarget2);
    renderer.render(trailScene, trailCamera);

    // 3. Depósito: Acumular depósitos de partículas sobre RT2
    particleVariable.material.uniforms.uTrailMap.value = trailRenderTarget2.texture;
    renderer.autoClear = false;
    renderer.render(particleScene, particleCamera);
    renderer.autoClear = true;

    // 4. Render Final: Proyectar colores al lienzo principal
    displayMaterial.uniforms.uTrailMap.value = trailRenderTarget2.texture;
    renderer.setRenderTarget(null);
    renderer.render(displayScene, displayCamera);

    // Intercambio Ping-Pong
    const temp = trailRenderTarget1;
    trailRenderTarget1 = trailRenderTarget2;
    trailRenderTarget2 = temp;
}