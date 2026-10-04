/**
 * Cardio3D AI — Vanilla Three.js Anatomical Heart & Coronary Viewer
 * Renders BodyParts3D 83,600-triangle heart model with real-time risk color-mapping.
 */

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/loaders/GLTFLoader.js';

let scene, camera, renderer, controls;
let ladMesh, lcxMesh, rcaMesh, aortaMesh, chamberMesh;
let selectedVessel = null;

const vesselBadges = {
  LAD: { pos: new THREE.Vector3(0.35, 0.15, 0.72), label: 'LAD', prob: 0.24, el: null },
  LCX: { pos: new THREE.Vector3(0.68, 0.35, -0.12), label: 'LCX', prob: 0.06, el: null },
  RCA: { pos: new THREE.Vector3(-0.62, 0.22, 0.38), label: 'RCA', prob: 0.02, el: null },
};

export function getRiskColor(prob) {
  const p = Math.max(0, Math.min(1, prob));
  const c1 = new THREE.Color();
  const c2 = new THREE.Color();

  if (p < 0.25) {
    c1.set('#10b981');
    c2.set('#84cc16');
    return c1.lerp(c2, p / 0.25);
  } else if (p < 0.50) {
    c1.set('#84cc16');
    c2.set('#f59e0b');
    return c1.lerp(c2, (p - 0.25) / 0.25);
  } else if (p < 0.75) {
    c1.set('#f59e0b');
    c2.set('#f97316');
    return c1.lerp(c2, (p - 0.50) / 0.25);
  } else {
    c1.set('#f97316');
    c2.set('#ef4444');
    return c1.lerp(c2, (p - 0.75) / 0.25);
  }
}

export function initHeartViewer() {
  const container = document.getElementById('heart-canvas-container');
  if (!container) return;

  const width = container.clientWidth || 600;
  const height = container.clientHeight || 560;

  // Scene
  scene = new THREE.Scene();

  // Camera
  camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 100);
  camera.position.set(0, 0.35, 3.2);

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;
  container.appendChild(renderer.domElement);

  // OrbitControls
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.minDistance = 1.2;
  controls.maxDistance = 6.0;

  // Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 1.2);
  scene.add(ambientLight);

  const keyLight = new THREE.DirectionalLight(0xffffff, 1.8);
  keyLight.position.set(4, 5, 4);
  scene.add(keyLight);

  const fillLight = new THREE.DirectionalLight(0x60a5fa, 0.7);
  fillLight.position.set(-4, -3, -4);
  scene.add(fillLight);

  const rimLight = new THREE.PointLight(0xffffff, 0.8);
  rimLight.position.set(0, 2, 2);
  scene.add(rimLight);

  // Create badge DOM elements
  createBadgeElements(container);

  // Load GLB
  const loader = new GLTFLoader();
  loader.load('/models/heart.glb', (gltf) => {
    const root = gltf.scene;

    root.traverse((child) => {
      if (child.isMesh) {
        if (child.name === 'LAD') {
          ladMesh = child;
          setupVesselMaterial(ladMesh, vesselBadges.LAD.prob, false);
        } else if (child.name === 'LCX') {
          lcxMesh = child;
          setupVesselMaterial(lcxMesh, vesselBadges.LCX.prob, false);
        } else if (child.name === 'RCA') {
          rcaMesh = child;
          setupVesselMaterial(rcaMesh, vesselBadges.RCA.prob, false);
        } else if (child.name === 'Aorta') {
          aortaMesh = child;
          child.material = new THREE.MeshStandardMaterial({
            color: '#be123c',
            roughness: 0.3,
            metalness: 0.15,
            opacity: 0.88,
            transparent: true,
          });
        } else if (child.name === 'Heart_Muscle') {
          chamberMesh = child;
          child.material = new THREE.MeshStandardMaterial({
            color: '#641026',
            roughness: 0.55,
            metalness: 0.05,
            opacity: 0.25,
            transparent: true,
            depthWrite: false,
          });
        }
      }
    });

    scene.add(root);
    animate();
  });

  // Window resize
  window.addEventListener('resize', onWindowResize);

  // Raycasting for vessel click
  setupRaycasting(container);

  // Listen to HTMX updates
  document.body.addEventListener('vesselRiskUpdated', (e) => {
    const data = e.detail;
    if (data) {
      updateVessels(data);
    }
  });

  // Viewport buttons
  const resetBtn = document.getElementById('btn-reset-camera');
  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      controls.reset();
      camera.position.set(0, 0.35, 3.2);
    });
  }

  const opacitySlider = document.getElementById('myo-opacity-slider');
  if (opacitySlider) {
    opacitySlider.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      if (chamberMesh && chamberMesh.material) {
        chamberMesh.material.opacity = val;
      }
    });
  }

  const toggleMyoBtn = document.getElementById('btn-toggle-myocardium');
  if (toggleMyoBtn) {
    toggleMyoBtn.addEventListener('click', () => {
      if (chamberMesh) {
        chamberMesh.visible = !chamberMesh.visible;
        toggleMyoBtn.style.opacity = chamberMesh.visible ? '1' : '0.5';
      }
    });
  }
}

function setupVesselMaterial(mesh, prob, isSelected) {
  const col = getRiskColor(prob);
  mesh.material = new THREE.MeshStandardMaterial({
    color: col,
    roughness: 0.2,
    metalness: 0.2,
    emissive: col,
    emissiveIntensity: isSelected ? 0.75 : 0.18,
  });
}

export function updateVessels(probs) {
  if (probs.LAD !== undefined) {
    vesselBadges.LAD.prob = probs.LAD;
    if (ladMesh) setupVesselMaterial(ladMesh, probs.LAD, selectedVessel === 'LAD');
    updateBadgeText('LAD', probs.LAD);
  }
  if (probs.LCX !== undefined) {
    vesselBadges.LCX.prob = probs.LCX;
    if (lcxMesh) setupVesselMaterial(lcxMesh, probs.LCX, selectedVessel === 'LCX');
    updateBadgeText('LCX', probs.LCX);
  }
  if (probs.RCA !== undefined) {
    vesselBadges.RCA.prob = probs.RCA;
    if (rcaMesh) setupVesselMaterial(rcaMesh, probs.RCA, selectedVessel === 'RCA');
    updateBadgeText('RCA', probs.RCA);
  }
}

export function setSelectedVessel(vesselId) {
  selectedVessel = vesselId;
  if (ladMesh) setupVesselMaterial(ladMesh, vesselBadges.LAD.prob, selectedVessel === 'LAD');
  if (lcxMesh) setupVesselMaterial(lcxMesh, vesselBadges.LCX.prob, selectedVessel === 'LCX');
  if (rcaMesh) setupVesselMaterial(rcaMesh, vesselBadges.RCA.prob, selectedVessel === 'RCA');
}

function createBadgeElements(container) {
  Object.keys(vesselBadges).forEach((key) => {
    const badge = vesselBadges[key];
    const el = document.createElement('div');
    el.className = 'vessel-3d-badge';
    el.id = `badge-${key}`;
    const colHex = '#' + getRiskColor(badge.prob).getHexString();
    el.style.border = `1.5px solid ${colHex}`;

    el.innerHTML = `
      <span class="badge-dot" style="background-color: ${colHex}; box-shadow: 0 0 6px ${colHex};"></span>
      <span style="color: #cbd5e1;">${badge.label}:</span>
      <span style="color: ${colHex};" id="badge-val-${key}">${Math.round(badge.prob * 100)}%</span>
    `;

    el.addEventListener('click', (e) => {
      e.stopPropagation();
      selectVesselTrigger(key);
    });

    container.appendChild(el);
    badge.el = el;
  });
}

function updateBadgeText(key, prob) {
  const badge = vesselBadges[key];
  if (!badge || !badge.el) return;
  const colHex = '#' + getRiskColor(prob).getHexString();
  badge.el.style.border = `1.5px solid ${colHex}`;
  const valEl = document.getElementById(`badge-val-${key}`);
  if (valEl) {
    valEl.textContent = `${Math.round(prob * 100)}%`;
    valEl.style.color = colHex;
  }
  const dotEl = badge.el.querySelector('.badge-dot');
  if (dotEl) {
    dotEl.style.backgroundColor = colHex;
    dotEl.style.boxShadow = `0 0 6px ${colHex}`;
  }
}

function updateBadgePositions() {
  if (!camera || !renderer) return;
  const canvas = renderer.domElement;
  const widthHalf = canvas.clientWidth / 2;
  const heightHalf = canvas.clientHeight / 2;

  Object.keys(vesselBadges).forEach((key) => {
    const badge = vesselBadges[key];
    if (!badge.el) return;

    const screenPos = badge.pos.clone().project(camera);

    // Behind camera check
    if (screenPos.z > 1) {
      badge.el.style.display = 'none';
      return;
    }

    badge.el.style.display = 'flex';
    const x = screenPos.x * widthHalf + widthHalf;
    const y = -(screenPos.y * heightHalf) + heightHalf;
    badge.el.style.left = `${x}px`;
    badge.el.style.top = `${y}px`;
  });
}

function setupRaycasting(container) {
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();

  container.addEventListener('click', (event) => {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects([ladMesh, lcxMesh, rcaMesh].filter(Boolean), true);

    if (intersects.length > 0) {
      const hit = intersects[0].object;
      if (hit === ladMesh) selectVesselTrigger('LAD');
      else if (hit === lcxMesh) selectVesselTrigger('LCX');
      else if (hit === rcaMesh) selectVesselTrigger('RCA');
    }
  });
}

function selectVesselTrigger(vesselId) {
  setSelectedVessel(selectedVessel === vesselId ? null : vesselId);
  // Trigger HTMX request to switch SHAP target
  const target = selectedVessel || 'Cath';
  const badge = document.getElementById('active-target-badge');
  if (badge) {
    badge.textContent = target === 'Cath' ? 'Overall CAD (Cath)' : `${target} Vessel`;
  }
  if (window.htmx) {
    window.htmx.ajax('GET', `/api/shap?target=${target}`, { target: '#shap-container', swap: 'innerHTML' });
  }
}

function onWindowResize() {
  const container = document.getElementById('heart-canvas-container');
  if (!container || !renderer || !camera) return;
  const w = container.clientWidth;
  const h = container.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  updateBadgePositions();
  renderer.render(scene, camera);
}

// Auto init on load
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initHeartViewer);
} else {
  initHeartViewer();
}
