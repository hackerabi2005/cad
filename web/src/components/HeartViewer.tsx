import React, { Suspense, useRef, useState, useEffect } from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { OrbitControls, useGLTF, Html, Center } from '@react-three/drei';
import * as THREE from 'three';
import { Eye, EyeOff, RotateCcw, Info, Activity, Tag, Layers, Compass } from 'lucide-react';
import vesselsConfig from '../vessels.json';
import { VesselsConfig, PredictResponse } from '../types';
import { LumenCrossSection } from './LumenCrossSection';

interface HeartViewerProps {
  prediction: PredictResponse | null;
  selectedVessel: string | null;
  onSelectVessel: (vesselId: string | null) => void;
  mode?: 'balanced' | 'screening';
}

const HEART_GLB_PATH = `${(import.meta as any).env?.BASE_URL || '/'}models/heart.glb`;

export function hasWebGL(): boolean {
  if (typeof window === 'undefined') return true;
  try {
    const canvas = document.createElement('canvas');
    return !!(window.WebGLRenderingContext && (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')));
  } catch (e) {
    return false;
  }
}

// Continuous risk color interpolation:
// Standard: green (#10b981) -> lime (#84cc16) -> amber (#f59e0b) -> orange (#f97316) -> red (#ef4444)
// Colorblind-safe (Viridis): navy (#3b528b) -> teal (#21918c) -> lime (#35b779) -> yellow (#fde725)
export function getRiskColor(prob: number, colorblind: boolean = false): string {
  const p = Math.max(0, Math.min(1, prob));
  if (colorblind) {
    if (p < 0.25) return interpolateColor('#3b528b', '#21918c', p / 0.25);
    if (p < 0.50) return interpolateColor('#21918c', '#35b779', (p - 0.25) / 0.25);
    if (p < 0.75) return interpolateColor('#35b779', '#90d743', (p - 0.50) / 0.25);
    return interpolateColor('#90d743', '#fde725', (p - 0.75) / 0.25);
  }
  if (p < 0.25) {
    const t = p / 0.25;
    return interpolateColor('#10b981', '#84cc16', t);
  } else if (p < 0.50) {
    const t = (p - 0.25) / 0.25;
    return interpolateColor('#84cc16', '#f59e0b', t);
  } else if (p < 0.75) {
    const t = (p - 0.50) / 0.25;
    return interpolateColor('#f59e0b', '#f97316', t);
  } else {
    const t = (p - 0.75) / 0.25;
    return interpolateColor('#f97316', '#ef4444', t);
  }
}

// Color scale anchored to individual vessel operating threshold or absolute probability
export function getVesselRiskColor(
  prob: number,
  cutoff: number = 0.5,
  colorblind: boolean = false,
  absolute: boolean = false
): string {
  if (absolute || !cutoff || cutoff <= 0) return getRiskColor(prob, colorblind);
  const ratio = prob / cutoff;
  if (colorblind) {
    if (ratio < 0.6) return interpolateColor('#3b528b', '#21918c', ratio / 0.6);
    if (ratio < 1.0) return interpolateColor('#21918c', '#35b779', (ratio - 0.6) / 0.4);
    if (ratio < 1.35) return interpolateColor('#35b779', '#90d743', (ratio - 1.0) / 0.35);
    return interpolateColor('#90d743', '#fde725', Math.min(1, (ratio - 1.35) / 0.45));
  }
  if (ratio < 0.6) {
    const t = ratio / 0.6;
    return interpolateColor('#10b981', '#84cc16', t);
  } else if (ratio < 1.0) {
    const t = (ratio - 0.6) / 0.4;
    return interpolateColor('#84cc16', '#f59e0b', t);
  } else if (ratio < 1.35) {
    const t = (ratio - 1.0) / 0.35;
    return interpolateColor('#f59e0b', '#f97316', t);
  } else {
    const t = Math.min(1, (ratio - 1.35) / 0.45);
    return interpolateColor('#f97316', '#ef4444', t);
  }
}

function interpolateColor(color1: string, color2: string, factor: number): string {
  const c1 = new THREE.Color(color1);
  const c2 = new THREE.Color(color2);
  c1.lerp(c2, factor);
  return '#' + c1.getHexString();
}

// Smooth animated camera framing for focused vessel inspection and presets
function CameraController({
  selectedVessel,
  viewPreset,
  controlsRef,
}: {
  selectedVessel: string | null;
  viewPreset: 'Anterior' | 'Left Lateral' | 'Posterior' | 'Inferior' | null;
  controlsRef: React.RefObject<any>;
}) {
  const { camera } = useThree();
  const targetPos = useRef(new THREE.Vector3(0, 0.4, 3.2));
  const targetLookAt = useRef(new THREE.Vector3(0, 0, 0));
  const isAnimating = useRef(false);

  useEffect(() => {
    if (selectedVessel === 'LAD') {
      targetPos.current.set(0.3, 0.35, 1.9);
      targetLookAt.current.set(0.18, 0.12, 0.45);
    } else if (selectedVessel === 'LCX') {
      targetPos.current.set(1.4, 0.45, 1.5);
      targetLookAt.current.set(0.55, 0.25, -0.05);
    } else if (selectedVessel === 'RCA') {
      targetPos.current.set(-1.3, 0.35, 1.7);
      targetLookAt.current.set(-0.45, 0.18, 0.25);
    } else if (viewPreset === 'Left Lateral') {
      targetPos.current.set(3.2, 0.4, 0.2);
      targetLookAt.current.set(0, 0, 0);
    } else if (viewPreset === 'Posterior') {
      targetPos.current.set(0, 0.4, -3.2);
      targetLookAt.current.set(0, 0, 0);
    } else if (viewPreset === 'Inferior') {
      targetPos.current.set(0, -3.2, 0.4);
      targetLookAt.current.set(0, 0, 0);
    } else {
      // Anterior default
      targetPos.current.set(0, 0.4, 3.2);
      targetLookAt.current.set(0, 0, 0);
    }
    isAnimating.current = true;
  }, [selectedVessel, viewPreset]);

  useFrame((_, delta) => {
    if (!isAnimating.current) return;
    const lerpFactor = Math.min(delta * 4.5, 0.2);
    camera.position.lerp(targetPos.current, lerpFactor);
    if (controlsRef.current) {
      controlsRef.current.target.lerp(targetLookAt.current, lerpFactor);
      controlsRef.current.update();
    }
    if (
      camera.position.distanceTo(targetPos.current) < 0.02 &&
      (!controlsRef.current || controlsRef.current.target.distanceTo(targetLookAt.current) < 0.02)
    ) {
      isAnimating.current = false;
    }
  });

  return null;
}

// Individual vessel mesh with normal-inflated hit proxy and selection dimming
function VesselMeshItem({
  vessel,
  node,
  prob,
  cutoff,
  isSelected,
  selectedVessel,
  showLabels,
  isPulsing,
  colorblind = false,
  absolute = false,
  onSelectVessel,
}: {
  vessel: any;
  node: any;
  prob: number | undefined;
  cutoff?: number;
  isSelected: boolean;
  selectedVessel: string | null;
  showLabels: boolean;
  isPulsing: boolean;
  colorblind?: boolean;
  absolute?: boolean;
  onSelectVessel: (id: string | null) => void;
}) {
  const [isHovered, setIsHovered] = useState(false);
  const materialRef = useRef<THREE.MeshStandardMaterial>(null);
  const hasProb = prob !== undefined && prob !== null;
  const effectiveCutoff = cutoff ?? 0.5;
  const colorHex = hasProb ? getVesselRiskColor(prob, effectiveCutoff, colorblind, absolute) : '#64748b';
  const isAboveCutoff = hasProb && prob >= effectiveCutoff;
  const badgeText = hasProb ? `${(prob * 100).toFixed(0)}%` : '—';
  const isDimmed = selectedVessel !== null && !isSelected;

  // C1: Inflated in-place hit proxy along vertex normals (delta = 0.012)
  const hitProxyGeometry = React.useMemo(() => {
    if (!node?.geometry) return null;
    const geom = node.geometry.clone();
    const pos = geom.attributes.position;
    const norm = geom.attributes.normal;
    if (pos && norm) {
      const delta = 0.012;
      for (let i = 0; i < pos.count; i++) {
        pos.setXYZ(
          i,
          pos.getX(i) + norm.getX(i) * delta,
          pos.getY(i) + norm.getY(i) * delta,
          pos.getZ(i) + norm.getZ(i) * delta
        );
      }
      pos.needsUpdate = true;
    }
    return geom;
  }, [node?.geometry]);

  useFrame((state) => {
    if (!materialRef.current) return;
    if (isDimmed) {
      materialRef.current.opacity = 0.28;
      materialRef.current.emissiveIntensity = isHovered ? 0.35 : 0.04;
      return;
    }
    materialRef.current.opacity = 1.0;
    const prefersReducedMotion = typeof window !== 'undefined' && window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (isPulsing && !prefersReducedMotion && isAboveCutoff) {
      const pulse = (Math.sin(state.clock.elapsedTime * 4.5) + 1) * 0.5;
      materialRef.current.emissiveIntensity = (isSelected ? 0.65 : (isHovered ? 0.45 : 0.15)) + 0.35 * pulse;
    } else {
      materialRef.current.emissiveIntensity = isSelected ? 0.85 : (isHovered ? 0.55 : 0.15);
    }
  });

  return (
    <group key={vessel.id}>
      {/* High-fidelity Vessel Geometry */}
      <mesh
        geometry={node.geometry}
        onClick={(e) => {
          e.stopPropagation();
          onSelectVessel(isSelected ? null : vessel.id);
        }}
        onPointerOver={(e) => {
          e.stopPropagation();
          setIsHovered(true);
          document.body.style.cursor = 'pointer';
        }}
        onPointerOut={() => {
          setIsHovered(false);
          document.body.style.cursor = 'auto';
        }}
      >
        <meshStandardMaterial
          ref={materialRef}
          color={colorHex}
          roughness={0.2}
          metalness={0.2}
          transparent={isDimmed}
          opacity={isDimmed ? 0.28 : 1.0}
          emissive={new THREE.Color(colorHex)}
          emissiveIntensity={isSelected ? 0.85 : (isHovered ? 0.55 : (isDimmed ? 0.04 : 0.15))}
        />
      </mesh>

      {/* Invisible In-place Hit-Mesh inflated along normals for accurate picking */}
      <mesh
        geometry={hitProxyGeometry || node.geometry}
        visible={true}
        onClick={(e) => {
          e.stopPropagation();
          onSelectVessel(isSelected ? null : vessel.id);
        }}
        onPointerOver={(e) => {
          e.stopPropagation();
          setIsHovered(true);
          document.body.style.cursor = 'pointer';
        }}
        onPointerOut={() => {
          setIsHovered(false);
          document.body.style.cursor = 'auto';
        }}
      >
        <meshBasicMaterial transparent opacity={0} depthWrite={false} />
      </mesh>

      {/* 3D Floating Risk Badge */}
      {showLabels && (
        <Html
          position={vessel.badgePosition}
          center
          distanceFactor={6}
          zIndexRange={[100, 0]}
        >
          <div
            onClick={(e) => {
              e.stopPropagation();
              onSelectVessel(isSelected ? null : vessel.id);
            }}
            className={`cursor-pointer transition-all transform hover:scale-105 flex items-center select-none shadow-md ${
              isSelected ? 'scale-110 shadow-cyan-500/50' : 'opacity-95'
            }`}
            style={{
              backgroundColor: 'rgba(11, 18, 33, 0.94)',
              border: `1.5px solid ${isSelected ? '#38bdf8' : colorHex}`,
              borderRadius: '9999px',
              padding: '3px 8px',
              fontSize: '11px',
              fontFamily: 'monospace',
              fontWeight: 700,
              whiteSpace: 'nowrap',
              gap: '5px',
              boxShadow: isSelected ? '0 0 14px rgba(56, 189, 248, 0.6)' : '0 2px 8px rgba(0, 0, 0, 0.5)',
              cursor: 'pointer',
              pointerEvents: 'auto',
            }}
          >
            <span
              style={{
                width: '7px',
                height: '7px',
                borderRadius: '9999px',
                backgroundColor: colorHex,
                display: 'inline-block',
                flexShrink: 0,
                boxShadow: `0 0 6px ${colorHex}`,
              }}
            />
            <span style={{ color: '#e2e8f0', letterSpacing: '0.02em' }}>{vessel.abbreviation}:</span>
            <span style={{ color: colorHex, fontWeight: 800 }}>{badgeText}</span>
          </div>
        </Html>
      )}
    </group>
  );
}

function HeartModel({
  prediction,
  selectedVessel,
  onSelectVessel,
  showChambers,
  chamberOpacity,
  showLabels,
  showTerritories,
  isPulsing,
  colorblind = false,
  absolute = false,
  mode = 'balanced',
}: {
  prediction: PredictResponse | null;
  selectedVessel: string | null;
  onSelectVessel: (id: string | null) => void;
  showChambers: boolean;
  chamberOpacity: number;
  showLabels: boolean;
  showTerritories: boolean;
  isPulsing: boolean;
  colorblind?: boolean;
  absolute?: boolean;
  mode?: 'balanced' | 'screening';
}) {
  const { nodes } = useGLTF(HEART_GLB_PATH) as any;
  const config = vesselsConfig as VesselsConfig;

  return (
    <group dispose={null}>
      {/* 1. Coronary Arteries */}
      {config.vessels.map((vessel) => {
        const node = nodes[vessel.nodeName];
        if (!node || !node.geometry) return null;

        const vPred = prediction?.vessels[vessel.id as 'LAD' | 'LCX' | 'RCA'];
        const prob = vPred?.prob;
        const cutoff = mode === 'screening' ? (vPred?.high_sensitivity_threshold ?? 0.42) : 0.50;
        const isSelected = selectedVessel === vessel.id;

        return (
          <VesselMeshItem
            key={vessel.id}
            vessel={vessel}
            node={node}
            prob={prob}
            cutoff={cutoff}
            isSelected={isSelected}
            selectedVessel={selectedVessel}
            showLabels={showLabels}
            isPulsing={isPulsing}
            colorblind={colorblind}
            absolute={absolute}
            onSelectVessel={onSelectVessel}
          />
        );
      })}

      {/* 2. Dynamic anatomy structures from config */}
      {config.anatomy?.map((anat) => {
        const node = nodes[anat.nodeName];
        if (!node?.geometry) return null;
        const isMuscle = anat.id === 'Heart_Muscle';
        if (isMuscle && !showChambers) return null;
        const baseOpacity = isMuscle ? chamberOpacity : (anat.defaultOpacity ?? 0.9);
        const opacity = selectedVessel && !isMuscle ? baseOpacity * 0.35 : baseOpacity;
        return (
          <mesh key={anat.id} geometry={node.geometry}>
            <meshStandardMaterial
              color={anat.defaultColor}
              roughness={isMuscle ? 0.5 : 0.3}
              metalness={0.05}
              opacity={opacity}
              transparent
              depthWrite={!isMuscle}
            />
          </mesh>
        );
      })}

      {/* 3. Myocardium vascular territories overlay markers */}
      {showTerritories && (
        <group>
          {/* LAD Territory: Anterior & Septal */}
          <Html position={[0.22, -0.05, 0.65]} center distanceFactor={5.5}>
            <div className="px-2 py-0.5 rounded-md bg-emerald-950/90 border border-emerald-500/60 text-[10px] text-emerald-300 font-bold whitespace-nowrap shadow-lg">
              LAD: Anterior / Septal
            </div>
          </Html>
          {/* LCX Territory: Lateral & Posterior */}
          <Html position={[0.75, 0.05, -0.25]} center distanceFactor={5.5}>
            <div className="px-2 py-0.5 rounded-md bg-indigo-950/90 border border-indigo-500/60 text-[10px] text-indigo-300 font-bold whitespace-nowrap shadow-lg">
              LCX: Lateral / Posterior
            </div>
          </Html>
          {/* RCA Territory: Right & Inferior */}
          <Html position={[-0.55, -0.15, 0.35]} center distanceFactor={5.5}>
            <div className="px-2 py-0.5 rounded-md bg-amber-950/90 border border-amber-500/60 text-[10px] text-amber-300 font-bold whitespace-nowrap shadow-lg">
              RCA: Right Ventricle / Inferior
            </div>
          </Html>
        </group>
      )}
    </group>
  );
}

export function HeartViewer({
  prediction,
  selectedVessel,
  onSelectVessel,
  mode = 'balanced',
}: HeartViewerProps) {
  const controlsRef = useRef<any>(null);
  const [showChambers, setShowChambers] = useState(true);
  const [chamberOpacity, setChamberOpacity] = useState(0.25);
  const [isPulsing, setIsPulsing] = useState(true);
  const [pulseActive, setPulseActive] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [showTerritories, setShowTerritories] = useState(false);
  const [showLumenModal, setShowLumenModal] = useState(false);
  const [viewPreset, setViewPreset] = useState<'Anterior' | 'Left Lateral' | 'Posterior' | 'Inferior' | null>('Anterior');
  const [isColorblindSafe, setIsColorblindSafe] = useState(false);
  const [useAbsoluteProb, setUseAbsoluteProb] = useState(false);

  // Settle continuous pulse after 4.5s on new predictions to conserve CPU under software rendering
  useEffect(() => {
    setPulseActive(true);
    const timer = setTimeout(() => {
      setPulseActive(false);
    }, 4500);
    return () => clearTimeout(timer);
  }, [prediction]);

  const isScreening = mode === 'screening';

  const resetCamera = () => {
    onSelectVessel(null);
    setViewPreset('Anterior');
    if (controlsRef.current) {
      controlsRef.current.reset();
    }
  };

  const activeVesselForLumen = selectedVessel || 'LAD';
  const activeVesselProb =
    prediction?.vessels[activeVesselForLumen as 'LAD' | 'LCX' | 'RCA']?.prob ?? 0.5;
  const activeVesselCutoff = isScreening
    ? (prediction?.vessels[activeVesselForLumen as 'LAD' | 'LCX' | 'RCA']?.high_sensitivity_threshold ?? 0.42)
    : 0.50;

  return (
    <div
      className="relative w-full h-[520px] lg:h-full min-h-[480px] bg-gradient-to-b from-[#0a0f1d] to-[#050811] rounded-2xl border border-slate-800/80 overflow-hidden shadow-2xl flex flex-col"
      style={{ minHeight: '480px', height: '100%', position: 'relative' }}
    >
      {/* 3D Viewport Controls Overlay */}
      <div className="absolute top-4 left-4 z-10 flex flex-wrap items-center gap-2">
        {/* Toggle Myocardium */}
        <button
          onClick={() => setShowChambers(!showChambers)}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border backdrop-blur-md transition-colors ${
            showChambers
              ? 'bg-slate-800/80 text-cyan-400 border-cyan-500/40 shadow-sm'
              : 'bg-slate-900/60 text-slate-400 border-slate-700/50'
          }`}
          title="Toggle Heart Muscle / Chambers"
        >
          {showChambers ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
          <span>Myocardium</span>
        </button>

        {showChambers && (
          <div className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg bg-slate-900/70 border border-slate-800/70 text-xs backdrop-blur-md">
            <span className="text-[11px] text-slate-400">Opacity:</span>
            <input
              type="range"
              min="0.05"
              max="0.6"
              step="0.05"
              value={chamberOpacity}
              onChange={(e) => setChamberOpacity(parseFloat(e.target.value))}
              className="w-16 h-1 bg-slate-700 rounded-lg cursor-pointer"
            />
          </div>
        )}

        {/* C4: Toggle Territories */}
        <button
          onClick={() => setShowTerritories(!showTerritories)}
          className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium border backdrop-blur-md transition-colors ${
            showTerritories
              ? 'bg-indigo-500/25 text-indigo-300 border-indigo-500/40 shadow-sm'
              : 'bg-slate-900/70 text-slate-400 border-slate-800/70'
          }`}
          title="Toggle Myocardium Vascular Territories Overlay"
        >
          <Compass className="w-3.5 h-3.5" />
          <span>Territories</span>
        </button>

        {/* Toggle Pulsing Glow */}
        <button
          onClick={() => {
            const next = !isPulsing;
            setIsPulsing(next);
            if (next) setPulseActive(true);
          }}
          className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium border backdrop-blur-md transition-colors ${
            isPulsing
              ? 'bg-amber-500/20 text-amber-300 border-amber-500/40 shadow-sm'
              : 'bg-slate-900/70 text-slate-400 border-slate-800/70'
          }`}
          title="Toggle Pulsating Glow on Stenotic Arteries"
        >
          <Activity className="w-3.5 h-3.5" />
          <span>Pulse {isPulsing ? 'On' : 'Off'}</span>
        </button>

        {/* Toggle 3D Labels */}
        <button
          onClick={() => setShowLabels(!showLabels)}
          className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium border backdrop-blur-md transition-colors ${
            showLabels
              ? 'bg-slate-800/80 text-cyan-400 border-cyan-500/40'
              : 'bg-slate-900/70 text-slate-400 border-slate-800/70'
          }`}
          title="Toggle 3D Floating Branch Labels"
        >
          <Tag className="w-3.5 h-3.5" />
          <span>Labels</span>
        </button>

        {/* Open Lumen Cross-Section Visualizer */}
        <button
          onClick={() => setShowLumenModal(!showLumenModal)}
          className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium border backdrop-blur-md transition-colors ${
            showLumenModal
              ? 'bg-rose-500/25 text-rose-300 border-rose-500/40 shadow-sm'
              : 'bg-slate-900/70 text-slate-300 border-slate-800/70 hover:bg-slate-800'
          }`}
          title="Inspect 2D Lumen Cross-Section Schematic"
        >
          <Layers className="w-3.5 h-3.5" />
          <span>Lumen View</span>
        </button>

        {/* Toggle Accessible Colorblind Palette */}
        <button
          onClick={() => setIsColorblindSafe(!isColorblindSafe)}
          className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium border backdrop-blur-md transition-colors ${
            isColorblindSafe
              ? 'bg-emerald-500/25 text-emerald-300 border-emerald-500/40 shadow-sm'
              : 'bg-slate-900/70 text-slate-400 border-slate-800/70'
          }`}
          title="Toggle Colorblind-Safe Viridis Palette"
        >
          <span>{isColorblindSafe ? 'Viridis (Accessible)' : 'Standard Ramp'}</span>
        </button>

        {/* Toggle Absolute Probability Mode */}
        <button
          onClick={() => setUseAbsoluteProb(!useAbsoluteProb)}
          className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium border backdrop-blur-md transition-colors ${
            useAbsoluteProb
              ? 'bg-cyan-500/25 text-cyan-300 border-cyan-500/40 shadow-sm'
              : 'bg-slate-900/70 text-slate-400 border-slate-800/70'
          }`}
          title="Toggle Absolute Probability vs Cutoff-Relative Coloring"
        >
          <span>{useAbsoluteProb ? 'Absolute %' : 'Relative Cutoff'}</span>
        </button>

        {/* Reset Camera */}
        <button
          onClick={resetCamera}
          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium bg-slate-900/70 text-slate-300 border border-slate-800/70 hover:bg-slate-800 backdrop-blur-md transition-colors"
          title="Reset Camera Orientation"
        >
          <RotateCcw className="w-3.5 h-3.5 text-slate-400" />
          <span>Reset</span>
        </button>
      </div>

      {/* C5: View Presets Bar */}
      <div className="absolute top-16 left-4 z-10 flex items-center gap-1 bg-slate-950/80 backdrop-blur-md p-1 rounded-xl border border-slate-800/80 text-[10px]">
        <span className="text-slate-400 px-1 font-semibold">View:</span>
        {(['Anterior', 'Left Lateral', 'Posterior', 'Inferior'] as const).map((preset) => (
          <button
            key={preset}
            onClick={() => {
              setViewPreset(preset);
              onSelectVessel(null);
            }}
            className={`px-2 py-0.5 rounded-lg font-medium transition-all ${
              viewPreset === preset && selectedVessel === null
                ? 'bg-cyan-500/25 text-cyan-300 font-bold border border-cyan-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            {preset}
          </button>
        ))}
      </div>

      {/* Selected Vessel HUD Pill / In-Canvas Educational Watermark */}
      {selectedVessel ? (
        <div className="absolute top-4 right-4 z-10 flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-900/90 border border-cyan-500/50 shadow-lg backdrop-blur-md animate-fade-in">
          <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-ping" />
          <span className="text-xs text-slate-300 font-medium">Focused:</span>
          <span className="text-xs font-bold text-cyan-300 font-mono">{selectedVessel}</span>
          <button
            onClick={() => onSelectVessel(null)}
            className="ml-1 text-slate-400 hover:text-white text-xs px-1"
            title="Deselect"
          >
            ✕
          </button>
        </div>
      ) : (
        <div className="absolute top-4 right-4 z-10 pointer-events-none hidden sm:flex flex-col items-end opacity-75">
          <span className="text-[10px] font-mono tracking-wider text-cyan-300/80 bg-slate-950/80 px-2 py-0.5 rounded-md border border-slate-800 backdrop-blur-sm">
            Decision Support / Educational
          </span>
          <span className="text-[9px] text-slate-500 font-mono mt-0.5">
            BodyParts3D Mesh (CC BY-SA 2.1 JP)
          </span>
        </div>
      )}

      {/* 3D Canvas or C2: 2D SVG schematic fallback */}
      {!hasWebGL() ? (
        <div className="w-full h-full flex flex-col items-center justify-center p-6 text-center select-none" data-testid="webgl-fallback">
          <span className="text-xs text-amber-400 font-semibold mb-2">
            WebGL Unavailable — 2D Coronary Schematic Fallback
          </span>
          <svg viewBox="0 0 400 320" className="w-80 h-72 max-w-full">
            {/* Myocardium chambers shell */}
            <path
              d="M 200,60 C 130,20 60,100 100,200 C 130,260 200,300 200,300 C 200,300 270,260 300,200 C 340,100 270,20 200,60 Z"
              fill={showChambers ? '#be123c' : '#1e293b'}
              fillOpacity={chamberOpacity}
              stroke="#475569"
              strokeWidth="2"
            />
            {/* Aorta */}
            <path d="M 180,60 C 180,10 220,10 220,60" fill="none" stroke="#be123c" strokeWidth="14" strokeLinecap="round" />

            {/* Vascular Territories partition overlay */}
            {showTerritories && (
              <g opacity="0.35">
                <path d="M 200,60 L 195,270 L 140,210 Z" fill="#10b981" />
                <path d="M 200,60 L 260,190 L 195,270 Z" fill="#6366f1" />
                <path d="M 200,60 L 140,210 L 100,200 Z" fill="#f59e0b" />
              </g>
            )}

            {/* LAD branch */}
            <path
              data-testid="vessel-path-LAD"
              d="M 195,60 C 190,120 185,180 195,270"
              fill="none"
              stroke={prediction?.vessels?.LAD?.prob !== undefined ? getRiskColor(prediction.vessels.LAD.prob, isColorblindSafe) : '#64748b'}
              strokeWidth={selectedVessel === 'LAD' ? 10 : (selectedVessel ? 4 : 6)}
              opacity={selectedVessel && selectedVessel !== 'LAD' ? 0.35 : 1.0}
              className="cursor-pointer"
              onClick={() => onSelectVessel(selectedVessel === 'LAD' ? null : 'LAD')}
            />
            {/* LCX branch */}
            <path
              data-testid="vessel-path-LCX"
              d="M 205,65 C 240,90 270,140 260,190"
              fill="none"
              stroke={prediction?.vessels?.LCX?.prob !== undefined ? getRiskColor(prediction.vessels.LCX.prob, isColorblindSafe) : '#64748b'}
              strokeWidth={selectedVessel === 'LCX' ? 10 : (selectedVessel ? 4 : 6)}
              opacity={selectedVessel && selectedVessel !== 'LCX' ? 0.35 : 1.0}
              className="cursor-pointer"
              onClick={() => onSelectVessel(selectedVessel === 'LCX' ? null : 'LCX')}
            />
            {/* RCA branch */}
            <path
              data-testid="vessel-path-RCA"
              d="M 185,65 C 150,90 130,150 140,210"
              fill="none"
              stroke={prediction?.vessels?.RCA?.prob !== undefined ? getRiskColor(prediction.vessels.RCA.prob, isColorblindSafe) : '#64748b'}
              strokeWidth={selectedVessel === 'RCA' ? 10 : (selectedVessel ? 4 : 6)}
              opacity={selectedVessel && selectedVessel !== 'RCA' ? 0.35 : 1.0}
              className="cursor-pointer"
              onClick={() => onSelectVessel(selectedVessel === 'RCA' ? null : 'RCA')}
            />
            <text x="195" y="285" fill="#f8fafc" fontSize="11" textAnchor="middle" fontWeight="bold">
              LAD: {prediction?.vessels?.LAD?.prob !== undefined ? `${Math.round(prediction.vessels.LAD.prob * 100)}%` : '—'}
            </text>
            <text x="270" y="205" fill="#f8fafc" fontSize="11" textAnchor="middle" fontWeight="bold">
              LCX: {prediction?.vessels?.LCX?.prob !== undefined ? `${Math.round(prediction.vessels.LCX.prob * 100)}%` : '—'}
            </text>
            <text x="130" y="225" fill="#f8fafc" fontSize="11" textAnchor="middle" fontWeight="bold">
              RCA: {prediction?.vessels?.RCA?.prob !== undefined ? `${Math.round(prediction.vessels.RCA.prob * 100)}%` : '—'}
            </text>
          </svg>
        </div>
      ) : (
        <Canvas
          frameloop={isPulsing && pulseActive ? 'always' : 'demand'}
          dpr={[1, 1.5]}
          camera={{ position: [0, 0.4, 3.2], fov: 42 }}
          className="w-full h-full cursor-grab active:cursor-grabbing"
        >
          <ambientLight intensity={1.2} />
          <directionalLight position={[4, 5, 4]} intensity={1.8} />
          <directionalLight position={[-4, -3, -4]} intensity={0.6} color="#60a5fa" />
          <pointLight position={[0, 2, 2]} intensity={0.8} />

          <Suspense
            fallback={
              <Html center>
                <div className="flex flex-col items-center gap-2 bg-slate-900/90 p-4 rounded-xl border border-slate-800 text-slate-300 text-xs">
                  <div className="w-6 h-6 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
                  <span>Loading Anatomical Model...</span>
                </div>
              </Html>
            }
          >
            <Center>
              <HeartModel
                prediction={prediction}
                selectedVessel={selectedVessel}
                onSelectVessel={onSelectVessel}
                showChambers={showChambers}
                chamberOpacity={chamberOpacity}
                showLabels={showLabels}
                showTerritories={showTerritories}
                isPulsing={isPulsing && pulseActive}
                colorblind={isColorblindSafe}
                absolute={useAbsoluteProb}
                mode={mode}
              />
            </Center>
          </Suspense>

          <CameraController
            selectedVessel={selectedVessel}
            viewPreset={viewPreset}
            controlsRef={controlsRef}
          />

          <OrbitControls
            ref={controlsRef}
            enableDamping
            dampingFactor={0.05}
            minDistance={1.2}
            maxDistance={7.0}
            autoRotate={false}
          />
        </Canvas>
      )}

      {/* Floating Lumen Cross-Section Overlay */}
      {showLumenModal && (
        <div className="absolute inset-0 z-30 bg-slate-950/70 backdrop-blur-sm p-4 flex items-center justify-center animate-fade-in">
          <LumenCrossSection
            vesselId={activeVesselForLumen}
            prob={activeVesselProb}
            cutoff={activeVesselCutoff}
            mode={mode}
            onClose={() => setShowLumenModal(false)}
            onSelectVessel={(vId) => onSelectVessel(vId)}
          />
        </div>
      )}

      {/* Risk Color Spectrum Bar & Legend */}
      <div className="absolute bottom-3 left-4 right-4 z-10 flex flex-col md:flex-row items-center justify-between gap-2 p-2.5 rounded-xl bg-slate-900/85 border border-slate-800/80 backdrop-blur-md text-[11px]">
        <div className="flex items-center gap-2 text-slate-300">
          <Info className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0" />
          <span>3D: Drag rotate · Scroll zoom · Right-drag pan · Click mesh to inspect</span>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <span className="text-slate-400 text-[10px] uppercase font-bold tracking-wider" title="Predicted probability of ≥50% stenosis">
            {useAbsoluteProb ? 'Absolute P(≥50% Stenosis):' : 'Risk vs Cutoff:'}
          </span>
          {(isColorblindSafe
            ? [
                { stop: 0.15, color: '#3b528b', label: '<25%' },
                { stop: 0.35, color: '#21918c', label: '25–50%' },
                { stop: 0.65, color: '#35b779', label: '50–75%' },
                { stop: 0.85, color: '#fde725', label: '>75%' },
              ]
            : (vesselsConfig as VesselsConfig).colorScale || []
          ).map((cs: any) => (
            <div key={cs.stop} className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: cs.color }} />
              <span className="text-slate-300 text-[10px]">{cs.label}</span>
            </div>
          ))}
          {prediction?.vessels && (
            <div className="flex flex-wrap items-center gap-1.5 pl-2 border-l border-slate-800 text-[10px] text-slate-400 font-mono">
              <span className="text-cyan-400 font-bold">[{isScreening ? 'Screening' : 'Default (0.50)'}]:</span>
              <span>LAD {((prediction.vessels.LAD?.prob ?? 0) * 100).toFixed(0)}% (cut: {isScreening ? `${Math.round((prediction.vessels.LAD?.high_sensitivity_threshold ?? 0.42) * 100)}%` : '50%'})</span>
              <span>·</span>
              <span>LCX {((prediction.vessels.LCX?.prob ?? 0) * 100).toFixed(0)}% (cut: {isScreening ? `${Math.round((prediction.vessels.LCX?.high_sensitivity_threshold ?? 0.25) * 100)}%` : '50%'})</span>
              <span>·</span>
              <span>RCA {((prediction.vessels.RCA?.prob ?? 0) * 100).toFixed(0)}% (cut: {isScreening ? `${Math.round((prediction.vessels.RCA?.high_sensitivity_threshold ?? 0.23) * 100)}%` : '50%'})</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

useGLTF.preload(HEART_GLB_PATH);
