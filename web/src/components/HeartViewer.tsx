import React, { Suspense, useRef, useState, useEffect } from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { OrbitControls, useGLTF, Html, Center } from '@react-three/drei';
import * as THREE from 'three';
import { Eye, EyeOff, RotateCcw, Info, Activity, Tag, Layers } from 'lucide-react';
import vesselsConfig from '../vessels.json';
import { VesselsConfig, PredictResponse } from '../types';
import { LumenCrossSection } from './LumenCrossSection';

interface HeartViewerProps {
  prediction: PredictResponse | null;
  selectedVessel: string | null;
  onSelectVessel: (vesselId: string | null) => void;
}

// Continuous risk color interpolation: green (#10b981) -> lime (#84cc16) -> amber (#f59e0b) -> orange (#f97316) -> red (#ef4444)
export function getRiskColor(prob: number): string {
  const p = Math.max(0, Math.min(1, prob));
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

function interpolateColor(color1: string, color2: string, factor: number): string {
  const c1 = new THREE.Color(color1);
  const c2 = new THREE.Color(color2);
  c1.lerp(c2, factor);
  return '#' + c1.getHexString();
}

// Smooth animated camera framing for focused vessel inspection
function CameraController({
  selectedVessel,
  controlsRef,
}: {
  selectedVessel: string | null;
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
    } else {
      targetPos.current.set(0, 0.4, 3.2);
      targetLookAt.current.set(0, 0, 0);
    }
    isAnimating.current = true;
  }, [selectedVessel]);

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

// Individual vessel mesh with optional pulsating glow for stenotic alerts
function VesselMeshItem({
  vessel,
  node,
  prob,
  isSelected,
  showLabels,
  isPulsing,
  onSelectVessel,
}: {
  vessel: any;
  node: any;
  prob: number;
  isSelected: boolean;
  showLabels: boolean;
  isPulsing: boolean;
  onSelectVessel: (id: string | null) => void;
}) {
  const materialRef = useRef<THREE.MeshStandardMaterial>(null);
  const colorHex = getRiskColor(prob);
  const isHighRisk = prob >= 0.5;

  useFrame((state) => {
    if (!materialRef.current) return;
    if (isPulsing && (isSelected || isHighRisk)) {
      const pulse = (Math.sin(state.clock.elapsedTime * 4.5) + 1) * 0.5; // 0 to 1
      if (isSelected) {
        materialRef.current.emissiveIntensity = 0.5 + 0.5 * pulse;
      } else {
        materialRef.current.emissiveIntensity = 0.15 + 0.35 * pulse;
      }
    } else {
      materialRef.current.emissiveIntensity = isSelected ? 0.75 : 0.15;
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
          document.body.style.cursor = 'pointer';
        }}
        onPointerOut={() => {
          document.body.style.cursor = 'auto';
        }}
      >
        <meshStandardMaterial
          ref={materialRef}
          color={colorHex}
          roughness={0.2}
          metalness={0.2}
          emissive={new THREE.Color(colorHex)}
          emissiveIntensity={isSelected ? 0.75 : 0.15}
        />
      </mesh>

      {/* Invisible Fatter Hit-Mesh for effortless clicking */}
      <mesh
        geometry={node.geometry}
        scale={[1.5, 1.5, 1.5]}
        visible={false}
        onClick={(e) => {
          e.stopPropagation();
          onSelectVessel(isSelected ? null : vessel.id);
        }}
      >
        <meshBasicMaterial transparent opacity={0} />
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
            <span style={{ color: colorHex, fontWeight: 800 }}>{(prob * 100).toFixed(0)}%</span>
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
  isPulsing,
}: {
  prediction: PredictResponse | null;
  selectedVessel: string | null;
  onSelectVessel: (id: string | null) => void;
  showChambers: boolean;
  chamberOpacity: number;
  showLabels: boolean;
  isPulsing: boolean;
}) {
  const { nodes } = useGLTF('/models/heart.glb') as any;
  const config = vesselsConfig as VesselsConfig;

  return (
    <group dispose={null}>
      {/* 1. Coronary Arteries */}
      {config.vessels.map((vessel) => {
        const node = nodes[vessel.nodeName];
        if (!node || !node.geometry) return null;

        const prob = prediction?.vessels[vessel.id as 'LAD' | 'LCX' | 'RCA']?.prob ?? 0.2;
        const isSelected = selectedVessel === vessel.id;

        return (
          <VesselMeshItem
            key={vessel.id}
            vessel={vessel}
            node={node}
            prob={prob}
            isSelected={isSelected}
            showLabels={showLabels}
            isPulsing={isPulsing}
            onSelectVessel={onSelectVessel}
          />
        );
      })}

      {/* 2. Ascending Aorta */}
      {nodes.Aorta?.geometry && (
        <mesh geometry={nodes.Aorta.geometry}>
          <meshStandardMaterial
            color="#be123c"
            roughness={0.3}
            metalness={0.1}
            opacity={0.88}
            transparent
          />
        </mesh>
      )}

      {/* 3. Myocardium & Heart Chambers Shell */}
      {showChambers && nodes.Heart_Muscle?.geometry && (
        <mesh geometry={nodes.Heart_Muscle.geometry}>
          <meshStandardMaterial
            color="#881337"
            roughness={0.5}
            metalness={0.05}
            opacity={chamberOpacity}
            transparent
            depthWrite={false}
          />
        </mesh>
      )}
    </group>
  );
}

export function HeartViewer({ prediction, selectedVessel, onSelectVessel }: HeartViewerProps) {
  const controlsRef = useRef<any>(null);
  const [showChambers, setShowChambers] = useState(true);
  const [chamberOpacity, setChamberOpacity] = useState(0.25);
  const [isPulsing, setIsPulsing] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [showLumenModal, setShowLumenModal] = useState(false);

  const resetCamera = () => {
    onSelectVessel(null);
    if (controlsRef.current) {
      controlsRef.current.reset();
    }
  };

  const activeVesselForLumen = selectedVessel || 'LAD';
  const activeVesselProb =
    prediction?.vessels[activeVesselForLumen as 'LAD' | 'LCX' | 'RCA']?.prob ?? 0.5;

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

        {/* Toggle Pulsing Glow */}
        <button
          onClick={() => setIsPulsing(!isPulsing)}
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
          title="Inspect 2D Lumen Cross-Section & Area Stenosis"
        >
          <Layers className="w-3.5 h-3.5" />
          <span>Lumen View</span>
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

      {/* Selected Vessel HUD Pill */}
      {selectedVessel && (
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
      )}

      {/* 3D Canvas with continuous or demand frameloop */}
      <Canvas
        frameloop={isPulsing ? 'always' : 'demand'}
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
              isPulsing={isPulsing}
            />
          </Center>
        </Suspense>

        <CameraController selectedVessel={selectedVessel} controlsRef={controlsRef} />

        <OrbitControls
          ref={controlsRef}
          enableDamping
          dampingFactor={0.05}
          minDistance={1.2}
          maxDistance={7.0}
          autoRotate={false}
        />
      </Canvas>

      {/* Floating Lumen Cross-Section Overlay */}
      {showLumenModal && (
        <div className="absolute inset-0 z-30 bg-slate-950/70 backdrop-blur-sm p-4 flex items-center justify-center animate-fade-in">
          <LumenCrossSection
            vesselId={activeVesselForLumen}
            prob={activeVesselProb}
            onClose={() => setShowLumenModal(false)}
            onSelectVessel={(vId) => onSelectVessel(vId)}
          />
        </div>
      )}

      {/* Risk Color Spectrum Bar & Legend */}
      <div className="absolute bottom-3 left-4 right-4 z-10 flex flex-col md:flex-row items-center justify-between gap-2 p-2.5 rounded-xl bg-slate-900/85 border border-slate-800/80 backdrop-blur-md text-[11px]">
        <div className="flex items-center gap-2 text-slate-300">
          <Info className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0" />
          <span>Click vessels in 3D to fly camera and inspect localized SHAP features</span>
        </div>

        <div className="flex items-center gap-3">
          <span className="text-slate-400 text-[10px] uppercase font-bold tracking-wider">Risk Scale:</span>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#10b981]" />
            <span className="text-slate-300 text-[10px]">Low (&lt;25%)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#f59e0b]" />
            <span className="text-slate-300 text-[10px]">Moderate</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#ef4444]" />
            <span className="text-slate-300 text-[10px]">Severe (&ge;75%)</span>
          </div>
        </div>
      </div>
    </div>
  );
}

useGLTF.preload('/models/heart.glb');
