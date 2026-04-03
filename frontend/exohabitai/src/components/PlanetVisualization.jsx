import { useRef, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Sphere, OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

function Planet({ habitable, probability }) {
  const meshRef = useRef();
  const ringRef = useRef();
  const moonRef = useRef();
  const moonPivotRef = useRef();

  // Pick planet color based on habitability score
  const color = habitable
    ? new THREE.Color(0.1, 0.7, 0.4)   // green-blue
    : probability > 0.4
    ? new THREE.Color(0.9, 0.5, 0.1)   // amber
    : new THREE.Color(0.6, 0.1, 0.1);  // red

  const emissive = habitable
    ? new THREE.Color(0.05, 0.3, 0.15)
    : probability > 0.4
    ? new THREE.Color(0.3, 0.15, 0.02)
    : new THREE.Color(0.25, 0.03, 0.03);

  useFrame((state, delta) => {
    if (meshRef.current)      meshRef.current.rotation.y += delta * 0.3;
    if (ringRef.current)      ringRef.current.rotation.z += delta * 0.1;
    if (moonPivotRef.current) moonPivotRef.current.rotation.y += delta * 0.8;
  });

  return (
    <group>
      {/* Atmosphere glow */}
      <Sphere args={[1.15, 32, 32]}>
        <meshStandardMaterial
          color={color}
          transparent
          opacity={0.08}
          side={THREE.BackSide}
        />
      </Sphere>

      {/* Planet body */}
      <Sphere ref={meshRef} args={[1, 64, 64]}>
        <meshStandardMaterial
          color={color}
          emissive={emissive}
          emissiveIntensity={0.4}
          roughness={0.8}
          metalness={0.1}
        />
      </Sphere>

      {/* Ring */}
      <mesh ref={ringRef} rotation={[Math.PI / 2.5, 0, 0]}>
        <torusGeometry args={[1.6, 0.07, 16, 100]} />
        <meshStandardMaterial
          color={habitable ? '#00E5FF' : '#4C1D95'}
          emissive={habitable ? '#00E5FF' : '#4C1D95'}
          emissiveIntensity={0.5}
          transparent
          opacity={0.6}
        />
      </mesh>

      {/* Moon pivot */}
      <group ref={moonPivotRef}>
        <mesh ref={moonRef} position={[1.8, 0, 0]}>
          <sphereGeometry args={[0.18, 16, 16]} />
          <meshStandardMaterial color="#E0E7FF" roughness={0.9} />
        </mesh>
      </group>
    </group>
  );
}

export default function PlanetVisualization({ habitable = false, probability = 0 }) {
  return (
    <div className="planet-viz-wrap">
      <Canvas
        camera={{ position: [0, 0, 4], fov: 45 }}
        gl={{ antialias: true, alpha: true }}
      >
        {/* Lighting */}
        <ambientLight intensity={0.3} />
        <pointLight position={[5, 5, 5]} intensity={1.5} color="#ffffff" />
        <pointLight position={[-5, -3, -3]} intensity={0.4} color={habitable ? '#00E5FF' : '#7C3AED'} />

        <Planet habitable={habitable} probability={probability} />

        <OrbitControls
          enableZoom={false}
          enablePan={false}
          autoRotate
          autoRotateSpeed={0.5}
        />
      </Canvas>
    </div>
  );
}
