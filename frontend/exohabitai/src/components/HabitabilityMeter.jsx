import { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';

export default function HabitabilityMeter({ probability = 0, habitable = false, confidence = '—' }) {
  const circumference = 2 * Math.PI * 72; // ≈ 452.4
  const offset = circumference - probability * circumference;

  const color = habitable
    ? '#10B981'
    : probability > 0.4
    ? '#F59E0B'
    : '#EF4444';

  const pct = Math.round(probability * 100);

  return (
    <div className="meter-wrap">
      <span className="res-card-tag">HABITABILITY METER</span>

      <motion.div
        initial={{ opacity: 0, scale: 0.7 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, type: 'spring', stiffness: 200 }}
      >
        <svg className="gauge-svg" viewBox="0 0 200 200">
          {/* Decorative outer ring */}
          <circle cx="100" cy="100" r="92"
            fill="none"
            stroke="rgba(0,229,255,0.05)"
            strokeWidth="1"
            strokeDasharray="4 3"
          />

          {/* Tick marks */}
          {Array.from({ length: 20 }).map((_, i) => {
            const angle = (i / 20) * 360 - 90;
            const rad = angle * Math.PI / 180;
            const r1 = 82, r2 = 86;
            return (
              <line
                key={i}
                x1={100 + r1 * Math.cos(rad)}
                y1={100 + r1 * Math.sin(rad)}
                x2={100 + r2 * Math.cos(rad)}
                y2={100 + r2 * Math.sin(rad)}
                stroke="rgba(0,229,255,0.2)"
                strokeWidth={i % 5 === 0 ? 2 : 1}
              />
            );
          })}

          {/* Track */}
          <circle cx="100" cy="100" r="72"
            fill="none"
            stroke="#1a1f4e"
            strokeWidth="12"
          />

          {/* Fill arc */}
          <motion.circle
            cx="100" cy="100" r="72"
            fill="none"
            stroke={color}
            strokeWidth="12"
            strokeDasharray={circumference}
            strokeLinecap="round"
            transform="rotate(-90 100 100)"
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: offset }}
            transition={{ duration: 1.8, ease: [0.4, 0, 0.2, 1] }}
            style={{ filter: `drop-shadow(0 0 6px ${color})` }}
          />

          {/* Center score */}
          <text x="100" y="94"
            textAnchor="middle"
            fill={color}
            fontSize="30"
            fontFamily="Orbitron"
            fontWeight="700"
          >
            {pct}%
          </text>
          <text x="100" y="112"
            textAnchor="middle"
            fill="#94A3B8"
            fontSize="8"
            fontFamily="Rajdhani"
            letterSpacing="2"
          >
            PROBABILITY
          </text>
          <text x="100" y="126"
            textAnchor="middle"
            fill="#94A3B8"
            fontSize="8"
            fontFamily="Rajdhani"
            letterSpacing="1"
          >
            {confidence}
          </text>
        </svg>
      </motion.div>

      {/* Linear bar */}
      <div style={{ width: '100%', marginTop: 8 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: 'var(--text-muted)', marginBottom: 6 }}>
          <span>LOW</span><span>MODERATE</span><span>HIGH</span>
        </div>
        <div style={{ height: 6, background: 'rgba(255,255,255,0.06)', borderRadius: 4, overflow: 'visible', position: 'relative' }}>
          <motion.div
            style={{
              height: '100%',
              background: `linear-gradient(90deg, #EF4444, #F59E0B, #10B981)`,
              borderRadius: 4,
              clipPath: `inset(0 ${100 - pct}% 0 0 round 4px)`,
            }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5, duration: 0.8 }}
          />
          <motion.div
            style={{
              position: 'absolute',
              top: '50%',
              left: `${pct}%`,
              transform: 'translate(-50%, -50%)',
              width: 12, height: 12,
              background: color,
              borderRadius: '50%',
              border: '2px solid var(--space-dark)',
              boxShadow: `0 0 8px ${color}`,
            }}
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.8, type: 'spring', stiffness: 300 }}
          />
        </div>
      </div>
    </div>
  );
}
