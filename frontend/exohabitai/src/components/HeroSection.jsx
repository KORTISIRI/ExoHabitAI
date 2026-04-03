import { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';

function AnimatedCounter({ target, duration = 2000 }) {
  const ref = useRef(null);

  useEffect(() => {
    let start = 0;
    const step = target / (duration / 16);
    const timer = setInterval(() => {
      start += step;
      if (start >= target) {
        start = target;
        clearInterval(timer);
      }
      if (ref.current) ref.current.textContent = Math.floor(start).toLocaleString();
    }, 16);
    return () => clearInterval(timer);
  }, [target, duration]);

  return <span ref={ref}>0</span>;
}

export default function HeroSection({ onNav }) {
  return (
    <section className="hero-section section-page" id="home">
      {/* Floating planets */}
      <motion.div
        className="floating-planet fp-1"
        animate={{ y: [0, -20, 0], rotate: [0, 8, 0] }}
        transition={{ duration: 12, repeat: Infinity, ease: 'easeInOut' }}
      />
      <motion.div
        className="floating-planet fp-2"
        animate={{ y: [0, 15, 0], rotate: [0, -6, 0] }}
        transition={{ duration: 9, repeat: Infinity, ease: 'easeInOut' }}
      />
      <motion.div
        className="floating-planet fp-3"
        animate={{ y: [0, -10, 0], rotate: [0, 5, 0] }}
        transition={{ duration: 7, repeat: Infinity, ease: 'easeInOut' }}
      />

      <div className="hero-content">
        <motion.div
          className="hero-badge"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <span className="badge-pulse" />
          NASA EXOPLANET ARCHIVE · LIVE AI ANALYSIS
        </motion.div>

        <motion.h1
          className="hero-title"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.4 }}
        >
          Exo<span className="cyan">Habit</span>
          <span className="purple">AI</span>
          <br />
          <span style={{ fontSize: '0.45em', fontWeight: 600, letterSpacing: '0.1em', color: 'rgba(224,231,255,0.6)' }}>
            Exoplanet Habitability Predictor
          </span>
        </motion.h1>

        <motion.p
          className="hero-sub"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.6 }}
        >
          Advanced machine learning analysis of planetary and stellar parameters
          to assess the potential for extraterrestrial life.
        </motion.p>

        <motion.div
          className="hero-stats"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.8 }}
        >
          <div className="stat-item">
            <span className="stat-val">
              <AnimatedCounter target={39315} />
            </span>
            <span className="stat-label">Planets Analyzed</span>
          </div>
          <div className="stat-item">
            <span className="stat-val">90.6%</span>
            <span className="stat-label">Model AUC</span>
          </div>
          <div className="stat-item">
            <span className="stat-val">RF</span>
            <span className="stat-label">Algorithm</span>
          </div>
          <div className="stat-item">
            <span className="stat-val">16</span>
            <span className="stat-label">Features</span>
          </div>
        </motion.div>

        <motion.div
          className="hero-cta"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 1.0 }}
        >
          <button className="btn-primary" onClick={() => onNav('predict')}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Begin Analysis
          </button>
          <button className="btn-ghost" onClick={() => onNav('about')}>
            Learn More
          </button>
        </motion.div>
      </div>

      <div className="scroll-indicator">
        <span>SCROLL</span>
        <div className="scroll-line" />
      </div>
    </section>
  );
}
