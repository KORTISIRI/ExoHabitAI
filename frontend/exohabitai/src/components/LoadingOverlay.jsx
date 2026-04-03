import { motion } from 'framer-motion';

export default function LoadingOverlay({ progress = 0, statusMsg = 'Initializing...' }) {
  return (
    <motion.div
      className="loading-overlay"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
    >
      {/* Animated planet system */}
      <div className="loading-planet-wrap">
        <div className="loading-planet-body" />

        {/* Outer orbit */}
        <div className="loading-orbit">
          <div className="loading-moon" />
        </div>

        {/* Inner orbit */}
        <div className="loading-orbit-2">
          <div className="loading-moon-2" />
        </div>
      </div>

      <motion.div className="loading-title"
        animate={{ opacity: [0.7, 1, 0.7] }}
        transition={{ duration: 1.5, repeat: Infinity }}
      >
        ANALYZING EXOPLANET
      </motion.div>

      <p className="loading-sub">{statusMsg}</p>

      <div className="loading-bar-wrap">
        <motion.div
          className="loading-bar-fill"
          style={{ width: `${progress}%` }}
          transition={{ duration: 0.5, ease: 'easeInOut' }}
        />
      </div>

      <p style={{
        marginTop: 12,
        fontFamily: "'Share Tech Mono', monospace",
        fontSize: '0.65rem',
        color: 'rgba(100,116,139,0.6)',
        letterSpacing: 2,
      }}>
        {progress}% COMPLETE
      </p>
    </motion.div>
  );
}
