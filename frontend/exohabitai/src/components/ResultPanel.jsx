import { motion, AnimatePresence } from 'framer-motion';
import HabitabilityMeter from './HabitabilityMeter';
import PlanetVisualization from './PlanetVisualization';

export default function ResultPanel({ result, onReset, onNav }) {
  if (!result) {
    return (
      <motion.div
        className="glass-panel result-panel"
        initial={{ opacity: 0, x: 30 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5, delay: 0.3 }}
      >
        <div className="panel-corner tl" /><div className="panel-corner tr" />
        <div className="panel-corner bl" /><div className="panel-corner br" />

        <div className="result-empty">
          <div className="result-empty-icon">🔭</div>
          <p className="result-empty-text">
            NO ANALYSIS YET<br />
            <span style={{ fontSize: '0.72rem', marginTop: 8, display: 'block' }}>
              Configure parameters and run prediction
            </span>
          </p>
        </div>
      </motion.div>
    );
  }

  const { data, inputs, raw } = result;
  const isHab  = data.prediction === 1;
  const prob   = data.habitability_probability;
  const pct    = Math.round(prob * 100);

  const stTeff  = parseFloat(raw.st_teff)    || 0;
  const orbsmax = parseFloat(raw.pl_orbsmax) || 0;
  const radius  = parseFloat(raw.pl_rade)    || 0;

  const teffOk = stTeff >= 4000 && stTeff <= 7000;
  const orbOk  = orbsmax > 0.3;
  const radOk  = radius > 0 && radius < 2.0;

  const factors = [
    { icon: '🌡️', label: 'Star Temp',  val: `${stTeff}K`,   cls: teffOk ? 'good' : 'bad' },
    { icon: '🪐',  label: 'Orbit',     val: `${orbsmax}AU`, cls: orbOk  ? 'good' : 'bad' },
    { icon: '🌍',  label: 'Radius',    val: radius > 0 ? `${radius}R⊕` : '—', cls: radOk ? 'good' : radius > 0 ? 'bad' : 'neutral' },
    { icon: '⭐',  label: 'Spec Type', val: raw.st_spectype || '—', cls: ['G','K','F'].includes(raw.st_spectype) ? 'good' : 'neutral' },
  ];

  return (
    <motion.div
      className="glass-panel result-panel"
      initial={{ opacity: 0, x: 30 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="panel-corner tl" /><div className="panel-corner tr" />
      <div className="panel-corner bl" /><div className="panel-corner br" />

      {/* Verdict */}
      <AnimatePresence mode="wait">
        <motion.div
          key={data.prediction}
          className="verdict-wrap"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ type: 'spring', stiffness: 200 }}
        >
          <span className="verdict-planet-emoji">
            {isHab ? '🌍' : '🪐'}
          </span>
          <span className={`verdict-badge ${isHab ? 'habitable' : 'non-habitable'}`}>
            <span style={{ fontSize: '1rem' }}>{isHab ? '✓' : '✗'}</span>
            {data.label}
          </span>
          <p className="verdict-confidence">Confidence: {data.confidence}</p>
          <p className="verdict-desc">
            {isHab
              ? 'Promising habitability indicators detected. Stellar and orbital parameters within stable ranges.'
              : 'Habitability criteria not met. Extreme conditions or orbital instability detected.'}
          </p>
        </motion.div>
      </AnimatePresence>

      {/* 3D Planet */}
      <PlanetVisualization habitable={isHab} probability={prob} />

      {/* Meter */}
      <HabitabilityMeter probability={prob} habitable={isHab} confidence={data.confidence} />

      {/* Analysis factors */}
      <div style={{ marginTop: 16 }}>
        <span className="res-card-tag">FACTOR ANALYSIS</span>
        <div className="factors-grid">
          {factors.map(f => (
            <div key={f.label} className={`factor-chip ${f.cls}`}>
              <span className="factor-chip-icon">{f.icon}</span>
              <span className="factor-chip-text">{f.label}</span>
              <span className="factor-chip-val">{f.val}</span>
            </div>
          ))}
        </div>
      </div>

      {/* AI explanation */}
      <div className={`analysis-text ${isHab ? 'positive' : 'negative'}`} style={{ marginTop: 16 }}>
        {isHab
          ? `Positive Assessment: The model classified this exoplanet as potentially habitable with ${data.confidence} confidence. Stellar temperature, orbital parameters, and spectral type align with known habitable zones. Further spectroscopic analysis recommended.`
          : `Negative Assessment: The model classified this system as non-habitable with ${data.confidence} confidence. Key parameters fall outside habitable zone criteria — extreme stellar radiation or orbital instability likely prevents liquid water from persisting.`
        }
      </div>

      {/* Summary */}
      <div style={{ marginTop: 16 }}>
        <span className="res-card-tag">INPUT SUMMARY</span>
        <div className="summary-list">
          {Object.entries(inputs).map(([k, v]) => (
            <div key={k} className="sum-row">
              <span className="sum-key">{k}</span>
              <span className="sum-val">{v}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Actions */}
      <div className="result-actions">
        <button className="btn-ghost" onClick={onReset}>
          <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
            <path d="M2 6.5a4.5 4.5 0 1 0 1-2.8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
            <path d="M2 3v3.5H5.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          New Analysis
        </button>
        <button className="btn-primary" onClick={() => onNav('history')}>
          <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
            <path d="M6.5 1v7M4 6l2.5 2.5L9 6M2 11h9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          View History
        </button>
      </div>
    </motion.div>
  );
}
