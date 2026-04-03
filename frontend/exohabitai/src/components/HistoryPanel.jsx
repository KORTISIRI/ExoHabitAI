import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export default function HistoryPanel() {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const h = JSON.parse(localStorage.getItem('exohistory') || '[]');
    setHistory(h);
  }, []);

  const clearHistory = () => {
    localStorage.removeItem('exohistory');
    setHistory([]);
  };

  return (
    <motion.div
      className="history-panel"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.4 }}
    >
      <div className="sec-header">
        <span className="sec-tag">MISSION LOG · PREDICTION ARCHIVE</span>
        <h2 className="sec-title">Prediction History</h2>
        <p className="sec-sub">Stored locally via browser localStorage</p>
      </div>

      {history.length === 0 ? (
        <div className="glass-panel" style={{ padding: 60, textAlign: 'center', borderRadius: 12 }}>
          <div style={{ fontSize: '3rem', marginBottom: 16, opacity: 0.3 }}>📜</div>
          <p className="result-empty-text">No prediction history yet.<br/>Run an analysis to get started.</p>
        </div>
      ) : (
        <>
          <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 16 }}>
            <button className="history-clear-btn" onClick={clearHistory}>
              Clear All History
            </button>
          </div>

          <div className="history-entries">
            <AnimatePresence>
              {history.map((r, i) => (
                <motion.div
                  key={r.id}
                  className="history-entry"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  transition={{ delay: i * 0.05 }}
                >
                  <span className="history-num">{i + 1}</span>

                  <div className="history-info">
                    <strong>{r.id}</strong>
                    <small>
                      Orb: {r.pl_orbper}d · Axis: {r.pl_orbsmax}AU · Teff: {r.st_teff}K · {r.st_spectype}
                      {r.timestamp && ` · ${new Date(r.timestamp).toLocaleDateString()}`}
                    </small>
                  </div>

                  <span
                    className="history-prob"
                    style={{ color: r.habitable ? 'var(--green)' : 'var(--red)' }}
                  >
                    {(r.probability * 100).toFixed(1)}%
                  </span>

                  <span className={`history-verdict ${r.habitable ? 'hab' : 'inhab'}`}>
                    {r.habitable ? '✓ HABITABLE' : '✗ NON-HAB'}
                  </span>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </>
      )}
    </motion.div>
  );
}
