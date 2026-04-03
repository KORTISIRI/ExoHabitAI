import { useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { trainAndPredictFile, trainAndPredictCsv, trainAndPredictJson, ApiError } from '../api';
import './RawDataPanel.css';

const SAMPLE_CSV = `planet_radius,planet_mass,orbital_period,semi_major_axis,surface_temp,star_temperature,star_metallicity,luminosity,star_radius,spectral_type,habitability_label
1.1,1.0,365,1.0,288,5778,0.0,1.0,1.0,G,1
0.8,0.7,320,0.9,300,5500,-0.1,0.9,0.95,G,1
12.0,300,10,0.05,900,25000,0.2,500,5.0,B,0
0.5,0.3,180,0.7,320,4800,-0.2,0.6,0.8,K,1
8.0,50,5,0.03,1200,10000,0.3,50,2.0,A,0
1.5,2.5,420,1.2,275,5900,0.1,1.1,1.05,G,1
0.9,0.8,380,1.0,295,5600,0.05,0.95,1.0,G,1
3.0,15,60,0.2,650,6200,0.15,2.0,1.3,F,0
1.2,1.3,400,1.1,285,5700,0.0,1.0,1.0,G,1
0.7,0.5,280,0.85,310,5100,-0.15,0.75,0.9,K,1`;

const SAMPLE_JSON = JSON.stringify([
  {planet_radius:1.1,planet_mass:1.0,orbital_period:365,semi_major_axis:1.0,surface_temp:288,star_temperature:5778,star_metallicity:0.0,luminosity:1.0,star_radius:1.0,spectral_type:"G",habitability_label:1},
  {planet_radius:12.0,planet_mass:300,orbital_period:10,semi_major_axis:0.05,surface_temp:900,star_temperature:25000,star_metallicity:0.2,luminosity:500,star_radius:5.0,spectral_type:"B",habitability_label:0},
  {planet_radius:1.2,planet_mass:1.3,orbital_period:400,semi_major_axis:1.1,surface_temp:285,star_temperature:5700,star_metallicity:0.0,luminosity:1.0,star_radius:1.0,spectral_type:"G",habitability_label:1},
], null, 2);

const REQUIRED_COLS = [
  'planet_radius','planet_mass','orbital_period','semi_major_axis',
  'surface_temp','star_temperature','star_metallicity','luminosity',
  'star_radius','spectral_type','habitability_label'
];

// ── Main Panel ────────────────────────────────────────────────────────────────
export default function RawDataPanel() {
  const [tab, setTab]             = useState('paste');   // 'paste' | 'upload'
  const [format, setFormat]       = useState('csv');     // 'csv' | 'json'
  const [rawText, setRawText]     = useState('');
  const [file, setFile]           = useState(null);
  const [dragging, setDragging]   = useState(false);
  const [phase, setPhase]         = useState('idle');    // 'idle' | 'loading' | 'done' | 'error'
  const [trainResult, setTrain]   = useState(null);
  const [predictions, setPreds]   = useState([]);
  const [errorMsg, setError]      = useState('');
  const [filterMode, setFilterMode] = useState('all');
  const fileRef                   = useRef();

  const reset = () => {
    setPhase('idle'); setTrain(null); setPreds([]); setError(''); setFilterMode('all');
  };

  // ── Drag-and-drop ──────────────────────────────────────────────────────────
  const onDrop = useCallback((e) => {
    e.preventDefault(); setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) { setFile(f); setTab('upload'); }
  }, []);

  // ── Submit ─────────────────────────────────────────────────────────────────
  const handleSubmit = async () => {
    setPhase('loading'); setError(''); setTrain(null); setPreds([]);

    try {
      let data;
      if (tab === 'upload' && file) {
        data = await trainAndPredictFile(file);
      } else {
        const text = rawText.trim();
        if (!text) throw new Error('Please paste some data first.');
        if (format === 'csv') {
          data = await trainAndPredictCsv(text);
        } else {
          let parsed;
          try { parsed = JSON.parse(text); } catch { throw new Error('Invalid JSON — check your format.'); }
          data = await trainAndPredictJson(Array.isArray(parsed) ? parsed : [parsed]);
        }
      }

      setTrain(data.training);
      setPreds(data.predictions || []);
      setPhase('done');
    } catch (err) {
      if (err instanceof ApiError && err.code === 'NETWORK_ERROR') {
        setError(err.message);
      } else if (err instanceof ApiError && err.statusCode === 503) {
        setError('⚠ ML model not loaded on server. Run python ML_Model_Training.py first.');
      } else if (err instanceof ApiError) {
        setError(`Server error: ${err.message}`);
      } else {
        setError(err.message || 'Unknown error occurred.');
      }
      setPhase('error');
    }
  };

  // ── CSV export ─────────────────────────────────────────────────────────────
  const exportCSV = () => {
    const rows = [['Row','Score %','Status','Actual Label']];
    predictions.forEach(r => rows.push([
      r.row_index + 1,
      (r.habitability_score * 100).toFixed(2),
      r.status,
      r.actual_label ?? 'N/A',
    ]));
    const csv = rows.map(r => r.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url  = URL.createObjectURL(blob);
    const a    = Object.assign(document.createElement('a'), { href: url, download: 'predictions.csv' });
    a.click(); URL.revokeObjectURL(url);
  };

  // ── Derived: filtered rows for batch-style display ─────────────────────────
  const filteredRows = predictions.filter(r => {
    if (filterMode === 'habitable')     return r.prediction === 1;
    if (filterMode === 'non-habitable') return r.prediction === 0;
    return true;
  });

  return (
    <div className="raw-panel">
      {/* ── Header ── */}
      <div className="raw-header">
        <div>
          <h2 className="raw-title">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" className="raw-title-icon">
              <path d="M4 6h16M4 10h16M4 14h10M4 18h6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>
            </svg>
            Raw Data Training &amp; Prediction
          </h2>
          <p className="raw-subtitle">
            Paste CSV/JSON or upload a file — the model retrains on your data and predicts habitability for every row.
          </p>
        </div>
      </div>

      {phase !== 'done' ? (
        <div className="raw-input-area">
          {/* ── Input mode tabs ── */}
          <div className="raw-tabs">
            {['paste','upload'].map(t => (
              <button
                key={t}
                className={`raw-tab ${tab === t ? 'raw-tab--active' : ''}`}
                onClick={() => { setTab(t); reset(); }}
              >
                {t === 'paste' ? '✏️ Paste Data' : '📂 Upload File'}
              </button>
            ))}
          </div>

          {tab === 'paste' && (
            <>
              {/* Format toggle */}
              <div className="raw-format-row">
                <span className="raw-format-label">Format:</span>
                {['csv','json'].map(f => (
                  <button
                    key={f}
                    className={`raw-fmt-btn ${format === f ? 'raw-fmt-btn--active' : ''}`}
                    onClick={() => setFormat(f)}
                  >
                    {f.toUpperCase()}
                  </button>
                ))}
                <button
                  className="raw-sample-btn"
                  onClick={() => setRawText(format === 'csv' ? SAMPLE_CSV : SAMPLE_JSON)}
                >
                  Load Sample
                </button>
              </div>

              <textarea
                className="raw-textarea"
                placeholder={
                  format === 'csv'
                    ? `Paste CSV here…\n\nRequired columns:\n${REQUIRED_COLS.join(', ')}`
                    : `Paste JSON array here…\n[\n  { "planet_radius": 1.1, "planet_mass": 1.0, ... }\n]`
                }
                value={rawText}
                onChange={e => setRawText(e.target.value)}
                spellCheck={false}
              />
            </>
          )}

          {tab === 'upload' && (
            <div
              className={`raw-dropzone ${dragging ? 'raw-dropzone--over' : ''} ${file ? 'raw-dropzone--has-file' : ''}`}
              onDragOver={e => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => fileRef.current?.click()}
            >
              <input
                ref={fileRef}
                type="file"
                accept=".csv,.json"
                style={{ display: 'none' }}
                onChange={e => { setFile(e.target.files[0]); reset(); }}
              />
              {file ? (
                <>
                  <div className="raw-drop-icon">📄</div>
                  <p className="raw-drop-name">{file.name}</p>
                  <p className="raw-drop-size">{(file.size / 1024).toFixed(1)} KB</p>
                  <button className="raw-remove-file" onClick={e => { e.stopPropagation(); setFile(null); reset(); }}>
                    ✕ Remove
                  </button>
                </>
              ) : (
                <>
                  <div className="raw-drop-icon">☁️</div>
                  <p className="raw-drop-hint">Drag &amp; drop a <strong>.csv</strong> or <strong>.json</strong> file here</p>
                  <p className="raw-drop-sub">or click to browse</p>
                </>
              )}
            </div>
          )}

          {/* Required columns hint */}
          <details className="raw-cols-hint">
            <summary>📋 Required column names</summary>
            <div className="raw-cols-grid">
              {REQUIRED_COLS.map(c => <code key={c}>{c}</code>)}
            </div>
            <p className="raw-cols-note">NASA aliases also accepted: <code>pl_rade</code>, <code>st_teff</code>, <code>pl_orbper</code>, etc.</p>
          </details>

          {/* Error */}
          <AnimatePresence>
            {phase === 'error' && (
              <motion.div className="raw-error" initial={{opacity:0,y:-8}} animate={{opacity:1,y:0}} exit={{opacity:0}}>
                ⚠️ {errorMsg}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Submit */}
          <motion.button
            className="raw-submit-btn"
            onClick={handleSubmit}
            disabled={phase === 'loading'}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            {phase === 'loading' ? (
              <><span className="raw-spinner" /> Training &amp; Predicting…</>
            ) : (
              <>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                  <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
                </svg>
                Train &amp; Predict
              </>
            )}
          </motion.button>
        </div>
      ) : (
        /* ── Results (Batch-style display) ── */
        <motion.div
          className="batch-results"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          {/* Training accuracy banner — extra info not in batch */}
          {trainResult && (
            <div style={{
              background: 'rgba(0,229,255,0.05)',
              border: '1px solid rgba(0,229,255,0.15)',
              borderRadius: 10,
              padding: '12px 18px',
              display: 'flex',
              alignItems: 'center',
              gap: 16,
              flexWrap: 'wrap',
              marginBottom: 4,
            }}>
              <span style={{ fontFamily: 'Share Tech Mono, monospace', fontSize: '0.72rem', color: 'rgba(0,229,255,0.6)', letterSpacing: 2 }}>
                🧠 TRAINING COMPLETE
              </span>
              <span style={{ fontFamily: 'Orbitron, sans-serif', fontSize: '1rem', fontWeight: 700, color: '#00E5FF' }}>
                {trainResult.accuracy_pct || `${(trainResult.accuracy * 100).toFixed(2)}%`}
              </span>
              <span style={{ fontFamily: 'Share Tech Mono, monospace', fontSize: '0.72rem', color: 'rgba(224,231,255,0.4)' }}>
                accuracy · {trainResult.samples_used} samples · train/test {trainResult.train_size}/{trainResult.test_size}
              </span>
              {trainResult.note && (
                <span style={{ fontFamily: 'Share Tech Mono, monospace', fontSize: '0.65rem', color: 'rgba(245,158,11,0.7)' }}>
                  ⚠ {trainResult.note}
                </span>
              )}
            </div>
          )}

          {/* Stat cards — same as Batch */}
          <div className="batch-stats-row">
            <div className="batch-stat-card total">
              <span className="bsc-val">{predictions.length}</span>
              <span className="bsc-label">TOTAL PLANETS</span>
            </div>
            <div className="batch-stat-card habitable">
              <span className="bsc-val">{predictions.filter(p => p.prediction === 1).length}</span>
              <span className="bsc-label">HABITABLE</span>
            </div>
            <div className="batch-stat-card non-habitable">
              <span className="bsc-val">{predictions.filter(p => p.prediction === 0).length}</span>
              <span className="bsc-label">NON-HABITABLE</span>
            </div>
            <div className="batch-stat-card avg-score">
              <span className="bsc-val">
                {Math.round(
                  predictions.reduce((s, r) => s + (r.habitability_score ?? 0), 0) /
                  (predictions.length || 1) * 100
                )}%
              </span>
              <span className="bsc-label">AVG SCORE</span>
            </div>
          </div>

          {/* Toolbar — filter tabs + action buttons, same as Batch */}
          <div className="batch-toolbar">
            <div className="batch-filter-tabs">
              {['all', 'habitable', 'non-habitable'].map(mode => (
                <button
                  key={mode}
                  className={`batch-filter-tab ${filterMode === mode ? 'active' : ''}`}
                  onClick={() => setFilterMode(mode)}
                >
                  {mode === 'all'
                    ? `All (${predictions.length})`
                    : mode === 'habitable'
                    ? `🟢 Habitable (${predictions.filter(p => p.prediction === 1).length})`
                    : `🔴 Non-Habitable (${predictions.filter(p => p.prediction === 0).length})`}
                </button>
              ))}
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <button className="batch-action-btn" onClick={exportCSV}>
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                  <path d="M7 1v8M3 6l4 4 4-4M1 11h12v2H1z" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                EXPORT CSV
              </button>
              <button className="batch-action-btn" onClick={reset}>
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                  <path d="M2 7a5 5 0 1 1 1.5 3.5M2 11V7h4" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                NEW DATASET
              </button>
            </div>
          </div>

          {/* Results table — exact same layout as Batch */}
          <div className="batch-results-table-wrap">
            <table className="batch-results-table">
              <thead>
                <tr>
                  <th style={{ width: 55 }}>ROW</th>
                  <th>HABITABILITY SCORE</th>
                  <th style={{ width: 160 }}>STATUS</th>
                  <th style={{ width: 100 }}>CONFIDENCE</th>
                  {predictions.some(p => p.actual_label !== undefined) && (
                    <th style={{ width: 90 }}>ACTUAL</th>
                  )}
                </tr>
              </thead>
              <tbody>
                {filteredRows.map((row, i) => {
                  const score    = row.habitability_score ?? 0;
                  const pct      = Math.round(score * 100);
                  const isHab    = row.prediction === 1;
                  const color    = isHab ? '#10B981' : '#EF4444';
                  const barColor = isHab
                    ? 'linear-gradient(90deg, #065f46, #10B981)'
                    : 'linear-gradient(90deg, #7f1d1d, #EF4444)';
                  return (
                    <motion.tr
                      key={row.row_index}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: Math.min(i * 0.03, 0.5), duration: 0.3 }}
                      style={{ borderBottom: '1px solid rgba(0,229,255,0.07)' }}
                    >
                      <td style={{ padding: '10px 12px', color: 'rgba(100,116,139,0.9)', fontFamily: 'Share Tech Mono, monospace', fontSize: '0.78rem' }}>
                        #{row.row_index + 1}
                      </td>
                      <td style={{ padding: '10px 12px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <div style={{ flex: 1, height: 6, borderRadius: 3, background: 'rgba(255,255,255,0.06)', overflow: 'hidden' }}>
                            <motion.div
                              initial={{ width: 0 }}
                              animate={{ width: `${pct}%` }}
                              transition={{ delay: Math.min(i * 0.03, 0.5) + 0.2, duration: 0.6, ease: 'easeOut' }}
                              style={{ height: '100%', borderRadius: 3, background: barColor }}
                            />
                          </div>
                          <span style={{ fontFamily: 'Share Tech Mono, monospace', fontSize: '0.82rem', color, minWidth: 38, textAlign: 'right' }}>
                            {pct}%
                          </span>
                        </div>
                      </td>
                      <td style={{ padding: '10px 12px' }}>
                        <span style={{
                          display: 'inline-flex', alignItems: 'center', gap: 5,
                          padding: '3px 10px', borderRadius: 100,
                          fontSize: '0.72rem', fontFamily: 'Share Tech Mono, monospace', letterSpacing: 1,
                          background: isHab ? 'rgba(16,185,129,0.12)' : 'rgba(239,68,68,0.1)',
                          border: `1px solid ${isHab ? 'rgba(16,185,129,0.35)' : 'rgba(239,68,68,0.3)'}`,
                          color,
                        }}>
                          <span style={{ width: 6, height: 6, borderRadius: '50%', background: color, display: 'inline-block' }} />
                          {isHab ? 'HABITABLE' : 'NON-HABITABLE'}
                        </span>
                      </td>
                      <td style={{ padding: '10px 12px', color: 'rgba(224,231,255,0.6)', fontFamily: 'Share Tech Mono, monospace', fontSize: '0.78rem' }}>
                        {row.confidence}
                      </td>
                      {predictions.some(p => p.actual_label !== undefined) && (
                        <td style={{ padding: '10px 12px', fontFamily: 'Share Tech Mono, monospace', fontSize: '0.78rem' }}>
                          {row.actual_label !== undefined
                            ? <span style={{ color: Number(row.actual_label) === 1 ? '#10B981' : '#EF4444' }}>
                                {Number(row.actual_label) === 1 ? '🟢 1' : '🔴 0'}
                              </span>
                            : <span style={{ color: 'rgba(255,255,255,0.25)' }}>—</span>}
                        </td>
                      )}
                    </motion.tr>
                  );
                })}
              </tbody>
            </table>
            {filteredRows.length === 0 && (
              <div style={{ textAlign: 'center', padding: '32px', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                No planets match this filter.
              </div>
            )}
          </div>
        </motion.div>
      )}
    </div>
  );
}
