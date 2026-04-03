import { useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const API_BASE = 'http://localhost:5000';

const REQUIRED_COLUMNS = [
  'planet_name',
  'planet_radius',
  'planet_mass',
  'orbital_period',
  'semi_major_axis',
  'surface_temp',
  'star_temperature',
  'star_metallicity',
  'luminosity',
  'star_radius',
  'spectral_type',
];

// ── Spectral type encoding map (same as backend LabelEncoder order) ──
const SPECTRAL_MAP = { A: 0, B: 1, F: 2, G: 3, K: 4, M: 5, O: 6 };

// ── Parse raw text as JSON or CSV ──
function detectAndParse(text) {
  const trimmed = text.trim();
  if (!trimmed) throw new Error('Input is empty.');

  // Try JSON
  if (trimmed.startsWith('[') || trimmed.startsWith('{')) {
    try {
      const parsed = JSON.parse(trimmed);
      return Array.isArray(parsed) ? parsed : [parsed];
    } catch {
      throw new Error('Invalid JSON format. Check your brackets and commas.');
    }
  }

  // Try CSV
  const lines = trimmed.split('\n').map(l => l.trim()).filter(Boolean);
  if (lines.length < 2) throw new Error('CSV must have a header row and at least one data row.');

  const headers = lines[0].split(',').map(h => h.trim().toLowerCase()
    .replace(/ /g, '_').replace(/-/g, '_'));
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const vals = lines[i].split(',').map(v => v.trim());
    if (vals.length !== headers.length) throw new Error(`Row ${i + 1} has ${vals.length} columns but header has ${headers.length}.`);
    const obj = {};
    headers.forEach((h, idx) => { obj[h] = vals[idx]; });
    rows.push(obj);
  }
  return rows;
}

// ── Validate required columns ──
function validateColumns(rows) {
  if (!rows || rows.length === 0) throw new Error('No data rows found.');
  const keys = Object.keys(rows[0]).map(k => k.toLowerCase().replace(/ /g, '_').replace(/-/g, '_'));
  const missing = REQUIRED_COLUMNS.filter(c => c !== 'planet_name' && !keys.includes(c));
  if (missing.length) throw new Error(`Missing required columns: ${missing.join(', ')}`);
}

// ── Normalize rows to canonical keys ──
function normalizeRows(rows) {
  return rows.map(row => {
    const normalized = {};
    Object.keys(row).forEach(k => {
      normalized[k.toLowerCase().replace(/ /g, '_').replace(/-/g, '_')] = row[k];
    });
    return normalized;
  });
}

// ── Build payload for /predict-batch (raw data has NO "habitable" col) ──
function buildPayload(rows) {
  return rows.map(row => {
    const spectral = String(row.spectral_type ?? 'G').trim().toUpperCase().charAt(0);
    return {
      planet_radius:    parseFloat(row.planet_radius)    || 1.0,
      planet_mass:      parseFloat(row.planet_mass)      || 1.0,
      orbital_period:   parseFloat(row.orbital_period)   || 365.0,
      semi_major_axis:  parseFloat(row.semi_major_axis)  || 1.0,
      surface_temp:     parseFloat(row.surface_temp)     || 288.0,
      star_temperature: parseFloat(row.star_temperature) || 5778.0,
      star_metallicity: parseFloat(row.star_metallicity) || 0.0,
      luminosity:       parseFloat(row.luminosity)       || 1.0,
      star_radius:      parseFloat(row.star_radius)      || 1.0,
      spectral_type:    Object.keys(SPECTRAL_MAP).includes(spectral) ? spectral : 'G',
    };
  });
}

// Scoring helpers
function scoreColor(score) {
  if (score >= 0.7) return '#00e676';
  if (score >= 0.5) return '#ffeb3b';
  return '#ff5252';
}
function scoreBg(score) {
  if (score >= 0.7) return 'rgba(0,230,118,0.08)';
  if (score >= 0.5) return 'rgba(255,235,59,0.08)';
  return 'rgba(255,82,82,0.08)';
}
function statusLabel(score) {
  if (score >= 0.5) return 'HABITABLE';
  return 'NON-HABITABLE';
}

export default function RawDataAnalysisPanel() {
  const [inputMode, setInputMode]     = useState('paste'); // 'paste' | 'upload'
  const [rawText, setRawText]         = useState('');
  const [dragging, setDragging]       = useState(false);
  const [fileName, setFileName]       = useState('');
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState('');
  const [results, setResults]         = useState(null);   // array of result objects
  const [planetNames, setPlanetNames] = useState([]);
  const fileInputRef = useRef(null);

  // ── Process data ──
  const processData = useCallback(async (text) => {
    setError('');
    setResults(null);
    setLoading(true);

    try {
      // 1. Parse
      let rows = detectAndParse(text);

      // 2. Normalize keys
      rows = normalizeRows(rows);

      // 3. Validate columns
      validateColumns(rows);

      // 4. Store planet names for display
      const names = rows.map((r, i) => r.planet_name || `Planet ${i + 1}`);
      setPlanetNames(names);

      // 5. Build payload (only feature columns)
      const payload = buildPayload(rows);

      // 6. Call backend
      const res = await fetch(`${API_BASE}/predict-batch`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(payload),
      });

      const json = await res.json();

      if (!res.ok || json.status === 'error') {
        throw new Error(json.message || 'Prediction failed.');
      }

      setResults(json.predictions);
    } catch (err) {
      setError(err.message || 'An unexpected error occurred.');
    } finally {
      setLoading(false);
    }
  }, []);

  // ── Handle file read ──
  const handleFile = (file) => {
    if (!file) return;
    if (!file.name.match(/\.(csv|json)$/i)) {
      setError('Only .csv and .json files are supported.');
      return;
    }
    setFileName(file.name);
    const reader = new FileReader();
    reader.onload = (e) => {
      setRawText(e.target.result);
      processData(e.target.result);
    };
    reader.readAsText(file);
  };

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  }, []);

  const onDragOver = (e) => { e.preventDefault(); setDragging(true); };
  const onDragLeave = () => setDragging(false);

  // ── Analysis stats ──
  const stats = results ? (() => {
    const habitable = results.filter(r => r.habitability_score >= 0.5);
    const scores    = results.map(r => r.habitability_score);
    const avgScore  = scores.reduce((a, b) => a + b, 0) / scores.length;

    const sorted       = [...results].map((r, i) => ({ ...r, name: planetNames[i] }))
                           .sort((a, b) => b.habitability_score - a.habitability_score);
    const top3         = sorted.slice(0, 3);
    const leastHab     = sorted[sorted.length - 1];

    return { total: results.length, habitableCount: habitable.length, avgScore, top3, leastHab, sorted };
  })() : null;

  return (
    <div className="raw-analysis-panel">
      {/* ── Header ── */}
      <div className="rap-header">
        <div className="rap-header-icon">🧬</div>
        <div>
          <h3 className="rap-title">Raw Data Prediction &amp; Analysis</h3>
          <p className="rap-subtitle">
            Upload or paste raw astronomical data — no preprocessing needed
          </p>
        </div>
      </div>

      {/* ── Mode toggle ── */}
      <div className="rap-toggle-row">
        <button
          className={`rap-toggle-btn${inputMode === 'upload' ? ' active' : ''}`}
          onClick={() => { setInputMode('upload'); setError(''); setResults(null); }}
        >
          <span>📂</span> Upload File
        </button>
        <button
          className={`rap-toggle-btn${inputMode === 'paste' ? ' active' : ''}`}
          onClick={() => { setInputMode('paste'); setError(''); setResults(null); }}
        >
          <span>📋</span> Paste Data
        </button>
      </div>

      {/* ── Input area ── */}
      <AnimatePresence mode="wait">
        {inputMode === 'upload' ? (
          <motion.div
            key="upload"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.25 }}
          >
            {/* Drag-and-drop zone */}
            <div
              className={`rap-dropzone${dragging ? ' dragging' : ''}`}
              onDrop={onDrop}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv,.json"
                style={{ display: 'none' }}
                onChange={e => handleFile(e.target.files[0])}
              />
              <div className="rap-drop-icon">🚀</div>
              {fileName ? (
                <p className="rap-drop-filename">{fileName}</p>
              ) : (
                <>
                  <p className="rap-drop-primary">Drag &amp; drop your file here</p>
                  <p className="rap-drop-secondary">or click to browse · CSV and JSON accepted</p>
                </>
              )}
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="paste"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.25 }}
          >
            <div className="rap-textarea-wrap">
              <label className="rap-label">
                Paste raw JSON array or CSV (with header row)
              </label>
              <textarea
                className="rap-textarea"
                rows={10}
                value={rawText}
                onChange={e => { setRawText(e.target.value); setError(''); setResults(null); }}
                placeholder={`Example JSON:\n[\n  {\n    "planet_name": "Kepler-452b",\n    "planet_radius": 1.6,\n    "planet_mass": 5.0,\n    "orbital_period": 384.84,\n    "semi_major_axis": 1.046,\n    "surface_temp": 265,\n    "star_temperature": 5757,\n    "star_metallicity": 0.21,\n    "luminosity": 1.2,\n    "star_radius": 1.11,\n    "spectral_type": "G"\n  }\n]`}
              />
              <button
                className="rap-process-btn"
                onClick={() => processData(rawText)}
                disabled={loading || !rawText.trim()}
              >
                {loading ? (
                  <span className="rap-spinner" />
                ) : (
                  <>⚡ Process Data</>
                )}
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Loading ── */}
      <AnimatePresence>
        {loading && (
          <motion.div
            className="rap-loading-bar"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <div className="rap-loading-track">
              <motion.div
                className="rap-loading-fill"
                initial={{ width: '0%' }}
                animate={{ width: '90%' }}
                transition={{ duration: 2.5, ease: 'easeInOut' }}
              />
            </div>
            <p className="rap-loading-msg">Running ML inference · Please wait…</p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Error ── */}
      <AnimatePresence>
        {error && (
          <motion.div
            className="rap-error"
            initial={{ opacity: 0, scale: 0.97 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
          >
            <span className="rap-error-icon">⚠️</span>
            <span>{error}</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Results ── */}
      <AnimatePresence>
        {results && stats && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.4 }}
          >
            {/* ── Stats cards ── */}
            <div className="rap-stats-grid">
              <div className="rap-stat-card">
                <div className="rap-stat-value">{stats.total}</div>
                <div className="rap-stat-label">Planets Analyzed</div>
              </div>
              <div className="rap-stat-card green">
                <div className="rap-stat-value">{stats.habitableCount}</div>
                <div className="rap-stat-label">Habitable</div>
              </div>
              <div className="rap-stat-card yellow">
                <div className="rap-stat-value">{(stats.avgScore * 100).toFixed(1)}%</div>
                <div className="rap-stat-label">Avg Hab. Score</div>
              </div>
              <div className="rap-stat-card red">
                <div className="rap-stat-value">{stats.total - stats.habitableCount}</div>
                <div className="rap-stat-label">Non-Habitable</div>
              </div>
            </div>

            {/* ── Highlights ── */}
            <div className="rap-highlights">
              <div className="rap-hl-block">
                <div className="rap-hl-title">🏆 Top 3 Most Habitable</div>
                {stats.top3.map((p, i) => (
                  <div key={i} className="rap-hl-row" style={{ borderLeftColor: scoreColor(p.habitability_score) }}>
                    <span className="rap-hl-rank">#{i + 1}</span>
                    <span className="rap-hl-name">{p.name}</span>
                    <span className="rap-hl-score" style={{ color: scoreColor(p.habitability_score) }}>
                      {(p.habitability_score * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
              <div className="rap-hl-block">
                <div className="rap-hl-title">📉 Least Habitable</div>
                <div className="rap-hl-row" style={{ borderLeftColor: '#ff5252' }}>
                  <span className="rap-hl-rank">⚠</span>
                  <span className="rap-hl-name">{stats.leastHab.name}</span>
                  <span className="rap-hl-score" style={{ color: '#ff5252' }}>
                    {(stats.leastHab.habitability_score * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>

            {/* ── Results table ── */}
            <div className="rap-table-wrap">
              <div className="rap-table-header">
                <span>📊 Prediction Results — {stats.total} planets</span>
              </div>
              <div className="rap-table-scroll">
                <table className="rap-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Planet Name</th>
                      <th>Hab. Score</th>
                      <th>Status</th>
                      <th>Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((r, i) => {
                      const score = r.habitability_score;
                      const color = scoreColor(score);
                      const bg    = scoreBg(score);
                      return (
                        <motion.tr
                          key={i}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: i * 0.04 }}
                          style={{ background: bg }}
                        >
                          <td className="rap-td-row">{i + 1}</td>
                          <td className="rap-td-name">{planetNames[i] || `Planet ${i + 1}`}</td>
                          <td>
                            <div className="rap-score-wrap">
                              <div className="rap-score-bar-track">
                                <div
                                  className="rap-score-bar-fill"
                                  style={{ width: `${score * 100}%`, background: color }}
                                />
                              </div>
                              <span className="rap-score-text" style={{ color }}>
                                {(score * 100).toFixed(1)}%
                              </span>
                            </div>
                          </td>
                          <td>
                            <span className="rap-status-badge" style={{ color, borderColor: color + '55', background: color + '15' }}>
                              {statusLabel(score)}
                            </span>
                          </td>
                          <td className="rap-td-conf">{r.confidence}</td>
                        </motion.tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Format hint ── */}
      {!results && !loading && !error && (
        <div className="rap-hint">
          <div className="rap-hint-title">📐 Required Data Fields</div>
          <div className="rap-hint-cols">
            {REQUIRED_COLUMNS.map(c => (
              <code key={c} className="rap-hint-col">{c}</code>
            ))}
          </div>
          <p className="rap-hint-note">
            Note: <code>planet_name</code> is optional but recommended for display purposes.
            The <code>habitable</code> column is NOT required.
          </p>
        </div>
      )}
    </div>
  );
}
