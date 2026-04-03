import { useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const API = 'http://127.0.0.1:5000';

const JSON_PLACEHOLDER = `[
  {
    "pl_orbper": 365.25,
    "pl_orbsmax": 1.0,
    "pl_rade": 1.0,
    "pl_bmasse": 1.0,
    "pl_eqt": 288,
    "st_teff": 5778,
    "st_met": 0.0,
    "st_lum": 0.0,
    "st_rad": 1.0,
    "st_spectype": "G"
  },
  {
    "pl_orbper": 687,
    "pl_orbsmax": 1.52,
    "st_teff": 5000,
    "st_met": -0.2,
    "st_spectype": "K"
  }
]`;

/* ── helper: parse JSON text → array of objects ── */
function parseJSON(text) {
  const parsed = JSON.parse(text);
  if (Array.isArray(parsed)) return parsed;
  if (parsed && typeof parsed === 'object') return [parsed];
  throw new Error('JSON must be an array or object.');
}

/* ── helper: parse CSV text → array of objects ── */
function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) throw new Error('CSV must have a header row + at least 1 data row.');
  const headers = lines[0].split(',').map(h => h.trim());
  return lines.slice(1).map(line => {
    const vals = line.split(',');
    const obj = {};
    headers.forEach((h, i) => { obj[h] = vals[i]?.trim() ?? ''; });
    return obj;
  });
}

/* ── Row result card ── */
function ResultRow({ row, index }) {
  const score    = row.habitability_probability ?? row.habitability_score ?? 0;
  const pct      = Math.round(score * 100);
  const isHab    = row.prediction === 1;
  const color    = isHab ? '#10B981' : '#EF4444';
  const barColor = isHab
    ? 'linear-gradient(90deg, #065f46, #10B981)'
    : 'linear-gradient(90deg, #7f1d1d, #EF4444)';

  return (
    <motion.tr
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.03, duration: 0.3 }}
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
              transition={{ delay: index * 0.03 + 0.2, duration: 0.6, ease: 'easeOut' }}
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
    </motion.tr>
  );
}

/* ══ Main Component ══ */
export default function BatchUploadPanel() {
  // ── Input mode: 'file' | 'json'
  const [inputMode, setInputMode]   = useState('file');

  // ── File upload state
  const [dragOver, setDragOver]     = useState(false);
  const [file, setFile]             = useState(null);
  const fileRef                     = useRef();

  // ── JSON text input state
  const [jsonText, setJsonText]     = useState('');
  const [jsonValid, setJsonValid]   = useState(null); // null | true | false

  // ── Shared state
  const [preview, setPreview]       = useState(null);
  const [loading, setLoading]       = useState(false);
  const [progress, setProgress]     = useState(0);
  const [results, setResults]       = useState(null);
  const [error, setError]           = useState('');
  const [filterMode, setFilterMode] = useState('all');

  /* ── helpers ── */
  const resetAll = () => {
    setFile(null); setPreview(null); setResults(null);
    setError(''); setJsonText(''); setJsonValid(null);
  };

  /* ── drag handlers ── */
  const onDragOver  = useCallback(e => { e.preventDefault(); setDragOver(true);  }, []);
  const onDragLeave = useCallback(() => setDragOver(false), []);
  const onDrop      = useCallback(e  => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]); }, []);

  /* ── File picked ── */
  const handleFile = (f) => {
    if (!f) return;
    setError(''); setResults(null); setPreview(null);
    const ext = f.name.split('.').pop().toLowerCase();
    if (!['csv', 'json'].includes(ext)) { setError('Only .csv and .json files are supported.'); return; }
    setFile(f);
    const reader = new FileReader();
    reader.onload = e => {
      try {
        const rows = ext === 'json' ? parseJSON(e.target.result) : parseCSV(e.target.result);
        setPreview({ headers: Object.keys(rows[0] || {}), rows: rows.slice(0, 3), total: rows.length });
      } catch (err) {
        setError(`Could not read file: ${err.message}`); setFile(null);
      }
    };
    reader.readAsText(f);
  };

  /* ── JSON text changed ── */
  const handleJsonChange = (val) => {
    setJsonText(val);
    setError('');
    setResults(null);
    setPreview(null);
    if (!val.trim()) { setJsonValid(null); return; }
    try {
      const rows = parseJSON(val);
      setJsonValid(true);
      setPreview({ headers: Object.keys(rows[0] || {}), rows: rows.slice(0, 3), total: rows.length });
    } catch {
      setJsonValid(false);
    }
  };

  /* ── Submit ── */
  const isReady = inputMode === 'file' ? !!file : (jsonValid === true);

  const handleSubmit = async () => {
    if (!isReady) return;
    setError(''); setLoading(true); setProgress(0);

    const ticks = [
      { pct: 20, ms: 300  },
      { pct: 50, ms: 700  },
      { pct: 80, ms: 1200 },
    ];
    ticks.forEach(t => setTimeout(() => setProgress(t.pct), t.ms));

    try {
      let res, data;

      if (inputMode === 'file') {
        const fd = new FormData();
        fd.append('file', file);
        res  = await fetch(`${API}/predict-batch`, { method: 'POST', body: fd });
      } else {
        // Send JSON text as application/json
        res  = await fetch(`${API}/predict-batch`, {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    jsonText.trim(),
        });
      }

      data = await res.json();
      setProgress(100);
      await new Promise(r => setTimeout(r, 400));

      if (!res.ok || data.status === 'error') {
        setError(data.message || 'Backend returned an error.');
        return;
      }
      setResults(data);
    } catch {
      setError(`Cannot connect to Flask backend at ${API}. Make sure python App.py is running.`);
    } finally {
      setLoading(false); setProgress(0);
    }
  };

  /* ── Export CSV ── */
  const exportCSV = () => {
    if (!results) return;
    const hdr  = ['Row', 'Habitability %', 'Status', 'Confidence'];
    const rows = results.predictions.map(r => [
      r.row_index + 1,
      Math.round((r.habitability_probability ?? r.habitability_score) * 100),
      r.status, r.confidence,
    ]);
    const csv  = [hdr, ...rows].map(r => r.join(',')).join('\n');
    const url  = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    Object.assign(document.createElement('a'), { href: url, download: 'exohabitai_results.csv' }).click();
    URL.revokeObjectURL(url);
  };

  /* ── Derived stats ── */
  const filteredRows = results?.predictions?.filter(r => {
    if (filterMode === 'habitable')     return r.prediction === 1;
    if (filterMode === 'non-habitable') return r.prediction === 0;
    return true;
  }) ?? [];
  const habCount    = results?.predictions?.filter(r => r.prediction === 1).length ?? 0;
  const nonHabCount = results?.predictions?.filter(r => r.prediction === 0).length ?? 0;
  const avgScore    = results?.predictions
    ? Math.round((results.predictions.reduce((s, r) => s + (r.habitability_probability ?? r.habitability_score ?? 0), 0) / results.predictions.length) * 100)
    : 0;

  /* ════════════════ RENDER ════════════════ */
  return (
    <motion.div
      className="glass-panel batch-panel"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1 }}
    >
      <div className="panel-corner tl" /><div className="panel-corner tr" />
      <div className="panel-corner bl" /><div className="panel-corner br" />

      {/* ── Top bar ── */}
      <div className="panel-topbar">
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div className="topbar-dots">
            <div className="tb-dot red" /><div className="tb-dot yellow" /><div className="tb-dot green" />
          </div>
          <span className="tb-title">EXOHABITAI // BULK ANALYSIS ENGINE</span>
        </div>
        <span className="tb-right" style={{ color: 'var(--amber)', opacity: 0.8 }}>CSV · JSON</span>
      </div>

      <div className="batch-body">

        {/* ── Section title ── */}
        <div className="batch-section-title">
          <div style={{
            width: 30, height: 30, borderRadius: 6,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: 'rgba(245,158,11,0.1)', border: '1px solid rgba(245,158,11,0.25)',
          }}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M3 4h10M3 8h10M3 12h6" stroke="#F59E0B" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
          </div>
          <span style={{
            fontFamily: 'Rajdhani, sans-serif', fontWeight: 600,
            fontSize: '0.8rem', color: 'var(--text-muted)',
            letterSpacing: 3, textTransform: 'uppercase',
          }}>
            Batch Prediction Upload
          </span>
          <div style={{ height: 1, flex: 1, background: 'linear-gradient(to right, rgba(245,158,11,0.2), transparent)', marginLeft: 12 }} />
        </div>

        {/* ══ INPUT MODE TABS ══ */}
        <div className="batch-mode-tabs">
          <button
            className={`batch-mode-tab ${inputMode === 'file' ? 'active' : ''}`}
            onClick={() => { setInputMode('file'); resetAll(); }}
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path d="M2 2h6l3 3v7H2V2z" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round"/>
              <path d="M8 2v3h3" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round"/>
            </svg>
            Upload File
            <span className="batch-mode-tab-badge">CSV · JSON</span>
          </button>
          <button
            className={`batch-mode-tab ${inputMode === 'json' ? 'active' : ''}`}
            onClick={() => { setInputMode('json'); resetAll(); }}
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path d="M4 3L1 7l3 4M10 3l3 4-3 4M8 2l-2 10" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Paste JSON
            <span className="batch-mode-tab-badge">Text Input</span>
          </button>
        </div>

        <AnimatePresence mode="wait">

          {/* ══ FILE UPLOAD MODE ══ */}
          {inputMode === 'file' && (
            <motion.div key="file-mode"
              initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.25 }}
              style={{ display: 'flex', flexDirection: 'column', gap: 16 }}
            >
              {/* format hints */}
              <div className="batch-hint-row">
                <div className="batch-hint-card">
                  <span className="batch-hint-icon">📄</span>
                  <div>
                    <strong>CSV Format</strong>
                    <span>Headers: pl_orbper, pl_orbsmax, st_teff, st_met, st_spectype (+ optional fields)</span>
                  </div>
                </div>
                <div className="batch-hint-card">
                  <span className="batch-hint-icon">{'{}'}</span>
                  <div>
                    <strong>JSON Format</strong>
                    <span>Array of objects with the same field names</span>
                  </div>
                </div>
              </div>

              {/* drop zone */}
              <div
                className={`batch-dropzone ${dragOver ? 'drag-over' : ''} ${file ? 'has-file' : ''}`}
                onDragOver={onDragOver}
                onDragLeave={onDragLeave}
                onDrop={onDrop}
                onClick={() => !file && fileRef.current?.click()}
              >
                <input ref={fileRef} type="file" accept=".csv,.json" style={{ display: 'none' }}
                  onChange={e => handleFile(e.target.files[0])} />
                <AnimatePresence mode="wait">
                  {file ? (
                    <motion.div key="file" className="batch-file-info"
                      initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}
                    >
                      <div className="batch-file-icon">{file.name.endsWith('.csv') ? '📊' : '📋'}</div>
                      <div className="batch-file-meta">
                        <span className="batch-file-name">{file.name}</span>
                        <span className="batch-file-size">
                          {(file.size / 1024).toFixed(1)} KB
                          {preview && ` · ${preview.total} row${preview.total !== 1 ? 's' : ''}`}
                        </span>
                      </div>
                      <button className="batch-remove-btn"
                        onClick={e => { e.stopPropagation(); setFile(null); setPreview(null); setResults(null); setError(''); }}>
                        ✕
                      </button>
                    </motion.div>
                  ) : (
                    <motion.div key="empty" className="batch-drop-prompt"
                      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    >
                      <div className="batch-drop-icon">
                        <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
                          <circle cx="20" cy="20" r="19" stroke="rgba(0,229,255,0.2)" strokeWidth="1.5" strokeDasharray="4 3"/>
                          <path d="M20 28V16M14 22l6-6 6 6" stroke="#00E5FF" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                      </div>
                      <p className="batch-drop-text">
                        Drag &amp; drop your <span>.csv</span> or <span>.json</span> file here
                      </p>
                      <p className="batch-drop-sub">or click to browse your file system</p>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </motion.div>
          )}

          {/* ══ JSON TEXT INPUT MODE ══ */}
          {inputMode === 'json' && (
            <motion.div key="json-mode"
              initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.25 }}
              style={{ display: 'flex', flexDirection: 'column', gap: 12 }}
            >
              {/* instruction card */}
              <div className="batch-hint-card" style={{ background: 'rgba(0,229,255,0.04)', borderColor: 'rgba(0,229,255,0.12)' }}>
                <span className="batch-hint-icon">📡</span>
                <div>
                  <strong>Paste or type your JSON array</strong>
                  <span>
                    Each object is one planet. Required keys: <code>pl_orbper</code>, <code>pl_orbsmax</code>,{' '}
                    <code>st_teff</code>, <code>st_met</code>, <code>st_spectype</code>. Optional: <code>pl_rade</code>,{' '}
                    <code>pl_bmasse</code>, <code>pl_eqt</code>, <code>st_lum</code>, <code>st_rad</code>
                  </span>
                </div>
              </div>

              {/* textarea */}
              <div className="batch-json-wrap">
                {/* header bar */}
                <div className="batch-json-header">
                  <span>JSON INPUT</span>
                  <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                    {jsonText.trim() && (
                      <span style={{
                        fontFamily: 'Share Tech Mono, monospace', fontSize: '0.65rem', letterSpacing: 1,
                        color: jsonValid ? '#10B981' : '#EF4444',
                        display: 'flex', alignItems: 'center', gap: 4,
                      }}>
                        <span style={{
                          width: 6, height: 6, borderRadius: '50%',
                          background: jsonValid ? '#10B981' : '#EF4444',
                          display: 'inline-block',
                        }} />
                        {jsonValid ? 'VALID JSON' : 'INVALID JSON'}
                      </span>
                    )}
                    <button
                      className="batch-action-btn"
                      style={{ padding: '3px 10px', fontSize: '0.62rem' }}
                      onClick={() => { setJsonText(JSON_PLACEHOLDER); handleJsonChange(JSON_PLACEHOLDER); }}
                    >
                      LOAD EXAMPLE
                    </button>
                    {jsonText && (
                      <button
                        className="batch-action-btn"
                        style={{ padding: '3px 10px', fontSize: '0.62rem', borderColor: 'rgba(239,68,68,0.2)', color: '#FCA5A5' }}
                        onClick={() => { setJsonText(''); setJsonValid(null); setPreview(null); setError(''); }}
                      >
                        CLEAR
                      </button>
                    )}
                  </div>
                </div>

                {/* the actual textarea */}
                <textarea
                  className={`batch-json-textarea ${jsonText.trim() ? (jsonValid ? 'valid' : 'invalid') : ''}`}
                  value={jsonText}
                  onChange={e => handleJsonChange(e.target.value)}
                  placeholder={JSON_PLACEHOLDER}
                  rows={12}
                  spellCheck="false"
                  autoComplete="off"
                  autoCorrect="off"
                  autoCapitalize="off"
                />

                <div className="batch-json-footer">
                  <span>{jsonText.trim() ? `${jsonText.length} chars` : 'Empty'}</span>
                  {preview && jsonValid && (
                    <span style={{ color: '#10B981' }}>
                      ✓ {preview.total} planet{preview.total !== 1 ? 's' : ''} detected
                    </span>
                  )}
                </div>
              </div>
            </motion.div>
          )}

        </AnimatePresence>

        {/* ── Data preview (shared) ── */}
        <AnimatePresence>
          {preview && !results && (
            <motion.div className="batch-preview"
              initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}
            >
              <div className="batch-preview-header">
                <span>📡 DATA PREVIEW — First {Math.min(3, preview.rows.length)} of {preview.total} rows</span>
              </div>
              <div style={{ overflowX: 'auto' }}>
                <table className="batch-table">
                  <thead><tr>{preview.headers.map(h => <th key={h}>{h}</th>)}</tr></thead>
                  <tbody>
                    {preview.rows.map((row, i) => (
                      <tr key={i}>{preview.headers.map(h => <td key={h}>{row[h] ?? '—'}</td>)}</tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Error ── */}
        <AnimatePresence>
          {error && (
            <motion.div className="error-box" style={{ margin: '4px 0' }}
              initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            >
              <span>⚠</span><span>{error}</span>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Submit ── */}
        {!results && (
          <div>
            {loading && (
              <div className="batch-progress-wrap">
                <div className="batch-progress-bar">
                  <motion.div className="batch-progress-fill"
                    animate={{ width: `${progress}%` }} transition={{ duration: 0.4 }}
                  />
                </div>
                <span className="batch-progress-pct">{progress}%</span>
              </div>
            )}
            <motion.button
              className="predict-btn"
              disabled={!isReady || loading}
              onClick={handleSubmit}
              whileHover={isReady && !loading ? { scale: 1.01 } : {}}
              whileTap={isReady && !loading ? { scale: 0.99 } : {}}
            >
              {loading ? (
                <>
                  <div className="orbit-spinner"><div className="orbit-ring" /><div className="orbit-core" /></div>
                  ANALYZING {preview?.total ?? '...'} PLANETS...
                </>
              ) : (
                <>
                  <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
                    <path d="M3 9h12M9 3l6 6-6 6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  RUN BATCH PREDICTION
                </>
              )}
            </motion.button>
            <p className="predict-meta">
              Supports up to 10,000 rows · Required: pl_orbper, pl_orbsmax, st_teff, st_met, st_spectype
            </p>
          </div>
        )}

        {/* ══ RESULTS ══ */}
        <AnimatePresence>
          {results && (
            <motion.div className="batch-results"
              initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}
            >
              <div className="batch-stats-row">
                <div className="batch-stat-card total">
                  <span className="bsc-val">{results.total_rows}</span>
                  <span className="bsc-label">TOTAL PLANETS</span>
                </div>
                <div className="batch-stat-card habitable">
                  <span className="bsc-val">{habCount}</span>
                  <span className="bsc-label">HABITABLE</span>
                </div>
                <div className="batch-stat-card non-habitable">
                  <span className="bsc-val">{nonHabCount}</span>
                  <span className="bsc-label">NON-HABITABLE</span>
                </div>
                <div className="batch-stat-card avg-score">
                  <span className="bsc-val">{avgScore}%</span>
                  <span className="bsc-label">AVG SCORE</span>
                </div>
              </div>

              <div className="batch-toolbar">
                <div className="batch-filter-tabs">
                  {['all','habitable','non-habitable'].map(mode => (
                    <button key={mode}
                      className={`batch-filter-tab ${filterMode === mode ? 'active' : ''}`}
                      onClick={() => setFilterMode(mode)}
                    >
                      {mode === 'all' ? `All (${results.total_rows})`
                        : mode === 'habitable' ? `🟢 Habitable (${habCount})`
                        : `🔴 Non-Habitable (${nonHabCount})`}
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
                  <button className="batch-action-btn" onClick={resetAll}>
                    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                      <path d="M2 7a5 5 0 1 1 1.5 3.5M2 11V7h4" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                    NEW BATCH
                  </button>
                </div>
              </div>

              <div className="batch-results-table-wrap">
                <table className="batch-results-table">
                  <thead>
                    <tr>
                      <th style={{ width: 55 }}>ROW</th>
                      <th>HABITABILITY SCORE</th>
                      <th style={{ width: 160 }}>STATUS</th>
                      <th style={{ width: 100 }}>CONFIDENCE</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredRows.map((row, i) => (
                      <ResultRow key={row.row_index} row={row} index={i} />
                    ))}
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
        </AnimatePresence>

      </div>
    </motion.div>
  );
}
