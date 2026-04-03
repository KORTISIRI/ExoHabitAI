import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const API = 'http://127.0.0.1:5000';

function generateMissionId() {
  const letters = 'ABCDEFGHJKLMNPQRSTUVWXYZ';
  const suffix = Array.from({ length: 3 }, () => letters[Math.floor(Math.random() * letters.length)]).join('');
  const num = parseInt(localStorage.getItem('missionCounter') || '1000') + 1;
  localStorage.setItem('missionCounter', num);
  return `EXO-${num}-${suffix}`;
}

// ── Field component defined at MODULE level (NOT inside PredictionPanel) ──────
// This is critical: if defined inside the parent render function, React sees a
// brand-new component type on every render, unmounting the input and losing focus
// after each keystroke. At module level it stays stable.
function Field({ id, label, unit, placeholder, value, onChange, error }) {
  return (
    <div className="field">
      <label htmlFor={`field-${id}`}>
        {label} {unit && <em>{unit}</em>}
      </label>
      <input
        id={`field-${id}`}
        type="text"
        inputMode="decimal"
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        className={error ? 'error' : ''}
        autoComplete="off"
        autoCorrect="off"
        autoCapitalize="off"
        spellCheck="false"
      />
      {error
        ? <span className="field-hint" style={{ color: 'var(--red)' }}>{error}</span>
        : <span className="field-hint">{placeholder}</span>
      }
    </div>
  );
}

export default function PredictionPanel({ onResult, loading, setLoading }) {
  const [missionId, setMissionId] = useState(() => generateMissionId());
  const [form, setForm] = useState({
    pl_orbper:  '',
    pl_orbsmax: '',
    pl_rade:    '',
    pl_bmasse:  '',
    pl_eqt:     '',
    st_teff:    '',
    st_met:     '',
    st_lum:     '',
    st_rad:     '',
    st_spectype:'',
  });
  const [errors, setErrors]           = useState({});
  const [globalError, setGlobalError] = useState('');

  const delay = ms => new Promise(r => setTimeout(r, ms));
  const set   = (k, v) => setForm(f => ({ ...f, [k]: v }));

  const validate = () => {
    const e = {};
    if (!form.pl_orbper  || parseFloat(form.pl_orbper)  <= 0) e.pl_orbper  = 'Required';
    if (!form.pl_orbsmax || parseFloat(form.pl_orbsmax) <= 0) e.pl_orbsmax = 'Required';
    if (!form.st_teff    || parseFloat(form.st_teff)    < 300) e.st_teff   = 'Min 300 K';
    if (form.st_met === '' || isNaN(parseFloat(form.st_met))) e.st_met      = 'Required';
    if (!form.st_spectype) e.st_spectype = 'Select type';
    return e;
  };

  const handleSubmit = async e => {
    e.preventDefault();
    setGlobalError('');
    const errs = validate();
    setErrors(errs);
    if (Object.keys(errs).length > 0) return;

    setLoading(true);

    const payload = {
      pl_orbper:   parseFloat(form.pl_orbper),
      pl_orbsmax:  parseFloat(form.pl_orbsmax),
      st_teff:     parseFloat(form.st_teff),
      st_met:      parseFloat(form.st_met),
      st_spectype: form.st_spectype,
    };

    try {
      await delay(400);
      const res  = await fetch(`${API}/predict`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(payload),
      });
      await delay(300);
      const data = await res.json();

      if (!res.ok || data.status === 'error') {
        setGlobalError(data.message || 'Backend error. Try again.');
        setLoading(false);
        return;
      }

      const inputs = {
        'Orbital Period':  form.pl_orbper  ? `${parseFloat(form.pl_orbper).toFixed(2)} days`  : '—',
        'Semi-Major Axis': form.pl_orbsmax ? `${parseFloat(form.pl_orbsmax).toFixed(3)} AU`   : '—',
        'Planet Radius':   form.pl_rade    ? `${parseFloat(form.pl_rade).toFixed(2)} R⊕`      : '—',
        'Planet Mass':     form.pl_bmasse  ? `${parseFloat(form.pl_bmasse).toFixed(2)} M⊕`    : '—',
        'Surface Temp':    form.pl_eqt     ? `${parseFloat(form.pl_eqt).toFixed(0)} K`        : '—',
        'Star Temp':       form.st_teff    ? `${parseFloat(form.st_teff).toFixed(0)} K`       : '—',
        'Metallicity':     `${parseFloat(form.st_met).toFixed(2)} dex`,
        'Luminosity':      form.st_lum     ? `${parseFloat(form.st_lum).toFixed(2)} log L☉`  : '—',
        'Star Radius':     form.st_rad     ? `${parseFloat(form.st_rad).toFixed(2)} R☉`       : '—',
        'Spectral Type':   form.st_spectype,
      };

      const record = {
        id:          missionId,
        pl_orbper:   parseFloat(form.pl_orbper).toFixed(2),
        pl_orbsmax:  parseFloat(form.pl_orbsmax).toFixed(3),
        st_teff:     parseFloat(form.st_teff).toFixed(0),
        st_spectype: form.st_spectype,
        probability: data.habitability_probability,
        label:       data.label,
        habitable:   data.prediction === 1,
        raw:         { ...form },
        timestamp:   new Date().toISOString(),
      };

      const hist = JSON.parse(localStorage.getItem('exohistory') || '[]');
      hist.unshift(record);
      if (hist.length > 20) hist.pop();
      localStorage.setItem('exohistory', JSON.stringify(hist));

      onResult({ data, inputs, raw: form });
      setMissionId(generateMissionId());

    } catch (err) {
      setGlobalError(`Cannot connect to Flask backend at ${API}. Run: python App.py`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.div
      className="glass-panel prediction-panel"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
    >
      <div className="panel-corner tl" /><div className="panel-corner tr" />
      <div className="panel-corner bl" /><div className="panel-corner br" />

      {/* Top Bar */}
      <div className="panel-topbar">
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div className="topbar-dots">
            <div className="tb-dot red" />
            <div className="tb-dot yellow" />
            <div className="tb-dot green" />
          </div>
          <span className="tb-title">EXOHABITAI // PARAMETER CONSOLE</span>
        </div>
        <span className="tb-right">MISSION-ID: {missionId}</span>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="panel-body">

          {/* Planetary Parameters */}
          <div className="param-section">
            <div className="param-section-title">
              <div className="param-section-icon cyan-bg">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                  <circle cx="8" cy="8" r="4.5" stroke="#00E5FF" strokeWidth="1.5"/>
                  <ellipse cx="8" cy="8" rx="7.5" ry="2.5" stroke="#00E5FF" strokeWidth="0.8"
                    transform="rotate(-20 8 8)" opacity="0.6"/>
                </svg>
              </div>
              <span className="param-section-label">Planetary Parameters</span>
              <div className="param-divider" />
            </div>

            <div className="field-grid">
              <Field
                id="pl_rade" label="Planet Radius" unit="R⊕"
                placeholder="1.0 (Earth = 1.0)"
                value={form.pl_rade}
                onChange={e => set('pl_rade', e.target.value)}
                error={errors.pl_rade}
              />
              <Field
                id="pl_bmasse" label="Planet Mass" unit="M⊕"
                placeholder="1.0 (Earth = 1.0)"
                value={form.pl_bmasse}
                onChange={e => set('pl_bmasse', e.target.value)}
                error={errors.pl_bmasse}
              />
              <Field
                id="pl_orbper" label="Orbital Period" unit="days"
                placeholder="365.25 (Earth)"
                value={form.pl_orbper}
                onChange={e => set('pl_orbper', e.target.value)}
                error={errors.pl_orbper}
              />
              <Field
                id="pl_orbsmax" label="Semi-Major Axis" unit="AU"
                placeholder="1.0 (Earth)"
                value={form.pl_orbsmax}
                onChange={e => set('pl_orbsmax', e.target.value)}
                error={errors.pl_orbsmax}
              />

              {/* Surface Temperature — inline (no Field wrapper needed, already stable) */}
              <div className="field field-full">
                <label htmlFor="field-pl_eqt">Surface Temperature <em>K</em></label>
                <input
                  id="field-pl_eqt"
                  type="text"
                  inputMode="decimal"
                  value={form.pl_eqt}
                  onChange={e => set('pl_eqt', e.target.value)}
                  placeholder="288 (Earth ≈ 288 K)"
                  autoComplete="off"
                  autoCorrect="off"
                  autoCapitalize="off"
                  spellCheck="false"
                />
                <span className="field-hint">Earth ≈ 288 K  (optional)</span>
              </div>
            </div>
          </div>

          {/* Divider */}
          <div className="params-divider">
            <div className="params-divider-line" />
            <div className="params-divider-icon">
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                <path d="M4 7h6M7 4v6" stroke="#00E5FF" strokeWidth="1.5" strokeLinecap="round"/>
              </svg>
            </div>
            <div className="params-divider-line" />
          </div>

          {/* Stellar Parameters */}
          <div className="param-section" style={{ marginBottom: 0 }}>
            <div className="param-section-title">
              <div className="param-section-icon amber-bg">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                  <circle cx="8" cy="8" r="3.5" fill="#F59E0B" opacity="0.9"/>
                  {[0,45,90,135].map(a => (
                    <line key={a}
                      x1={8 + 5.5 * Math.cos(a*Math.PI/180)}
                      y1={8 + 5.5 * Math.sin(a*Math.PI/180)}
                      x2={8 + 7.5 * Math.cos(a*Math.PI/180)}
                      y2={8 + 7.5 * Math.sin(a*Math.PI/180)}
                      stroke="#F59E0B" strokeWidth="1.5" strokeLinecap="round"
                    />
                  ))}
                </svg>
              </div>
              <span className="param-section-label">Stellar Parameters</span>
              <div className="param-divider" style={{ background: 'linear-gradient(to right, rgba(245,158,11,0.2), transparent)' }} />
            </div>

            <div className="field-grid">
              <Field
                id="st_teff" label="Star Temperature" unit="K"
                placeholder="5778 (Sun = 5778 K)"
                value={form.st_teff}
                onChange={e => set('st_teff', e.target.value)}
                error={errors.st_teff}
              />
              <Field
                id="st_met" label="Star Metallicity" unit="dex"
                placeholder="0.0 (Sun = 0.0)"
                value={form.st_met}
                onChange={e => set('st_met', e.target.value)}
                error={errors.st_met}
              />
              <Field
                id="st_lum" label="Luminosity" unit="log₁₀ L☉"
                placeholder="0.0 (Sun = 0.0)"
                value={form.st_lum}
                onChange={e => set('st_lum', e.target.value)}
                error={errors.st_lum}
              />
              <Field
                id="st_rad" label="Star Radius" unit="R☉"
                placeholder="1.0 (Sun = 1.0)"
                value={form.st_rad}
                onChange={e => set('st_rad', e.target.value)}
                error={errors.st_rad}
              />

              {/* Spectral Type dropdown */}
              <div className="field field-full">
                <label htmlFor="field-st_spectype">Spectral Type</label>
                <select
                  id="field-st_spectype"
                  value={form.st_spectype}
                  onChange={e => set('st_spectype', e.target.value)}
                  className={errors.st_spectype ? 'error' : ''}
                  style={{
                    background: 'rgba(0,0,0,0.35)',
                    border: '1px solid rgba(0,229,255,0.15)',
                    borderRadius: 6,
                    color: 'var(--soft-white)',
                    fontFamily: "'Share Tech Mono', monospace",
                    fontSize: '0.9rem',
                    padding: '10px 14px',
                    outline: 'none',
                    width: '100%',
                  }}
                >
                  <option value="">Select spectral type</option>
                  <option value="G">G — Sun-like (most habitable)</option>
                  <option value="K">K — Orange dwarf (habitable)</option>
                  <option value="M">M — Red dwarf</option>
                  <option value="F">F — Yellow-white</option>
                  <option value="B">B — Blue-white (hot)</option>
                  <option value="A">A — White</option>
                </select>
                {errors.st_spectype
                  ? <span className="field-hint" style={{ color: 'var(--red)' }}>{errors.st_spectype}</span>
                  : <span className="field-hint">G/K types most likely habitable</span>
                }
              </div>
            </div>
          </div>
        </div>

        {/* Error */}
        <AnimatePresence>
          {globalError && (
            <motion.div
              className="error-box"
              style={{ margin: '0 24px 16px' }}
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
            >
              <span>⚠</span>
              <span>{globalError}</span>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Submit */}
        <div className="predict-btn-wrap">
          <motion.button
            type="submit"
            className="predict-btn"
            disabled={loading}
            whileHover={!loading ? { scale: 1.01 } : {}}
            whileTap={!loading ? { scale: 0.99 } : {}}
          >
            {loading ? (
              <>
                <div className="orbit-spinner">
                  <div className="orbit-ring" />
                  <div className="orbit-core" />
                </div>
                ANALYZING SYSTEM...
              </>
            ) : (
              <>
                <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
                  <circle cx="9" cy="9" r="7.5" stroke="currentColor" strokeWidth="1.5"/>
                  <path d="M6 9l2.5 2.5L13 6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                PREDICT HABITABILITY
              </>
            )}
          </motion.button>
          <p className="predict-meta">
            Model: Random Forest (Tuned) · AUC 0.906 · 39,315 Training Samples
          </p>
        </div>
      </form>
    </motion.div>
  );
}