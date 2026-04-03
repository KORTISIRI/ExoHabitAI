import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

function generateAvatar(name) {
  return name ? name.charAt(0).toUpperCase() : '?';
}

export default function AuthPage({ onAuth }) {
  const [mode, setMode]         = useState('login');   // 'login' | 'signup'
  const [form, setForm]         = useState({ name: '', email: '', password: '', confirm: '' });
  const [errors, setErrors]     = useState({});
  const [loading, setLoading]   = useState(false);
  const [success, setSuccess]   = useState(false);

  const set = (k, v) => {
    setForm(f => ({ ...f, [k]: v }));
    setErrors(e => ({ ...e, [k]: '' }));
  };

  const validate = () => {
    const e = {};
    if (mode === 'signup' && !form.name.trim())
      e.name = 'Name is required';
    if (!form.email.trim() || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email))
      e.email = 'Valid email required';
    if (!form.password || form.password.length < 6)
      e.password = 'Min 6 characters';
    if (mode === 'signup' && form.password !== form.confirm)
      e.confirm = 'Passwords do not match';
    return e;
  };

  const handleSubmit = async e => {
    e.preventDefault();
    const errs = validate();
    if (Object.keys(errs).length) { setErrors(errs); return; }

    setLoading(true);
    await new Promise(r => setTimeout(r, 1200));   // simulate auth call

    if (mode === 'login') {
      // Check localStorage for registered user
      const stored = JSON.parse(localStorage.getItem('exo_user') || 'null');
      if (stored && stored.email === form.email && stored.password === form.password) {
        setSuccess(true);
        setTimeout(() => onAuth({ name: stored.name, email: stored.email }), 1000);
      } else if (!stored) {
        // Auto-login demo account
        const user = { name: form.email.split('@')[0], email: form.email };
        setSuccess(true);
        setTimeout(() => onAuth(user), 1000);
      } else {
        setErrors({ password: 'Invalid email or password' });
        setLoading(false);
      }
    } else {
      // Register
      const user = { name: form.name, email: form.email, password: form.password };
      localStorage.setItem('exo_user', JSON.stringify(user));
      setSuccess(true);
      setTimeout(() => onAuth({ name: form.name, email: form.email }), 1000);
    }
  };

  const switchMode = (m) => {
    setMode(m);
    setErrors({});
    setForm({ name: '', email: '', password: '', confirm: '' });
  };

  return (
    <motion.div
      className="auth-page"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.6 }}
    >
      {/* Background nebulas */}
      <div className="auth-nebula n1" />
      <div className="auth-nebula n2" />
      <div className="auth-nebula n3" />

      {/* Grid overlay */}
      <div className="grid-overlay" />

      {/* Floating planets */}
      <motion.div className="auth-planet p1"
        animate={{ y: [0,-18,0], rotate:[0,6,0] }}
        transition={{ duration:10, repeat:Infinity, ease:'easeInOut' }}
      />
      <motion.div className="auth-planet p2"
        animate={{ y:[0,14,0], rotate:[0,-5,0] }}
        transition={{ duration:8, repeat:Infinity, ease:'easeInOut' }}
      />

      {/* Card */}
      <div className="auth-card-wrap">

        {/* Logo */}
        <motion.div
          className="auth-logo"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.5 }}
        >
          <div className="auth-logo-icon">
            <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
              <circle cx="14" cy="14" r="5" fill="#00E5FF" opacity="0.9"/>
              <circle cx="14" cy="14" r="12" fill="none" stroke="#00E5FF" strokeWidth="1"
                strokeDasharray="3 2" opacity="0.4"/>
              <ellipse cx="14" cy="14" rx="13.5" ry="4" fill="none" stroke="#7C3AED"
                strokeWidth="1.2" transform="rotate(-25 14 14)" opacity="0.7"/>
            </svg>
          </div>
          <span className="auth-logo-name">
            Exo<span className="auth-cyan">Habit</span><span className="auth-purple">AI</span>
          </span>
          <span className="auth-logo-tag">Mission Control — v2.0</span>
        </motion.div>

        {/* Tab switcher */}
        <motion.div
          className="auth-tabs"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.35 }}
        >
          <button
            className={`auth-tab ${mode === 'login' ? 'active' : ''}`}
            onClick={() => switchMode('login')}
          >
            SIGN IN
          </button>
          <button
            className={`auth-tab ${mode === 'signup' ? 'active' : ''}`}
            onClick={() => switchMode('signup')}
          >
            REGISTER
          </button>
          <motion.div
            className="auth-tab-slider"
            layout
            layoutId="authTabSlider"
            style={{ left: mode === 'login' ? 0 : '50%' }}
            transition={{ type:'spring', stiffness:400, damping:35 }}
          />
        </motion.div>

        {/* Form card */}
        <motion.div
          className="auth-card glass-panel"
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.45, duration: 0.5 }}
        >
          <div className="panel-corner tl" />
          <div className="panel-corner tr" />
          <div className="panel-corner bl" />
          <div className="panel-corner br" />

          {/* Card header */}
          <div className="auth-card-header">
            <div className="topbar-dots">
              <div className="tb-dot red" />
              <div className="tb-dot yellow" />
              <div className="tb-dot green" />
            </div>
            <span className="tb-title">
              EXOHABITAI // {mode === 'login' ? 'AUTHENTICATION' : 'CREW REGISTRATION'}
            </span>
          </div>

          <div className="auth-card-body">
            <AnimatePresence mode="wait">
              <motion.form
                key={mode}
                onSubmit={handleSubmit}
                noValidate
                initial={{ opacity: 0, x: mode === 'login' ? -20 : 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: mode === 'login' ? 20 : -20 }}
                transition={{ duration: 0.25 }}
              >
                {/* Name (signup only) */}
                {mode === 'signup' && (
                  <div className="auth-field">
                    <label>
                      <span className="auth-field-icon">👤</span> Commander Name
                    </label>
                    <input
                      type="text"
                      placeholder="Dr. Ellen Ripley"
                      value={form.name}
                      onChange={e => set('name', e.target.value)}
                      className={errors.name ? 'error' : ''}
                      autoFocus
                    />
                    {errors.name && <span className="auth-err">{errors.name}</span>}
                  </div>
                )}

                {/* Email */}
                <div className="auth-field">
                  <label>
                    <span className="auth-field-icon">📡</span> Mission Email
                  </label>
                  <input
                    type="email"
                    placeholder="commander@nasa.gov"
                    value={form.email}
                    onChange={e => set('email', e.target.value)}
                    className={errors.email ? 'error' : ''}
                    autoFocus={mode === 'login'}
                  />
                  {errors.email && <span className="auth-err">{errors.email}</span>}
                </div>

                {/* Password */}
                <div className="auth-field">
                  <label>
                    <span className="auth-field-icon">🔐</span> Access Code
                  </label>
                  <input
                    type="password"
                    placeholder="••••••••"
                    value={form.password}
                    onChange={e => set('password', e.target.value)}
                    className={errors.password ? 'error' : ''}
                  />
                  {errors.password && <span className="auth-err">{errors.password}</span>}
                </div>

                {/* Confirm password */}
                {mode === 'signup' && (
                  <div className="auth-field">
                    <label>
                      <span className="auth-field-icon">🔑</span> Confirm Access Code
                    </label>
                    <input
                      type="password"
                      placeholder="••••••••"
                      value={form.confirm}
                      onChange={e => set('confirm', e.target.value)}
                      className={errors.confirm ? 'error' : ''}
                    />
                    {errors.confirm && <span className="auth-err">{errors.confirm}</span>}
                  </div>
                )}

                {/* Submit */}
                <motion.button
                  type="submit"
                  className="auth-submit-btn"
                  disabled={loading || success}
                  whileHover={!loading && !success ? { scale: 1.01 } : {}}
                  whileTap={!loading && !success ? { scale: 0.99 } : {}}
                >
                  {success ? (
                    <motion.span
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      style={{ display:'flex', alignItems:'center', gap:8 }}
                    >
                      <span style={{ fontSize:'1.1rem' }}>✓</span>
                      ACCESS GRANTED
                    </motion.span>
                  ) : loading ? (
                    <>
                      <span style={{
                        display: 'inline-block',
                        width: 16, height: 16,
                        border: '2px solid rgba(255,255,255,0.3)',
                        borderTopColor: '#00E5FF',
                        borderRadius: '50%',
                        animation: 'spin 0.7s linear infinite',
                        flexShrink: 0,
                      }} />
                      AUTHENTICATING...
                    </>
                  ) : (
                    <>
                      <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                        <path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" strokeWidth="1.6"
                          strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                      {mode === 'login' ? 'LAUNCH MISSION' : 'JOIN THE CREW'}
                    </>
                  )}
                </motion.button>

                {/* Divider */}
                <div className="auth-divider">
                  <div className="auth-divider-line" />
                  <span>OR</span>
                  <div className="auth-divider-line" />
                </div>

                {/* Guest access */}
                <button
                  type="button"
                  className="auth-guest-btn"
                  onClick={() => onAuth({ name: 'Guest Commander', email: 'guest@exohabitai.space' })}
                >
                  Continue as Guest →
                </button>

                <p className="auth-switch">
                  {mode === 'login' ? "New to ExoHabitAI? " : "Already have access? "}
                  <button type="button" className="auth-switch-btn"
                    onClick={() => switchMode(mode === 'login' ? 'signup' : 'login')}
                  >
                    {mode === 'login' ? 'Register here' : 'Sign in'}
                  </button>
                </p>
              </motion.form>
            </AnimatePresence>
          </div>
        </motion.div>

        {/* Security note */}
        <motion.p
          className="auth-security-note"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
        >
          🔒 Credentials stored locally · No data sent to external servers
        </motion.p>
      </div>
    </motion.div>
  );
}
