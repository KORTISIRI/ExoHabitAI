import { useState, useCallback, useEffect } from 'react';
import { AnimatePresence, motion } from 'framer-motion';

import './App.css';

import { healthCheck } from './api';
import GalaxyBackground from './components/GalaxyBackground';
import Navbar           from './components/Navbar';
import StoryPage        from './components/StoryPage';
import AuthPage         from './components/AuthPage';
import HeroSection      from './components/HeroSection';
import GalaxyNavigation from './components/GalaxyNavigation';
import PredictionPanel  from './components/PredictionPanel';
import ResultPanel      from './components/ResultPanel';
import HistoryPanel     from './components/HistoryPanel';
import AboutSection     from './components/AboutSection';
import LoadingOverlay   from './components/LoadingOverlay';
import BatchUploadPanel from './components/BatchUploadPanel';
import RawDataPanel     from './components/RawDataPanel';

/* ── App-level flow states ─────────────────────────
   'story'     → Cinematic intro
   'auth'      → Login / Register
   'dashboard' → Main app (home, predict, results…)
   ─────────────────────────────────────────────── */

const pageVariants = {
  initial: { opacity: 0, y: 24 },
  animate: { opacity: 1, y: 0 },
  exit:    { opacity: 0, y: -16 },
};

export default function App() {
  // ── Onboarding flow (persisted so refresh keeps you logged in)
  const [appFlow, setAppFlow] = useState(() => {
    const user = localStorage.getItem('exo_session');
    if (user) return 'dashboard';   // already logged in → go straight to dashboard
    return 'story';                 // always show the cinematic intro first
  });

  const [backendStatus, setBackendStatus] = useState('checking'); // 'checking' | 'ok' | 'down' | 'no-model'

  // ── Health check on dashboard mount ───────────────────────────
  useEffect(() => {
    if (appFlow !== 'dashboard') return;
    let cancelled = false;
    healthCheck()
      .then(data => {
        if (cancelled) return;
        setBackendStatus(data.model_loaded ? 'ok' : 'no-model');
      })
      .catch(() => {
        if (!cancelled) setBackendStatus('down');
      });
    return () => { cancelled = true; };
  }, [appFlow]);

  const [user, setUser] = useState(() => {
    const s = localStorage.getItem('exo_session');
    return s ? JSON.parse(s) : null;
  });

  // ── Auth mode pre-selected from story page ('login' | 'signup')
  const [authMode, setAuthMode] = useState('login');

  // ── Dashboard state
  const [section, setSection] = useState('home');
  const [loading, setLoading] = useState(false);
  const [loadPct, setLoadPct] = useState(0);
  const [loadMsg, setLoadMsg] = useState('Initializing...');
  const [result,  setResult]  = useState(null);

  /* ── handlers ────────────────────────────────── */
  const handleStoryDone = useCallback((mode = 'login') => {
    localStorage.setItem('exo_story_seen', '1');
    if (mode === 'guest') {
      // Guest login — skip auth page entirely
      const guestUser = { name: 'Guest Commander', email: 'guest@exohabitai.space' };
      localStorage.setItem('exo_session', JSON.stringify(guestUser));
      setUser(guestUser);
      setAppFlow('dashboard');
      setSection('home');
    } else {
      setAuthMode(mode);   // 'login' or 'signup'
      setAppFlow('auth');
    }
  }, []);

  const handleAuth = useCallback((u) => {
    localStorage.setItem('exo_session', JSON.stringify(u));
    setUser(u);
    setAppFlow('dashboard');
    setSection('home');
  }, []);

  const handleLogout = useCallback(() => {
    localStorage.removeItem('exo_session');
    setUser(null);
    setAppFlow('auth');
    setResult(null);
  }, []);

  const handleSetLoading = useCallback(async (on) => {
    if (on) {
      setLoading(true);
      setLoadPct(0);
      setLoadMsg('Validating parameters...');
      setTimeout(() => { setLoadPct(20); setLoadMsg('Connecting to ML model...'); },       400);
      setTimeout(() => { setLoadPct(50); setLoadMsg('Running Random Forest inference...'); }, 900);
      setTimeout(() => { setLoadPct(80); setLoadMsg('Processing results...'); },            1400);
      setTimeout(() => { setLoadPct(100); setLoadMsg('Analysis complete!'); },              1800);
    } else {
      setLoadPct(100);
      await new Promise(r => setTimeout(r, 300));
      setLoading(false);
      setLoadPct(0);
    }
  }, []);

  const handleResult = useCallback((res) => {
    setResult(res);
    setSection('results');
  }, []);

  const handleNav = useCallback((id) => {
    setSection(id);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, []);

  const handleReset = useCallback(() => {
    setResult(null);
    setSection('predict');
  }, []);

  /* ── render ──────────────────────────────────── */
  return (
    <div className="app-root">

      {/* Galaxy background — always visible behind everything */}
      <GalaxyBackground />

      {/* ── Backend health warning banner ── */}
      <AnimatePresence>
        {appFlow === 'dashboard' && backendStatus !== 'ok' && backendStatus !== 'checking' && (
          <motion.div
            key="health-banner"
            initial={{ y: -48, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: -48, opacity: 0 }}
            transition={{ duration: 0.4 }}
            style={{
              position: 'fixed', top: 0, left: 0, right: 0, zIndex: 99999,
              background: backendStatus === 'down'
                ? 'linear-gradient(90deg, rgba(239,68,68,0.95), rgba(185,28,28,0.95))'
                : 'linear-gradient(90deg, rgba(234,179,8,0.95), rgba(161,98,7,0.95))',
              color: '#fff',
              fontSize: '0.78rem',
              fontFamily: "'Share Tech Mono', monospace",
              letterSpacing: 1,
              padding: '8px 20px',
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 12,
            }}
          >
            {backendStatus === 'down' ? (
              <><span>⚠</span><span>Flask backend is offline — start it with <strong>python app.py</strong> in the backend folder</span></>
            ) : (
              <><span>⚠</span><span>Backend is up but ML model is not loaded — run <strong>python ML_Model_Training.py</strong> first</span></>
            )}
            <button
              onClick={() => setBackendStatus('ok')}
              style={{ marginLeft: 16, background: 'rgba(255,255,255,0.2)', border: 'none', borderRadius: 4, color: '#fff', fontSize: '0.7rem', padding: '2px 8px', cursor: 'pointer' }}
            >✕</button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ══ SINGLE AnimatePresence with mode="wait" ══
          This ensures: Story FULLY exits → Auth mounts → Auth FULLY exits → Dashboard mounts
          Prevents z-index overlap where Story (9000) blocked Auth (8000) */}
      <AnimatePresence mode="wait">

        {/* STORY */}
        {appFlow === 'story' && (
          <StoryPage key="story" onComplete={handleStoryDone} />
        )}

        {/* AUTH */}
        {appFlow === 'auth' && (
          <AuthPage key="auth" onAuth={handleAuth} initialMode={authMode} />
        )}

        {/* DASHBOARD */}
        {appFlow === 'dashboard' && (
          <motion.div
            key="dashboard-root"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
            style={{ position: 'relative', zIndex: 1 }}
          >
            {/* Loading overlay */}
            <AnimatePresence>
              {loading && (
                <LoadingOverlay progress={loadPct} statusMsg={loadMsg} />
              )}
            </AnimatePresence>

            {/* Navbar */}
            <Navbar
              activeSection={section}
              onNav={handleNav}
              user={user}
              onLogout={handleLogout}
            />

            <main className="main-layout">
              <AnimatePresence mode="wait">

                {/* HOME */}
                {section === 'home' && (
                  <motion.div key="home"
                    variants={pageVariants}
                    initial="initial" animate="animate" exit="exit"
                    transition={{ duration: 0.4 }}
                  >
                    <HeroSection onNav={handleNav} />
                  </motion.div>
                )}

                {/* PREDICT + RESULTS */}
                {(section === 'predict' || section === 'results') && (
                  <motion.div key="dashboard-cols"
                    variants={pageVariants}
                    initial="initial" animate="animate" exit="exit"
                    transition={{ duration: 0.4 }}
                    className="dashboard-section"
                  >
                    <div className="sec-header">
                      <span className="sec-tag">
                        {section === 'predict'
                          ? 'MISSION CONTROL · ANALYSIS INPUT'
                          : 'MISSION CONTROL · ANALYSIS OUTPUT'}
                      </span>
                      <h2 className="sec-title">
                        {section === 'predict'
                          ? 'Planetary Analysis Console'
                          : 'Habitability Assessment Report'}
                      </h2>
                      {section === 'predict' && (
                        <p className="sec-sub">
                          Configure stellar and planetary parameters for AI habitability assessment
                        </p>
                      )}
                    </div>

                    <div className="dashboard-3col">
                      <GalaxyNavigation activeSection={section} onNav={handleNav} />

                      <PredictionPanel
                        onResult={handleResult}
                        loading={loading}
                        setLoading={handleSetLoading}
                      />

                      <div className="result-col">
                        <ResultPanel
                          result={result}
                          onReset={handleReset}
                          onNav={handleNav}
                        />
                      </div>
                    </div>
                  </motion.div>
                )}

                {/* HISTORY */}
                {section === 'history' && (
                  <motion.div key="history"
                    variants={pageVariants}
                    initial="initial" animate="animate" exit="exit"
                    transition={{ duration: 0.4 }}
                  >
                    <div className="dashboard-section" style={{ paddingTop: 40 }}>
                      <div className="dashboard-3col">
                        <GalaxyNavigation activeSection={section} onNav={handleNav} />
                        <div style={{ gridColumn: '2 / -1' }}>
                          <HistoryPanel />
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}

                {/* BATCH UPLOAD */}
                {section === 'batch' && (
                  <motion.div key="batch"
                    variants={pageVariants}
                    initial="initial" animate="animate" exit="exit"
                    transition={{ duration: 0.4 }}
                  >
                    <div className="dashboard-section" style={{ paddingTop: 40 }}>
                      <div className="sec-header">
                        <span className="sec-tag">MISSION CONTROL · BULK ANALYSIS</span>
                        <h2 className="sec-title">Batch Habitability Predictor</h2>
                        <p className="sec-sub">Upload a CSV or JSON file to predict habitability for multiple planets at once</p>
                      </div>
                      <div className="dashboard-3col">
                        <GalaxyNavigation activeSection={section} onNav={handleNav} />
                        <div style={{ gridColumn: '2 / -1' }}>
                          <BatchUploadPanel />
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}


                {/* RAW DATA TRAINING */}
                {section === 'rawdata' && (
                  <motion.div key="rawdata"
                    variants={pageVariants}
                    initial="initial" animate="animate" exit="exit"
                    transition={{ duration: 0.4 }}
                  >
                    <div className="dashboard-section" style={{ paddingTop: 40 }}>
                      <div className="sec-header">
                        <span className="sec-tag">MISSION CONTROL · RAW ANALYSIS</span>
                        <h2 className="sec-title">Raw Analysis — Train &amp; Predict</h2>
                        <p className="sec-sub">Upload a CSV or JSON file — the model retrains on your data and predicts habitability for every row</p>
                      </div>
                      <div className="dashboard-3col">
                        <GalaxyNavigation activeSection={section} onNav={handleNav} />
                        <div style={{ gridColumn: '2 / -1' }}>
                          <RawDataPanel />
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}

                {/* ABOUT */}
                {section === 'about' && (
                  <motion.div key="about"
                    variants={pageVariants}
                    initial="initial" animate="animate" exit="exit"
                    transition={{ duration: 0.4 }}
                  >
                    <div className="dashboard-section" style={{ paddingTop: 40 }}>
                      <div className="dashboard-3col">
                        <GalaxyNavigation activeSection={section} onNav={handleNav} />
                        <div style={{ gridColumn: '2 / -1' }}>
                          <AboutSection />
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}

              </AnimatePresence>
            </main>
          </motion.div>
        )}

      </AnimatePresence>

    </div>
  );
}
