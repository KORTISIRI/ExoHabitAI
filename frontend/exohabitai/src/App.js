import { useState, useCallback } from 'react';
import { AnimatePresence, motion } from 'framer-motion';

import './App.css';

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
import RawDataAnalysisPanel from './components/RawDataAnalysisPanel';

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
  // Always start with the story on every page load / refresh.
  // After story completes: go to dashboard if already logged in, else auth.
  const [appFlow, setAppFlow] = useState('story');

  const [user, setUser] = useState(() => {
    const s = localStorage.getItem('exo_session');
    return s ? JSON.parse(s) : null;
  });

  // ── Dashboard state
  const [section, setSection] = useState('home');
  const [loading, setLoading] = useState(false);
  const [loadPct, setLoadPct] = useState(0);
  const [loadMsg, setLoadMsg] = useState('Initializing...');
  const [result,  setResult]  = useState(null);

  /* ── handlers ────────────────────────────────── */
  const handleStoryDone = useCallback(() => {
    // If already logged in, skip auth and go straight to dashboard
    const session = localStorage.getItem('exo_session');
    setAppFlow(session ? 'dashboard' : 'auth');
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
          <AuthPage key="auth" onAuth={handleAuth} />
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

                {/* RAW DATA ANALYSIS */}
                {section === 'rawdata' && (
                  <motion.div key="rawdata"
                    variants={pageVariants}
                    initial="initial" animate="animate" exit="exit"
                    transition={{ duration: 0.4 }}
                  >
                    <div className="dashboard-section" style={{ paddingTop: 40 }}>
                      <div className="sec-header">
                        <span className="sec-tag">MISSION CONTROL · RAW DATA INTELLIGENCE</span>
                        <h2 className="sec-title">Raw Data Prediction &amp; Analysis</h2>
                        <p className="sec-sub">Upload or paste unprocessed astronomical data for instant ML-powered habitability assessment</p>
                      </div>
                      <div className="dashboard-3col">
                        <GalaxyNavigation activeSection={section} onNav={handleNav} />
                        <div style={{ gridColumn: '2 / -1' }}>
                          <RawDataAnalysisPanel />
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
