import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const STORY_LINES = [
  { delay: 600,   text: 'The year is 2047.', tag: 'year' },
  { delay: 2000,  text: 'Humanity has catalogued over 39,000 worlds beyond our solar system.', tag: 'body' },
  { delay: 4000,  text: 'Most are silent. Barren. Hostile to life as we know it.', tag: 'body' },
  { delay: 6200,  text: 'But some... might not be.', tag: 'highlight' },
  { delay: 8500,  text: 'A question echoes across observatories, labs, and mission control rooms:', tag: 'body' },
  { delay: 11000, text: '"Which worlds could harbour life?"', tag: 'quote' },
  { delay: 13500, text: 'To answer it — we built an AI.', tag: 'body' },
  { delay: 15500, text: 'Trained on stellar temperatures, orbital mechanics, and planetary mass.', tag: 'body-small' },
  { delay: 17500, text: 'Powered by Earth\'s most advanced machine learning models.', tag: 'body-small' },
  { delay: 19500, text: 'This is', tag: 'pre-title' },
  { delay: 20800, text: 'ExoHabitAI', tag: 'title' },
  { delay: 22500, text: 'Exoplanet Habitability Predictor', tag: 'subtitle' },
];

/* ── Stars canvas ───────────────────────────── */
function StarCanvas() {
  const ref = useRef(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let id;
    const stars = Array.from({ length: 220 }, () => ({
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
      r: Math.random() * 1.3 + 0.2,
      a: Math.random() * 0.6 + 0.15,
      ph: Math.random() * Math.PI * 2,
    }));
    const resize = () => {
      canvas.width  = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);
    const draw = (t) => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      stars.forEach(s => {
        const tw = 0.45 + 0.55 * Math.sin(t * 0.0008 + s.ph);
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(200,220,255,${s.a * tw})`;
        ctx.fill();
      });
      id = requestAnimationFrame(draw);
    };
    id = requestAnimationFrame(draw);
    return () => { cancelAnimationFrame(id); window.removeEventListener('resize', resize); };
  }, []);
  return <canvas ref={ref} className="story-canvas" />;
}

/* ── Main StoryPage ─────────────────────────── */
export default function StoryPage({ onComplete }) {
  // visibleCount drives which lines are showing
  const [visibleCount, setVisibleCount] = useState(0);
  const [done, setDone]                 = useState(false);  // all lines revealed + CTA visible
  const timersRef                       = useRef([]);

  // Schedule each line to appear
  const scheduleLines = () => {
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];
    STORY_LINES.forEach((line, i) => {
      const t = setTimeout(() => {
        setVisibleCount(i + 1);
        if (i === STORY_LINES.length - 1) {
          // small extra pause before showing CTA
          timersRef.current.push(setTimeout(() => setDone(true), 1200));
        }
      }, line.delay);
      timersRef.current.push(t);
    });
  };

  useEffect(() => {
    scheduleLines();
    return () => timersRef.current.forEach(clearTimeout);
  }, []); // eslint-disable-line

  // Skip: reveal all lines then go straight to auth after a brief flash
  const handleSkip = () => {
    timersRef.current.forEach(clearTimeout);
    setVisibleCount(STORY_LINES.length);
    setDone(true);
    // Auto-advance to auth after 600ms so user sees the title before transitioning
    setTimeout(() => onComplete(), 600);
  };

  const cls = {
    year:        'sl-year',
    body:        'sl-body',
    'body-small':'sl-body-small',
    highlight:   'sl-highlight',
    quote:       'sl-quote',
    'pre-title': 'sl-pre-title',
    title:       'sl-title',
    subtitle:    'sl-subtitle',
  };

  return (
    <motion.div
      className="story-page"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.35 }}
    >
      <StarCanvas />
      <div className="story-nebula n1" />
      <div className="story-nebula n2" />
      <div className="story-nebula n3" />
      <div className="cinema-bar top" />
      <div className="cinema-bar bottom" />

      {/* Story lines */}
      <div className="story-content">
        <div className="story-lines">
          {STORY_LINES.slice(0, visibleCount).map((line, i) => (
            <motion.div
              key={i}
              className={`story-line ${cls[line.tag] || 'sl-body'}`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              {line.tag === 'title'
                ? <>Exo<span className="sl-cyan">Habit</span><span className="sl-purple">AI</span></>
                : line.text
              }
            </motion.div>
          ))}
        </div>

        {/* CTA — only shown after all lines appear */}
        <AnimatePresence>
          {done && (
            <motion.div
              className="story-cta"
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
            >
              <p className="story-cta-label">Your mission awaits, Commander.</p>
              <button
                className="story-begin-btn"
                onClick={onComplete}
              >
                <span className="story-btn-glow" />
                <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
                  <path d="M9 2L16 9L9 16M2 9h14" stroke="currentColor" strokeWidth="1.8"
                    strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                INITIATE MISSION
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Skip button — always visible until CTA appears */}
      {!done && (
        <button className="story-skip-btn" onClick={handleSkip}>
          SKIP INTRO ›
        </button>
      )}

      {/* Progress bar */}
      <motion.div
        className="story-progress"
        initial={{ scaleX: 0 }}
        animate={{ scaleX: visibleCount / STORY_LINES.length }}
        transition={{ duration: 0.4 }}
      />
    </motion.div>
  );
}
