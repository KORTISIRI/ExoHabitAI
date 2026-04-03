import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence, useMotionValue, useTransform } from 'framer-motion';

/* ─── Narrative script ────────────────────────────────────────────────────── */
const SCRIPT = [
  { id: 0,  delay: 600,   text: 'THE YEAR IS 2047.',                                                          tag: 'year'       },
  { id: 1,  delay: 1800,  text: 'Humanity has catalogued over 39,000 worlds beyond our solar system.',        tag: 'body'       },
  { id: 2,  delay: 3200,  text: 'Most are silent. Barren. Inhospitable to life as we know it.',               tag: 'body'       },
  { id: 3,  delay: 4600,  text: 'But hidden among the stars…',                                                tag: 'pause'      },
  { id: 4,  delay: 5800,  text: '…some worlds might not be.',                                                 tag: 'highlight'  },
  { id: 5,  delay: 7400,  text: '"Which of these worlds could harbour life?"',                                 tag: 'quote'      },
  { id: 6,  delay: 9000,  text: 'To answer that question — we built an AI.',                                  tag: 'body'       },
  { id: 7,  delay: 10400, text: 'Trained on stellar temperatures, orbital mechanics, and planetary physics.', tag: 'body-small' },
  { id: 8,  delay: 11600, text: 'This is',                                                                    tag: 'pre-title'  },
  { id: 9,  delay: 12400, text: 'ExoHabitAI',                                                                 tag: 'title'      },
  { id: 10, delay: 13400, text: 'Exoplanet Habitability Intelligence',                                        tag: 'subtitle'   },
];

/* ─── Typewriter hook ─────────────────────────────────────────────────────── */
function useTypewriter(text, startTyping, speed = 38) {
  const [displayed, setDisplayed] = useState('');
  useEffect(() => {
    if (!startTyping) { setDisplayed(''); return; }
    setDisplayed('');
    let i = 0;
    const tick = setInterval(() => {
      i++;
      setDisplayed(text.slice(0, i));
      if (i >= text.length) clearInterval(tick);
    }, speed);
    return () => clearInterval(tick);
  }, [text, startTyping, speed]);
  return displayed;
}

/* ─── Particle canvas ────────────────────────────────────────────────────── */
function StarField() {
  const ref = useRef(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let raf;
    const COUNT = 280;
    const stars = Array.from({ length: COUNT }, () => ({
      x:  Math.random() * window.innerWidth,
      y:  Math.random() * window.innerHeight,
      r:  Math.random() * 1.4 + 0.2,
      a:  Math.random() * 0.7 + 0.1,
      ph: Math.random() * Math.PI * 2,
      speed: Math.random() * 0.0006 + 0.0003,
    }));

    const resize = () => {
      canvas.width  = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    const draw = t => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      stars.forEach(s => {
        const tw = 0.3 + 0.7 * Math.sin(t * s.speed + s.ph);
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(200,220,255,${s.a * tw})`;
        ctx.fill();
      });
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', resize);
    };
  }, []);
  return <canvas ref={ref} className="story-canvas" />;
}

/* ─── Shooting star ──────────────────────────────────────────────────────── */
function ShootingStars() {
  const [stars, setStars] = useState([]);
  useEffect(() => {
    const spawn = () => {
      const id = Date.now() + Math.random();
      const startX = Math.random() * window.innerWidth * 0.6;
      const startY = Math.random() * window.innerHeight * 0.4;
      setStars(s => [...s, { id, x: startX, y: startY }]);
      setTimeout(() => setStars(s => s.filter(st => st.id !== id)), 1200);
    };
    const interval = setInterval(spawn, 2800);
    spawn();
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', overflow: 'hidden', zIndex: 2 }}>
      {stars.map(s => (
        <motion.div
          key={s.id}
          initial={{ x: s.x, y: s.y, opacity: 0, scaleX: 0 }}
          animate={{ x: s.x + 320, y: s.y + 160, opacity: [0, 1, 1, 0], scaleX: [0, 1, 1, 0] }}
          transition={{ duration: 1.1, ease: 'easeOut' }}
          style={{
            position: 'absolute',
            width: 90, height: 1.5,
            borderRadius: 2,
            background: 'linear-gradient(90deg, transparent, rgba(0,229,255,0.8), white)',
            transformOrigin: 'left center',
            rotate: 30,
          }}
        />
      ))}
    </div>
  );
}

/* ─── Nebula blobs ───────────────────────────────────────────────────────── */
function NebulaLayer() {
  return (
    <>
      <motion.div
        className="story-nebula n1"
        animate={{ scale: [1, 1.12, 1], x: [0, 30, 0], y: [0, 20, 0] }}
        transition={{ duration: 22, repeat: Infinity, ease: 'easeInOut' }}
      />
      <motion.div
        className="story-nebula n2"
        animate={{ scale: [1, 1.09, 1], x: [0, -25, 0], y: [0, -15, 0] }}
        transition={{ duration: 28, repeat: Infinity, ease: 'easeInOut' }}
      />
      <motion.div
        className="story-nebula n3"
        animate={{ scale: [1, 1.15, 1], rotate: [0, 20, 0] }}
        transition={{ duration: 18, repeat: Infinity, ease: 'easeInOut' }}
      />
      {/* Extra pink wisp for depth */}
      <motion.div
        style={{
          position: 'absolute', width: 380, height: 380,
          top: '60%', left: '10%',
          borderRadius: '50%', filter: 'blur(120px)', pointerEvents: 'none', zIndex: 1,
          background: 'radial-gradient(circle, rgba(255,0,180,0.09) 0%, transparent 70%)',
        }}
        animate={{ scale: [1, 1.2, 1], opacity: [0.6, 1, 0.6] }}
        transition={{ duration: 14, repeat: Infinity, ease: 'easeInOut' }}
      />
    </>
  );
}

/* ─── Orbit icon (for Register button) ──────────────────────────────────── */
function OrbitIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 22 22" fill="none">
      <circle cx="11" cy="11" r="4" fill="currentColor" opacity="0.9"/>
      <ellipse cx="11" cy="11" rx="10.5" ry="3.5" fill="none"
        stroke="currentColor" strokeWidth="1.2" opacity="0.5"
        transform="rotate(-30 11 11)"/>
    </svg>
  );
}

/* ─── Planet visual ──────────────────────────────────────────────────────── */
function FloatingPlanet() {
  return (
    <motion.div
      style={{
        position: 'absolute', right: '8%', top: '50%', transform: 'translateY(-50%)',
        pointerEvents: 'none', zIndex: 2,
      }}
      animate={{ y: [0, -18, 0] }}
      transition={{ duration: 6, repeat: Infinity, ease: 'easeInOut' }}
    >
      {/* Planet body */}
      <div style={{
        width: 180, height: 180, borderRadius: '50%',
        background: 'radial-gradient(circle at 35% 35%, #2a6bbf, #0a1240)',
        boxShadow: '0 0 60px rgba(0,100,255,0.25), inset -30px -10px 60px rgba(0,0,80,0.8)',
        position: 'relative', overflow: 'visible',
      }}>
        {/* Ring */}
        <div style={{
          position: 'absolute', top: '50%', left: '50%',
          width: 270, height: 40,
          borderRadius: '50%',
          border: '2px solid rgba(0,229,255,0.2)',
          background: 'transparent',
          transform: 'translate(-50%,-50%) rotateX(72deg)',
          boxShadow: '0 0 20px rgba(0,229,255,0.1)',
        }} />
        {/* Surface bands */}
        {[30, 50, 65, 80].map(top => (
          <div key={top} style={{
            position: 'absolute', left: 0, right: 0, top: `${top}%`, height: '3px',
            background: 'rgba(0,229,255,0.06)',
          }} />
        ))}
      </div>
    </motion.div>
  );
}

/* ─── CTA block ──────────────────────────────────────────────────────────── */
function CtaBlock({ onComplete }) {
  const [hoverLogin, setHoverLogin] = useState(false);
  const [hoverReg, setHoverReg]     = useState(false);

  return (
    <motion.div
      className="story-cta"
      initial={{ opacity: 0, y: 32 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8, ease: 'easeOut' }}
    >
      <p className="story-cta-label">Your mission awaits, Commander.</p>

      <div className="story-cta-divider">
        <span className="story-cta-divider-line" />
        <span className="story-cta-divider-text">CHOOSE YOUR ACCESS</span>
        <span className="story-cta-divider-line" />
      </div>

      <div className="story-auth-btns">
        {/* Sign In */}
        <motion.button
          className="story-auth-btn story-auth-btn--login"
          onClick={() => onComplete('login')}
          onHoverStart={() => setHoverLogin(true)}
          onHoverEnd={() => setHoverLogin(false)}
          whileHover={{ scale: 1.04, y: -3 }}
          whileTap={{ scale: 0.97 }}
        >
          <span className="story-btn-glow login-glow" />
          {/* Animated border shimmer on hover */}
          <AnimatePresence>
            {hoverLogin && (
              <motion.span
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                style={{
                  position: 'absolute', inset: 0, borderRadius: 10, pointerEvents: 'none',
                  background: 'linear-gradient(135deg, rgba(0,229,255,0.12), transparent 60%)',
                }}
              />
            )}
          </AnimatePresence>
          <span className="story-auth-btn-icon">
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
              <path d="M9 2a4 4 0 100 8 4 4 0 000-8zM2 16c0-3.314 3.134-6 7-6s7 2.686 7 6"
                stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"/>
            </svg>
          </span>
          <span className="story-auth-btn-text">
            <strong>SIGN IN</strong>
            <small>Already a Commander</small>
          </span>
        </motion.button>

        {/* Register */}
        <motion.button
          className="story-auth-btn story-auth-btn--register"
          onClick={() => onComplete('signup')}
          onHoverStart={() => setHoverReg(true)}
          onHoverEnd={() => setHoverReg(false)}
          whileHover={{ scale: 1.04, y: -3 }}
          whileTap={{ scale: 0.97 }}
        >
          <span className="story-btn-glow register-glow" />
          <AnimatePresence>
            {hoverReg && (
              <motion.span
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                style={{
                  position: 'absolute', inset: 0, borderRadius: 10, pointerEvents: 'none',
                  background: 'linear-gradient(135deg, rgba(124,58,237,0.15), transparent 60%)',
                }}
              />
            )}
          </AnimatePresence>
          <span className="story-auth-btn-icon"><OrbitIcon /></span>
          <span className="story-auth-btn-text">
            <strong>REGISTER</strong>
            <small>Join the Mission</small>
          </span>
        </motion.button>
      </div>

      <motion.button
        className="story-guest-link"
        onClick={() => onComplete('guest')}
        whileHover={{ opacity: 0.85 }}
      >
        Continue as Guest →
      </motion.button>
    </motion.div>
  );
}

/* ─── Main component ─────────────────────────────────────────────────────── */
export default function StoryPage({ onComplete }) {
  const [visibleCount, setVisibleCount]   = useState(0);
  const [done, setDone]                   = useState(false);
  const [typingIndex, setTypingIndex]     = useState(-1);
  const timersRef                         = useRef([]);

  // Mouse parallax for planet
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);
  const pX = useTransform(mouseX, [0, window.innerWidth],  [-18, 18]);
  const pY = useTransform(mouseY, [0, window.innerHeight], [-10, 10]);

  const handleMouseMove = useCallback(e => {
    mouseX.set(e.clientX);
    mouseY.set(e.clientY);
  }, [mouseX, mouseY]);

  useEffect(() => {
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, [handleMouseMove]);

  // Sequence timers
  useEffect(() => {
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];

    SCRIPT.forEach((line, i) => {
      const t = setTimeout(() => {
        setVisibleCount(i + 1);
        setTypingIndex(i);
        if (i === SCRIPT.length - 1) {
          timersRef.current.push(setTimeout(() => setDone(true), 1200));
        }
      }, line.delay);
      timersRef.current.push(t);
    });

    return () => timersRef.current.forEach(clearTimeout);
  }, []);

  const handleSkip = useCallback(() => {
    timersRef.current.forEach(clearTimeout);
    setVisibleCount(SCRIPT.length);
    setTypingIndex(SCRIPT.length - 1);
    setDone(true);
  }, []);

  // Display: when done, only show last 3 lines (title block) + CTA
  const displayLines = done
    ? SCRIPT.slice(-3)
    : SCRIPT.slice(0, visibleCount);

  return (
    <motion.div
      className="story-page"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0, scale: 0.97 }}
      transition={{ duration: 0.5 }}
      style={{ cursor: 'default' }}
    >
      {/* Layers */}
      <StarField />
      <ShootingStars />
      <NebulaLayer />

      {/* Scan line effect */}
      <div style={{
        position: 'absolute', inset: 0, zIndex: 3, pointerEvents: 'none',
        backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.04) 2px, rgba(0,0,0,0.04) 4px)',
      }} />

      {/* Letterbox bars */}
      <div className="cinema-bar top" />
      <div className="cinema-bar bottom" />

      {/* Floating planet (parallax) */}
      <motion.div style={{ x: pX, y: pY, position: 'absolute', right: '7%', top: '50%', marginTop: -90, zIndex: 2 }}>
        <FloatingPlanet />
      </motion.div>

      {/* Skip button */}
      {!done && (
        <motion.button
          className="story-skip-btn"
          onClick={handleSkip}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 2 }}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          SKIP INTRO ›
        </motion.button>
      )}

      {/* Progress bar */}
      <motion.div
        className="story-progress"
        initial={{ scaleX: 0 }}
        animate={{ scaleX: visibleCount / SCRIPT.length }}
        transition={{ duration: 0.5, ease: 'easeOut' }}
      />

      {/* Main story content */}
      <div className="story-content">
        <div className="story-lines">
          <AnimatePresence>
            {displayLines.map((line, i) => (
              <StoryLine
                key={done ? `done-${i}` : line.id}
                line={line}
                index={i}
                isTyping={!done && line.id === typingIndex}
                isDone={done}
              />
            ))}
          </AnimatePresence>
        </div>

        <AnimatePresence>
          {done && <CtaBlock onComplete={onComplete} />}
        </AnimatePresence>
      </div>
    </motion.div>
  );
}

/* ─── Individual story line ──────────────────────────────────────────────── */
function StoryLine({ line, index, isTyping, isDone }) {
  const typewriterText = useTypewriter(
    line.tag !== 'title' ? line.text : '',
    isTyping && line.tag !== 'title',
    35
  );

  const cls = {
    year:         'sl-year',
    body:         'sl-body',
    'body-small': 'sl-body-small',
    highlight:    'sl-highlight',
    quote:        'sl-quote',
    pause:        'sl-pause',
    'pre-title':  'sl-pre-title',
    title:        'sl-title',
    subtitle:     'sl-subtitle',
  };

  const displayText = (isTyping && line.tag !== 'title') ? typewriterText : line.text;

  // Title gets special animated reveal
  if (line.tag === 'title') {
    return (
      <motion.div
        className="sl-title story-line"
        initial={{ opacity: 0, scale: 0.85, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
      >
        <motion.span
          className="sl-cyan"
          animate={{ textShadow: ['0 0 30px rgba(0,229,255,0.5)', '0 0 60px rgba(0,229,255,0.9)', '0 0 30px rgba(0,229,255,0.5)'] }}
          transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut' }}
        >
          Exo
        </motion.span>
        <motion.span
          className="sl-cyan"
          animate={{ textShadow: ['0 0 30px rgba(0,229,255,0.5)', '0 0 60px rgba(0,229,255,0.9)', '0 0 30px rgba(0,229,255,0.5)'] }}
          transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut', delay: 0.1 }}
        >
          Habit
        </motion.span>
        <motion.span
          className="sl-purple"
          animate={{ textShadow: ['0 0 30px rgba(124,58,237,0.6)', '0 0 60px rgba(124,58,237,1)', '0 0 30px rgba(124,58,237,0.6)'] }}
          transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut', delay: 0.2 }}
        >
          AI
        </motion.span>
      </motion.div>
    );
  }

  // Quote gets left-border treatment
  if (line.tag === 'quote') {
    return (
      <motion.blockquote
        className="sl-quote story-line"
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.6, ease: 'easeOut' }}
      >
        <motion.span
          animate={{ borderColor: ['rgba(0,229,255,0.3)', 'rgba(0,229,255,0.7)', 'rgba(0,229,255,0.3)'] }}
          transition={{ duration: 3, repeat: Infinity }}
          style={{ display: 'block', borderLeft: '3px solid rgba(0,229,255,0.4)', paddingLeft: 20 }}
        >
          {displayText}
          {isTyping && <span className="story-cursor">|</span>}
        </motion.span>
      </motion.blockquote>
    );
  }

  // Highlight gets a glow pulse
  if (line.tag === 'highlight') {
    return (
      <motion.div
        className="sl-highlight story-line"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.6 }}
      >
        <motion.span
          animate={{ textShadow: ['0 0 20px rgba(0,229,255,0.4)', '0 0 50px rgba(0,229,255,0.8)', '0 0 20px rgba(0,229,255,0.4)'] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          {displayText}
          {isTyping && <span className="story-cursor">|</span>}
        </motion.span>
      </motion.div>
    );
  }

  return (
    <motion.div
      className={`${cls[line.tag] || 'sl-body'} story-line`}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.55, ease: 'easeOut' }}
    >
      {displayText}
      {isTyping && <span className="story-cursor">|</span>}
    </motion.div>
  );
}
