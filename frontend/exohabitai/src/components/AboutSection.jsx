import { motion } from 'framer-motion';

const cards = [
  {
    icon: '🤖',
    title: 'ML Model',
    desc: 'Random Forest (Tuned) classifier trained on 39,315 exoplanets from the NASA Exoplanet Archive with AUC of 0.906.',
  },
  {
    icon: '🔬',
    title: 'Science',
    desc: 'Habitability criteria based on stellar temperature (4000–7000 K) and planetary radius (<2.0 R⊕) thresholds.',
  },
  {
    icon: '⚡',
    title: 'Backend',
    desc: 'Flask REST API serving predictions via POST /predict endpoint with full input validation and JSON responses.',
  },
  {
    icon: '📊',
    title: 'Performance',
    desc: 'F1 Score: 0.71 · Recall: 0.71 · Precision: 0.71 · Cross-validation stable across 5 folds.',
  },
  {
    icon: '🌌',
    title: 'Data Source',
    desc: 'NASA Exoplanet Archive — the authoritative catalog of confirmed exoplanets with 16 selected features.',
  },
  {
    icon: '🚀',
    title: 'Frontend',
    desc: 'React.js + Framer Motion + Three.js for an immersive space simulation UI with real-time 3D visualization.',
  },
];

export default function AboutSection() {
  return (
    <motion.div
      className="about-section"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.4 }}
    >
      <div className="sec-header">
        <span className="sec-tag">ABOUT THE PROJECT</span>
        <h2 className="sec-title">ExoHabitAI</h2>
        <p className="sec-sub">Infosys Internship Project · 2026</p>
      </div>

      <div className="about-grid">
        {cards.map((card, i) => (
          <motion.div
            key={card.title}
            className="about-card"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 + i * 0.08 }}
            whileHover={{ y: -6, transition: { duration: 0.2 } }}
          >
            <div className="panel-corner tl" style={{ width: 10, height: 10, top: 4, left: 4 }} />
            <div className="panel-corner tr" style={{ width: 10, height: 10, top: 4, right: 4 }} />
            <span className="about-card-icon">{card.icon}</span>
            <h3>{card.title}</h3>
            <p>{card.desc}</p>
          </motion.div>
        ))}
      </div>

      {/* Tech stack */}
      <motion.div
        style={{ marginTop: 48, textAlign: 'center' }}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7 }}
      >
        <span className="sec-tag">TECHNOLOGY STACK</span>
        <div style={{ display: 'flex', gap: 12, justifyContent: 'center', flexWrap: 'wrap', marginTop: 20 }}>
          {['React.js', 'Framer Motion', 'Three.js', 'Flask', 'scikit-learn', 'Python', 'NASA Archive'].map(tech => (
            <span key={tech} style={{
              padding: '6px 16px',
              border: '1px solid rgba(0,229,255,0.2)',
              borderRadius: 100,
              fontSize: '0.78rem',
              color: 'var(--neon-cyan)',
              background: 'rgba(0,229,255,0.05)',
              fontFamily: "'Share Tech Mono', monospace",
              letterSpacing: 1,
            }}>
              {tech}
            </span>
          ))}
        </div>
      </motion.div>

      {/* Footer-style attribution */}
      <motion.div
        style={{
          marginTop: 60,
          padding: '24px',
          background: 'rgba(0,0,0,0.2)',
          borderRadius: 12,
          border: '1px solid rgba(0,229,255,0.08)',
          textAlign: 'center',
        }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.9 }}
      >
        <span style={{ fontFamily: "'Orbitron', sans-serif", fontSize: '1rem', fontWeight: 700 }}>
          Exo<span style={{ color: 'var(--neon-cyan)' }}>Habit</span>
          <span style={{ color: 'var(--purple-glow)' }}>AI</span>
        </span>
        <br />
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', letterSpacing: 2 }}>
          INFOSYS INTERNSHIP · 2026 · NASA EXOPLANET ARCHIVE · RANDOM FOREST · FLASK API · AUC 0.906
        </span>
      </motion.div>
    </motion.div>
  );
}
