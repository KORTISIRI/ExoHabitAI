import { motion } from 'framer-motion';

const navItems = [
  {
    id: 'predict',
    emoji: '🌍',
    label: 'Predict',
    sub: 'Analyze System',
    cls: 'predict',
  },
  {
    id: 'results',
    emoji: '🔭',
    label: 'Results',
    sub: 'View Analysis',
    cls: 'results',
  },
  {
    id: 'history',
    emoji: '🪐',
    label: 'History',
    sub: 'Mission Log',
    cls: 'history',
  },
  {
    id: 'batch',
    emoji: '📂',
    label: 'Batch Upload',
    sub: 'CSV / JSON File',
    cls: 'batch',
  },
  {
    id: 'rawdata',
    emoji: '🧬',
    label: 'Raw Analysis',
    sub: 'Train & Predict',
    cls: 'rawdata',
  },
  {
    id: 'about',
    emoji: '⭐',
    label: 'About',
    sub: 'Project Info',
    cls: 'about',
  },
];

export default function GalaxyNavigation({ activeSection, onNav }) {
  return (
    <aside className="galaxy-nav glass-panel">
      <div className="panel-corner tl" />
      <div className="panel-corner tr" />
      <div className="panel-corner bl" />
      <div className="panel-corner br" />

      <div className="galaxy-nav-inner">
        <div className="galaxy-nav-title">// GALAXY NAV</div>

        {navItems.map((item, i) => (
          <motion.button
            key={item.id}
            className={`planet-nav-btn ${item.cls}${activeSection === item.id ? ' active' : ''}`}
            onClick={() => onNav(item.id)}
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 + i * 0.08, duration: 0.4 }}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <div className="planet-orb">{item.emoji}</div>
            <div className="planet-nav-text">
              <strong>{item.label}</strong>
              <small>{item.sub}</small>
            </div>
            {activeSection === item.id && (
              <motion.div
                className="nav-active-bar"
                layoutId="activeBar"
                transition={{ type: 'spring', stiffness: 400, damping: 30 }}
              />
            )}
          </motion.button>
        ))}
      </div>
    </aside>
  );
}
