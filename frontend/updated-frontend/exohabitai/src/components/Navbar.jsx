import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export default function Navbar({ activeSection, onNav, user, onLogout }) {
  const [menuOpen, setMenuOpen] = useState(false);

  const links = [
    { id: 'home',    label: 'HOME' },
    { id: 'predict', label: 'PREDICT' },
    { id: 'results', label: 'RESULTS' },
    { id: 'history', label: 'HISTORY' },
    { id: 'batch',   label: 'BATCH' },
    { id: 'rawdata', label: 'RAW ANALYSIS' },
    { id: 'about',   label: 'ABOUT' },
  ];

  const handleNav = (id) => {
    onNav(id);
    setMenuOpen(false);
  };

  const avatar = user?.name ? user.name.charAt(0).toUpperCase() : '?';
  const displayName = user?.name
    ? (user.name.length > 14 ? user.name.slice(0, 12) + '…' : user.name)
    : 'Commander';

  return (
    <>
      <motion.header
        className="navbar"
        initial={{ y: -80, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6, ease: 'easeOut' }}
      >
        {/* Brand */}
        <button
          className="nav-brand"
          onClick={() => handleNav('home')}
          style={{ background: 'none', border: 'none', cursor: 'pointer' }}
        >
          <div className="nav-brand-icon">
            <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
              <circle cx="11" cy="11" r="4.5" fill="#00E5FF" opacity="0.9"/>
              <circle cx="11" cy="11" r="9" fill="none" stroke="#00E5FF"
                strokeWidth="1" strokeDasharray="3 2" opacity="0.4"/>
              <ellipse cx="11" cy="11" rx="10.5" ry="3.5" fill="none" stroke="#7C3AED"
                strokeWidth="1.2" transform="rotate(-25 11 11)" opacity="0.7"/>
              <circle cx="18" cy="7" r="1.5" fill="#E0E7FF" opacity="0.8"/>
            </svg>
          </div>
          <span className="nav-brand-name">
            ExoHabit<span>AI</span>
          </span>
        </button>

        {/* Desktop nav links */}
        <nav className="nav-links">
          {links.map(l => (
            <button
              key={l.id}
              className={`nav-link${activeSection === l.id ? ' active' : ''}`}
              onClick={() => handleNav(l.id)}
            >
              {l.label}
            </button>
          ))}
        </nav>

        {/* Right side */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginLeft: 'auto' }}>
          {/* User badge */}
          {user && (
            <motion.div
              className="nav-user-badge"
              initial={{ opacity: 0, x: 10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
            >
              <div className="nav-user-avatar">{avatar}</div>
              <span className="nav-user-name" style={{ display: window.innerWidth > 768 ? 'block' : 'none' }}>
                {displayName}
              </span>
            </motion.div>
          )}

          {/* System status */}
          <div className="nav-status">
            <span className="status-dot" />
            <span>ONLINE</span>
          </div>

          {/* Logout */}
          {user && onLogout && (
            <button className="nav-logout-btn" onClick={onLogout} title="Sign out">
              EXIT
            </button>
          )}

          {/* Hamburger */}
          <button className="hamburger" onClick={() => setMenuOpen(v => !v)}>
            <span style={{ transform: menuOpen ? 'rotate(45deg) translate(5px,5px)' : 'none', transition: 'all 0.3s' }} />
            <span style={{ opacity: menuOpen ? 0 : 1, transition: 'all 0.3s' }} />
            <span style={{ transform: menuOpen ? 'rotate(-45deg) translate(5px,-5px)' : 'none', transition: 'all 0.3s' }} />
          </button>
        </div>
      </motion.header>

      {/* Mobile menu */}
      <AnimatePresence>
        {menuOpen && (
          <motion.nav
            className="mobile-nav-menu"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
          >
            {links.map(l => (
              <button
                key={l.id}
                className={`nav-link${activeSection === l.id ? ' active' : ''}`}
                onClick={() => handleNav(l.id)}
              >
                {l.label}
              </button>
            ))}
            {user && (
              <button
                className="nav-link"
                onClick={() => { onLogout(); setMenuOpen(false); }}
                style={{ color: 'rgba(239,68,68,0.7)', borderTop: '1px solid rgba(0,229,255,0.08)', marginTop: 8, paddingTop: 16 }}
              >
                SIGN OUT
              </button>
            )}
          </motion.nav>
        )}
      </AnimatePresence>
    </>
  );
}
