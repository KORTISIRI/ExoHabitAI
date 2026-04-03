import { useEffect, useRef } from 'react';

export default function GalaxyBackground() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let animId;
    let stars = [];
    const N = 300;

    function resize() {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      createStars();
    }

    function createStars() {
      stars = Array.from({ length: N }, () => ({
        x:     Math.random() * canvas.width,
        y:     Math.random() * canvas.height,
        r:     Math.random() * 1.5 + 0.2,
        alpha: Math.random() * 0.8 + 0.1,
        speed: Math.random() * 0.3 + 0.05,
        phase: Math.random() * Math.PI * 2,
        drift: (Math.random() - 0.5) * 0.08,
      }));
    }

    function draw(t) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Shooting stars occasionally
      if (Math.random() < 0.003) {
        const sx = Math.random() * canvas.width;
        const sy = Math.random() * canvas.height * 0.5;
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(sx + 80, sy + 30);
        const grad = ctx.createLinearGradient(sx, sy, sx + 80, sy + 30);
        grad.addColorStop(0, 'rgba(0,229,255,0.8)');
        grad.addColorStop(1, 'transparent');
        ctx.strokeStyle = grad;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      stars.forEach(s => {
        const tw = 0.5 + 0.5 * Math.sin(t * 0.001 * s.speed * 5 + s.phase);
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);

        // occasional cyan tinted stars
        const isCyan = s.r > 1.2;
        ctx.fillStyle = isCyan
          ? `rgba(180,240,255,${s.alpha * tw})`
          : `rgba(224,231,255,${s.alpha * tw})`;
        ctx.fill();

        s.x += s.drift;
        if (s.x < 0) s.x = canvas.width;
        if (s.x > canvas.width) s.x = 0;
      });

      animId = requestAnimationFrame(draw);
    }

    resize();
    animId = requestAnimationFrame(draw);
    window.addEventListener('resize', resize);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return (
    <div className="galaxy-bg">
      <canvas ref={canvasRef} className="galaxy-canvas" />
      <div className="nebula nebula-1" />
      <div className="nebula nebula-2" />
      <div className="nebula nebula-3" />
      <div className="grid-overlay" />
    </div>
  );
}
