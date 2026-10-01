const canvas = document.getElementById('particles');
const ctx = canvas.getContext('2d');
const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const COLORS = ['124, 252, 174', '255, 79, 216'];

let width = 0;
let height = 0;
let particles = [];

function resize() {
  width = canvas.width = window.innerWidth;
  height = canvas.height = window.innerHeight;
}

function makeParticles(count) {
  return Array.from({ length: count }, () => ({
    x: Math.random() * width,
    y: Math.random() * height,
    r: 1 + Math.random() * 2.5,
    speed: 0.15 + Math.random() * 0.35,
    drift: (Math.random() - 0.5) * 0.3,
    color: COLORS[Math.floor(Math.random() * COLORS.length)],
    alpha: 0.15 + Math.random() * 0.35,
  }));
}

function draw() {
  ctx.clearRect(0, 0, width, height);
  particles.forEach((p) => {
    p.y -= p.speed;
    p.x += p.drift;
    if (p.y < -10) {
      p.y = height + 10;
      p.x = Math.random() * width;
    }
    if (p.x < -10) p.x = width + 10;
    if (p.x > width + 10) p.x = -10;

    ctx.beginPath();
    ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(${p.color}, ${p.alpha})`;
    ctx.shadowBlur = 8;
    ctx.shadowColor = `rgba(${p.color}, ${p.alpha})`;
    ctx.fill();
  });
  requestAnimationFrame(draw);
}

resize();
window.addEventListener('resize', resize);

if (ctx && !prefersReducedMotion) {
  const density = Math.min(70, Math.floor((width * height) / 18000));
  particles = makeParticles(density);
  requestAnimationFrame(draw);
}
