const tiles = document.querySelectorAll('[data-tile]');

tiles.forEach((tile, i) => {
  tile.style.transitionDelay = `${i * 0.08}s`;
});

if (typeof IntersectionObserver === 'undefined') {
  tiles.forEach((tile) => tile.classList.add('is-visible'));
} else {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.2 }
  );

  tiles.forEach((tile) => observer.observe(tile));
}
