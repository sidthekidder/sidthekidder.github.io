import { gsap } from 'https://esm.sh/gsap@3.12.5';
import { ScrollTrigger } from 'https://esm.sh/gsap@3.12.5/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

const title = document.querySelector('.hero-title');
const subtitle = document.querySelector('.hero-subtitle');

if (title && subtitle) {
  gsap.set([title, subtitle], { opacity: 0, y: 40 });

  gsap
    .timeline({ defaults: { ease: 'power3.out' } })
    .to(title, { opacity: 1, y: 0, duration: 1 })
    .to(subtitle, { opacity: 1, y: 0, duration: 0.8 }, '-=0.5');

  gsap.to('.hero', {
    opacity: 0.2,
    scale: 0.9,
    scrollTrigger: {
      trigger: '.hero',
      start: 'top top',
      end: 'bottom top',
      scrub: true,
    },
  });
}
