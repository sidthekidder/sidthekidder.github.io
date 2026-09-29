import { gsap } from 'https://esm.sh/gsap@3.12.5';
import { ScrollTrigger } from 'https://esm.sh/gsap@3.12.5/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

const hero = document.querySelector('.hero');

if (hero) {
  gsap.to(hero, {
    opacity: 0.2,
    scale: 0.9,
    scrollTrigger: {
      trigger: hero,
      start: 'top top',
      end: 'bottom top',
      scrub: true,
    },
  });
}
