const words = document.querySelectorAll('.word');

function showWordsImmediately() {
  words.forEach((word) => word.classList.add('is-visible'));
}

try {
  const { gsap } = await import('https://esm.sh/gsap@3.12.5');
  const { ScrollTrigger } = await import('https://esm.sh/gsap@3.12.5/ScrollTrigger');
  gsap.registerPlugin(ScrollTrigger);

  const chapters = document.querySelectorAll('[data-chapter]');
  chapters.forEach((chapter, i) => {
    const isLast = i === chapters.length - 1;

    ScrollTrigger.create({
      trigger: chapter,
      start: 'top top',
      end: 'bottom top',
      pin: !isLast,
      pinSpacing: !isLast,
    });

    const chapterWords = chapter.querySelectorAll('.word');
    if (chapterWords.length) {
      gsap.fromTo(
        chapterWords,
        { opacity: 0, y: 30 },
        {
          opacity: 1,
          y: 0,
          stagger: 0.06,
          ease: 'power2.out',
          scrollTrigger: {
            trigger: chapter,
            start: 'top 70%',
            toggleActions: 'play none none reverse',
          },
        }
      );
    }
  });
} catch (err) {
  console.warn('Kinetic scroll enhancement unavailable, showing static content:', err);
  showWordsImmediately();
}
