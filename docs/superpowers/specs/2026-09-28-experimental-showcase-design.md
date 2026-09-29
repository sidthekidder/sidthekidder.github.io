# Experimental Showcase Page — Design Spec

Date: 2026-09-28

## Purpose

A hidden, experimental page at `/new` on `sidthekidder.github.io`, separate
from the main Jekyll blog/portfolio. Its purpose is to be a personal
playground showcasing standalone, flashy, technically interesting web
demos/experiments/mini-games — not a rebuild of the existing site, and not
tied to existing real projects (`_projects`). Success looks like: a visually
striking landing experience with scroll-driven motion, leading into a grid
of independent experiment tiles, each demonstrating one specific piece of
cutting-edge browser tech.

"Hidden" means unlinked from site navigation — the page is publicly
reachable by URL but not advertised or discoverable through the existing
site.

## Scope

**V1 ships two experiments:**
1. Matrix-rain / generative shader art (raw WebGL2 fragment shader)
2. Audio-reactive visualizer (Web Audio API + Three.js)

**Deferred to later (not part of this spec/plan):**
- WebLLM in-browser chat
- Three.js mini-game
- Face/hand-tracking avatar
- WebXR mini-scene

The landing page and tile-grid structure must make adding these later a
matter of dropping in a new folder + tile, not restructuring existing code.

## Architecture

No build step. Plain HTML/CSS/JS using ES modules, importing Three.js and
GSAP from a CDN (esm.sh or unpkg). Rationale: only two demos in v1, so
npm/Vite tooling is overhead without payoff; CDN imports mean editing files
and refreshing, no build/deploy pipeline to maintain or keep in sync with
Jekyll's own build.

Everything lives under a new `/new/` folder at the repo root. Jekyll
already copies non-underscore top-level folders through untouched during
its own build, so this deploys automatically via the existing GitHub Pages
pipeline — no CI/config changes required. Confirm during implementation
that `_config.yml` has no `exclude`/`include` rule that would interfere.

## File structure

```
/new/
  index.html          — landing page: hero + tile grid
  style.css            — shared visual system (dark, bold type, neon/gradient accent)
  js/
    hero.js            — scroll-driven entrance animation (GSAP ScrollTrigger)
    tiles.js           — grid reveal-on-scroll + hover/click-to-open behavior
  experiments/
    matrix/
      index.html       — standalone page for the shader demo
      matrix.js
    audio-visualizer/
      index.html       — standalone page for the audio-reactive demo
      visualizer.js
```

Each experiment is a fully standalone page under `experiments/<name>/`,
reached by clicking its tile from the landing grid. This isolation means a
bug or performance issue in one demo cannot affect the landing page or any
other demo, and adding a future experiment is purely additive: new folder,
new tile, no changes to existing files.

## Components

### Landing page (`/new/index.html`, `hero.js`, `tiles.js`)
- Hero section with a scroll-driven entrance animation (GSAP ScrollTrigger),
  establishing the visual tone before the grid appears.
- Grid of experiment tiles below the hero, animating in on scroll
  (reveal-on-scroll), with hover feedback. Clicking a tile navigates to that
  experiment's standalone page.
- Visual language: dark background, bold oversized type, one neon/gradient
  accent, motion concentrated in the hero and tile transitions rather than
  scattered across the whole page (per 2026 trend of committing to one
  strong idea cleanly rather than stacking effects). Exact colors,
  typography, and animation timing to be finalized during implementation
  with the `frontend-design` skill rather than pinned in this spec.

### Matrix / generative shader demo (`experiments/matrix/`)
- Fullscreen WebGL2 fragment shader (raw GLSL, no Three.js — a single
  fullscreen quad is simplest here).
- Matrix-rain-style or generative pattern, reactive to mouse movement.

### Audio-reactive visualizer (`experiments/audio-visualizer/`)
- Web Audio API `AnalyserNode` reads frequency data from either microphone
  input (`getUserMedia`) or an uploaded audio file.
- Frequency data drives a Three.js scene (particles/bars/waveform reacting
  to the audio).
- If mic permission is denied or unavailable, the UI falls back to an
  audio-file upload input rather than presenting a dead/broken page.

## Data flow

No backend, no persistence. All state (audio analysis buffers, animation
state, shader uniforms) is ephemeral and client-side only for the lifetime
of the page.

## Error handling & compatibility

WebGL2 has broad support (~97%+ of browsers) so no feature-gating is
needed for the v1 demos. However, the tile/experiment pattern should be
built so each experiment independently checks its own requirements before
launching and shows a friendly "not supported in this browser/device"
message instead of crashing — this is what makes future experiments
(WebLLM/WebXR, which do need real feature detection for WebGPU/XR support)
drop in cleanly without reworking the landing page.

The audio visualizer specifically must handle a denied or unavailable
microphone by falling back to file upload, per above.

## Testing

This is a visual/creative page — no automated test suite. Verification is
manual: exercise both demos in Chrome, Safari, and Firefox on desktop plus
one mobile viewport; confirm reasonable frame rate; confirm the
mic-denied fallback actually works in the audio visualizer.

## Out of scope for this spec

- Any change to the existing Jekyll site, its nav, posts, or About/projects
  pages.
- The four deferred experiments listed above.
- Analytics, SEO, or any promotion of the `/new` page.
