// Whether the reader has asked for less motion.
//
// Loaded as a classic global script on every page that animates: the
// generator, Analytics, the menu and Settings. A file of its own so a
// page can ask without loading the visual code, which the menu, the
// one among them that draws no tokens, has no other use for.

"use strict";

// Every page that animates anything has to ask this, so it lives
// here rather than being reimplemented per page. Unprefixed because
// it is a plain predicate about the environment, not part of the
// overlay model. Total: an environment without matchMedia is treated
// as having no preference, which is the same answer a browser that
// does not know the query gives.
function prefersReducedMotion() {
  try {
    return window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;
  } catch (_e) {
    return false;
  }
}
