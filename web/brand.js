// The product's name and the look of its wordmark, in one place. Everything that shows the name reads it from here:
// the pages (this file: BRAND.name, t()'s {app}, data-brand elements, theme.css's .wordmark through the --brand-*
// properties set below), and the Python side (about.py parses the object below: the window and menu titles, the
// splash, the export's corner mark and its metadata). Change the name's case here and nowhere else.
//
//   name      the name as people see it, in prose and in the wordmark ("MojiMask", "Mojimask", "mojimask", ...)
//   split     where the wordmark changes colour: name[:split] in `first`, the rest in `second`
//   font      the wordmark's typeface: a family theme.css loads (@font-face), with its file in fonts/<font>.woff2
//   weight    its weight (Nunito is variable: 200-1000)
//   tracking  letter-spacing, in em (a fraction of the font size), so every size is spaced alike; negative is tighter
//   first, second   the two parts' colours
//
// The object below must stay plain JSON (double quotes, no comments, no trailing comma): Python reads it as JSON.
// Not branding, and not changed from here: the bundle (Mojimask.app, org.mojimask.app), the MOJIMASK_* variables,
// the folders (~/Library/Application Support/Mojimask, ~/Library/Logs/Mojimask) and the repository's name.
const BRAND = {
  "name": "mojimask",
  "split": 4,
  "font": "Nunito",
  "weight": 900,
  "tracking": -0.01,
  "first": "#FFFFFF",
  "second": "#FFD23F"
};

// the wordmark's two parts, e.g. ["Moji", "Mask"]
function brandParts() { return [BRAND.name.slice(0, BRAND.split), BRAND.name.slice(BRAND.split)]; }

// The wordmark's look as CSS properties on the page (theme.css's .wordmark uses them). Set as this file loads, so load
// it in the page's <head>, after theme.css.
(function () {
  const s = document.documentElement.style;
  s.setProperty("--brand-font", `"${BRAND.font}", ui-rounded, "SF Pro Rounded", -apple-system, sans-serif`);
  s.setProperty("--brand-weight", String(BRAND.weight));
  s.setProperty("--brand-tracking", `${BRAND.tracking}em`);
  s.setProperty("--brand-first", BRAND.first);
  s.setProperty("--brand-second", BRAND.second);
})();

// Fills the page's brand elements (i18n() calls this):
//   <span class="wordmark" data-brand="wordmark"></span>   the two-colour wordmark (Moji<b>Mask</b>)
//   <title data-brand="name"></title>                      the plain name
// A wordmark marked as a window drag region (pywebview-drag-region) passes that on to its <b>, so it drags too.
function brandFill(root = document) {
  const [a, b] = brandParts();
  for (const el of root.querySelectorAll('[data-brand="name"]')) el.textContent = BRAND.name;
  for (const el of root.querySelectorAll('[data-brand="wordmark"]')) {
    const second = document.createElement("b");
    second.textContent = b;
    if (el.classList.contains("pywebview-drag-region")) second.className = "pywebview-drag-region";
    el.replaceChildren(document.createTextNode(a), second);
    el.setAttribute("aria-label", BRAND.name);
  }
}
