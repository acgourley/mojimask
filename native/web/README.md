# The app's words, icons, brand, palettes and pictures

Everything here is read at run time (`mm_core::paths::web`; in the Mac app, `Contents/Resources/web`), except
`brand.json`, which is also built into the export code. Nothing is fetched while the app runs.

| file | what it holds |
|---|---|
| `strings.json` | every English word the app shows, one key each (`"scan.left": "about {eta} left"`) |
| `icons.json` | the app's icons: Lucide shapes (https://lucide.dev, ISC licence) as SVG path data |
| `brand.json` | the product's name and its wordmark |
| `palettes.json` | the emoji sets |
| `emoji_eyes.json` | where each emoji's eyes are, for lining covers up with faces |
| `fonts/` | Nunito (headings, the wordmark) and Inter (everything else), with their OFL licences |
| `assets/` | public-domain paintings: the drop card's gallery, the icon's painting, the default portrait |

## strings.json

- `{name}` placeholders are filled from the caller's values (`{app}` is always the brand's name).
- A plural is an object, `{"one": "1 person", "other": "{n} people"}`, picked by the value `n`; `"other"` is the
  fallback.
- Some strings end in `_html` and carry simple markup: keep the tags and translate around them. Key names on keyboard
  shortcuts (S, R, [, ⌘Z) stay as they are.
- For a translation, add a file with the same keys; a key missing there falls back to English.

## brand.json

Change the name's case here and nowhere else.

| key | |
|---|---|
| `name` | the name as people see it, in prose and in the wordmark |
| `split` | where the wordmark changes colour: `name[..split]` in `first`, the rest in `second` |
| `font` | the wordmark's typeface: a file in `fonts/<font>.woff2` |
| `weight` | its weight (Nunito is variable: 200-1000) |
| `tracking` | letter-spacing, in em, so every size is spaced alike; negative is tighter |
| `first`, `second` | the two parts' colours |

Not branding, and not changed from here: the bundle id, the `MOJIMASK_*` variables, the folders
(`~/Library/Application Support/Mojimask`) and the repository's name.
