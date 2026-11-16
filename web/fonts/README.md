# Fonts

The app's two typefaces, bundled so nothing is fetched while it runs. Both are under the SIL Open Font License 1.1
(`Nunito-OFL.txt`, `Inter-OFL.txt`: these notices ship with the app).

| file | what | from | SHA-256 |
|---|---|---|---|
| `Nunito.woff2` | Nunito, variable weight 200–1000, Latin | `@fontsource-variable/nunito` 5.3.0, `files/nunito-latin-wght-normal.woff2` | `ba344451eab25b217a165363b1982048a5e5830a0daf36577973955a04cac793` |
| `Nunito-ext.woff2` | the same, Latin Extended | `files/nunito-latin-ext-wght-normal.woff2` | `2c8d792869818ecb253a46bc3c63c7013df7aac2f69291c3c85e5cdc94160960` |
| `Inter.woff2` | Inter, variable weight 100–900, Latin | `@fontsource-variable/inter` 5.3.0, `files/inter-latin-wght-normal.woff2` | `3100e775e8616cd2611beecfa23a4263d7037586789b43f035236a2e6fbd4c62` |
| `Inter-ext.woff2` | the same, Latin Extended | `files/inter-latin-ext-wght-normal.woff2` | `34b9c504cab7a73e37b746343a449132e56cf7b5481af2cb81dc74dcff25c956` |

Fetched 2026-11-14 from `https://cdn.jsdelivr.net/npm/<package>@5.3.0/<file>`. Other scripts (Cyrillic, Greek,
Vietnamese) fall back to the Mac's own fonts. `theme.css` loads these; `splash.py` puts Nunito's Latin file inside
the opening page.
