# Vendored browser assets

Written by `scripts/vendor_assets.py`. Do not edit these
files by hand: rerun the script to bump a version, and
commit the diff.

They are vendored rather than loaded from a CDN so that
every page works with no outbound network, and so that no
third-party origin gets to run code next to the model and
deletion APIs. See audit finding TRUST-02.

| File | Bytes | SHA-256 | Source |
|---|---|---|---|
| `chart.js.LICENSE.md` | 1093 | `41a84aa2caba645f966a18d9c2056b73e6d3a81d80bc0046bc0011a2634d4cce` | https://cdn.jsdelivr.net/npm/chart.js@4.4.7/LICENSE.md |
| `chart.umd.min.js` | 205889 | `206b6e8bb00fc7bba2c7ee80ca41db3e9e05ba7be0aa35abeba9cfd5357f5d0e` | https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js |
| `chartjs-plugin-zoom.LICENSE.md` | 1104 | `faf50ba4a21e0c740c96e9bbce1b862273cfe5dd2cdf200d3271a781094c6ba9` | https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.2.0/LICENSE.md |
| `chartjs-plugin-zoom.min.js` | 15203 | `e4a088e5bab93be6ee47c939eeb9ebaa80e0b39156d4bdfd1af9c844be81b6c4` | https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.2.0/dist/chartjs-plugin-zoom.min.js |
| `fonts/OFL.txt` | 4399 | `30f0c136e3c88e422d0791acd97238870f9054a9729bc34cf2ff0d4ed8cac4ad` | https://raw.githubusercontent.com/JetBrains/JetBrainsMono/v2.304/OFL.txt |
| `fonts/jetbrains-mono-cyrillic-ext.woff2` | 2028 | `593ccd6fe36e299e151aeae4521be9bfe83ac487d96d7cdb44f50be4a04041c7` | https://fonts.gstatic.com/s/jetbrainsmono/v24/tDbV2o-flEEny0FZhsfKu5WU4xD2OwG_TA.woff2 |
| `fonts/jetbrains-mono-cyrillic.woff2` | 12108 | `d274604c40757f98f07df623a8b8fb5194508a6fa4f5413a7a107f09e3b7452a` | https://fonts.gstatic.com/s/jetbrainsmono/v24/tDbV2o-flEEny0FZhsfKu5WU4xD_OwG_TA.woff2 |
| `fonts/jetbrains-mono-greek.woff2` | 9004 | `ee863198077f3093ffa81c072581cc4bd608a04e60a8a9c0b846396ce4f14109` | https://fonts.gstatic.com/s/jetbrainsmono/v24/tDbV2o-flEEny0FZhsfKu5WU4xD4OwG_TA.woff2 |
| `fonts/jetbrains-mono-latin-ext.woff2` | 15196 | `79bfdab9ba467e26eea4122e6f2567e188dd8a09a8c730d501fc487c4ab99c6e` | https://fonts.gstatic.com/s/jetbrainsmono/v24/tDbV2o-flEEny0FZhsfKu5WU4xD1OwG_TA.woff2 |
| `fonts/jetbrains-mono-latin.woff2` | 40404 | `18be452724bfdc236c074ca94a249a7f41a86752c7d04ab258ce9ed5651f6a7e` | https://fonts.gstatic.com/s/jetbrainsmono/v24/tDbV2o-flEEny0FZhsfKu5WU4xD7OwE.woff2 |
| `fonts/jetbrains-mono-vietnamese.woff2` | 7504 | `2288795da89dd97e648954b96e22b4501bd85dfc453264460b252f888b582ca2` | https://fonts.gstatic.com/s/jetbrainsmono/v24/tDbV2o-flEEny0FZhsfKu5WU4xD0OwG_TA.woff2 |
| `hammer.js.LICENSE.md` | 1104 | `4d93df6544df47a49b25add3aff67ff9ff47e4756d85dd0c3e1beba8520ab9f2` | https://cdn.jsdelivr.net/npm/hammerjs@2.0.8/LICENSE.md |
| `hammer.min.js` | 20765 | `7953631f0e54794d2352a3cfa591c0914d73e14f90141058e3cf16bee7939bcf` | https://cdn.jsdelivr.net/npm/hammerjs@2.0.8/hammer.min.js |
