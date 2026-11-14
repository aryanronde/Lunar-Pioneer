# CLPS Lunar Mission Browser — Static Demo

This folder is a self-contained, static frontend demonstration for sharing the current Site06 dashboard concept.

## View locally

Open `index.html` directly in a browser. The CSS, JavaScript, map, and sample rows are all local files; the page does not call an API or request external assets.

## Deploy to Netlify

Use `netlify-demo` as the site base/publish directory, or drag and drop this folder into Netlify's static deploy flow. The included `netlify.toml` publishes the folder contents.

## Demo data note

The page uses a few saved representative values from the October 1, 2026 Site06 dashboard run. Its controls do not run SPICE or recompute geometry, terrain horizons, visibility, or mission windows. The footer states this limitation in the page.
