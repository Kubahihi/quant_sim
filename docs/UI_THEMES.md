# Appearance contract

One source of truth: Streamlit's native Light/Dark choice. The sidebar component
selects the same native menu item as the user. It never reloads the page or
modifies React internals. Native persistence is browser-local; reported chart
state is session-local. The native System option remains available in the menu.

- `.streamlit/config.toml`: native widget/canvas palette.
- `ui/theme_tokens.toml`: additional semantic colors (custom HTML and figures).
- `ui/dashboard.css`: layout and custom classes; no global widget recoloring.
- `ui/theme_switch.js`: isolated native menu adapter. It depends on the menu
  test IDs in Streamlit 1.58–1.61; re-run browser acceptance before upgrading.
- `render_plotly_chart` / `render_matplotlib_chart`: theme copies, preserving
  original/cached figures and data. Native Streamlit charts stay native.

Never attempt to theme `st.dataframe` / `st.data_editor` by setting `--gdg-*`
or a CSS background on the canvas. Those cannot replace its painted pixels.
Never restore an independent Python toggle that only selects CSS colors.

## Verification

`python -m pytest tests/ui -q` checks palette contrast, adapters, data preservation,
login and existing UI behavior. AppTest is **not** a browser and cannot establish
that a table canvas or a popup is visually correct.

Run `python -m streamlit run tests/ui/theme_gallery.py` from the project root
for synthetic-data browser acceptance. Verify both modes, both directions:

1. Native editor and read-only table: header, rows, blank rows, editing overlay.
2. Number stepper, checkbox, multi/selectbox including open popup, expander.
3. Password/reveal, unsaved form text, primary/secondary/disabled buttons.
4. Plotly axes/annotations/3D, Matplotlib labels/legend, native line chart.
5. Native menu selection updates sidebar and server chart state; navigating
   between gallery pages preserves the selected mode.

For automation, install `playwright` in the test environment and use the installed
Edge browser. Set `THEME_GALLERY_URL=http://127.0.0.1:8501` (or the gallery's port),
then run `python -m pytest tests/ui/test_theme_browser.py -q -s`. The test checks
painted canvas pixels, form and edited-cell preservation, native menu sync and
browser persistence, and saves screenshots under pytest's temporary directory.

The fixture contains no authentication bypass and does not access user data.

## Applying the change

Restart Streamlit after updating: source watching is intentionally disabled in
this repository, and imported Python modules otherwise stay cached. Refreshing
only the browser is insufficient. Cloud must run the updated commit and restart.
For reproducible deployment install the pinned `requirements.txt` (1.61.1);
the implementation is also browser-checked with local Streamlit 1.58.0.
