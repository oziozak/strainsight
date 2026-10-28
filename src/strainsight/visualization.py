"""Portable, interactive track playback using only NumPy and the standard library."""
import base64
import json
from pathlib import Path
import struct
from typing import TYPE_CHECKING
import zlib

import numpy as np

if TYPE_CHECKING:
    from .experiment import ExperimentResult


def _png_url(frame: np.ndarray) -> str:
    """Encode the rendered grayscale or RGB pixels without an imaging dependency."""
    height, width = frame.shape[:2]
    color_type = 0 if frame.ndim == 2 else 2

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (struct.pack("!I", len(payload)) + kind + payload
                + struct.pack("!I", zlib.crc32(kind + payload)))

    rows = b"".join(b"\0" + row.tobytes() for row in frame)
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, color_type, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def write_tracking_view(result: "ExperimentResult", path: str | Path) -> Path:
    """Save frames and measurements together for offline playback, without re-tracking."""
    frames = result.sequence.frames
    positions = result.tracks.positions
    safe_positions = positions.astype(object)
    safe_positions[~np.isfinite(positions)] = None
    payload = {
        "run_id": result.record.run_id,
        "tracker": result.record.configuration["tracker"],
        "query_count": len(result.query_points),
        "width": frames.shape[2], "height": frames.shape[1],
        "frames": [_png_url(frame) for frame in frames],
        "queries": result.query_points.tolist(),
        "positions": safe_positions.tolist(),
        "visibility": result.tracks.visibility.tolist(),
    }
    # Script data must not be able to terminate its HTML element. Missing
    # positions become JSON null, preserving losses without inventing a position.
    data = json.dumps(payload, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_HTML.replace("__TRACKING_DATA__", data), encoding="utf-8")
    return path


_HTML = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StrainSight · Track playback</title>
<style>
  :root { color-scheme: dark; font-family: system-ui, sans-serif; background: #10151b; color: #e8eef4; }
  body { max-width: 1050px; margin: 32px auto; padding: 0 20px; }
  h1 { font-size: 24px; margin-bottom: 8px; }
  p { color: #afbdca; line-height: 1.5; }
  canvas { display: block; width: 100%; background: #202832; border-radius: 8px; }
  .controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin: 16px 0; }
  button, select { padding: 8px 12px; border: 1px solid #526274; border-radius: 6px; background: #243242; color: inherit; }
  button, input, select, label { cursor: pointer; }
  input[type=range] { flex: 1; min-width: 150px; accent-color: #66d9c4; }
  #status { font-variant-numeric: tabular-nums; color: #b9eee0; }
  #run { overflow-wrap: anywhere; }
</style>
<h1>Track playback</h1>
<p id="run"></p>
<canvas id="view" aria-label="Video frames with tracked query points and trajectories"></canvas>
<div class="controls">
  <button id="play" disabled>Loading…</button>
  <button id="previous" aria-label="Previous frame">←</button>
  <input id="frame" type="range" min="0" value="0" step="1" aria-label="Frame">
  <button id="next" aria-label="Next frame">→</button>
  <label>Speed <select id="speed"><option value="4">4 fps</option><option value="8" selected>8 fps</option><option value="16">16 fps</option><option value="32">32 fps</option></select></label>
</div>
<div class="controls">
  <label><input id="points" type="checkbox" checked>Tracked points</label>
  <label><input id="trails" type="checkbox" checked>Trails (last 8 frames)</label>
  <label><input id="reference" type="checkbox">Reference positions</label>
  <label><input id="lost" type="checkbox" checked>Invisible predictions</label>
</div>
<p id="status" role="status" aria-live="polite">Loading frames…</p>
<p>Each query point keeps its color throughout the sequence. Red crosses show invisible
predictions, which are excluded from strain estimates. Non-finite positions have no
marker. Trails break across losses. Only query points are shown; the tracker's
support grid is excluded.</p>
<p>Drag the frame slider to inspect a frame. Use ← / → to step and Space to play or pause.</p>
<script id="tracking-data" type="application/json">__TRACKING_DATA__</script>
<script>
const data = JSON.parse(document.getElementById('tracking-data').textContent);
const el = id => document.getElementById(id);
const canvas = el('view'), ctx = canvas.getContext('2d'), slider = el('frame');
canvas.width = data.width; canvas.height = data.height;
slider.max = data.frames.length - 1;
el('run').textContent = `${data.tracker} · ${data.query_count} query points · Run ${data.run_id}`;
let current = 0, playing = false, lastTime = null, ready = false, animation = null;
const colors = data.queries.map((_, i) => `hsl(${i * 137.508 % 360} 90% 65%)`);
const finite = p => p && p.every(v => v !== null && Number.isFinite(v));
const usable = (t, i) => data.visibility[t][i] && finite(data.positions[t][i]);
const images = data.frames.map(src => { const img = new Image(); img.src = src; return img; });
function line(a, b) { ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); }
function draw() {
  if (!ready) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(images[current], 0, 0);
  let visible = 0;
  for (let i = 0; i < data.query_count; i++) {
    const p = data.positions[current][i], valid = usable(current, i);
    if (valid) visible++;
    if (el('reference').checked) {
      const q = data.queries[i]; ctx.strokeStyle = '#dce5ee88'; ctx.lineWidth = 0.7;
      ctx.beginPath(); ctx.arc(q[0], q[1], 2.5, 0, Math.PI * 2); ctx.stroke();
    }
    if (el('trails').checked) {
      ctx.strokeStyle = colors[i]; ctx.lineWidth = 1; ctx.globalAlpha = 0.65;
      for (let t = Math.max(1, current - 7); t <= current; t++) {
        if (usable(t - 1, i) && usable(t, i)) line(data.positions[t - 1][i], data.positions[t][i]);
      }
      ctx.globalAlpha = 1;
    }
    if (valid && el('points').checked) {
      ctx.fillStyle = colors[i]; ctx.strokeStyle = '#10151b'; ctx.lineWidth = 0.5;
      ctx.beginPath(); ctx.arc(p[0], p[1], 1.8, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    } else if (!valid && finite(p) && el('lost').checked) {
      ctx.strokeStyle = '#ff5656'; ctx.lineWidth = 1.1;
      line([p[0]-2, p[1]-2], [p[0]+2, p[1]+2]);
      line([p[0]-2, p[1]+2], [p[0]+2, p[1]-2]);
    }
  }
  slider.value = current;
  el('status').textContent = `Frame ${current + 1} / ${images.length} · Visible ${visible} / ${data.query_count} · Lost ${data.query_count - visible}`;
}
function pause() {
  playing = false; lastTime = null; cancelAnimationFrame(animation); el('play').textContent = 'Play';
}
function toggle() {
  if (!ready) return;
  if (playing) { pause(); return; }
  playing = true; lastTime = null; el('play').textContent = 'Pause';
  animation = requestAnimationFrame(tick);
}
function tick(time) {
  if (!playing) return;
  if (lastTime === null) lastTime = time;
  if (time - lastTime >= 1000 / Number(el('speed').value)) {
    current = (current + 1) % images.length; lastTime = time; draw();
  }
  animation = requestAnimationFrame(tick);
}
function step(delta) { pause(); current = Math.max(0, Math.min(images.length - 1, current + delta)); draw(); }
el('play').onclick = toggle;
el('previous').onclick = () => step(-1); el('next').onclick = () => step(1);
slider.oninput = () => { pause(); current = Number(slider.value); draw(); };
for (const id of ['points', 'trails', 'reference', 'lost']) el(id).onchange = draw;
document.addEventListener('keydown', event => {
  if (['INPUT', 'SELECT', 'BUTTON', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
  if (event.code === 'Space') { event.preventDefault(); toggle(); }
  if (event.code === 'ArrowLeft') { event.preventDefault(); step(-1); }
  if (event.code === 'ArrowRight') { event.preventDefault(); step(1); }
});
Promise.all(images.map(img => img.decode())).then(() => {
  ready = true; el('play').disabled = false; el('play').textContent = 'Play'; draw();
}).catch(() => { el('status').textContent = 'Could not decode video frames. Reload this file to retry.'; });
</script>
</html>
"""
