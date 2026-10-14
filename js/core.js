/* ============================================================
   core.js — shared namespace, number formatting, RNG, CSV,
   synthetic datasets, and theme-colour lookup.
   Plain scripts (no modules) so the site also works when the
   HTML files are opened straight from disk.
   ============================================================ */
var MLV = (globalThis.MLV = globalThis.MLV || {});

/* ---------- number formatting ---------- */

MLV.MINUS = '−';

// Fixed-width-ish display of a number: unicode minus, exponent for
// very large / very small magnitudes so tables never blow out.
MLV.fmt = function (v, digits = 4) {
  if (v === undefined || v === null) return '–';
  if (Number.isNaN(v)) return 'NaN';
  if (!Number.isFinite(v)) return v > 0 ? '∞' : MLV.MINUS + '∞';
  const a = Math.abs(v);
  let s;
  if (a !== 0 && (a >= 1e5 || a < 1e-4)) s = v.toExponential(2);
  else s = v.toFixed(digits);
  if (/^-0\.?0*(e[+-]?\d+)?$/.test(s)) s = s.slice(1); // no "-0.0000"
  return s.replace('-', MLV.MINUS);
};

// Same, but wrapped in parentheses when negative — for "a × (−b)".
MLV.fmtP = function (v, digits = 4) {
  const s = MLV.fmt(v, digits);
  return v < 0 ? '(' + s + ')' : s;
};

// Signed delta, always with a leading + or −.
MLV.fmtDelta = function (v, digits = 4) {
  const s = MLV.fmt(v, digits);
  return v > 0 || (v === 0) ? '+' + s : s;
};

MLV.escape = function (s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
};

/* ---------- seeded RNG (mulberry32) ---------- */

MLV.rng = function (seed) {
  let a = (seed >>> 0) || 1;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

/* ---------- CSV (PapaParse when present, simple split otherwise) ---------- */

// Returns rows of exactly `numCols` numbers; header / malformed rows dropped.
MLV.parseNumericCSV = function (text, numCols) {
  const trimmed = (text || '').trim();
  if (!trimmed) return [];
  let raw;
  if (globalThis.Papa) raw = Papa.parse(trimmed, { skipEmptyLines: true }).data;
  else raw = trimmed.split(/\r?\n/).map((l) => l.split(/[,\t;]/));
  const rows = [];
  for (const r of raw) {
    if (r.length < numCols) continue;
    const nums = r.slice(0, numCols).map((v) => parseFloat(String(v).trim()));
    if (nums.some((v) => !Number.isFinite(v))) continue;
    rows.push(nums);
  }
  return rows;
};

MLV.downloadText = function (filename, text) {
  const blob = new Blob([text], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};

/* ---------- synthetic datasets ---------- */

MLV.datasets = {
  // y = m·x + b + noise, x in [0, 10]
  linear(n = 20, m = 2, b = 1, noise = 1.5, rand = Math.random) {
    const X = [], Y = [];
    for (let i = 0; i < n; i++) {
      const x = rand() * 10;
      X.push(x);
      Y.push(m * x + b + (rand() - 0.5) * 2 * noise);
    }
    return { X, Y };
  },

  // two roughly separable clouds, labels 0 / 1
  blobs(n = 40, spread = 1.1, rand = Math.random) {
    const X = [], Y = [];
    for (let i = 0; i < n; i++) {
      const c = i % 2;
      const cx = c ? 1 : -1, cy = c ? 1 : -1;
      X.push([cx + (rand() - 0.5) * 2 * spread, cy + (rand() - 0.5) * 2 * spread]);
      Y.push(c);
    }
    return { X, Y };
  },

  // four corner clusters, label = XOR of the quadrant signs
  xor(n = 60, noise = 0.45, rand = Math.random) {
    const X = [], Y = [];
    for (let i = 0; i < n; i++) {
      const sx = i % 2 ? 1 : -1;
      const sy = Math.floor(i / 2) % 2 ? 1 : -1;
      X.push([sx + (rand() - 0.5) * 2 * noise, sy + (rand() - 0.5) * 2 * noise]);
      Y.push(sx * sy < 0 ? 1 : 0);
    }
    return { X, Y };
  },

  // inner disc (class 0) inside a ring (class 1)
  circles(n = 60, rand = Math.random) {
    const X = [], Y = [];
    for (let i = 0; i < n; i++) {
      const c = i % 2;
      const r = c ? 1.3 + rand() * 0.5 : rand() * 0.7;
      const t = rand() * Math.PI * 2;
      X.push([r * Math.cos(t), r * Math.sin(t)]);
      Y.push(c);
    }
    return { X, Y };
  },
};

/* ---------- theme colours (read from CSS custom properties) ---------- */

MLV.css = function (name) {
  if (typeof document === 'undefined') return '#000';
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
};

MLV.seriesColor = function (slot) {
  return MLV.css('--series-' + ((slot % 4) + 1));
};

// hex "#rrggbb" -> [r, g, b]
MLV.hexToRgb = function (hex) {
  const h = hex.replace('#', '');
  const v = h.length === 3 ? h.split('').map((c) => c + c).join('') : h;
  return [0, 2, 4].map((i) => parseInt(v.slice(i, i + 2), 16));
};

MLV.mix = function (a, b, t) {
  return a.map((v, i) => Math.round(v + (b[i] - v) * t));
};
