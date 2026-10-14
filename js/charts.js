/* ============================================================
   charts.js — two canvas components:
     MLV.Plot2D     the data-space plot (points, fitted line,
                    decision field), click to add points
     MLV.LineChart  value-over-epochs chart for weight
                    trajectories and loss, with hover tooltip,
                    click-to-select-epoch and a selected-epoch marker
   Both size themselves from their container and redraw on resize,
   reading colours from CSS custom properties so dark mode works.
   ============================================================ */
var MLV = (globalThis.MLV = globalThis.MLV || {});

(function () {
  function setupCanvas(canvas, cssHeight) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.parentElement.clientWidth;
    canvas.style.width = w + 'px';
    canvas.style.height = cssHeight + 'px';
    canvas.width = Math.max(1, Math.round(w * dpr));
    canvas.height = Math.max(1, Math.round(cssHeight * dpr));
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h: cssHeight };
  }

  // "nice" tick step for a range
  function niceStep(range, target) {
    const raw = range / Math.max(1, target);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    const step = norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10;
    return step * mag;
  }

  function tickLabel(v, step) {
    const d = step >= 1 ? 0 : Math.min(4, Math.ceil(-Math.log10(step)));
    return MLV.fmt(v, d);
  }

  /* ======================================================
     Plot2D
     ====================================================== */
  class Plot2D {
    constructor(canvas, domain, height = 360) {
      this.canvas = canvas;
      this.domain = domain;
      this.height = height;
      this.pad = { l: 40, r: 14, t: 12, b: 28 };
      this.onDraw = null;
      const ro = new ResizeObserver(() => this.redraw());
      ro.observe(canvas.parentElement);
    }

    setDomain(d) { this.domain = d; }

    redraw() {
      // never taller than it is wide, so the plot keeps a sane aspect on phones
      const h = Math.min(this.height, Math.max(240, this.canvas.parentElement.clientWidth * 0.9));
      const s = setupCanvas(this.canvas, h);
      this.ctx = s.ctx; this.w = s.w; this.h = s.h;
      if (this.onDraw) this.onDraw();
    }

    toPx(x, y) {
      const { xMin, xMax, yMin, yMax } = this.domain, p = this.pad;
      return [
        p.l + ((x - xMin) / (xMax - xMin)) * (this.w - p.l - p.r),
        this.h - p.b - ((y - yMin) / (yMax - yMin)) * (this.h - p.t - p.b),
      ];
    }

    toData(px, py) {
      const { xMin, xMax, yMin, yMax } = this.domain, p = this.pad;
      return [
        xMin + ((px - p.l) / (this.w - p.l - p.r)) * (xMax - xMin),
        yMin + ((this.h - p.b - py) / (this.h - p.t - p.b)) * (yMax - yMin),
      ];
    }

    eventToData(evt) {
      const r = this.canvas.getBoundingClientRect();
      return this.toData(evt.clientX - r.left, evt.clientY - r.top);
    }

    clear() {
      this.ctx.fillStyle = MLV.css('--surface');
      this.ctx.fillRect(0, 0, this.w, this.h);
    }

    clip(fn) {
      const p = this.pad, c = this.ctx;
      c.save();
      c.beginPath();
      c.rect(p.l, p.t, this.w - p.l - p.r, this.h - p.t - p.b);
      c.clip();
      fn(c);
      c.restore();
    }

    axes() {
      const c = this.ctx, p = this.pad, { xMin, xMax, yMin, yMax } = this.domain;
      c.save();
      c.font = '11px "IBM Plex Mono", monospace';
      c.fillStyle = MLV.css('--text-muted');
      c.strokeStyle = MLV.css('--grid');
      c.lineWidth = 1;
      const xs = niceStep(xMax - xMin, 6), ys = niceStep(yMax - yMin, 5);
      for (let v = Math.ceil(xMin / xs) * xs; v <= xMax + 1e-9; v += xs) {
        const [px] = this.toPx(v, yMin);
        c.beginPath(); c.moveTo(px, p.t); c.lineTo(px, this.h - p.b); c.stroke();
        c.textAlign = 'center';
        c.fillText(tickLabel(v, xs), px, this.h - p.b + 16);
      }
      for (let v = Math.ceil(yMin / ys) * ys; v <= yMax + 1e-9; v += ys) {
        const [, py] = this.toPx(xMin, v);
        c.beginPath(); c.moveTo(p.l, py); c.lineTo(this.w - p.r, py); c.stroke();
        c.textAlign = 'right';
        c.fillText(tickLabel(v, ys), p.l - 6, py + 4);
      }
      c.strokeStyle = MLV.css('--axis');
      if (yMin < 0 && yMax > 0) { const [, py] = this.toPx(xMin, 0); c.beginPath(); c.moveTo(p.l, py); c.lineTo(this.w - p.r, py); c.stroke(); }
      if (xMin < 0 && xMax > 0) { const [px] = this.toPx(0, yMin); c.beginPath(); c.moveTo(px, p.t); c.lineTo(px, this.h - p.b); c.stroke(); }
      c.restore();
    }

    point(x, y, fill, r = 5) {
      const [px, py] = this.toPx(x, y), c = this.ctx;
      c.beginPath();
      c.arc(px, py, r, 0, Math.PI * 2);
      c.fillStyle = fill;
      c.fill();
      c.lineWidth = 2;
      c.strokeStyle = MLV.css('--surface');
      c.stroke();
    }

    // Shade the plane by P(class 1): class-0 colour → neutral → class-1 colour.
    field(prob, res = 56) {
      const p = this.pad;
      const c0 = MLV.hexToRgb(MLV.css('--series-1'));
      const c1 = MLV.hexToRgb(MLV.css('--series-2'));
      const mid = MLV.hexToRgb(MLV.css('--neutral-mid'));
      const surf = MLV.hexToRgb(MLV.css('--surface'));
      const iw = this.w - p.l - p.r, ih = this.h - p.t - p.b;
      const cw = iw / res, ch = ih / res;
      for (let i = 0; i < res; i++) {
        for (let j = 0; j < res; j++) {
          const [x, y] = this.toData(p.l + (i + 0.5) * cw, p.t + (j + 0.5) * ch);
          let q = prob(x, y);
          if (!Number.isFinite(q)) q = 0.5;
          const pole = q < 0.5 ? MLV.mix(mid, c0, (0.5 - q) * 2) : MLV.mix(mid, c1, (q - 0.5) * 2);
          const col = MLV.mix(surf, pole, 0.32);
          this.ctx.fillStyle = `rgb(${col[0]},${col[1]},${col[2]})`;
          this.ctx.fillRect(p.l + i * cw, p.t + j * ch, cw + 0.6, ch + 0.6);
        }
      }
    }

    line(slope, intercept, color, width = 2.5, dash = []) {
      const { xMin, xMax } = this.domain;
      if (!Number.isFinite(slope) || !Number.isFinite(intercept)) return;
      this.clip((c) => {
        const [x1, y1] = this.toPx(xMin, slope * xMin + intercept);
        const [x2, y2] = this.toPx(xMax, slope * xMax + intercept);
        c.beginPath(); c.moveTo(x1, y1); c.lineTo(x2, y2);
        c.setLineDash(dash);
        c.strokeStyle = color; c.lineWidth = width; c.stroke();
      });
    }

    // straight segment in data coordinates (used for residuals)
    segment(x1, y1, x2, y2, color, width = 1.5, dash = [3, 3]) {
      this.clip((c) => {
        const [a, b] = this.toPx(x1, y1), [d, e] = this.toPx(x2, y2);
        c.beginPath(); c.moveTo(a, b); c.lineTo(d, e);
        c.setLineDash(dash);
        c.strokeStyle = color; c.lineWidth = width; c.stroke();
      });
    }

    // contour of a scalar field at level 0.5, via marching squares
    contour(prob, color, res = 70) {
      const p = this.pad, iw = this.w - p.l - p.r, ih = this.h - p.t - p.b;
      const g = [];
      for (let i = 0; i <= res; i++) {
        g.push([]);
        for (let j = 0; j <= res; j++) {
          const [x, y] = this.toData(p.l + (i / res) * iw, p.t + (j / res) * ih);
          g[i].push(prob(x, y) - 0.5);
        }
      }
      const c = this.ctx;
      c.save();
      c.strokeStyle = color; c.lineWidth = 2;
      c.beginPath();
      const px = (i) => p.l + (i / res) * iw, py = (j) => p.t + (j / res) * ih;
      const lerp = (a, b) => a / (a - b);
      for (let i = 0; i < res; i++) {
        for (let j = 0; j < res; j++) {
          const v = [g[i][j], g[i + 1][j], g[i + 1][j + 1], g[i][j + 1]];
          const pts = [];
          const corners = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]];
          for (let k = 0; k < 4; k++) {
            const a = v[k], b = v[(k + 1) % 4];
            if ((a > 0) !== (b > 0)) {
              const t = lerp(a, b);
              const [ci, cj] = corners[k], [di, dj] = corners[(k + 1) % 4];
              pts.push([px(ci + (di - ci) * t), py(cj + (dj - cj) * t)]);
            }
          }
          for (let k = 0; k + 1 < pts.length; k += 2) {
            c.moveTo(pts[k][0], pts[k][1]); c.lineTo(pts[k + 1][0], pts[k + 1][1]);
          }
        }
      }
      c.stroke();
      c.restore();
    }
  }
  MLV.Plot2D = Plot2D;

  /* ======================================================
     LineChart — values over epochs
     ====================================================== */
  class LineChart {
    constructor(container, opts = {}) {
      this.container = container;
      container.classList.add('linechart');
      this.canvas = document.createElement('canvas');
      this.tip = document.createElement('div');
      this.tip.className = 'chart-tip';
      this.tip.hidden = true;
      container.appendChild(this.canvas);
      container.appendChild(this.tip);
      this.height = opts.height || 240;
      this.onSelect = opts.onSelect || null;
      this.directLabels = opts.directLabels !== false;
      this.emptyText = opts.emptyText || 'Step an epoch to start drawing';
      this.series = [];
      this.marker = null;
      this.focusKey = null;
      this.hoverEpoch = null;
      this.pad = { l: 56, r: 12, t: 10, b: 28 };

      new ResizeObserver(() => this.draw()).observe(container);
      this.canvas.addEventListener('pointermove', (e) => this.onMove(e));
      this.canvas.addEventListener('pointerleave', () => { this.hoverEpoch = null; this.hoverKey = null; this.tip.hidden = true; this.draw(); });
      this.canvas.addEventListener('click', (e) => {
        const ep = this.epochAt(e);
        if (ep !== null && this.onSelect) this.onSelect(ep);
      });
    }

    // series: [{ key, label, color, values: number[] , hidden }]
    setData(series, marker) {
      this.series = series;
      this.marker = marker;
      this.draw();
    }

    setFocus(key) { this.focusKey = key; this.draw(); }

    visible() { return this.series.filter((s) => !s.hidden); }

    count() { return this.series.length ? this.series[0].values.length : 0; }

    labelsOn() { return this.directLabels && this.visible().length <= 4 && this.visible().length > 1; }

    geom() {
      const n = this.count();
      const vis = this.visible();
      let lo = 0, hi = 0; // always include zero so sign changes are visible
      for (const s of vis) for (const v of s.values) if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
      if (hi - lo < 1e-9) { hi += 1; lo -= hi === 1 ? 0 : 1; }
      const span = hi - lo;
      lo -= span * 0.06; hi += span * 0.08;
      if (lo > 0) lo = 0;
      const right = this.labelsOn() ? 92 : this.pad.r;
      return { n, lo, hi, l: this.pad.l, r: this.w - right, t: this.pad.t, b: this.h - this.pad.b };
    }

    xPx(g, i) { return g.l + (g.n <= 1 ? 0 : (i / (g.n - 1)) * (g.r - g.l)); }
    yPx(g, v) { return g.b - ((v - g.lo) / (g.hi - g.lo)) * (g.b - g.t); }

    epochAt(e) {
      const n = this.count();
      if (!n) return null;
      const g = this.geom();
      const r = this.canvas.getBoundingClientRect();
      const x = e.clientX - r.left;
      if (n === 1) return 0;
      const i = Math.round(((x - g.l) / (g.r - g.l)) * (n - 1));
      return Math.max(0, Math.min(n - 1, i));
    }

    onMove(e) {
      const ep = this.epochAt(e);
      if (ep === null) return;
      const g = this.geom();
      const r = this.canvas.getBoundingClientRect();
      const my = e.clientY - r.top;
      // nearest visible line to the pointer at this epoch
      let best = null, bd = Infinity;
      for (const s of this.visible()) {
        const v = s.values[ep];
        if (!Number.isFinite(v)) continue;
        const d = Math.abs(this.yPx(g, v) - my);
        if (d < bd) { bd = d; best = s.key; }
      }
      this.hoverEpoch = ep;
      this.hoverKey = this.visible().length > 4 && bd < 18 ? best : null;
      this.draw();
      this.renderTip(ep, e.clientX - r.left, my);
    }

    renderTip(ep, mx, my) {
      const vis = this.visible();
      if (!vis.length) { this.tip.hidden = true; return; }
      let rows = vis.map((s) => ({ s, v: s.values[ep] }));
      rows.sort((a, b) => (b.v ?? -Infinity) - (a.v ?? -Infinity));
      if (this.hoverKey) {
        const k = rows.findIndex((r) => r.s.key === this.hoverKey);
        if (k > 0) rows.unshift(rows.splice(k, 1)[0]);
      }
      const more = rows.length > 10 ? rows.length - 10 : 0;
      rows = rows.slice(0, 10);
      this.tip.innerHTML =
        `<div class="tip-head">epoch ${ep}</div>` +
        rows.map((r) =>
          `<div class="tip-row${r.s.key === this.hoverKey ? ' is-hot' : ''}"><span class="swatch" style="background:${r.s.color}"></span>` +
          `<span class="tip-label">${MLV.escape(r.s.label)}</span><span class="tip-val">${MLV.fmt(r.v)}</span></div>`).join('') +
        (more ? `<div class="tip-more">+${more} more — hover a line to pick it out</div>` : '') +
        (this.onSelect ? '<div class="tip-more">click to inspect this epoch</div>' : '');
      this.tip.hidden = false;
      const tw = this.tip.offsetWidth, th = this.tip.offsetHeight;
      let x = mx + 14, y = my - th / 2;
      if (x + tw > this.w) x = mx - tw - 14;
      y = Math.max(0, Math.min(this.h - th, y));
      this.tip.style.transform = `translate(${x}px, ${y}px)`;
    }

    draw() {
      const s = setupCanvas(this.canvas, this.height);
      this.ctx = s.ctx; this.w = s.w; this.h = s.h;
      const c = this.ctx;
      c.fillStyle = MLV.css('--surface');
      c.fillRect(0, 0, this.w, this.h);
      c.font = '11px "IBM Plex Mono", monospace';

      const n = this.count();
      if (!n) {
        c.fillStyle = MLV.css('--text-muted');
        c.fillText(this.emptyText, this.pad.l, this.h / 2);
        return;
      }
      const g = this.geom();

      // y grid
      const ys = niceStep(g.hi - g.lo, 4);
      c.strokeStyle = MLV.css('--grid'); c.lineWidth = 1;
      c.fillStyle = MLV.css('--text-muted');
      c.textAlign = 'right';
      for (let v = Math.ceil(g.lo / ys) * ys; v <= g.hi + 1e-12; v += ys) {
        const y = this.yPx(g, v);
        c.beginPath(); c.moveTo(g.l, y); c.lineTo(g.r, y); c.stroke();
        c.fillText(tickLabel(v, ys), g.l - 6, y + 4);
      }
      // zero line
      c.strokeStyle = MLV.css('--axis');
      const y0 = this.yPx(g, 0);
      c.beginPath(); c.moveTo(g.l, y0); c.lineTo(g.r, y0); c.stroke();

      // x ticks (epochs)
      const last = n - 1;
      const xs = Math.max(1, niceStep(Math.max(1, last), 6));
      c.textAlign = 'center';
      c.fillStyle = MLV.css('--text-muted');
      for (let v = 0; v <= last; v += xs) c.fillText(String(Math.round(v)), this.xPx(g, v), this.h - 8);

      // selected-epoch marker
      if (this.marker !== null && this.marker !== undefined && this.marker <= last) {
        const x = this.xPx(g, this.marker);
        c.save();
        c.setLineDash([4, 4]);
        c.strokeStyle = MLV.css('--text-secondary');
        c.beginPath(); c.moveTo(x, g.t); c.lineTo(x, g.b); c.stroke();
        c.restore();
      }

      // lines
      const vis = this.visible();
      const focus = this.focusKey || this.hoverKey;
      const ordered = focus ? vis.filter((s) => s.key !== focus).concat(vis.filter((s) => s.key === focus)) : vis;
      for (const sr of ordered) {
        const dim = focus && sr.key !== focus;
        c.save();
        c.globalAlpha = dim ? 0.18 : 1;
        c.strokeStyle = sr.color;
        c.lineWidth = sr.key === focus ? 3 : 2;
        c.lineJoin = 'round';
        c.beginPath();
        let started = false;
        for (let i = 0; i < n; i++) {
          const v = sr.values[i];
          if (!Number.isFinite(v)) { started = false; continue; }
          const x = this.xPx(g, i), y = this.yPx(g, Math.max(g.lo, Math.min(g.hi, v)));
          if (!started) { c.moveTo(x, y); started = true; } else c.lineTo(x, y);
        }
        c.stroke();
        c.restore();
      }

      // dots at marker / hover epoch
      const dotAt = this.hoverEpoch ?? this.marker;
      if (dotAt !== null && dotAt !== undefined && dotAt <= last) {
        for (const sr of vis) {
          const v = sr.values[dotAt];
          if (!Number.isFinite(v)) continue;
          if (focus && sr.key !== focus && vis.length > 4) continue;
          c.beginPath();
          c.arc(this.xPx(g, dotAt), this.yPx(g, v), 4, 0, Math.PI * 2);
          c.fillStyle = sr.color; c.fill();
          c.lineWidth = 2; c.strokeStyle = MLV.css('--surface'); c.stroke();
        }
      }
      if (this.hoverEpoch !== null) {
        const x = this.xPx(g, this.hoverEpoch);
        c.strokeStyle = MLV.css('--axis');
        c.beginPath(); c.moveTo(x, g.t); c.lineTo(x, g.b); c.stroke();
      }

      // direct labels at the line ends (≤ 4 series)
      if (this.labelsOn()) {
        const items = vis
          .map((sr) => ({ sr, v: sr.values[last] }))
          .filter((o) => Number.isFinite(o.v))
          .map((o) => ({ ...o, y: this.yPx(g, o.v) }))
          .sort((a, b) => a.y - b.y);
        for (let i = 1; i < items.length; i++) if (items[i].y - items[i - 1].y < 14) items[i].y = items[i - 1].y + 14;
        c.textAlign = 'left';
        for (const it of items) {
          c.fillStyle = it.sr.color;
          c.fillRect(g.r + 8, it.y - 4, 8, 8);
          c.fillStyle = MLV.css('--text-secondary');
          c.fillText(`${it.sr.short || it.sr.label} ${MLV.fmt(it.v, 2)}`, g.r + 20, it.y + 4);
        }
      }
    }
  }
  MLV.LineChart = LineChart;
})();
