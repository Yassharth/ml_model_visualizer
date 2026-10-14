/* Neural network page: architecture controls, data, decision field and network diagram. */
(function () {
  const $ = (id) => document.getElementById(id);
  const SVG = 'http://www.w3.org/2000/svg';
  const DOMAIN = { xMin: -2.4, xMax: 2.4, yMin: -2.4, yMax: 2.4 };

  function makePreset(name) {
    if (name === 'circles') return MLV.datasets.circles(60);
    if (name === 'blobs') return MLV.datasets.blobs(40, 1.0);
    return MLV.datasets.xor(60, 0.45);
  }

  const seed = () => parseInt($('seed').value, 10) || 0;
  const scale = () => { const v = parseFloat($('scale').value); return Number.isFinite(v) ? v : 1; };

  let model = MLV.models.mlp(4, 'tanh');
  model.data = makePreset('xor');
  model.setParams(model.randomParams(seed(), scale()));

  const plot = new MLV.Plot2D($('main-plot'), DOMAIN, 360);
  let view = { p: model.getParams(), prev: null };

  plot.onDraw = function () {
    const { X, Y } = model.data;
    const m = model, p = view.p;
    plot.clear();
    plot.field((a, b) => m.predict(p, a, b), 48);
    plot.axes();
    if (view.prev) {
      plot.ctx.save(); plot.ctx.setLineDash([6, 5]);
      plot.contour((a, b) => m.predict(view.prev, a, b), MLV.css('--text-muted'), 50);
      plot.ctx.restore();
    }
    plot.contour((a, b) => m.predict(p, a, b), MLV.css('--text-primary'), 70);
    const c0 = MLV.css('--series-1'), c1 = MLV.css('--series-2');
    for (let i = 0; i < X.length; i++) plot.point(X[i][0], X[i][1], Y[i] ? c1 : c0);
  };

  function el(tag, attrs, text) {
    const e = document.createElementNS(SVG, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    return e;
  }

  function drawDiagram(p) {
    const svg = $('net-diagram');
    svg.innerHTML = '';
    const H = model.H, { iW1, ib1, iW2, ib2 } = model.idx;
    const W = 460, Ht = 340, top = 34, bottom = 30;
    const colX = [60, 230, 400];
    const ys = (n) => Array.from({ length: n }, (_, i) => top + ((i + 0.5) / n) * (Ht - top - bottom));
    const inY = ys(2), hY = ys(H), oY = ys(1);
    const up = MLV.css('--up'), down = MLV.css('--down');
    const edge = (x1, y1, x2, y2, w, label) => {
      const g = el('line', {
        x1, y1, x2, y2,
        stroke: w >= 0 ? up : down,
        'stroke-width': Math.min(7, 0.6 + Math.abs(w) * 1.8),
        'stroke-opacity': Math.min(1, 0.25 + Math.abs(w) / 2.5),
        'stroke-linecap': 'round',
      });
      g.appendChild(el('title', {}, `${label} = ${MLV.fmt(w)}`));
      svg.appendChild(g);
    };
    for (let h = 0; h < H; h++) for (let i = 0; i < 2; i++) edge(colX[0], inY[i], colX[1], hY[h], p[iW1(h, i)], `W¹[h${h + 1},x${i + 1}]`);
    for (let h = 0; h < H; h++) edge(colX[1], hY[h], colX[2], oY[0], p[iW2(h)], `W²[ŷ,h${h + 1}]`);

    const node = (x, y, label, sub) => {
      svg.appendChild(el('circle', { cx: x, cy: y, r: 15, class: 'node' }));
      svg.appendChild(el('text', { x, y: y + 4, 'text-anchor': 'middle' }, label));
      if (sub) svg.appendChild(el('text', { x: x + 20, y: y + 4, 'text-anchor': 'start', style: 'font-size:10px' }, sub));
    };
    inY.forEach((y, i) => node(colX[0], y, `x${i + 1}`));
    hY.forEach((y, h) => node(colX[1], y, `h${h + 1}`, `b¹ ${MLV.fmt(p[ib1(h)], 2)}`));
    node(colX[2], oY[0], 'ŷ', '');
    svg.appendChild(el('text', { x: colX[2], y: oY[0] + 32, 'text-anchor': 'middle', style: 'font-size:10px' }, `b² ${MLV.fmt(p[ib2], 2)}`));
    ['input', `hidden · ${model.act.name}`, 'output · σ'].forEach((t, i) =>
      svg.appendChild(el('text', { x: colX[i], y: 16, 'text-anchor': 'middle' }, t)));
  }

  const app = MLV.createTrainer({
    model,
    drawMain(p, prev) { view = { p, prev }; plot.redraw(); drawDiagram(p); },
  });

  function rebuild() {
    const data = model.data;
    model = MLV.models.mlp(parseInt($('hidden').value, 10), $('activation').value);
    model.data = data;
    const p = model.randomParams(seed(), scale());
    model.setParams(p);
    app.setModel(model, p);
  }

  $('hidden').addEventListener('input', () => { $('hidden-val').textContent = $('hidden').value; });
  $('hidden').addEventListener('change', rebuild);
  $('activation').addEventListener('change', rebuild);
  $('btn-randomize-w').addEventListener('click', () => app.setInitial(model.randomParams(seed(), scale())));

  function setData(d) { model.data = d; app.dataChanged(); }
  $('preset').addEventListener('change', () => setData(makePreset($('preset').value)));
  $('btn-randomize').addEventListener('click', () => setData(makePreset($('preset').value)));
  $('btn-clear').addEventListener('click', () => setData({ X: [], Y: [] }));
  $('btn-load-csv').addEventListener('click', () => {
    const rows = MLV.parseNumericCSV($('csv-input').value, 3);
    if (!rows.length) { alert('No valid "x1, x2, label" rows found.'); return; }
    setData({ X: rows.map((r) => [r[0], r[1]]), Y: rows.map((r) => (r[2] >= 0.5 ? 1 : 0)) });
  });

  $('main-plot').addEventListener('click', (e) => {
    const [x, y] = plot.eventToData(e);
    const mode = document.querySelector('input[name="click-mode"]:checked').value;
    const cls = (mode === '1') !== e.shiftKey ? 1 : 0;
    model.data.X.push([x, y]);
    model.data.Y.push(cls);
    app.dataChanged();
  });
})();
