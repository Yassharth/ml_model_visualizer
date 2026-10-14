/* Linear regression page: data handling and the main plot. */
(function () {
  const $ = (id) => document.getElementById(id);
  const model = MLV.models.linear();
  model.data = MLV.datasets.linear();
  model.setParams([0, 0]);

  function domainFor(d) {
    if (!d.X.length) return { xMin: -1, xMax: 11, yMin: -5, yMax: 25 };
    const x0 = Math.min(...d.X), x1 = Math.max(...d.X), y0 = Math.min(...d.Y), y1 = Math.max(...d.Y);
    const px = Math.max((x1 - x0) * 0.15, 1), py = Math.max((y1 - y0) * 0.15, 1);
    return { xMin: Math.min(0, x0 - px), xMax: x1 + px, yMin: Math.min(0, y0 - py), yMax: y1 + py };
  }

  const plot = new MLV.Plot2D($('main-plot'), domainFor(model.data), 380);
  let view = { p: model.getParams(), prev: null };

  plot.onDraw = function () {
    const { X, Y } = model.data;
    const [m, b] = view.p;
    plot.clear();
    plot.axes();
    // residuals for the shown line
    const errCol = MLV.css('--series-loss');
    for (let i = 0; i < X.length; i++) plot.segment(X[i], Y[i], X[i], m * X[i] + b, errCol, 1.2, [3, 3]);
    if (view.prev) plot.line(view.prev[0], view.prev[1], MLV.css('--text-muted'), 1.6, [6, 5]);
    plot.line(m, b, MLV.css('--series-2'), 2.5);
    const ptCol = MLV.css('--series-1');
    for (let i = 0; i < X.length; i++) plot.point(X[i], Y[i], ptCol);
  };

  const app = MLV.createTrainer({
    model,
    drawMain(p, prev) { view = { p, prev }; plot.redraw(); },
  });

  function setData(d) {
    model.data = d;
    plot.setDomain(domainFor(d));
    app.dataChanged();
  }

  $('btn-randomize').addEventListener('click', () => {
    const m = +(Math.random() * 4 - 1).toFixed(1), b = +(Math.random() * 6 - 2).toFixed(1);
    setData(MLV.datasets.linear(20, m, b, 1.5));
  });
  $('btn-clear').addEventListener('click', () => setData({ X: [], Y: [] }));
  $('btn-load-csv').addEventListener('click', () => {
    const rows = MLV.parseNumericCSV($('csv-input').value, 2);
    if (!rows.length) { alert('No valid "x, y" rows found.'); return; }
    setData({ X: rows.map((r) => r[0]), Y: rows.map((r) => r[1]) });
  });

  $('main-plot').addEventListener('click', (e) => {
    const [x, y] = plot.eventToData(e);
    const mode = document.querySelector('input[name="click-mode"]:checked').value;
    const { X, Y } = model.data;
    if ((mode === '1') !== e.shiftKey) {
      if (!X.length) return;
      let k = 0, best = Infinity;
      X.forEach((xi, i) => { const d = (xi - x) ** 2 + (Y[i] - y) ** 2; if (d < best) { best = d; k = i; } });
      X.splice(k, 1); Y.splice(k, 1);
    } else {
      X.push(x); Y.push(y);
    }
    app.dataChanged();
  });
})();
