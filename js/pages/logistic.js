/* Logistic regression page: data handling and the decision-boundary plot. */
(function () {
  const $ = (id) => document.getElementById(id);
  const DOMAIN = { xMin: -3, xMax: 3, yMin: -3, yMax: 3 };
  const model = MLV.models.logistic();

  function makePreset(name) {
    if (name === 'overlap') return MLV.datasets.blobs(40, 1.8);
    if (name === 'xor') return MLV.datasets.xor(60, 0.45);
    return MLV.datasets.blobs(40, 1.1);
  }
  model.data = makePreset('blobs');
  model.setParams([0, 0, 0]);

  const plot = new MLV.Plot2D($('main-plot'), DOMAIN, 380);
  let view = { p: model.getParams(), prev: null };

  // p = 0.5 where w1·x1 + w2·x2 + b = 0
  function boundary(p, color, width, dash) {
    const [w1, w2, b] = p;
    if (Math.abs(w2) > 1e-9) plot.line(-w1 / w2, -b / w2, color, width, dash);
    else if (Math.abs(w1) > 1e-9) plot.segment(-b / w1, DOMAIN.yMin, -b / w1, DOMAIN.yMax, color, width, dash);
  }

  plot.onDraw = function () {
    const { X, Y } = model.data;
    const p = view.p;
    plot.clear();
    plot.field((x1, x2) => model.predict(p, x1, x2));
    plot.axes();
    if (view.prev) boundary(view.prev, MLV.css('--text-muted'), 1.6, [6, 5]);
    boundary(p, MLV.css('--text-primary'), 2.5, []);
    const c0 = MLV.css('--series-1'), c1 = MLV.css('--series-2');
    for (let i = 0; i < X.length; i++) plot.point(X[i][0], X[i][1], Y[i] ? c1 : c0);
  };

  const app = MLV.createTrainer({
    model,
    drawMain(p, prev) { view = { p, prev }; plot.redraw(); },
  });

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
