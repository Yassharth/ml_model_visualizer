/* ============================================================
   models.js — the three models. Each one exposes the same shape
   so the generic trainer/UI can drive any of them:

     meta            [{ key, sym, label, group }]   one per parameter
     groups          group names (legend for big models)
     getParams()     -> number[]   (a copy)
     setParams(arr)
     size()          number of training samples
     evaluate(p)     -> { loss, grads }   full-batch, does NOT mutate
     describe(rec, view) -> { forward, loss, grads }   HTML for the
                     "inside this epoch" panel, using the numbers in
                     rec = { epoch, before, grads, loss, lr, after }

   Training is full-batch gradient descent: one epoch = one pass over
   all n samples = exactly one update  θ ← θ − η·∇L(θ).
   ============================================================ */
var MLV = (globalThis.MLV = globalThis.MLV || {});
MLV.models = {};

(function () {
  const f = (v, d) => MLV.fmt(v, d);
  const fp = (v, d) => MLV.fmtP(v, d);
  const sigmoid = (z) => 1 / (1 + Math.exp(-z));
  const EPS = 1e-12;
  const clampP = (p) => Math.min(1 - EPS, Math.max(EPS, p));
  const bce = (y, p) => -(y * Math.log(clampP(p)) + (1 - y) * Math.log(1 - clampP(p)));

  /* ---------- small HTML helpers shared by every describe() ---------- */

  // A data table that shows the first `visible` rows and tucks the rest
  // into a <details> so a 100-row dataset doesn't swamp the panel.
  function table(headers, rows, footer, visible = 6) {
    const th = '<tr>' + headers.map((h) => `<th>${h}</th>`).join('') + '</tr>';
    const tr = (r) => '<tr>' + r.map((c) => `<td>${c}</td>`).join('') + '</tr>';
    const head = rows.slice(0, visible).map(tr).join('');
    const tail = rows.slice(visible).map(tr).join('');
    const foot = footer ? '<tfoot><tr>' + footer.map((c) => `<td>${c}</td>`).join('') + '</tr></tfoot>' : '';
    let html = `<div class="table-wrap"><table class="math-table"><thead>${th}</thead><tbody>${head}</tbody>${foot}</table></div>`;
    if (tail) {
      html += `<details class="more-rows"><summary>Show the other ${rows.length - visible} rows</summary>` +
        `<div class="table-wrap"><table class="math-table"><thead>${th}</thead><tbody>${tail}</tbody></table></div></details>`;
    }
    return html;
  }

  function eq(lhs, symbolic, numeric, result) {
    return `<div class="eq"><span class="eq-lhs">${lhs}</span><span class="eq-op">=</span>` +
      `<span class="eq-sym">${symbolic}</span>` +
      (numeric ? `<span class="eq-op">=</span><span class="eq-num">${numeric}</span>` : '') +
      `<span class="eq-op">=</span><b class="eq-res">${result}</b></div>`;
  }

  function weightsLine(meta, values) {
    return '<p class="given">Weights going into this epoch: ' +
      meta.map((m, i) => `<code>${m.sym} = ${f(values[i])}</code>`).join(' ') + '</p>';
  }

  MLV.mathTable = table;
  MLV.mathEq = eq;

  /* ======================================================
     Linear regression   ŷ = m·x + b      loss = MSE
     ====================================================== */
  MLV.models.linear = function () {
    const model = {
      kind: 'linear',
      meta: [
        { key: 'm', sym: 'm', label: 'm (slope)', group: 'weight' },
        { key: 'b', sym: 'b', label: 'b (intercept)', group: 'bias' },
      ],
      data: { X: [], Y: [] },
      p: [0, 0],
      getParams() { return this.p.slice(); },
      setParams(arr) { this.p = arr.slice(); },
      size() { return this.data.X.length; },
      predict(p, x) { return p[0] * x + p[1]; },

      evaluate(p) {
        const { X, Y } = this.data, n = X.length;
        if (!n) return { loss: 0, grads: [0, 0] };
        let loss = 0, gm = 0, gb = 0;
        for (let i = 0; i < n; i++) {
          const e = p[0] * X[i] + p[1] - Y[i];
          loss += e * e; gm += e * X[i]; gb += e;
        }
        return { loss: loss / n, grads: [(2 / n) * gm, (2 / n) * gb] };
      },

      describe(rec) {
        const { X, Y } = this.data, n = X.length;
        const [m, b] = rec.before;
        const rows = [];
        let sE = 0, sE2 = 0, sEX = 0;
        for (let i = 0; i < n; i++) {
          const yh = m * X[i] + b, e = yh - Y[i];
          sE += e; sE2 += e * e; sEX += e * X[i];
          rows.push([i + 1, f(X[i], 3), f(Y[i], 3), f(yh), f(e), f(e * e), f(e * X[i])]);
        }
        const forward =
          weightsLine(this.meta, rec.before) +
          '<p>Every sample is pushed through the current line, <code>ŷ = m·x + b</code>, and compared with its target. ' +
          'The error <code>e = ŷ − y</code> is what the rest of the epoch is built from.</p>' +
          table(['i', 'x', 'y', 'ŷ = m·x + b', 'e = ŷ − y', 'e²', 'e·x'], rows,
            ['Σ', '', '', '', f(sE), f(sE2), f(sEX)]);

        const loss =
          '<p>Mean squared error averages the squared errors over all n samples.</p>' +
          eq('L', '(1/n) Σ eᵢ²', `(1/${n}) × ${fp(sE2)}`, f(rec.loss));

        const grads =
          '<p>Differentiate <code>(m·x + b − y)²</code>: the chain rule gives <code>2e·x</code> for m and <code>2e</code> for b. ' +
          'Averaging over the batch:</p>' +
          eq('∂L/∂m', '(2/n) Σ eᵢ·xᵢ', `(2/${n}) × ${fp(sEX)}`, f(rec.grads[0])) +
          eq('∂L/∂b', '(2/n) Σ eᵢ', `(2/${n}) × ${fp(sE)}`, f(rec.grads[1])) +
          '<p class="aside">A negative gradient means the loss falls as the parameter grows, so the update will push it up — and vice versa.</p>';

        return { forward, loss, grads };
      },
    };
    return model;
  };

  /* ======================================================
     Logistic regression   p = σ(w₁x₁ + w₂x₂ + b)   loss = BCE
     ====================================================== */
  MLV.models.logistic = function () {
    const model = {
      kind: 'logistic',
      meta: [
        { key: 'w1', sym: 'w₁', label: 'w₁', group: 'weight' },
        { key: 'w2', sym: 'w₂', label: 'w₂', group: 'weight' },
        { key: 'b', sym: 'b', label: 'b (bias)', group: 'bias' },
      ],
      data: { X: [], Y: [] },
      p: [0, 0, 0],
      getParams() { return this.p.slice(); },
      setParams(arr) { this.p = arr.slice(); },
      size() { return this.data.X.length; },
      predict(p, x1, x2) { return sigmoid(p[0] * x1 + p[1] * x2 + p[2]); },

      evaluate(p) {
        const { X, Y } = this.data, n = X.length;
        if (!n) return { loss: 0, grads: [0, 0, 0] };
        let loss = 0, g1 = 0, g2 = 0, gb = 0;
        for (let i = 0; i < n; i++) {
          const q = sigmoid(p[0] * X[i][0] + p[1] * X[i][1] + p[2]);
          const d = q - Y[i];
          g1 += d * X[i][0]; g2 += d * X[i][1]; gb += d;
          loss += bce(Y[i], q);
        }
        return { loss: loss / n, grads: [g1 / n, g2 / n, gb / n] };
      },

      describe(rec) {
        const { X, Y } = this.data, n = X.length;
        const [w1, w2, b] = rec.before;
        const rows = [];
        let sD = 0, sD1 = 0, sD2 = 0, sL = 0;
        for (let i = 0; i < n; i++) {
          const z = w1 * X[i][0] + w2 * X[i][1] + b, q = sigmoid(z), d = q - Y[i], l = bce(Y[i], q);
          sD += d; sD1 += d * X[i][0]; sD2 += d * X[i][1]; sL += l;
          rows.push([i + 1, f(X[i][0], 3), f(X[i][1], 3), Y[i], f(z), f(q), f(d), f(d * X[i][0]), f(d * X[i][1]), f(l)]);
        }
        const forward =
          weightsLine(this.meta, rec.before) +
          '<p>Each point gets a score <code>z = w₁x₁ + w₂x₂ + b</code>, squashed by the sigmoid into a probability ' +
          '<code>p = σ(z) = 1/(1+e^(−z))</code> of being class 1.</p>' +
          table(['i', 'x₁', 'x₂', 'y', 'z', 'p = σ(z)', 'p − y', '(p − y)·x₁', '(p − y)·x₂', 'ℓᵢ'], rows,
            ['Σ', '', '', '', '', '', f(sD), f(sD1), f(sD2), f(sL)]);

        const loss =
          '<p>Binary cross-entropy: each point costs <code>ℓᵢ = −[y·log p + (1−y)·log(1−p)]</code> — small when the model is confidently right, large when confidently wrong.</p>' +
          eq('L', '(1/n) Σ ℓᵢ', `(1/${n}) × ${fp(sL)}`, f(rec.loss));

        const grads =
          '<p>The sigmoid\'s derivative and the log cancel neatly, so <code>∂ℓ/∂z = p − y</code>. ' +
          'Then <code>∂z/∂w₁ = x₁</code>, <code>∂z/∂w₂ = x₂</code>, <code>∂z/∂b = 1</code>:</p>' +
          eq('∂L/∂w₁', '(1/n) Σ (pᵢ − yᵢ)·x₁ᵢ', `(1/${n}) × ${fp(sD1)}`, f(rec.grads[0])) +
          eq('∂L/∂w₂', '(1/n) Σ (pᵢ − yᵢ)·x₂ᵢ', `(1/${n}) × ${fp(sD2)}`, f(rec.grads[1])) +
          eq('∂L/∂b', '(1/n) Σ (pᵢ − yᵢ)', `(1/${n}) × ${fp(sD)}`, f(rec.grads[2]));

        return { forward, loss, grads };
      },
    };
    return model;
  };

  /* ======================================================
     Neural network   2 → H → 1
       z¹ = W¹x + b¹,  a¹ = act(z¹)
       z² = W²·a¹ + b², ŷ = σ(z²)          loss = BCE
     Flat parameter layout:
       W¹ (H×2, row-major: [h][input]) | b¹ (H) | W² (H) | b²
     ====================================================== */
  const ACTS = {
    tanh: { name: 'tanh', f: Math.tanh, d: (z, a) => 1 - a * a, dText: "tanh′(z) = 1 − a²" },
    sigmoid: { name: 'σ', f: sigmoid, d: (z, a) => a * (1 - a), dText: "σ′(z) = a(1 − a)" },
    relu: { name: 'ReLU', f: (z) => (z > 0 ? z : 0), d: (z) => (z > 0 ? 1 : 0), dText: "ReLU′(z) = 1 if z > 0, else 0" },
  };
  MLV.ACTS = ACTS;

  MLV.models.mlp = function (H = 4, actName = 'tanh') {
    const act = ACTS[actName];
    const meta = [];
    for (let h = 0; h < H; h++) for (let i = 0; i < 2; i++)
      meta.push({ key: `W1_${h}_${i}`, sym: `W¹[h${h + 1},x${i + 1}]`, label: `W¹ h${h + 1}←x${i + 1}`, group: 'W¹' });
    for (let h = 0; h < H; h++) meta.push({ key: `b1_${h}`, sym: `b¹[h${h + 1}]`, label: `b¹ h${h + 1}`, group: 'b¹' });
    for (let h = 0; h < H; h++) meta.push({ key: `W2_${h}`, sym: `W²[ŷ,h${h + 1}]`, label: `W² ŷ←h${h + 1}`, group: 'W²' });
    meta.push({ key: 'b2', sym: 'b²', label: 'b²', group: 'b²' });

    const iW1 = (h, i) => h * 2 + i;
    const ib1 = (h) => 2 * H + h;
    const iW2 = (h) => 3 * H + h;
    const ib2 = 4 * H;

    function forward(p, x) {
      const z1 = new Array(H), a1 = new Array(H);
      let z2 = p[ib2];
      for (let h = 0; h < H; h++) {
        z1[h] = p[iW1(h, 0)] * x[0] + p[iW1(h, 1)] * x[1] + p[ib1(h)];
        a1[h] = act.f(z1[h]);
        z2 += p[iW2(h)] * a1[h];
      }
      return { z1, a1, z2, yhat: sigmoid(z2) };
    }

    // per-sample gradient, written out exactly as backprop does it
    function backward(p, x, y, fw) {
      const g = new Array(4 * H + 1).fill(0);
      const d2 = fw.yhat - y;                 // ∂ℓ/∂z²  (sigmoid + BCE)
      g[ib2] = d2;
      const d1 = new Array(H), dact = new Array(H);
      for (let h = 0; h < H; h++) {
        g[iW2(h)] = d2 * fw.a1[h];
        dact[h] = act.d(fw.z1[h], fw.a1[h]);
        d1[h] = p[iW2(h)] * d2 * dact[h];     // ∂ℓ/∂z¹_h
        g[ib1(h)] = d1[h];
        g[iW1(h, 0)] = d1[h] * x[0];
        g[iW1(h, 1)] = d1[h] * x[1];
      }
      return { g, d2, d1, dact };
    }

    const model = {
      kind: 'mlp',
      H, actName, act, meta,
      groups: ['W¹', 'b¹', 'W²', 'b²'],
      idx: { iW1, ib1, iW2, ib2 },
      data: { X: [], Y: [] },
      p: new Array(4 * H + 1).fill(0),
      getParams() { return this.p.slice(); },
      setParams(arr) { this.p = arr.slice(); },
      size() { return this.data.X.length; },
      forward,
      predict(p, x1, x2) { return forward(p, [x1, x2]).yhat; },

      randomParams(seed, scale) {
        const r = MLV.rng(seed);
        return meta.map((m) => (m.group === 'b¹' || m.group === 'b²') ? 0 : (r() * 2 - 1) * scale);
      },

      evaluate(p) {
        const { X, Y } = this.data, n = X.length;
        const grads = new Array(4 * H + 1).fill(0);
        if (!n) return { loss: 0, grads };
        let loss = 0;
        for (let i = 0; i < n; i++) {
          const fw = forward(p, X[i]);
          loss += bce(Y[i], fw.yhat);
          const { g } = backward(p, X[i], Y[i], fw);
          for (let k = 0; k < g.length; k++) grads[k] += g[k];
        }
        for (let k = 0; k < grads.length; k++) grads[k] /= n;
        return { loss: loss / n, grads };
      },

      describe(rec, view) {
        const { X, Y } = this.data, n = X.length;
        const p = rec.before;
        const rows = [];
        let sL = 0;
        for (let i = 0; i < n; i++) {
          const fw = forward(p, X[i]);
          const l = bce(Y[i], fw.yhat);
          sL += l;
          rows.push([i + 1, f(X[i][0], 3), f(X[i][1], 3), Y[i], f(fw.yhat), f(l)]);
        }
        const forwardHtml =
          `<p>Each sample flows input → ${H} hidden units (<code>${act.name}</code>) → one sigmoid output. ` +
          'The full layer-by-layer arithmetic for one sample is in step ③; here is the output for every sample.</p>' +
          table(['i', 'x₁', 'x₂', 'y', 'ŷ', 'ℓᵢ'], rows, ['Σ', '', '', '', '', f(sL)]);

        const lossHtml =
          '<p>Same binary cross-entropy as logistic regression, averaged over the batch.</p>' +
          eq('L', '(1/n) Σ ℓᵢ', `(1/${n}) × ${fp(sL)}`, f(rec.loss));

        // ---- backprop trace for one chosen sample ----
        const s = Math.min(Math.max(1, view.traceSample || 1), n) - 1;
        const x = X[s], y = Y[s];
        const fw = forward(p, x);
        const bw = backward(p, x, y, fw);
        const hRows = [];
        for (let h = 0; h < H; h++) {
          hRows.push([
            `h${h + 1}`,
            `${fp(p[iW1(h, 0)], 3)}·${fp(x[0], 3)} + ${fp(p[iW1(h, 1)], 3)}·${fp(x[1], 3)} + ${fp(p[ib1(h)], 3)} = <b>${f(fw.z1[h])}</b>`,
            f(fw.a1[h]),
            f(p[iW2(h)]),
            f(bw.dact[h]),
            f(bw.d1[h]),
            f(bw.g[iW2(h)]),
            f(bw.g[iW1(h, 0)]),
            f(bw.g[iW1(h, 1)]),
          ]);
        }
        const opts = Array.from({ length: n }, (_, i) =>
          `<option value="${i + 1}"${i === s ? ' selected' : ''}>#${i + 1}</option>`).join('');
        const gradsHtml =
          `<div class="trace-head"><label>Trace backprop for sample <select data-trace-sample>${opts}</select></label>` +
          `<code>x = (${f(x[0], 3)}, ${f(x[1], 3)}), y = ${y}</code></div>` +
          '<h4>Forward, one unit at a time</h4>' +
          table(['unit', 'z¹ = W¹x + b¹', `a¹ = ${act.name}(z¹)`, 'W²', act.dText.split(' =')[0], 'δ¹ = W²·δ²·act′', '∂ℓ/∂W²', '∂ℓ/∂W¹[·,x₁]', '∂ℓ/∂W¹[·,x₂]'], hRows, null, 16) +
          eq('z²', 'Σ W²ₕ·a¹ₕ + b²', null, f(fw.z2)) +
          eq('ŷ', 'σ(z²)', null, f(fw.yhat)) +
          '<h4>Backward</h4>' +
          eq('δ² = ∂ℓ/∂z²', 'ŷ − y', `${f(fw.yhat)} − ${y}`, f(bw.d2)) +
          eq('∂ℓ/∂b²', 'δ²', null, f(bw.d2)) +
          `<p>For each hidden unit: <code>∂ℓ/∂W²ₕ = δ²·a¹ₕ</code>; the error flows back through W² and the activation, ` +
          `<code>δ¹ₕ = W²ₕ · δ² · ${act.dText.split(' =')[0]}</code> with <code>${act.dText}</code>; ` +
          'then <code>∂ℓ/∂W¹[h,xᵢ] = δ¹ₕ·xᵢ</code> and <code>∂ℓ/∂b¹ₕ = δ¹ₕ</code>. All of those are in the table above.</p>' +
          `<p class="aside">That is the gradient for <em>one</em> sample. The batch gradient used in the update is the average of these over all ${n} samples — it is the ∂L/∂θ column in step ④.</p>`;

        return { forward: forwardHtml, loss: lossHtml, grads: gradsHtml };
      },
    };
    return model;
  };
})();
