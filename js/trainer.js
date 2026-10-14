/* ============================================================
   trainer.js — the training loop and everything that shows it.

   Owns the epoch history for whichever model the page hands it:
     points[k]  = { params: θ_k, loss: L(θ_k) }      k = 0..N
     records[k-1] = what happened in epoch k:
                    { epoch: k, before: θ_{k-1}, loss: L(θ_{k-1}),
                      grads: ∇L(θ_{k-1}), lr: η, after: θ_k }

   and renders: status, weight-trajectory chart, loss chart,
   the four-phase "inside epoch k" panel, and the epoch log.

   Phase-by-phase mode splits one epoch over four presses of Step:
   ① forward pass → ② loss → ③ gradients → ④ update (applied).
   ============================================================ */
var MLV = (globalThis.MLV = globalThis.MLV || {});

(function () {
  const $ = (id) => document.getElementById(id);
  const f = (v, d) => MLV.fmt(v, d);
  const fp = (v, d) => MLV.fmtP(v, d);
  const MAX_EPOCHS = 3000;
  const SPEEDS = [1, 2, 5, 10, 20, 50, 100, 200];
  const PHASES = ['forward', 'loss', 'grads', 'update'];

  MLV.createTrainer = function (cfg) {
    const st = {
      model: cfg.model,
      initParams: cfg.model.getParams(),
      points: [],
      records: [],
      pending: null,       // { rec, phase: 1..3 } while stepping phase by phase
      selected: null,      // epoch being inspected; null = follow the latest
      playing: false,
      diverged: false,
      hidden: new Set(),   // hidden series keys / groups
      focusKey: null,
      view: { traceSample: 1 },
    };

    /* ---------------- charts ---------------- */

    const traj = new MLV.LineChart($('traj-chart'), {
      height: 260,
      onSelect: (ep) => select(ep),
    });
    const lossChart = new MLV.LineChart($('loss-chart'), {
      height: 260,
      directLabels: false,
      onSelect: (ep) => select(ep),
    });

    /* ---------------- core loop ---------------- */

    function lr() { return parseFloat($('lr').value); }

    function computeRecord() {
      const m = st.model;
      const before = m.getParams();
      const { loss, grads } = m.evaluate(before);
      const eta = lr();
      const after = before.map((v, k) => v - eta * grads[k]);
      return { epoch: st.records.length + 1, before, loss, grads, lr: eta, after };
    }

    function commit(rec) {
      st.model.setParams(rec.after);
      st.records.push(rec);
      const lossAfter = st.model.evaluate(rec.after).loss;
      st.points.push({ params: rec.after.slice(), loss: lossAfter });
      if (!rec.after.every(Number.isFinite) || !Number.isFinite(lossAfter) || lossAfter > 1e12) {
        st.diverged = true;
        stopPlay();
      }
    }

    function canStep() {
      if (st.diverged) return false;
      if (!st.model.size()) { warn('Add some data points first — click the plot, load a CSV or randomise.'); return false; }
      if (st.records.length >= MAX_EPOCHS) { warn(`History is capped at ${MAX_EPOCHS} epochs. Press Reset to start again.`); stopPlay(); return false; }
      return true;
    }

    function stepOne() {
      if (!canStep()) return;
      if ($('phase-mode').checked) {
        if (!st.pending) {
          st.pending = { rec: computeRecord(), phase: 1 };
        } else if (st.pending.phase < 3) {
          st.pending.phase += 1;
        } else {
          commit(st.pending.rec);
          st.pending = null;
          st.selected = null;
        }
      } else {
        st.pending = null;
        commit(computeRecord());
        st.selected = null;
      }
      render();
    }

    function stepMany(k) {
      st.pending = null;
      for (let i = 0; i < k; i++) {
        if (!canStep()) break;
        commit(computeRecord());
      }
      st.selected = null;
      render();
    }

    let raf = null, acc = 0, lastT = 0;
    function loop(t) {
      if (!st.playing) return;
      const dt = Math.min(0.25, (t - lastT) / 1000);
      lastT = t;
      acc += dt * SPEEDS[parseInt($('speed').value, 10)];
      const k = Math.floor(acc);
      if (k > 0) { acc -= k; stepMany(k); }
      raf = requestAnimationFrame(loop);
    }
    function startPlay() {
      if (!canStep()) return;
      st.pending = null;
      st.playing = true;
      acc = 1; lastT = performance.now();
      $('btn-play').textContent = 'Pause';
      $('btn-play').setAttribute('aria-pressed', 'true');
      raf = requestAnimationFrame(loop);
    }
    function stopPlay() {
      st.playing = false;
      if (raf) cancelAnimationFrame(raf);
      $('btn-play').textContent = 'Play';
      $('btn-play').setAttribute('aria-pressed', 'false');
    }

    // Start a fresh history from the model's current weights.
    function restart(params) {
      stopPlay();
      if (params) st.model.setParams(params);
      const p = st.model.getParams();
      st.points = [{ params: p, loss: st.model.evaluate(p).loss }];
      st.records = [];
      st.pending = null;
      st.selected = null;
      st.diverged = false;
      st.view.traceSample = 1;
      render();
    }

    function select(ep) {
      if (st.pending) return;
      if (!st.records.length) return;
      st.selected = Math.max(1, Math.min(st.records.length, ep));
      if (st.selected === st.records.length) st.selected = null;
      render();
    }

    function inspected() {
      if (st.pending) return null;
      if (!st.records.length) return null;
      return st.selected ?? st.records.length;
    }

    /* ---------------- initial weights panel ---------------- */

    function buildInitInputs() {
      const box = $('init-weights');
      box.innerHTML = st.model.meta.map((m, i) =>
        `<label class="w-input"><span>${m.sym}</span>` +
        `<input type="number" step="any" data-k="${i}" value="${+st.initParams[i].toFixed(4)}"></label>`).join('');
      box.classList.toggle('dense', st.model.meta.length > 4);
    }

    function readInitInputs() {
      return Array.from($('init-weights').querySelectorAll('input')).map((el) => {
        const v = parseFloat(el.value);
        return Number.isFinite(v) ? v : 0;
      });
    }

    /* ---------------- rendering ---------------- */

    function warn(msg) {
      const el = $('stat-warn');
      el.textContent = msg || '';
      el.hidden = !msg;
    }

    function seriesColor(m, i) {
      const big = st.model.meta.length > 4;
      const slot = big ? st.model.groups.indexOf(m.group) : i;
      return MLV.seriesColor(slot);
    }

    function renderLegend() {
      const meta = st.model.meta;
      const big = meta.length > 4;
      const entries = big
        ? st.model.groups.map((g, i) => ({ key: g, label: `${g} (${meta.filter((m) => m.group === g).length})`, color: MLV.seriesColor(i) }))
        : meta.map((m, i) => ({ key: m.key, label: m.label, color: MLV.seriesColor(i) }));
      $('traj-legend').innerHTML = entries.map((e) =>
        `<button type="button" class="chip${st.hidden.has(e.key) ? ' is-off' : ''}" data-key="${e.key}" aria-pressed="${!st.hidden.has(e.key)}">` +
        `<span class="swatch" style="background:${e.color}"></span>${MLV.escape(e.label)}</button>`).join('') +
        (big ? '<span class="legend-note">Lines share their layer\'s colour — hover one to name it, or click a row in step ④.</span>' : '');
    }

    function renderCharts() {
      const meta = st.model.meta;
      const big = meta.length > 4;
      const n = st.points.length;
      const series = meta.map((m, i) => ({
        key: m.key,
        label: m.label,
        short: m.sym,
        color: seriesColor(m, i),
        hidden: st.hidden.has(big ? m.group : m.key),
        values: st.points.map((pt) => pt.params[i]),
      }));
      const marker = st.pending ? n - 1 : inspected();
      traj.focusKey = st.focusKey;
      traj.setData(n ? series : [], marker);
      lossChart.setData(n ? [{ key: 'loss', label: 'loss', color: MLV.css('--series-loss'), values: st.points.map((p) => p.loss) }] : [], marker);
    }

    function renderStatus() {
      const N = st.records.length;
      $('stat-epoch').textContent = st.pending ? `${N} → ${N + 1}` : N;
      const cur = st.points[st.points.length - 1];
      $('stat-loss').textContent = cur ? f(cur.loss) : '–';
      if (st.diverged) warn('Training diverged: the loss blew up, so the learning rate is too large for this data. Lower η and press Reset.');
      else warn('');
      $('btn-step').textContent = $('phase-mode').checked
        ? (st.pending ? ['Next: ② loss', 'Next: ③ gradients', 'Next: ④ update'][st.pending.phase - 1] : 'Start epoch: ① forward')
        : 'Step 1 epoch';
    }

    function updateSection(rec) {
      const meta = st.model.meta;
      const lines = meta.length <= 4;
      let html = `<p>Gradient descent moves every parameter a small step <em>against</em> its gradient, scaled by the learning rate <code>η = ${f(rec.lr)}</code>:</p>` +
        '<p class="rule"><code>θ ← θ − η · ∂L/∂θ</code></p>';
      if (lines) {
        html += meta.map((m, i) => {
          const d = rec.after[i] - rec.before[i];
          return `<div class="eq"><span class="eq-lhs">${m.sym}</span><span class="eq-op">←</span>` +
            `<span class="eq-sym">${m.sym} − η·∂L/∂${m.sym}</span><span class="eq-op">=</span>` +
            `<span class="eq-num">${f(rec.before[i])} − ${f(rec.lr)} × ${fp(rec.grads[i])}</span><span class="eq-op">=</span>` +
            `<b class="eq-res">${f(rec.after[i])}</b><span class="delta ${d >= 0 ? 'up' : 'down'}">Δ ${MLV.fmtDelta(d)}</span></div>`;
        }).join('');
      } else {
        const rows = meta.map((m, i) => {
          const d = rec.after[i] - rec.before[i];
          return `<tr data-key="${m.key}" class="${st.focusKey === m.key ? 'is-focus' : ''}">` +
            `<td><span class="swatch" style="background:${seriesColor(m, i)}"></span>${m.sym}</td>` +
            `<td>${f(rec.before[i])}</td><td>${f(rec.grads[i])}</td><td>${f(rec.lr * rec.grads[i])}</td>` +
            `<td><b>${f(rec.after[i])}</b></td><td class="delta ${d >= 0 ? 'up' : 'down'}">${MLV.fmtDelta(d)}</td></tr>`;
        }).join('');
        html += '<div class="table-wrap"><table class="math-table update-table"><thead><tr>' +
          '<th>θ</th><th>θ before</th><th>∂L/∂θ</th><th>η·∂L/∂θ</th><th>θ after</th><th>Δ</th></tr></thead>' +
          `<tbody>${rows}</tbody></table></div><p class="aside">Click a row to pick that parameter out in the trajectory chart.</p>`;
      }
      html += `<p class="aside">These become the weights going into epoch ${rec.epoch + 1}. Loss there: ${f(st.points[rec.epoch] ? st.points[rec.epoch].loss : NaN)} (was ${f(rec.loss)}).</p>`;
      return html;
    }

    function renderMath() {
      const N = st.records.length;
      const range = $('epoch-range');
      let rec = null, reveal = 0;
      if (st.pending) { rec = st.pending.rec; reveal = st.pending.phase; }
      else if (N) { rec = st.records[inspected() - 1]; reveal = 4; }

      // scrubber
      range.max = Math.max(1, N);
      range.disabled = !N || !!st.pending;
      range.value = inspected() || 1;
      $('epoch-prev').disabled = !N || !!st.pending || (inspected() || 1) <= 1;
      $('epoch-next').disabled = !N || !!st.pending || (inspected() || 0) >= N;
      $('epoch-latest').disabled = !N || !!st.pending || st.selected === null;
      $('epoch-label').innerHTML = !rec
        ? 'No epochs yet'
        : st.pending
          ? `Epoch ${rec.epoch} <span class="muted">in progress · phase ${reveal} of 4</span>`
          : `Epoch ${rec.epoch} <span class="muted">of ${N} · θ<sub>${rec.epoch - 1}</sub> → θ<sub>${rec.epoch}</sub></span>`;

      // loop strip
      document.querySelectorAll('#loop-strip [data-phase]').forEach((el, i) => {
        el.classList.toggle('is-done', !!rec && i < reveal);
        el.classList.toggle('is-current', !!st.pending && i === reveal - 1);
      });

      const empty = $('math-empty');
      empty.hidden = !!rec;
      if (!rec) {
        PHASES.forEach((p) => { $('phase-' + p).hidden = true; });
        return;
      }
      const d = st.model.describe(rec, st.view);
      const bodies = { forward: d.forward, loss: d.loss, grads: d.grads, update: null };
      PHASES.forEach((p, i) => {
        const sec = $('phase-' + p);
        sec.hidden = false;
        const body = sec.querySelector('.phase-body');
        const shown = i < reveal;
        sec.classList.toggle('is-locked', !shown);
        if (!shown) body.innerHTML = '<p class="locked">Press <b>Next</b> to compute this step.</p>';
        else if (p === 'update') body.innerHTML = st.pending ? '' : updateSection(rec);
        else body.innerHTML = bodies[p];
      });
      // in phase mode the update is applied on the press that reveals it,
      // so phase 4 is never shown as pending.
    }

    function renderLog() {
      const meta = st.model.meta;
      const N = st.points.length - 1;
      const sel = st.pending ? -1 : (inspected() ?? -1);
      const head = '<tr><th>epoch</th>' + meta.map((m) => `<th>${m.sym}</th>`).join('') + '<th>loss</th></tr>';
      const LIMIT = 150;
      let rows = '';
      for (let k = N; k >= Math.max(0, N - LIMIT + 1); k--) {
        const pt = st.points[k];
        rows += `<tr data-epoch="${k}" class="${k === sel ? 'is-sel' : ''}"><td>${k}</td>` +
          pt.params.map((v) => `<td>${f(v)}</td>`).join('') + `<td>${f(pt.loss)}</td></tr>`;
      }
      $('log-table').innerHTML = `<table class="log"><thead>${head}</thead><tbody>${rows}</tbody></table>` +
        (N + 1 > LIMIT ? `<p class="aside">Showing the latest ${LIMIT} of ${N + 1} rows — download the CSV for all of them.</p>` : '');
    }

    function render() {
      renderStatus();
      renderCharts();
      renderMath();
      renderLog();
      const insp = inspected();
      if (st.pending) cfg.drawMain(st.model.getParams(), null);
      else if (insp) cfg.drawMain(st.points[insp].params, st.points[insp - 1].params);
      else cfg.drawMain(st.model.getParams(), null);
    }

    /* ---------------- wiring ---------------- */

    $('lr').addEventListener('input', () => { $('lr-val').textContent = f(lr(), 3); });
    $('speed').addEventListener('input', () => { $('speed-val').textContent = SPEEDS[parseInt($('speed').value, 10)] + ' / s'; });
    $('btn-step').addEventListener('click', stepOne);
    $('btn-step10').addEventListener('click', () => { stopPlay(); stepMany(10); });
    $('btn-play').addEventListener('click', () => (st.playing ? stopPlay() : startPlay()));
    $('btn-reset').addEventListener('click', () => restart(st.initParams));
    $('phase-mode').addEventListener('change', () => { st.pending = null; render(); });
    $('btn-apply-init').addEventListener('click', () => { st.initParams = readInitInputs(); restart(st.initParams); });

    $('epoch-range').addEventListener('input', (e) => select(parseInt(e.target.value, 10)));
    $('epoch-prev').addEventListener('click', () => select((inspected() || 1) - 1));
    $('epoch-next').addEventListener('click', () => select((inspected() || 0) + 1));
    $('epoch-latest').addEventListener('click', () => { st.selected = null; render(); });

    $('traj-legend').addEventListener('click', (e) => {
      const b = e.target.closest('[data-key]');
      if (!b) return;
      const k = b.dataset.key;
      st.hidden.has(k) ? st.hidden.delete(k) : st.hidden.add(k);
      renderLegend(); renderCharts();
    });

    document.querySelector('.math-panel').addEventListener('change', (e) => {
      if (e.target.matches('[data-trace-sample]')) {
        st.view.traceSample = parseInt(e.target.value, 10);
        renderMath();
      }
    });
    document.querySelector('.math-panel').addEventListener('click', (e) => {
      const tr = e.target.closest('.update-table tr[data-key]');
      if (!tr) return;
      st.focusKey = st.focusKey === tr.dataset.key ? null : tr.dataset.key;
      renderCharts(); renderMath();
    });

    $('log-table').addEventListener('click', (e) => {
      const tr = e.target.closest('tr[data-epoch]');
      if (tr) select(parseInt(tr.dataset.epoch, 10) || 1);
    });

    $('btn-download').addEventListener('click', () => {
      const meta = st.model.meta;
      const lines = ['epoch,' + meta.map((m) => m.key).join(',') + ',loss,learning_rate'];
      st.points.forEach((pt, k) => {
        const eta = k > 0 ? st.records[k - 1].lr : '';
        lines.push([k, ...pt.params, pt.loss, eta].join(','));
      });
      MLV.downloadText(`${st.model.kind}-epochs.csv`, lines.join('\n'));
    });

    document.addEventListener('keydown', (e) => {
      if (e.target.closest('input, textarea, select, button')) return;
      if (e.key === 'ArrowRight') { e.preventDefault(); stepOne(); }
    });

    // re-read colours when the OS theme flips
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { renderLegend(); render(); });

    $('lr-val').textContent = f(lr(), 3);
    $('speed-val').textContent = SPEEDS[parseInt($('speed').value, 10)] + ' / s';

    /* ---------------- public API ---------------- */

    const api = {
      state: st,
      render,
      // data changed: keep the current weights, restart the history from them
      dataChanged() { restart(); },
      // new initial weights (e.g. randomised by seed): fill inputs and restart
      setInitial(params) {
        st.initParams = params.slice();
        buildInitInputs();
        restart(params);
      },
      // swap the model (e.g. new hidden size); data is carried over by the page
      setModel(model, params) {
        st.model = model;
        st.hidden.clear();
        st.focusKey = null;
        renderLegend();
        api.setInitial(params || model.getParams());
      },
      get model() { return st.model; },
    };

    renderLegend();
    buildInitInputs();
    restart(st.initParams);
    return api;
  };
})();
