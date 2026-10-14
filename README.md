# ML Model Visualizer

**Live site:** https://yassharth.github.io/ml_model_visualizer/

An interactive learning tool for the classic training loop. Pick a model, set your own
starting weights and data, then step through gradient descent and watch every weight and
bias change, epoch by epoch, with the real numbers behind each update.

| Page | Model | Parameters |
|---|---|---|
| `linear-regression.html` | ŷ = m·x + b, mean squared error | 2 |
| `logistic-regression.html` | p = σ(w₁x₁ + w₂x₂ + b), binary cross-entropy | 3 |
| `neural-network.html` | 2 → H → 1 MLP (tanh / sigmoid / ReLU hidden, sigmoid out), BCE | 4H + 1 (up to 33) |

## What each page shows

- **Weights & biases, epoch by epoch** – one line per parameter across all epochs (the
  neural network colours lines by layer; hover a line to name it). Click any epoch to inspect it.
- **Loss** – L(θ) after every update.
- **Inside the epoch** – the four steps of one epoch, filled in with real numbers:
  1. **Forward pass** – a table of every sample's prediction and error.
  2. **Loss** – the loss formula with the actual sums plugged in.
  3. **Gradients** – each partial derivative worked out. On the neural network page this is a
     unit-by-unit backpropagation trace for any sample you pick.
  4. **Update** – θ ← θ − η·∂L/∂θ for every parameter: old value, gradient, step, new value, Δ.
- **Phase by phase** mode – each press of Step reveals one of the four steps; the update is
  applied on the fourth press.
- **Epoch log** – every parameter and the loss at every epoch, downloadable as CSV.
- Scrub back to any earlier epoch; the plot shows the model after that epoch with the
  previous one dashed underneath.

Training is **full-batch gradient descent**: one epoch = one pass over all samples = exactly
one update. That keeps each step deterministic and easy to check by hand.

## Bring your own

- **Weights** – every page has an *Initial weights* panel with one input per parameter.
  The neural network can also be randomised from a seed and scale, so runs are reproducible.
- **Data** – click the plot to add points, pick a preset, or paste CSV
  (`x, y` for linear regression; `x1, x2, label` for the classifiers).
- **Learning rate** – push it high enough and training diverges; the page says so.

## Run locally

No build step and no dependencies to install. Open `index.html` in a browser, or serve the
folder:

```bash
python3 -m http.server 8000   # then open http://localhost:8000
```

## Project layout

```
index.html                 landing page
linear-regression.html     ┐
logistic-regression.html   ├ one page per model, same layout
neural-network.html        ┘
css/style.css              design tokens (light + dark), layout, tables
js/core.js                 formatting, seeded RNG, CSV parsing, dataset generators
js/charts.js               Plot2D (data plot) and LineChart (values over epochs)
js/models.js               the models: evaluate() = loss + gradients, describe() = worked math
js/trainer.js              the training loop, epoch history, and all the panels above
js/pages/*.js              per-page wiring: data controls and the main plot
```

### Adding a model (CNN, RNN, LSTM…)

A model is an object with `meta` (one entry per parameter), `getParams` / `setParams`,
`size()`, `evaluate(params) → { loss, grads }` and `describe(record) → { forward, loss, grads }`
(HTML strings for steps ①–③). `MLV.createTrainer` handles everything else — history,
trajectory and loss charts, the update step, the epoch log — so a new page is mostly a new
model file plus its own main plot. Copy `neural-network.html` and `js/pages/nn.js` as a starting point.

All gradients are checked against finite differences; the analytic and numeric values agree
to within 1e-6 relative error for every model and activation.

## Deploying

The site is plain static files, served by GitHub Pages from the `main` branch root
(**Settings → Pages → Deploy from a branch → main / (root)**). The empty `.nojekyll`
file tells Pages to serve the files as they are.
