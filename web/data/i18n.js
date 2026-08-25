// RegimeFlow — English-only UI strings
const I18N = {

  // ===== Navigation =====
  'nav.overview':     { en: 'About' },
  'nav.graph':        { en: 'Models' },
  'nav.predict':      { en: 'Trajectory Prediction' },
  'nav.models':       { en: 'Models' },

  // ===== Overview =====
  'overview.heroTitle':    { en: 'Cross-System Trajectory Prediction' },
  'overview.heroSub':      { en: 'for 1000+ Biological Systems' },
  'overview.heroDesc':     { en: 'RegimeFlow is a regime-aware flow matching framework that encodes macroscopic behavioral patterns of biological systems as generative priors, enabling probabilistic trajectory prediction with uncertainty quantification across heterogeneous systems.' },
  'overview.paperTag':     { en: '📄 ICML 2026 Accepted Paper' },
  'overview.paperBtn':     { en: 'Read Paper (OpenReview)' },
  'overview.codeBtn':      { en: 'GitHub Repository' },
  'overview.datasetBtn':   { en: 'Dataset (HuggingFace)' },
  'overview.demoBtn':      { en: '🔬 Try Online Prediction →' },

  // Key metrics
  'overview.metricsTitle': { en: 'Key Performance' },
  'overview.metric1.val':  { en: '1,050' },
  'overview.metric1.desc': { en: 'ODE Biological Systems' },
  'overview.metric2.val':  { en: '31%' },
  'overview.metric2.desc': { en: 'MAE Reduction' },
  'overview.metric3.val':  { en: '17%' },
  'overview.metric3.desc': { en: 'CRPS Improvement' },
  'overview.metric4.val':  { en: '9.3×' },
  'overview.metric4.desc': { en: 'Faster Inference' },

  // Method cards
  'overview.methodTitle':  { en: 'How It Works' },
  'overview.step1.title':  { en: '① BLR Prior' },
  'overview.step1.desc':   { en: 'Bayesian Linear Regression constructs structured initial states from regime-specific basis functions (exponential decay, Fourier, polynomial), replacing standard Gaussian noise.' },
  'overview.step2.title':  { en: '② Conditional FM' },
  'overview.step2.desc':   { en: 'Conditional Flow Matching learns optimal transport paths from regime-aware priors to target trajectories, enabling high-quality predictions with single-step sampling.' },
  'overview.step3.title':  { en: '③ SSM + AdaLN Backbone' },
  'overview.step3.desc':   { en: 'Mamba state-space model backbone with linear time complexity. Adaptive Layer Normalization (AdaLN) dynamically modulates network behavior based on regime information.' },

  // Regime types
  'overview.regimeTitle':  { en: 'Biological Regime Types' },
  'overview.regime.stable':      { en: 'Stable' },
  'overview.regime.stable.desc': { en: 'Asymptotic relaxation to steady state' },
  'overview.regime.osc':         { en: 'Oscillatory' },
  'overview.regime.osc.desc':    { en: 'Sustained periodic fluctuations' },
  'overview.regime.mono':        { en: 'Monotonic' },
  'overview.regime.mono.desc':   { en: 'Irreversible directional change' },

  // Benchmark
  'overview.benchTitle':  { en: 'SysBio-Traj Benchmark' },
  'overview.benchDesc':   { en: '1,050 ODE-based systems spanning diverse organisms, biological processes, and dynamical behaviors in a standardized Python framework. RegimeFlow outperforms all 17 baselines comprehensively.' },
  'overview.benchBtn':    { en: 'Explore All Models →' },

  // ===== Bubble chart =====
  'graph.title':      { en: 'Model Bubble Chart' },
  'graph.desc':       { en: '17 time-series forecasting models · Grouped by family · Click to zoom · Click background to go back' },
  'graph.modelsCount':{ en: '{n} models' },
  'graph.clickToExplore': { en: 'Click to explore' },

  // legend
  'legend.fm':        { en: 'Flow Matching' },
  'legend.diff':      { en: 'Diffusion' },
  'legend.tf':        { en: 'Transformer' },
  'legend.mlp':       { en: 'MLP-Mixer' },
  'legend.linear':    { en: 'Linear' },
  'legend.mamba':     { en: 'Mamba/SSM' },
  'legend.zero':      { en: 'Zero-shot' },
  'legend.size':      { en: 'Bubble size ≈ model capacity' },
  'legend.hint':      { en: 'Click to zoom in  ·  Click background to go back  ·  Hover for details' },

  // tooltip
  'tooltip.clickHint':  { en: 'Click for details' },
  'tooltip.pretrained': { en: 'Pretrained (T5)' },

  // model type labels
  'type.probabilistic': { en: 'Probabilistic' },
  'type.point':         { en: 'Point Forecast' },
  'type.zero-shot':     { en: 'Zero-shot' },

  // ===== Modal =====
  'modal.family':       { en: 'Family' },
  'modal.hiddenDim':    { en: 'Hidden Dim' },
  'modal.layers':       { en: 'Layers' },
  'modal.lr':           { en: 'Learning Rate' },
  'modal.ctxLen':       { en: 'Context Length' },
  'modal.predLen':      { en: 'Prediction Length' },
  'modal.dState':       { en: 'State Dim' },
  'modal.nHeads':       { en: 'Attention Heads' },
  'modal.numSteps':     { en: 'Sampling Steps' },
  'modal.solver':       { en: 'Solver' },
  'modal.prior':        { en: 'Prior' },
  'modal.pretrained':   { en: 'Pretrained Model' },
  'modal.btnPredict':   { en: '🔬 Try Prediction →' },
  'modal.pretrainedNA': { en: 'Pretrained Model' },

  // ===== Prediction page =====
  'predict.title':        { en: 'Systems Biology Trajectory Prediction' },
  'predict.desc':         { en: 'Select system → First 96 steps as input trajectory → RegimeFlow engine generates next 256-step forecast with confidence intervals' },
  'predict.tabModels':    { en: 'Models' },
  'predict.tabCustom':    { en: 'Custom Data' },
  'predict.inputLabel':   { en: 'Paste time series (one value per line, or time,value CSV)' },
  'predict.inputPlaceholder': { en: 'e.g.:\n0, 1.2\n0.1, 1.5\n0.2, 1.8\n0.3, 2.1\n...\n(at least 10 data points)' },
  'predict.ctxLen':       { en: 'Context length:' },
  'predict.ctxUnit':      { en: 'pts' },
  'predict.btnPredict':   { en: '🔮 Start Predicting' },
  'predict.btnPredicting':{ en: '⏳ Predicting… wait a few seconds' },
  'predict.inputHint':    { en: 'First N points = input trajectory (blue) · Remaining = baseline (gray dashed)\nSupports CSV / TSV / plain numbers' },
  'predict.sidebarNote':  { en: '🔄 Connecting to backend…\nClick a species to run\nRegimeFlow prediction engine.' },
  'predict.sidebarReady': { en: '✅ RegimeFlow engine ready\nDevice: {device}\nClick a species for\nAI-predicted trajectories.' },
  'predict.sidebarError': { en: '⚠️ Backend offline\n{error}\nShowing synthetic data only.' },
  'predict.selectSpecies':{ en: 'Species:' },
  'predict.titleText':    { en: '{sys} — Trajectory Prediction' },
  'predict.subtitleAPI':  { en: '{sp} · First 96 steps (blue) = Input Trajectory · Next 256 = Prediction\nModel: {model} · Inference: {time}ms · Samples: {samples}' },
  'predict.subtitleOffline': { en: '{sp} · First 96 steps (blue) = Input Trajectory · Next 256 = Prediction\nBackend offline — synthetic baseline only' },

  // chart series names
  'chart.context':      { en: 'Input Trajectory' },
  'chart.groundTruth':  { en: 'Ground Truth' },
  'chart.chronos':      { en: 'RegimeFlow Prediction' },
  'chart.confidence':   { en: '90% Confidence interval' },

  // custom predict status
  'status.parseError':       { en: '❌ ' },
  'status.noData':           { en: 'Please paste data first' },
  'status.tooFew':           { en: 'Not enough data points (min 10), found {n} values' },
  'status.predicting':       { en: 'Calling RegimeFlow engine…' },
  'status.done':             { en: '✅ Done · Model: {model} · {time}ms' },
  'status.apiError':         { en: '⚠️ API error: ' },
  'status.offline':          { en: '⚠️ Backend offline — raw data only' },

  // prediction custom data chart
  'predict.customData':      { en: 'Custom Data' },

  // ===== Biological system names =====
  'sys.oscillation':         { en: 'Ca²⁺ Oscillation Model' },
  'sys.increasingStable':    { en: 'Bacterial Growth Curve' },
  'sys.decreasingStable':    { en: 'Protein Degradation Model' },
  'sys.monotonicInc':        { en: 'Cell Volume Growth' },
  'sys.monotonicDec':        { en: 'Signaling Cascade Decay' },
  'sys.complex':             { en: 'Circadian-Growth Coupling' },

  // system descriptions
  'sys.oscillation.desc':    { en: 'Cytosolic Ca²⁺ oscillations, ER calcium store & IP3 periodic dynamics (synthetic demo)' },
  'sys.increasingStable.desc': { en: 'Biomass growth to steady state, substrate depletion & oxygen utilization (synthetic demo)' },
  'sys.decreasingStable.desc': { en: 'Ubiquitin-mediated protein degradation, ubiquitin consumption & proteasome activity (synthetic demo)' },
  'sys.monotonicInc.desc':   { en: 'Continuous cell volume growth, protein accumulation & ATP supply (synthetic demo)' },
  'sys.monotonicDec.desc':   { en: 'Kinase activity decay, substrate phosphorylation & phosphatase action (synthetic demo)' },
  'sys.complex.desc':        { en: 'Circadian mRNA/protein oscillation coupled with growth factor dynamics (synthetic demo)' },

  // ===== Model family names =====
  'family.Flow Matching': { en: 'Flow Matching' },
  'family.Diffusion':     { en: 'Diffusion' },
  'family.Transformer':   { en: 'Transformer' },
  'family.MLP-Mixer':     { en: 'MLP-Mixer' },
  'family.Linear':        { en: 'Linear' },
  'family.Mamba':         { en: 'Mamba' },
  'family.Zero-shot':     { en: 'Zero-shot' },

  // ===== Regime types =====
  'regime.oscillation':         { en: 'Oscillation' },
  'regime.increasing-stable':   { en: 'Increase → Stable' },
  'regime.decreasing-stable':   { en: 'Decay → Stable' },
  'regime.monotonic increasing':{ en: 'Monotonic Increase' },
  'regime.monotonic decreasing':{ en: 'Monotonic Decrease' },
  'regime.complex':             { en: 'Complex' },

  // species count
  'sys.speciesCount': { en: '{n} species' },
};
