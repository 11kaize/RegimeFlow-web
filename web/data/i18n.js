// RegimeFlow — 中英文翻译词典
const I18N = {

  // ===== 导航栏 =====
  'nav.overview':     { zh: '关于',               en: 'About' },
  'nav.graph':        { zh: '探索模型',           en: 'Models' },
  'nav.predict':      { zh: '轨迹预测',           en: 'Prediction' },
  'nav.models':      { zh: '模型库',             en: 'Models' },

  // ===== 概览页 =====
  'overview.heroTitle':    { zh: '面向 1000+ 生物系统的', en: 'Cross-System Trajectory Prediction' },
'overview.heroSub':      { zh: '跨系统轨迹预测框架', en: 'for 1000+ Biological Systems' },
  'overview.heroDesc':     { zh: 'RegimeFlow 是一个 regime-aware 的流匹配框架，通过将生物系统的宏观行为模式编码为生成先验，实现对异构生物系统轨迹的概率预测与不确定度量化。', en: 'RegimeFlow is a regime-aware flow matching framework that encodes macroscopic behavioral patterns of biological systems as generative priors, enabling probabilistic trajectory prediction with uncertainty quantification across heterogeneous systems.' },
  'overview.paperTag':     { zh: '📄 ICML 2026 接收论文', en: '📄 ICML 2026 Accepted Paper' },
  'overview.paperBtn':     { zh: '阅读论文 (OpenReview)', en: 'Read Paper (OpenReview)' },
  'overview.codeBtn':      { zh: 'GitHub 代码仓库', en: 'GitHub Repository' },
  'overview.datasetBtn':   { zh: '数据集 (HuggingFace)', en: 'Dataset (HuggingFace)' },
  'overview.demoBtn':      { zh: '🔬 试试在线预测 →', en: '🔬 Try Online Prediction →' },

  // 关键指标
  'overview.metricsTitle': { zh: '核心表现', en: 'Key Performance' },
  'overview.metric1.val':  { zh: '1,050', en: '1,050' },
  'overview.metric1.desc': { zh: 'ODE 生物系统', en: 'ODE Biological Systems' },
  'overview.metric2.val':  { zh: '31%', en: '31%' },
  'overview.metric2.desc': { zh: 'MAE 降低', en: 'MAE Reduction' },
  'overview.metric3.val':  { zh: '17%', en: '17%' },
  'overview.metric3.desc': { zh: 'CRPS 提升', en: 'CRPS Improvement' },
  'overview.metric4.val':  { zh: '9.3×', en: '9.3×' },
  'overview.metric4.desc': { zh: '推理加速', en: 'Faster Inference' },

  // 方法卡片
  'overview.methodTitle':  { zh: '核心方法', en: 'How It Works' },
  'overview.step1.title':  { zh: '① BLR 先验构造', en: '① BLR Prior' },
  'overview.step1.desc':   { zh: '贝叶斯线性回归从 regime-specific 基函数（指数衰减、傅里叶、多项式）构造结构化初始状态，替代标准高斯噪声。', en: 'Bayesian Linear Regression constructs structured initial states from regime-specific basis functions (exponential decay, Fourier, polynomial), replacing standard Gaussian noise.' },
  'overview.step2.title':  { zh: '② 条件流匹配', en: '② Conditional FM' },
  'overview.step2.desc':   { zh: '条件流匹配学习从 regime-aware 先验分布到目标轨迹的最优传输路径，单步采样即可生成高质量预测。', en: 'Conditional Flow Matching learns optimal transport paths from regime-aware priors to target trajectories, enabling high-quality predictions with single-step sampling.' },
  'overview.step3.title':  { zh: '③ SSM + AdaLN 骨干', en: '③ SSM + AdaLN Backbone' },
  'overview.step3.desc':   { zh: 'Mamba 状态空间模型骨干网络，线性时间复杂度。自适应层归一化 (AdaLN) 根据 regime 信息动态调制网络行为。', en: 'Mamba state-space model backbone with linear time complexity. Adaptive Layer Normalization (AdaLN) dynamically modulates network behavior based on regime information.' },

  // Regime 类型
  'overview.regimeTitle':  { zh: '生物 Regime 类型', en: 'Biological Regime Types' },
  'overview.regime.stable':      { zh: '稳态型', en: 'Stable' },
  'overview.regime.stable.desc': { zh: '渐进松弛至稳态', en: 'Asymptotic relaxation to steady state' },
  'overview.regime.osc':         { zh: '振荡型', en: 'Oscillatory' },
  'overview.regime.osc.desc':    { zh: '持续周期性波动', en: 'Sustained periodic fluctuations' },
  'overview.regime.mono':        { zh: '单调型', en: 'Monotonic' },
  'overview.regime.mono.desc':   { zh: '不可逆方向性变化', en: 'Irreversible directional change' },

  // 基准对比
  'overview.benchTitle':  { zh: 'SysBio-Traj 基准', en: 'SysBio-Traj Benchmark' },
  'overview.benchDesc':   { zh: '覆盖多样生物、过程和动态行为的 1,050 个 ODE 系统，标准化 Python 框架。RegimeFlow 在此基准上全面超越 17 个对比模型。', en: '1,050 ODE-based systems spanning diverse organisms, biological processes, and dynamical behaviors in a standardized Python framework. RegimeFlow outperforms all 17 baselines comprehensively.' },
  'overview.benchBtn':    { zh: '探索全部模型 →', en: 'Explore All Models →' },

  // ===== 气泡图 =====
  'graph.title':      { zh: '模型气泡图',         en: 'Model Bubble Chart' },
  'graph.desc':       { zh: '17 个时序预测模型 · 按家族分组 · 点击气泡深入探索 · 点击空白返回',
                             en: '17 time-series forecasting models · Grouped by family · Click to zoom · Click background to go back' },
  'graph.modelsCount':{ zh: '{n} 个模型',          en: '{n} models' },
  'graph.clickToExplore': { zh: '点击深入探索',   en: 'Click to explore' },

  // legend
  'legend.fm':        { zh: '流匹配',             en: 'Flow Matching' },
  'legend.diff':      { zh: '扩散模型',           en: 'Diffusion' },
  'legend.tf':        { zh: 'Transformer',        en: 'Transformer' },
  'legend.mlp':       { zh: 'MLP-Mixer',          en: 'MLP-Mixer' },
  'legend.linear':    { zh: '线性模型',           en: 'Linear' },
  'legend.mamba':     { zh: 'Mamba/SSM',          en: 'Mamba/SSM' },
  'legend.zero':      { zh: '零样本',             en: 'Zero-shot' },
  'legend.size':      { zh: '泡泡大小 ≈ 模型容量', en: 'Bubble size ≈ model capacity' },
  'legend.hint':      { zh: '点击气泡深入探索  ·  点击空白返回  ·  悬停查看详情', en: 'Click to zoom in  ·  Click background to go back  ·  Hover for details' },

  // tooltip
  'tooltip.clickHint':  { zh: '点击查看详情', en: 'Click for details' },
  'tooltip.pretrained': { zh: '预训练模型 (T5)',   en: 'Pretrained (T5)' },

  // model type labels
  'type.probabilistic': { zh: '概率预测',         en: 'Probabilistic' },
  'type.point':         { zh: '点预测',           en: 'Point Forecast' },
  'type.zero-shot':     { zh: '零样本',           en: 'Zero-shot' },

  // ===== 模态框 =====
  'modal.family':       { zh: '模型家族',         en: 'Family' },
  'modal.hiddenDim':    { zh: '隐藏维度',         en: 'Hidden Dim' },
  'modal.layers':       { zh: '层数',             en: 'Layers' },
  'modal.lr':           { zh: '学习率',           en: 'Learning Rate' },
  'modal.ctxLen':       { zh: '上下文长度',       en: 'Context Length' },
  'modal.predLen':      { zh: '预测长度',         en: 'Prediction Length' },
  'modal.dState':       { zh: '状态维度',         en: 'State Dim' },
  'modal.nHeads':       { zh: '注意力头数',       en: 'Attention Heads' },
  'modal.numSteps':     { zh: '采样步数',         en: 'Sampling Steps' },
  'modal.solver':       { zh: '求解器',           en: 'Solver' },
  'modal.prior':        { zh: '先验分布',         en: 'Prior' },
  'modal.pretrained':   { zh: '预训练模型',       en: 'Pretrained Model' },
  'modal.btnPredict':   { zh: '🔬 试预测 —— 查看轨迹演示', en: '🔬 Try Prediction →' },
  'modal.pretrainedNA': { zh: '预训练模型',       en: 'Pretrained Model' },

  // ===== 预测页 =====
  'predict.title':        { zh: '轨迹预测演示',   en: 'Trajectory Prediction' },
  'predict.desc':         { zh: '选择生物系统 → 前96步为上下文 → RegimeFlow 预测引擎生成后256步轨迹预测与置信区间',
                                 en: 'Select system → First 96 steps as context → RegimeFlow engine generates next 256-step forecast with confidence intervals' },
  'predict.tabExamples':  { zh: '示例系统',       en: 'Examples' },
  'predict.tabCustom':    { zh: '自定义数据',     en: 'Custom Data' },
  'predict.inputLabel':   { zh: '粘贴时序数据（每行一个值，或 time,value 两列）',
                                 en: 'Paste time series (one value per line, or time,value CSV)' },
  'predict.inputPlaceholder': { zh: '例如：\n0, 1.2\n0.1, 1.5\n0.2, 1.8\n0.3, 2.1\n...\n（至少10个数据点）',
                                 en: 'e.g.:\n0, 1.2\n0.1, 1.5\n0.2, 1.8\n0.3, 2.1\n...\n(at least 10 data points)' },
  'predict.ctxLen':       { zh: '上下文长度:',    en: 'Context length:' },
  'predict.ctxUnit':      { zh: '个点',           en: 'pts' },
  'predict.btnPredict':   { zh: '🔮 开始预测',    en: '🔮 Predict' },
  'predict.btnPredicting':{ zh: '⏳ 预测中…',     en: '⏳ Predicting…' },
  'predict.inputHint':    { zh: '前 N 个点 = 上下文（蓝色）· 剩余 = 对比基准（灰色虚线）\n支持 CSV / TSV / 纯数值格式',
                                 en: 'First N points = context (blue) · Remaining = baseline (gray dashed)\nSupports CSV / TSV / plain numbers' },
  'predict.sidebarNote':  { zh: '🔄 正在连接后端…\n点击物种自动调用\nRegimeFlow 预测引擎。',
                                 en: '🔄 Connecting to backend…\nClick a species to run\nRegimeFlow prediction engine.' },
  'predict.sidebarReady': { zh: '✅ RegimeFlow 预测引擎已就绪\n设备: {device}\n点击物种即可获得\nAI 预测轨迹。',
                                 en: '✅ RegimeFlow engine ready\nDevice: {device}\nClick a species for\nAI-predicted trajectories.' },
  'predict.sidebarError': { zh: '⚠️ 后端未连接\n{error}\n当前显示合成数据。',
                                 en: '⚠️ Backend offline\n{error}\nShowing synthetic data only.' },
  'predict.selectSpecies':{ zh: '选择物种:',      en: 'Species:' },
  'predict.titleText':    { zh: '{sys} — 轨迹预测', en: '{sys} — Trajectory Prediction' },
  'predict.subtitleAPI':  { zh: '{sp} · 前96步 (蓝) = 上下文 · 后256步 = 预测区域\n模型: {model} · 推理耗时: {time}ms · 样本数: {samples}',
                                 en: '{sp} · First 96 steps (blue) = Context · Next 256 = Prediction\nModel: {model} · Inference: {time}ms · Samples: {samples}' },
  'predict.subtitleOffline': { zh: '{sp} · 前96步 (蓝) = 上下文 · 后256步 = 预测区域\n后端未连接 — 仅显示合成基准数据',
                                 en: '{sp} · First 96 steps (blue) = Context · Next 256 = Prediction\nBackend offline — synthetic baseline only' },

  // chart series names
  'chart.context':      { zh: '输入上下文 (Context)',      en: 'Context' },
  'chart.groundTruth':  { zh: '合成基准 (Ground Truth)',   en: 'Ground Truth' },
  'chart.chronos':      { zh: 'RegimeFlow 预测',         en: 'RegimeFlow Prediction' },
  'chart.confidence':   { zh: '90% 置信区间',              en: '90% Confidence interval' },

  // custom predict status
  'status.parseError':       { zh: '❌ ',           en: '❌ ' },
  'status.noData':           { zh: '请先粘贴数据',  en: 'Please paste data first' },
  'status.tooFew':           { zh: '有效数据点不足（至少需要 10 个），当前解析到 {n} 个数值',
                                   en: 'Not enough data points (min 10), found {n} values' },
  'status.predicting':       { zh: '正在调用 RegimeFlow 预测引擎…', en: 'Calling RegimeFlow engine…' },
  'status.done':             { zh: '✅ 预测完成 · 模型: {model} · 耗时: {time}ms', en: '✅ Done · Model: {model} · {time}ms' },
  'status.apiError':         { zh: '⚠️ API 错误: ', en: '⚠️ API error: ' },
  'status.offline':          { zh: '⚠️ 后端未连接 — 仅显示原始数据', en: '⚠️ Backend offline — raw data only' },

  // prediction custom data chart
  'predict.customData':      { zh: '自定义数据',   en: 'Custom Data' },

  // ===== 生物系统名称 =====
  'sys.oscillation':         { zh: '钙离子振荡模型',       en: 'Ca²⁺ Oscillation Model' },
  'sys.increasingStable':    { zh: '细菌生长曲线',         en: 'Bacterial Growth Curve' },
  'sys.decreasingStable':    { zh: '蛋白质降解模型',       en: 'Protein Degradation Model' },
  'sys.monotonicInc':        { zh: '细胞体积增长',         en: 'Cell Volume Growth' },
  'sys.monotonicDec':        { zh: '信号级联衰减',         en: 'Signaling Cascade Decay' },
  'sys.complex':             { zh: '昼夜节律与生长耦合',   en: 'Circadian-Growth Coupling' },

  // system descriptions
  'sys.oscillation.desc':    { zh: '胞质钙振荡、内质网钙库与IP3的周期性动力学 (合成演示数据)', en: 'Cytosolic Ca²⁺ oscillations, ER calcium store & IP3 periodic dynamics (synthetic demo)' },
  'sys.increasingStable.desc': { zh: '生物量增长至稳态、底物消耗与氧利用 (合成演示数据)', en: 'Biomass growth to steady state, substrate depletion & oxygen utilization (synthetic demo)' },
  'sys.decreasingStable.desc': { zh: '泛素化介导的蛋白质降解、泛素消耗与蛋白酶体活性 (合成演示数据)', en: 'Ubiquitin-mediated protein degradation, ubiquitin consumption & proteasome activity (synthetic demo)' },
  'sys.monotonicInc.desc':   { zh: '细胞体积持续增长、蛋白质累积与ATP供给 (合成演示数据)', en: 'Continuous cell volume growth, protein accumulation & ATP supply (synthetic demo)' },
  'sys.monotonicDec.desc':   { zh: '磷酸激酶活性衰减、底物磷酸化与磷酸酶作用 (合成演示数据)', en: 'Kinase activity decay, substrate phosphorylation & phosphatase action (synthetic demo)' },
  'sys.complex.desc':        { zh: '昼夜节律的mRNA/蛋白振荡与生长因子的复杂耦合 (合成演示数据)', en: 'Circadian mRNA/protein oscillation coupled with growth factor dynamics (synthetic demo)' },

  // ===== 模型家族名称（数据字段翻译） =====
  'family.Flow Matching': { zh: '流匹配',       en: 'Flow Matching' },
  'family.Diffusion':     { zh: '扩散模型',     en: 'Diffusion' },
  'family.Transformer':   { zh: 'Transformer',  en: 'Transformer' },
  'family.MLP-Mixer':     { zh: 'MLP-Mixer',    en: 'MLP-Mixer' },
  'family.Linear':        { zh: '线性模型',     en: 'Linear' },
  'family.Mamba':         { zh: 'Mamba/SSM',    en: 'Mamba' },
  'family.Zero-shot':     { zh: '零样本',       en: 'Zero-shot' },

  // ===== 模态类型 =====
  'regime.oscillation':         { zh: '振荡型',         en: 'Oscillation' },
  'regime.increasing-stable':   { zh: '增长→稳态',     en: 'Increase → Stable' },
  'regime.decreasing-stable':   { zh: '衰减→稳态',     en: 'Decay → Stable' },
  'regime.monotonic increasing':{ zh: '单调增长',       en: 'Monotonic Increase' },
  'regime.monotonic decreasing':{ zh: '单调衰减',       en: 'Monotonic Decrease' },
  'regime.complex':             { zh: '复合型',         en: 'Complex' },

  // 系统物种数
  'sys.speciesCount': { zh: '{n} 物种', en: '{n} species' },
};
