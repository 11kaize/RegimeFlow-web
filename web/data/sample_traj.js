// 6 种合成演示轨迹 —— 对应 RegimeFlow 论文中的典型 regime 类型
// ⚠️ 这些是 JS 客户端生成的合成数据,仅用于 UI 演示
// 论文真实数据: SysBio-Traj 基准包含 1,050 个 ODE 生物系统
//   (e.g. Queralt2006 双稳态, Novak2022 振荡型, dePillis2005 肿瘤免疫, Lopez2013 细胞凋亡)
// 每个系统: { id, name, regime, time[], species: {name: values[]}[] }
// 总长度 = 362 point (96 context + 256 prediction + 10 overlap)

const SAMPLE_SYSTEMS = [];

function generateTrajectory(config) {
  const total = 362;
  const t = Array.from({length: total}, (_, i) => i * 0.1);
  const species = {};

  config.species.forEach(sp => {
    const vals = [];
    for (let i = 0; i < total; i++) {
      let v = 0;
      if (sp.type === 'oscillation') {
        v = sp.amplitude * Math.sin(2 * Math.PI * sp.freq * t[i] + sp.phase) +
            sp.amplitude * 0.3 * Math.sin(2 * Math.PI * sp.freq * 2.5 * t[i]) +
            sp.baseline + sp.noise * (Math.random() - 0.5);
      } else if (sp.type === 'growth') {
        v = sp.baseline + sp.scale * (1 - Math.exp(-sp.rate * t[i])) +
            sp.noise * (Math.random() - 0.5);
      } else if (sp.type === 'decay') {
        v = sp.baseline + sp.scale * Math.exp(-sp.rate * t[i]) +
            sp.noise * (Math.random() - 0.5);
      } else if (sp.type === 'logistic') {
        v = sp.baseline + sp.scale / (1 + Math.exp(-sp.rate * (t[i] - sp.midpoint))) +
            sp.noise * (Math.random() - 0.5);
      } else if (sp.type === 'linear') {
        v = sp.baseline + sp.slope * t[i] + sp.noise * (Math.random() - 0.5);
      } else if (sp.type === 'complex') {
        v = sp.baseline +
            sp.scale * (1 - Math.exp(-sp.rate * t[i])) +
            sp.amplitude * Math.sin(2 * Math.PI * sp.freq * t[i]) *
            Math.exp(-0.003 * t[i]) +
            sp.amplitude * 0.5 * Math.sin(2 * Math.PI * sp.freq * 3.1 * t[i] + 1.5) +
            sp.noise * (Math.random() - 0.5);
      }
      vals.push(parseFloat(v.toFixed(4)));
    }
    species[sp.name] = vals;
  });

  const regimeLabels = [];
  for (let i = 0; i < total; i++) {
    if (i < 96) regimeLabels.push('context');
    else if (i < 106) regimeLabels.push('overlap');
    else regimeLabels.push('prediction');
  }

  return {
    id: config.id,
    name: config.name,
    regime: config.regime,
    description: config.description,
    speciesCount: config.species.length,
    speciesNames: config.species.map(s => s.name),
    time: t,
    species: species,
    regimeLabels: regimeLabels
  };
}

// 1. 振荡型 — 钙离子振荡 (Calcium oscillation)
SAMPLE_SYSTEMS.push(generateTrajectory({
  id: 'sys_oscillation', name: '钙离子振荡模型', regime: 'oscillation',
  description: '典型的生物振荡系统，模拟细胞内钙离子浓度的周期性波动。多频率叠加，振幅随时间缓慢衰减。',
  species: [
    { name: 'Ca_cyt', type: 'complex', baseline: 0.5, scale: 1.2, rate: 0.005, amplitude: 0.4, freq: 0.15, noise: 0.05 },
    { name: 'Ca_ER', type: 'oscillation', baseline: 2.0, amplitude: 0.6, freq: 0.15, phase: 3.14, noise: 0.04 },
    { name: 'IP3', type: 'oscillation', baseline: 0.8, amplitude: 0.25, freq: 0.15, phase: 1.57, noise: 0.03 }
  ]
}));

// 2. 增长稳态型 — 细菌生长曲线 (Bacterial growth)
SAMPLE_SYSTEMS.push(generateTrajectory({
  id: 'sys_growth_stable', name: '细菌生长曲线', regime: 'increasing-stable',
  description: '典型 logistic 生长模型，模拟细菌种群从指数增长到环境容纳量 (carrying capacity) 的饱和过程。',
  species: [
    { name: 'Biomass', type: 'logistic', baseline: 0.1, scale: 3.5, rate: 0.04, midpoint: 60, noise: 0.06 },
    { name: 'Substrate', type: 'decay', baseline: 0.5, scale: 3.0, rate: 0.03, noise: 0.05 },
    { name: 'Oxygen', type: 'decay', baseline: 0.3, scale: 1.5, rate: 0.025, noise: 0.04 }
  ]
}));

// 3. 衰减稳态型 — 蛋白质降解 (Protein degradation)
SAMPLE_SYSTEMS.push(generateTrajectory({
  id: 'sys_decay_stable', name: '蛋白质降解模型', regime: 'decreasing-stable',
  description: '模拟蛋白质在细胞内的降解过程——初始浓度快速下降后趋于稳态基线水平。',
  species: [
    { name: 'Protein', type: 'decay', baseline: 1.0, scale: 4.0, rate: 0.05, noise: 0.07 },
    { name: 'Ubiquitin', type: 'growth', baseline: 0.2, scale: 1.8, rate: 0.04, noise: 0.04 },
    { name: 'Proteasome', type: 'linear', baseline: 2.5, slope: 0.002, noise: 0.03 }
  ]
}));

// 4. 单调增长型 — 细胞体积增长 (Cell volume growth)
SAMPLE_SYSTEMS.push(generateTrajectory({
  id: 'sys_mono_increase', name: '细胞体积增长', regime: 'monotonic increasing',
  description: '模拟细胞在营养充足条件下的持续体积增长，伴有周期性的轻微波动（细胞周期）。',
  species: [
    { name: 'Volume', type: 'growth', baseline: 1.0, scale: 2.5, rate: 0.015, noise: 0.08 },
    { name: 'Protein_content', type: 'growth', baseline: 1.2, scale: 2.8, rate: 0.013, noise: 0.06 },
    { name: 'ATP', type: 'oscillation', baseline: 3.0, amplitude: 0.35, freq: 0.08, phase: 0, noise: 0.05 }
  ]
}));

// 5. 单调衰减型 — 信号衰减 (Signal attenuation)
SAMPLE_SYSTEMS.push(generateTrajectory({
  id: 'sys_mono_decrease', name: '信号级联衰减', regime: 'monotonic decreasing',
  description: '模拟细胞信号转导中的信号衰减——磷酸化信号沿级联路径逐步减弱。',
  species: [
    { name: 'pKinase', type: 'decay', baseline: 0.5, scale: 5.0, rate: 0.06, noise: 0.06 },
    { name: 'pSubstrate', type: 'decay', baseline: 0.8, scale: 3.5, rate: 0.045, noise: 0.05 },
    { name: 'Phosphatase', type: 'linear', baseline: 2.0, slope: -0.003, noise: 0.04 }
  ]
}));

// 6. 复合型 — 昼夜节律 + 生长 (Circadian + growth)
SAMPLE_SYSTEMS.push(generateTrajectory({
  id: 'sys_complex', name: '昼夜节律与生长耦合', regime: 'complex',
  description: '最复杂的系统——将昼夜节律振荡与长期生长趋势耦合，多个频率叠加，物种间相互影响。',
  species: [
    { name: 'mRNA_Per', type: 'complex', baseline: 1.0, scale: 2.0, rate: 0.008, amplitude: 0.7, freq: 0.1, noise: 0.06 },
    { name: 'PER_protein', type: 'complex', baseline: 1.5, scale: 1.5, rate: 0.006, amplitude: 0.55, freq: 0.1, noise: 0.05 },
    { name: 'Growth_factor', type: 'growth', baseline: 0.5, scale: 3.0, rate: 0.01, noise: 0.07 },
    { name: 'ATP', type: 'oscillation', baseline: 2.5, amplitude: 0.3, freq: 0.1, phase: 1.0, noise: 0.04 }
  ]
}));
