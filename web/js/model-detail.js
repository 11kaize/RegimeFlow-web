// Model detail modal
function showModelDetail(model) {
  const content = document.getElementById('modal-content');
  const typeClass = model.type === 'probabilistic' ? 'type-probabilistic'
    : model.type === 'point' ? 'type-point' : 'type-zero-shot';

  const lang = getLang();

  // 参数行
  let paramRows = '';
  const params = [
    [t('modal.family'), t('family.' + model.family)],
    [t('modal.hiddenDim'), model.hidden_dim > 0 ? model.hidden_dim : t('modal.pretrainedNA')],
    [t('modal.layers'), model.layers > 0 ? model.layers : t('modal.pretrainedNA')],
    [t('modal.lr'), model.lr],
    [t('modal.ctxLen'), model.context_len],
    [t('modal.predLen'), model.pred_len],
  ];
  if (model.d_state) params.push([t('modal.dState'), model.d_state]);
  if (model.n_heads) params.push([t('modal.nHeads'), model.n_heads]);
  if (model.num_steps) params.push([t('modal.numSteps'), model.num_steps]);
  if (model.solver) params.push([t('modal.solver'), model.solver]);
  if (model.prior) params.push([t('modal.prior'), model.prior]);
  if (model.pretrained) params.push([t('modal.pretrained'), model.pretrained]);

  params.forEach(function(pair) {
    paramRows += '<tr><td>' + pair[0] + '</td><td><b>' + pair[1] + '</b></td></tr>';
  });

  // 论文指标
  let metricsHtml = '';
  if (model.paper_metrics && Object.keys(model.paper_metrics).length > 0) {
    metricsHtml = '<div style="margin-bottom:14px;">';
    Object.entries(model.paper_metrics).forEach(function(entry) {
      metricsHtml += '<span style="display:inline-block;background:#1a3550;color:#4A90D9;padding:4px 12px;border-radius:8px;margin-right:8px;font-size:13px;font-weight:600;">' + entry[0] + ': ' + entry[1] + '</span>';
    });
    metricsHtml += '</div>';
  }

  // 特性标签（根据语言选择中/英文）
  const features = (lang === 'en' && model.features_en) ? model.features_en : model.features;
  let featuresHtml = '';
  if (features) {
    featuresHtml = '<div class="features">' + features.map(function(f) {
      return '<span class="feature-tag">' + f + '</span>';
    }).join('') + '</div>';
  }

  // 描述（根据语言选择）
  const desc = (lang === 'en' && model.description_en) ? model.description_en : model.description;

  content.innerHTML =
    '<span class="model-type ' + typeClass + '">' + t('type.' + model.type) + '</span>\n' +
    '<h3>' + model.name + '</h3>\n' +
    metricsHtml +
    '<p class="desc">' + desc + '</p>\n' +
    featuresHtml +
    '<table class="params-table">' + paramRows + '</table>\n' +
    '<button class="btn-predict" onclick="switchToPredict(\'' + model.id + '\')">' +
    t('modal.btnPredict') +
    '</button>';

  document.getElementById('modal-overlay').classList.add('active');
}

function switchToPredict(modelId) {
  document.getElementById('modal-overlay').classList.remove('active');
  // 切换到预测视图
  document.querySelectorAll('.nav-btn').forEach(function(b) { b.classList.remove('active'); });
  document.querySelector('.nav-btn[data-view="predict"]').classList.add('active');
  document.querySelectorAll('.view').forEach(function(v) { v.classList.remove('active'); });
  document.getElementById('view-predict').classList.add('active');
  setTimeout(function() { window._predictChart && window._predictChart.resize(); }, 200);
}
