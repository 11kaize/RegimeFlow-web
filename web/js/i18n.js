// RegimeFlow — i18n 工具函数
let currentLang = 'en';

function t(key, vars) {
  var entry = I18N[key];
  if (!entry) return key;
  var str = entry[currentLang] || entry['en'] || entry['zh'] || key;
  if (vars) {
    Object.keys(vars).forEach(function(k) {
      str = str.replace('{' + k + '}', vars[k]);
    });
  }
  return str;
}

function setLang(lang) {
  currentLang = lang;
  try { localStorage.setItem('regimeflow_lang', lang); } catch (e) {}
  applyTranslations();
}

function getLang() {
  return currentLang;
}

function toggleLang() {
  setLang(currentLang === 'zh' ? 'en' : 'zh');
}

// 将 HTML 中 data-i18n / data-i18n-html / data-i18n-placeholder 替换为当前语言文本
function applyTranslations() {
  // 更新文本内容
  document.querySelectorAll('[data-i18n]').forEach(function(el) {
    var key = el.getAttribute('data-i18n');
    var str = t(key);
    if (str) el.textContent = str;
  });

  // 支持 HTML 标签（如 <br>）
  document.querySelectorAll('[data-i18n-html]').forEach(function(el) {
    var key = el.getAttribute('data-i18n-html');
    var str = t(key);
    if (str) el.innerHTML = str;
  });

  // 更新 placeholder
  document.querySelectorAll('[data-i18n-placeholder]').forEach(function(el) {
    var key = el.getAttribute('data-i18n-placeholder');
    var str = t(key);
    if (str) el.setAttribute('placeholder', str);
  });

  // 更新语言按钮文字
  var btn = document.getElementById('lang-switch');
  if (btn) btn.textContent = currentLang === 'zh' ? 'EN' : '中';

  // 触发全局事件，通知图表组件重绘
  window.dispatchEvent(new CustomEvent('langchange', { detail: currentLang }));
}

// 页面加载完成后应用翻译
document.addEventListener('DOMContentLoaded', function() {
  applyTranslations();
});
