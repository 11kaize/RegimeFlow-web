// RegimeFlow — i18n helpers (English-only)
const currentLang = 'en';

function t(key, vars) {
  var entry = I18N[key];
  var str = (entry && entry.en) ? entry.en : key;
  if (vars) {
    Object.keys(vars).forEach(function(k) {
      str = str.replace('{' + k + '}', vars[k]);
    });
  }
  return str;
}

function getLang() {
  return currentLang;
}

// Replace data-i18n / data-i18n-html / data-i18n-placeholder attributes with text
function applyTranslations() {
  // text content
  document.querySelectorAll('[data-i18n]').forEach(function(el) {
    var key = el.getAttribute('data-i18n');
    var str = t(key);
    if (str) el.textContent = str;
  });

  // support inline HTML (e.g. <br>)
  document.querySelectorAll('[data-i18n-html]').forEach(function(el) {
    var key = el.getAttribute('data-i18n-html');
    var str = t(key);
    if (str) el.innerHTML = str;
  });

  // placeholder
  document.querySelectorAll('[data-i18n-placeholder]').forEach(function(el) {
    var key = el.getAttribute('data-i18n-placeholder');
    var str = t(key);
    if (str) el.setAttribute('placeholder', str);
  });

  // notify chart components to redraw
  window.dispatchEvent(new CustomEvent('langchange', { detail: currentLang }));
}

// apply translations on load
document.addEventListener('DOMContentLoaded', function() {
  applyTranslations();
});
