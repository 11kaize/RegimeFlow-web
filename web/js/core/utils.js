/* ================================================================
   RegimeFlow — Shared Utilities
   Zero-dependency toolkit: debounce, throttle, text wrapping, HTML escape.
   ================================================================ */

// ── Debounce ──────────────────────────────────────────────────
// Delays fn until `delay` ms after the *last* call. Ideal for
// search input, resize handlers, and zoom events.
function debounce(fn, delay) {
  delay = delay || 200;
  var timer = null;
  return function () {
    var ctx = this, args = arguments;
    clearTimeout(timer);
    timer = setTimeout(function () { fn.apply(ctx, args); }, delay);
  };
}

// ── Throttle ──────────────────────────────────────────────────
// Fires fn at most once per `interval` ms. Use for scroll / mousemove.
function throttle(fn, interval) {
  interval = interval || 100;
  var last = 0;
  return function () {
    var now = Date.now();
    if (now - last < interval) return;
    last = now;
    fn.apply(this, arguments);
  };
}

// ── Text Wrap (CJK-aware) ─────────────────────────────────────
// Splits a string into ≤maxLines lines, each ≤maxChars characters.
// Recognises CJK ideographs as individual breakable units and
// treats Latin words / numeric runs as atomic tokens.
//
//   wrapText('Hodgkin1952', 8, 2)  → ['Hodgkin', '1952']
//   wrapText('拟南芥模型', 4, 2)    → ['拟南芥', '模型']
//
function wrapText(text, maxChars, maxLines) {
  maxLines = maxLines || 2;
  if (!text) return [''];

  // Tokenise: CJK chars become individual tokens; everything else
  // is split on word boundaries (space, underscore, slash, hyphen).
  var segs = text.match(/[一-鿿]|[^一-鿿]+/g) || [text];
  var words = [];
  segs.forEach(function (seg) {
    if (/[一-鿿]/.test(seg)) {
      for (var i = 0; i < seg.length; i++) words.push(seg[i]);
    } else {
      var sub = seg.match(/[^ _/\-]+|[ _/\-]+/g) || [seg];
      words = words.concat(sub);
    }
  });

  // Greedy line-fill with leading-separator trimming.
  var lines = [], cur = '';
  for (var i = 0; i < words.length; i++) {
    var test = cur ? cur + words[i] : words[i];
    if (test.length > maxChars && cur.length > 0) {
      lines.push(cur);
      cur = words[i].replace(/^[ _/\-]+/, '');
      if (lines.length >= maxLines) break;
    } else {
      cur = test;
    }
  }
  if (cur && lines.length < maxLines) lines.push(cur);

  // Truncate overflow lines with ellipsis.
  for (var l = 0; l < lines.length; l++) {
    if (lines[l].length > maxChars + 2) {
      lines[l] = lines[l].substring(0, maxChars - 1) + '…';
    }
  }
  return lines;
}

// ── HTML Escape ────────────────────────────────────────────────
// Prevents XSS when inserting user / dataset strings into innerHTML.
function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}
