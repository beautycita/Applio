// A project opened from Achiever (?p=<slug>): every studio request, audio and link of this page works in
// that project's folder (its own script, takes, picks and mixes). Training and the mic test always use
// the main folder. Without ?p= this does nothing and the page is exactly as before.
(() => {
  const P = new URLSearchParams(location.search).get('p');
  if (!P || !/^[a-z0-9][a-z0-9-]{0,59}$/.test(P)) return;
  const skip = /^\/voz\/(entreno|entrenamiento|entrenar|prueba)/;
  const add = u => (typeof u !== 'string' || !u.startsWith('/voz/') || skip.test(u) || /[?&]p=/.test(u))
    ? u : u + (u.includes('?') ? '&' : '?') + 'p=' + encodeURIComponent(P);
  const f = window.fetch.bind(window);
  window.fetch = (u, o) => f(add(u), o);
  const fix = el => { if (el.dataset && el.dataset.nop) return; for (const a of ['src', 'href']) { const v = el.getAttribute && el.getAttribute(a); if (v && v !== add(v)) el.setAttribute(a, add(v)); } };
  new MutationObserver(ms => { for (const m of ms) {
    if (m.type === 'attributes') fix(m.target);
    for (const n of m.addedNodes || []) if (n.nodeType === 1) { fix(n); n.querySelectorAll('[src],[href]').forEach(fix); }
  } }).observe(document.documentElement, { subtree: true, childList: true, attributes: true, attributeFilter: ['src', 'href'] });
  addEventListener('DOMContentLoaded', async () => {
    let t = P;
    try { t = ((await (await f('/voz/proyectos')).json()).find(x => x.p === P) || {}).titulo || P; } catch (e) {}
    const b = document.createElement('div');
    b.style.cssText = 'position:sticky;top:0;z-index:50;background:#3478f6;color:#fff;font:600 15px system-ui,sans-serif;padding:10px 16px;display:flex;gap:12px;align-items:center;justify-content:space-between';
    const s = document.createElement('span'); s.textContent = '🎙️ Proyecto: ' + t;
    const a = document.createElement('a'); a.dataset.nop = '1'; a.href = location.pathname; a.textContent = 'Guion principal'; a.style.cssText = 'color:#fff;opacity:.9';
    b.append(s, a); document.body.prepend(b);
  });
})();
