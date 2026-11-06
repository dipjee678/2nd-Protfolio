(() => {
  'use strict';
  const root = document.documentElement;
  root.classList.add('js');
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---------- year ---------- */
  document.querySelectorAll('.js-year').forEach(el => { el.textContent = new Date().getFullYear(); });

  /* ---------- toast ---------- */
  const toastEl = document.querySelector('.toast');
  let toastTimer;
  function toast(msg) {
    toastEl.textContent = msg;
    toastEl.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toastEl.classList.remove('show'), 2200);
  }
  async function copy(text) {
    try { await navigator.clipboard.writeText(text); return true; }
    catch (e) {
      const ta = document.createElement('textarea');
      ta.value = text; ta.setAttribute('readonly', ''); ta.style.position = 'fixed'; ta.style.opacity = '0';
      document.body.appendChild(ta); ta.select();
      let ok = false; try { ok = document.execCommand('copy'); } catch (_) {}
      ta.remove(); return ok;
    }
  }

  /* ---------- theme ---------- */
  const isDark = () => {
    const t = root.getAttribute('data-theme');
    return t ? t === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
  };
  document.querySelector('.theme-toggle').addEventListener('click', () => {
    const next = isDark() ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem('db-theme', next); } catch (e) {}
    document.dispatchEvent(new Event('themechange'));
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener?.('change', () => document.dispatchEvent(new Event('themechange')));

  /* ---------- mobile contents menu ---------- */
  const tocBtn = document.querySelector('.toc-toggle');
  const tocList = document.getElementById('toc-list');
  const setToc = open => { tocList.classList.toggle('open', open); tocBtn.setAttribute('aria-expanded', String(open)); };
  tocBtn.addEventListener('click', e => { e.stopPropagation(); setToc(!tocList.classList.contains('open')); });
  tocList.addEventListener('click', e => { if (e.target.closest('a')) setToc(false); });
  document.addEventListener('click', e => { if (!e.target.closest('.toc')) setToc(false); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') setToc(false); });

  /* ---------- scroll: progress bar + masthead ---------- */
  const bar = document.getElementById('progress-bar');
  const masthead = document.querySelector('.masthead');
  let ticking = false;
  function onScroll() {
    const max = document.documentElement.scrollHeight - innerHeight;
    bar.style.transform = `scaleX(${max > 0 ? Math.min(1, scrollY / max) : 0})`;
    masthead.classList.toggle('is-scrolled', scrollY > 8);
    ticking = false;
  }
  addEventListener('scroll', () => { if (!ticking) { ticking = true; requestAnimationFrame(onScroll); } }, { passive: true });
  onScroll();

  /* ---------- active section in contents ---------- */
  const tocLinks = [...tocList.querySelectorAll('a')];
  const secObs = new IntersectionObserver(entries => {
    entries.forEach(en => {
      if (en.isIntersecting) {
        tocLinks.forEach(a => a.classList.toggle('is-active', a.getAttribute('href') === '#' + en.target.id));
      }
    });
  }, { rootMargin: '-45% 0px -50% 0px' });
  tocLinks.forEach(a => { const s = document.querySelector(a.getAttribute('href')); if (s) secObs.observe(s); });

  /* ---------- reveal on scroll ---------- */
  const revealTargets = document.querySelectorAll('.sec-head, .prose > *, .nn-fig, .ref, .proj, .table-fig, .cv li, .correspondence .wrap > *');
  if (!reduceMotion) {
    // position check on scroll: robust even when IntersectionObserver is throttled
    let pending = [...revealTargets].filter(el => el.getBoundingClientRect().top >= innerHeight);
    pending.forEach(el => el.classList.add('reveal'));
    const check = () => {
      const limit = innerHeight * .94;
      pending = pending.filter(el => {
        if (el.getBoundingClientRect().top < limit) { el.classList.add('in'); return false; }
        return true;
      });
      if (!pending.length) removeEventListener('scroll', onRevealScroll);
    };
    let rTick = false;
    const onRevealScroll = () => { if (!rTick) { rTick = true; requestAnimationFrame(() => { rTick = false; check(); }); } };
    addEventListener('scroll', onRevealScroll, { passive: true });
    addEventListener('resize', onRevealScroll);
    // anchor jumps can land without rAF firing promptly; also check directly
    addEventListener('hashchange', check);
    setTimeout(check, 50);
  }

  /* ---------- count-up (Table 1) ---------- */
  if (!reduceMotion) {
    document.querySelectorAll('.count').forEach(el => {
      const to = parseFloat(el.dataset.to), dec = +(el.dataset.dec || 0);
      const t0 = performance.now(), dur = 1300;
      el.textContent = (0).toFixed(dec);
      const step = now => {
        const k = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - k, 3);
        el.textContent = (to * e).toFixed(dec);
        if (k < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
  }

  /* ---------- cite / copy ---------- */
  document.querySelectorAll('.js-cite').forEach(btn => {
    btn.addEventListener('click', async () => {
      const bib = btn.closest('.ref').dataset.bib;
      toast(await copy(bib) ? 'BibTeX copied to clipboard' : 'Could not copy, sorry');
    });
  });
  document.querySelectorAll('.js-copy-email').forEach(btn => {
    btn.addEventListener('click', async () => {
      const ok = await copy(btn.dataset.email);
      btn.textContent = ok ? 'Copied ✓' : 'Copy failed';
      setTimeout(() => { btn.textContent = 'Copy'; }, 1800);
    });
  });

  /* ==========================================================
     Fig. 2 — live 2–16–1 MLP, trained by full-batch GD + momentum
     ========================================================== */
  const canvas = document.getElementById('nn-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const stage = canvas.parentElement;
  const H = 16;
  const LR = 0.12, MOM = 0.9, STEPS_PER_FRAME = 6;

  let W1, b1, W2, b2, vW1, vb1, vW2, vb2;
  let points = [];
  let epoch = 0, running = !reduceMotion, visible = false, cls = 0;
  let cw = 0, ch = 0, aspect = 0.5;

  const rnd = (a, b) => a + Math.random() * (b - a);
  const gauss = () => { let u = 0, v = 0; while (!u) u = Math.random(); while (!v) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };

  function initWeights() {
    W1 = Array.from({ length: H }, () => [rnd(-2, 2), rnd(-2, 2)]);
    b1 = Array.from({ length: H }, () => rnd(-1, 1));
    W2 = Array.from({ length: H }, () => rnd(-.5, .5));
    b2 = 0;
    vW1 = Array.from({ length: H }, () => [0, 0]); vb1 = new Array(H).fill(0); vW2 = new Array(H).fill(0); vb2 = 0;
    epoch = 0;
  }

  const hid = new Float64Array(H);
  function forward(x, y) {
    let z = b2;
    for (let j = 0; j < H; j++) { const h = Math.tanh(W1[j][0] * x + W1[j][1] * y + b1[j]); hid[j] = h; z += W2[j] * h; }
    return 1 / (1 + Math.exp(-z));
  }

  function trainStep() {
    const n = points.length;
    if (!n) return { loss: NaN, acc: NaN };
    const gW1 = Array.from({ length: H }, () => [0, 0]), gb1 = new Array(H).fill(0), gW2 = new Array(H).fill(0);
    let gb2 = 0, loss = 0, correct = 0;
    for (const p of points) {
      const out = forward(p.x, p.y);
      const pc = Math.min(1 - 1e-7, Math.max(1e-7, out));
      loss += -(p.c * Math.log(pc) + (1 - p.c) * Math.log(1 - pc));
      if ((out > .5 ? 1 : 0) === p.c) correct++;
      const dz = out - p.c;
      gb2 += dz;
      for (let j = 0; j < H; j++) {
        gW2[j] += dz * hid[j];
        const dh = dz * W2[j] * (1 - hid[j] * hid[j]);
        gW1[j][0] += dh * p.x; gW1[j][1] += dh * p.y; gb1[j] += dh;
      }
    }
    const k = LR / n;
    for (let j = 0; j < H; j++) {
      vW2[j] = MOM * vW2[j] - k * gW2[j]; W2[j] += vW2[j];
      vW1[j][0] = MOM * vW1[j][0] - k * gW1[j][0]; W1[j][0] += vW1[j][0];
      vW1[j][1] = MOM * vW1[j][1] - k * gW1[j][1]; W1[j][1] += vW1[j][1];
      vb1[j] = MOM * vb1[j] - k * gb1[j]; b1[j] += vb1[j];
    }
    vb2 = MOM * vb2 - k * gb2; b2 += vb2;
    epoch++;
    return { loss: loss / n, acc: correct / n };
  }

  /* datasets live in x∈[-1,1], y∈[-aspect, aspect] */
  function dataset(name) {
    const pts = [];
    if (name === 'moons') {
      for (let i = 0; i < 70; i++) {
        const t = Math.random() * Math.PI, c = i % 2;
        let x = c ? 1 - Math.cos(t) : Math.cos(t), y = c ? .5 - Math.sin(t) : Math.sin(t);
        x = (x - .5) / 1.5 * .82 + gauss() * .04; y = -(y - .25) / .75 * aspect * .72 + gauss() * .04;
        pts.push({ x, y, c });
      }
    } else if (name === 'rings') {
      const R = aspect * .9;
      for (let i = 0; i < 80; i++) {
        const c = i % 2, a = Math.random() * Math.PI * 2;
        const r = c ? R * rnd(.72, .98) : R * rnd(0, .38);
        pts.push({ x: Math.cos(a) * r, y: Math.sin(a) * r, c });
      }
    } else if (name === 'xor') {
      const m = aspect * .9;
      for (let i = 0; i < 80; i++) {
        const x = rnd(-m, m), y = rnd(-m, m);
        if (Math.abs(x) < .05 || Math.abs(y) < .05) { i--; continue; }
        pts.push({ x, y, c: (x > 0) !== (y > 0) ? 1 : 0 });
      }
    }
    return pts;
  }

  /* colours from CSS custom properties */
  let colA, colB, colPaper, colInk;
  function parseColor(str) {
    const probe = document.createElement('span');
    probe.style.color = str; document.body.appendChild(probe);
    const m = getComputedStyle(probe).color.match(/[\d.]+/g).map(Number);
    probe.remove(); return m.slice(0, 3);
  }
  function readColors() {
    const cs = getComputedStyle(root);
    colA = parseColor(cs.getPropertyValue('--cls-a').trim());
    colB = parseColor(cs.getPropertyValue('--cls-b').trim());
    colPaper = parseColor(cs.getPropertyValue('--paper-2').trim());
    colInk = cs.getPropertyValue('--ink').trim();
  }

  const field = document.createElement('canvas');
  const fctx = field.getContext('2d');
  let GW = 96, GH = 42, img;

  function resize() {
    const r = stage.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    cw = r.width; ch = r.height; aspect = ch / cw;
    canvas.width = Math.round(cw * dpr); canvas.height = Math.round(ch * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    GW = 96; GH = Math.max(24, Math.round(GW * aspect));
    field.width = GW; field.height = GH;
    img = fctx.createImageData(GW, GH);
  }

  const toPx = (x, y) => [(x + 1) / 2 * cw, (y / aspect + 1) / 2 * ch];
  const fromPx = (px, py) => [px / cw * 2 - 1, (py / ch * 2 - 1) * aspect];

  function draw() {
    // confidence field
    const d = img.data;
    for (let gy = 0; gy < GH; gy++) {
      for (let gx = 0; gx < GW; gx++) {
        const [x, y] = fromPx((gx + .5) / GW * cw, (gy + .5) / GH * ch);
        const p = forward(x, y);
        const conf = Math.abs(p - .5) * 2;
        const c = p > .5 ? colB : colA;
        const a = .08 + .34 * conf;
        const i = (gy * GW + gx) * 4;
        d[i] = colPaper[0] + (c[0] - colPaper[0]) * a;
        d[i + 1] = colPaper[1] + (c[1] - colPaper[1]) * a;
        d[i + 2] = colPaper[2] + (c[2] - colPaper[2]) * a;
        d[i + 3] = 255;
      }
    }
    fctx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(field, 0, 0, cw, ch);

    // axes
    ctx.strokeStyle = colInk; ctx.globalAlpha = .18; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cw / 2, 0); ctx.lineTo(cw / 2, ch); ctx.moveTo(0, ch / 2); ctx.lineTo(cw, ch / 2); ctx.stroke();
    ctx.globalAlpha = 1;

    // points
    const rad = cw < 500 ? 4.5 : 5.5;
    for (const p of points) {
      const [px, py] = toPx(p.x, p.y);
      ctx.beginPath(); ctx.arc(px, py, rad, 0, Math.PI * 2);
      ctx.fillStyle = `rgb(${(p.c ? colB : colA).join(',')})`; ctx.fill();
      ctx.lineWidth = 1.5; ctx.strokeStyle = `rgb(${colPaper.join(',')})`; ctx.stroke();
    }

    if (!points.length) {
      ctx.fillStyle = colInk; ctx.globalAlpha = .55;
      ctx.font = '500 13px "IBM Plex Mono", monospace'; ctx.textAlign = 'center';
      ctx.fillText('click anywhere to add points', cw / 2, ch / 2 - 10);
      ctx.globalAlpha = 1; ctx.textAlign = 'start';
    }
  }

  const elEpoch = document.getElementById('nn-epoch'), elLoss = document.getElementById('nn-loss'), elAcc = document.getElementById('nn-acc');
  let last = { loss: NaN, acc: NaN };
  function updateReadout() {
    elEpoch.textContent = epoch.toLocaleString();
    elLoss.textContent = isNaN(last.loss) ? '—' : last.loss.toFixed(3);
    elAcc.textContent = isNaN(last.acc) ? '—' : Math.round(last.acc * 100) + '%';
  }

  let raf = 0;
  function loop() {
    raf = 0;
    if (running && points.length) { for (let i = 0; i < STEPS_PER_FRAME; i++) last = trainStep(); }
    draw(); updateReadout();
    if (running && visible && !document.hidden) raf = requestAnimationFrame(loop);
  }
  const kick = () => { if (!raf) raf = requestAnimationFrame(loop); };

  // interaction
  stage.addEventListener('contextmenu', e => e.preventDefault());
  stage.addEventListener('pointerdown', e => {
    if (e.button === 1) return;
    const r = canvas.getBoundingClientRect();
    const [x, y] = fromPx(e.clientX - r.left, e.clientY - r.top);
    const c = (e.shiftKey || e.button === 2) ? 1 - cls : cls;
    points.push({ x, y, c });
    kick();
  });

  const fig = document.getElementById('fig-nn');
  fig.querySelectorAll('[data-cls]').forEach(b => b.addEventListener('click', () => {
    cls = +b.dataset.cls;
    fig.querySelectorAll('[data-cls]').forEach(o => o.classList.toggle('is-on', o === b));
  }));
  fig.querySelectorAll('[data-set]').forEach(b => b.addEventListener('click', () => {
    points = b.dataset.set === 'clear' ? [] : dataset(b.dataset.set);
    initWeights(); last = { loss: NaN, acc: NaN }; kick();
  }));
  const playBtn = document.getElementById('nn-play');
  const syncPlay = () => { playBtn.textContent = running ? 'Pause' : 'Train'; playBtn.setAttribute('aria-pressed', String(running)); };
  playBtn.addEventListener('click', () => { running = !running; syncPlay(); kick(); });
  document.getElementById('nn-reset').addEventListener('click', () => { initWeights(); last = { loss: NaN, acc: NaN }; kick(); });

  new IntersectionObserver(([en]) => { visible = en.isIntersecting; if (visible) kick(); }, { threshold: 0 }).observe(stage);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) kick(); });
  document.addEventListener('themechange', () => { readColors(); kick(); });
  if ('ResizeObserver' in window) new ResizeObserver(() => { resize(); kick(); }).observe(stage);
  else addEventListener('resize', () => { resize(); kick(); });

  readColors(); resize(); initWeights();
  points = dataset('moons');
  syncPlay(); kick();
})();
