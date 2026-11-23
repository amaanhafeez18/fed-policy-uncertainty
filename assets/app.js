/* Charts and controls for the Fed policy-uncertainty index page. Data: data/site.json. */
(function () {
  'use strict';

  /* ---------- theme ---------- */
  const root = document.documentElement;
  const toggle = document.getElementById('theme-toggle');
  const label = document.getElementById('theme-label');
  const setTheme = (t) => {
    root.setAttribute('data-theme', t);
    if (label) label.textContent = t === 'dark' ? 'Light mode' : 'Dark mode';
    if (toggle) toggle.setAttribute('aria-pressed', String(t === 'dark'));
    try { localStorage.setItem('fed-index-theme', t); } catch (e) { /* storage unavailable */ }
  };
  setTheme(root.getAttribute('data-theme') === 'dark' ? 'dark' : 'light');
  if (toggle) toggle.addEventListener('click', () => { setTheme(root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark'); drawAll(); });

  const css = (name) => getComputedStyle(root).getPropertyValue(name).trim();
  const MODELS = {luna: 'GPT-6 Luna', gemini: 'Gemini 3.1 Flash-Lite', haiku: 'Claude Haiku 5.5'};
  const parse = d3.utcParse('%Y-%m-%d');
  const fmtDay = d3.utcFormat('%b %-d, %Y');
  const fmtMonth = d3.utcFormat('%B %Y');
  const f2 = d3.format('.2f');
  const signed = (v) => (v >= 0 ? '+' : '−') + f2(Math.abs(v));
  const trailing = (vals, k) => vals.map((_, i) => {
    if (i < k - 1) return null;
    const w = vals.slice(i - k + 1, i + 1);
    return w.some((v) => v == null) ? null : w.reduce((s, v) => s + v, 0) / k;
  });

  let DATA = null;

  /* ---------- one line chart ---------- */
  function lineChart(el, opts) {
    const {series, yDomain, aria, events = false, height: H0 = 340, tip, yTitle} = opts;
    const W = Math.max(300, el.clientWidth);
    const narrow = W < 620;
    const H = narrow ? Math.round(H0 * 0.85) : H0;
    const m = {top: events && !narrow ? 62 : yTitle ? 30 : 14, right: 16, bottom: 34, left: 50};
    const svg = d3.create('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('role', 'img').attr('aria-label', aria);
    const dates = series[0].values.map((d) => d.date);
    const x = d3.scaleUtc().domain(d3.extent(dates)).range([m.left, W - m.right]);
    const allY = series.flatMap((s) => s.values.map((d) => d.y)).filter((v) => v != null);
    const y = d3.scaleLinear().domain(yDomain || d3.extent(allY)).nice().range([H - m.bottom, m.top]);

    for (const [a, b] of DATA.recessions) {
      const xa = Math.max(x(parse(a)), m.left), xb = Math.min(x(parse(b)), W - m.right);
      if (xb > xa) svg.append('rect').attr('class', 'band').attr('x', xa).attr('width', xb - xa).attr('y', m.top).attr('height', H - m.top - m.bottom);
    }
    svg.append('g').attr('transform', `translate(${m.left},0)`)
      .call(d3.axisLeft(y).ticks(narrow ? 4 : 6).tickSize(-(W - m.left - m.right)).tickPadding(8))
      .call((g) => g.select('.domain').remove());
    svg.append('g').attr('transform', `translate(0,${H - m.bottom})`)
      .call(d3.axisBottom(x).ticks(narrow ? 5 : 11).tickSizeOuter(0).tickPadding(8));
    if (yTitle) svg.append('text').attr('class', 'axlabel').attr('x', m.left).attr('y', m.top - (events && !narrow ? 50 : 14)).text(yTitle);

    if (events) {
      const lastEnd = [-Infinity, -Infinity, -Infinity];
      for (const [d, text] of DATA.events) {
        const xe = x(parse(d));
        svg.append('line').attr('class', 'event').attr('x1', xe).attr('x2', xe).attr('y1', m.top).attr('y2', H - m.bottom);
        if (narrow) continue;
        const w = text.length * 6.6;
        const row = lastEnd.findIndex((end) => xe - w / 2 > end + 6);
        if (row < 0) continue;
        svg.append('text').attr('class', 'evlabel').attr('x', xe).attr('y', m.top - 8 - row * 15).attr('text-anchor', 'middle').text(text);
        lastEnd[row] = xe + w / 2;
      }
    }

    const line = d3.line().defined((d) => d.y != null).x((d) => x(d.date)).y((d) => y(d.y));
    for (const s of series) {
      svg.append('path').attr('fill', 'none').attr('stroke', s.color).attr('stroke-width', s.width || 2)
        .attr('stroke-opacity', s.opacity ?? 1).attr('stroke-dasharray', s.dash || null)
        .attr('stroke-linejoin', 'round').attr('d', line(s.values));
    }

    const cross = svg.append('line').attr('class', 'cross').attr('y1', m.top).attr('y2', H - m.bottom).style('display', 'none');
    const dots = series.filter((s) => s.dot).map((s) => svg.append('circle').attr('r', 4.5).attr('fill', s.color).attr('stroke', css('--panel')).attr('stroke-width', 2).style('display', 'none'));
    const box = document.createElement('div'); box.className = 'tip'; box.hidden = true;
    const bis = d3.bisector((d) => d).center;
    const show = (ev) => {
      const [px] = d3.pointer(ev, svg.node());
      const i = bis(dates, x.invert(px));
      if (i < 0 || i >= dates.length) return;
      const xv = x(dates[i]);
      cross.attr('x1', xv).attr('x2', xv).style('display', null);
      series.filter((s) => s.dot).forEach((s, k) => {
        const v = s.values[i].y;
        dots[k].attr('cx', xv).attr('cy', v == null ? -99 : y(v)).style('display', v == null ? 'none' : null);
      });
      box.replaceChildren(...tip(i));
      box.hidden = false;
      const rect = svg.node().getBoundingClientRect(), sc = rect.width / W;
      const left = xv * sc + 14;
      box.style.left = (left + box.offsetWidth > rect.width ? xv * sc - box.offsetWidth - 14 : left) + 'px';
      box.style.top = Math.max(0, m.top * sc) + 'px';
    };
    const hide = () => { box.hidden = true; cross.style('display', 'none'); dots.forEach((d) => d.style('display', 'none')); };
    svg.append('rect').attr('x', m.left).attr('y', m.top).attr('width', W - m.left - m.right).attr('height', H - m.top - m.bottom)
      .attr('fill', 'transparent').style('touch-action', 'pan-y')
      .on('pointermove', show).on('pointerdown', show).on('pointerleave', hide);
    el.replaceChildren(svg.node(), box);
  }

  const tipRow = (name, value, color) => {
    const d = document.createElement('div');
    if (color) { const s = document.createElement('span'); s.className = 'swatch'; s.style.background = color; d.append(s); }
    const b = document.createElement('b'); b.textContent = value;
    d.append(name + '  ', b);
    return d;
  };
  const tipWhen = (text) => { const d = document.createElement('div'); d.className = 'when'; d.textContent = text; return d; };

  /* ---------- the four charts ---------- */
  function drawModel(el) {
    const mid = el.getAttribute('data-model');
    const dates = DATA.weeks.map(parse);
    const vals = DATA.index[mid];
    const avg = trailing(vals, 13);
    const c = css('--' + mid);
    lineChart(el, {
      series: [{values: dates.map((d, i) => ({date: d, y: vals[i]})), color: c, width: 1.3, opacity: 0.35},
        {values: dates.map((d, i) => ({date: d, y: avg[i]})), color: c, width: 3, dot: true}],
      yDomain: [0, 8], events: true, yTitle: 'Policy uncertainty (0–10)',
      aria: MODELS[mid] + ': weekly policy-uncertainty index from December 1993 to May 2015, with its 13-week average',
      tip: (i) => [tipWhen('Week ending ' + fmtDay(dates[i])), tipRow('Weekly', vals[i] == null ? '—' : f2(vals[i])),
        tipRow('13-week average', avg[i] == null ? '—' : f2(avg[i]), c)],
    });
  }

  function drawCompare(el) {
    const view = (document.getElementById('cmp-view') || {}).value || 'main';
    const suffix = view === 'ms' ? '_ms' : '';
    const dates = DATA.weeks.map(parse);
    const ids = ['luna', 'gemini', 'haiku'];
    const avg = Object.fromEntries(ids.map((k) => [k, trailing(DATA.index[k + suffix], 13)]));
    lineChart(el, {
      series: ids.map((k) => ({values: dates.map((d, i) => ({date: d, y: avg[k][i]})), color: css('--' + k), width: 2.6, dot: true})),
      yDomain: [1, 5.5], yTitle: '13-week average, 0–10 scale',
      aria: '13-week averages of the three models\' weekly index, ' + (view === 'ms' ? 'minutes and statements only' : 'all Fed text'),
      tip: (i) => [tipWhen('Week ending ' + fmtDay(dates[i])), ...ids.map((k) => tipRow(MODELS[k], avg[k][i] == null ? '—' : f2(avg[k][i]), css('--' + k)))],
    });
  }

  function drawMeasure(el) {
    const mid = (document.getElementById('msr-model') || {}).value || 'luna';
    const meid = (document.getElementById('msr-measure') || {}).value || 'hrs';
    const ms = DATA.measures[meid];
    const s = ms.series[mid];
    const dates = ms.dates.map(parse);
    const c = css('--' + mid);
    const when = ms.freq === 'weekly' ? (d) => 'Week ending ' + fmtDay(d) : (d) => fmtMonth(d);
    lineChart(el, {
      series: [{values: dates.map((d, i) => ({date: d, y: s.measure[i]})), color: css('--measure'), width: 2.2, dash: '7 5', dot: true},
        {values: dates.map((d, i) => ({date: d, y: s.model[i]})), color: c, width: 2.8, dot: true}],
      yTitle: 'Standard deviations from each series\' own average',
      aria: MODELS[mid] + ' compared with ' + ms.label + ', both standardized',
      tip: (i) => [tipWhen(when(dates[i])), tipRow(MODELS[mid], s.model[i] == null ? '—' : f2(s.model[i]), c),
        tipRow(ms.short, s.measure[i] == null ? '—' : f2(s.measure[i]), css('--measure'))],
    });
    const lg = document.getElementById('msr-legend');
    if (lg) {
      lg.querySelector('[data-k="model"] i').style.borderColor = c;
      lg.querySelector('[data-k="model"] em').textContent = MODELS[mid];
      lg.querySelector('[data-k="measure"] i').style.borderColor = css('--measure');
      lg.querySelector('[data-k="measure"] em').textContent = ms.label;
    }
    const r = document.getElementById('msr-r');
    if (r) {
      const unit = ms.freq === 'weekly' ? 'weeks' : 'months';
      r.textContent = 'Correlation in levels ' + signed(s.level) + ', in changes ' + signed(s.change) + ' — '
        + (Math.abs(s.change) > s.crit ? 'beyond' : 'within') + ' the 5% cut-off of ±' + f2(s.crit) + ' (' + d3.format(',')(s.n) + ' ' + unit + ').';
    }
  }

  function drawDocs(el) {
    const types = [['minutes', 'Minutes', '--t-minutes'], ['statement', 'Statements', '--t-statement'],
      ['speech', 'Speeches', '--t-speech'], ['press_conference', 'Press conferences', '--t-press']];
    const rows = DATA.docs_per_year;
    const W = Math.max(300, el.clientWidth), narrow = W < 620, H = narrow ? 230 : 260;
    const m = {top: 12, right: 10, bottom: 32, left: 44};
    const x = d3.scaleBand().domain(rows.map((r) => r.year)).range([m.left, W - m.right]).padding(0.2);
    const stack = d3.stack().keys(types.map((t) => t[0]))(rows);
    const y = d3.scaleLinear().domain([0, d3.max(rows, (r) => types.reduce((s, [t]) => s + r[t], 0))]).nice().range([H - m.bottom, m.top]);
    const svg = d3.create('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('role', 'img').attr('aria-label', 'Fed documents released each year, 1993 to 2015, by type');
    svg.append('g').attr('transform', `translate(${m.left},0)`).call(d3.axisLeft(y).ticks(5).tickSize(-(W - m.left - m.right)).tickPadding(8)).call((g) => g.select('.domain').remove());
    svg.append('g').attr('transform', `translate(0,${H - m.bottom})`).call(d3.axisBottom(x).tickValues(rows.map((r) => r.year).filter((yr) => yr % (narrow ? 4 : 2) === 0)).tickSizeOuter(0).tickPadding(8));
    stack.forEach((layer, k) => {
      svg.append('g').selectAll('rect').data(layer).join('rect').attr('x', (d) => x(d.data.year)).attr('width', x.bandwidth())
        .attr('y', (d) => y(d[1])).attr('height', (d) => Math.max(0, y(d[0]) - y(d[1]))).attr('fill', css(types[k][2]));
    });
    const box = document.createElement('div'); box.className = 'tip'; box.hidden = true;
    svg.append('g').selectAll('rect').data(rows).join('rect').attr('x', (d) => x(d.year)).attr('width', x.bandwidth())
      .attr('y', m.top).attr('height', H - m.top - m.bottom).attr('fill', 'transparent')
      .on('pointerenter pointerdown', (ev, d) => {
        const total = types.reduce((s, [t]) => s + d[t], 0);
        box.replaceChildren(tipWhen(d.year + ' · ' + total + ' documents'), ...types.slice().reverse().map(([t, name, v]) => tipRow(name, String(d[t]), css(v))));
        box.hidden = false;
        const rect = svg.node().getBoundingClientRect(), sc = rect.width / W;
        const left = (x(d.year) + x.bandwidth()) * sc + 8;
        box.style.left = (left + box.offsetWidth > rect.width ? x(d.year) * sc - box.offsetWidth - 8 : left) + 'px';
        box.style.top = '10px';
      }).on('pointerleave', () => { box.hidden = true; });
    el.replaceChildren(svg.node(), box);
  }

  function paintLegends() {
    for (const el of document.querySelectorAll('[data-color]')) {
      const v = css(el.getAttribute('data-color'));
      if (el.tagName === 'I' && !el.classList.contains('box')) el.style.borderColor = v; else el.style.background = v;
    }
  }

  function drawAll() {
    if (!DATA) return;
    paintLegends();
    for (const el of document.querySelectorAll('.chart[data-chart="model"]')) drawModel(el);
    for (const el of document.querySelectorAll('.chart[data-chart="compare"]')) drawCompare(el);
    for (const el of document.querySelectorAll('.chart[data-chart="measure"]')) drawMeasure(el);
    for (const el of document.querySelectorAll('.chart[data-chart="docs"]')) drawDocs(el);
  }

  for (const id of ['cmp-view', 'msr-model', 'msr-measure']) {
    const el = document.getElementById(id);
    if (el) el.addEventListener('change', drawAll);
  }
  let resizeTimer = null;
  window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(drawAll, 150); });

  fetch('data/site.json').then((r) => {
    if (!r.ok) throw new Error('HTTP ' + r.status);
    return r.json();
  }).then((json) => { DATA = json; drawAll(); }).catch(() => {
    for (const el of document.querySelectorAll('.chart')) {
      const p = document.createElement('p'); p.className = 'note';
      p.textContent = 'The chart data could not be loaded. The numbers in the tables and the PDF report are unaffected.';
      el.replaceChildren(p);
    }
  });
})();
