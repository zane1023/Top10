// AFL HP multi engine: probabilities, estimated bookie odds, optimiser, request parser, learning.
// Pure functions: the model JSON (built weekly by tools/build_model.py) and bet history are passed in.

export const STAT_NAMES = { disposals: 'disposals', goals: 'goals', marks: 'marks', tackles: 'tackles', kicks: 'kicks', handballs: 'handballs' };
const STAT_WORDS = {
  disposals: 'disposals', disposal: 'disposals', disp: 'disposals', disps: 'disposals', touches: 'disposals', possessions: 'disposals', possies: 'disposals',
  goals: 'goals', goal: 'goals', majors: 'goals', marks: 'marks', mark: 'marks', grabs: 'marks', tackles: 'tackles', tackle: 'tackles',
  kicks: 'kicks', kick: 'kicks', handballs: 'handballs', handball: 'handballs', handpasses: 'handballs',
};
export const TEAM_ALIASES = {
  Adelaide: ['adelaide', 'crows', 'adelaide crows'], Brisbane: ['brisbane', 'lions', 'brisbane lions'], Carlton: ['carlton', 'blues'],
  Collingwood: ['collingwood', 'pies', 'magpies', 'woods'], Essendon: ['essendon', 'bombers', 'dons'], Fremantle: ['fremantle', 'freo', 'dockers'],
  Geelong: ['geelong', 'cats'], 'Gold Coast': ['gold coast', 'suns', 'gc'], GWS: ['gws', 'giants', 'greater western sydney'],
  Hawthorn: ['hawthorn', 'hawks'], Melbourne: ['melbourne', 'demons', 'dees'], 'North Melbourne': ['north melbourne', 'north', 'kangaroos', 'roos'],
  'Port Adelaide': ['port adelaide', 'port', 'power'], Richmond: ['richmond', 'tigers', 'tiges'], 'St Kilda': ['st kilda', 'saints', 'stkilda'],
  Sydney: ['sydney', 'swans', 'bloods'], 'West Coast': ['west coast', 'eagles', 'weagles'], 'Western Bulldogs': ['western bulldogs', 'bulldogs', 'dogs', 'doggies', 'footscray'],
  Tasmania: ['tasmania', 'devils'],
};
const TEAM_LINES = [0, 13, 25, 40];
const NICKNAMES = { bont: 'Marcus Bontempelli', dusty: 'Dustin Martin', danger: 'Patrick Dangerfield', trac: 'Christian Petracca', gawn: 'Max Gawn', gawny: 'Max Gawn',
  nicky: 'Nick Daicos', 'nick daicos': 'Nick Daicos', 'jdaicos': 'Josh Daicos', cripps: 'Patrick Cripps', curnow: 'Charlie Curnow', hawkins: 'Tom Hawkins', 'the tomahawk': 'Tom Hawkins',
  zorko: 'Dayne Zorko', 'lachie neale': 'Lachie Neale', neale: 'Lachie Neale', heeney: 'Isaac Heeney', 'jezza': 'Jeremy Cameron', sicily: 'James Sicily', 'the bont': 'Marcus Bontempelli' };
const P_MIN = 0.015, P_MAX = 0.985;

// ---------------------------------------------------------------- maths
export function phi(x) { // standard normal CDF
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989422804014327 * Math.exp(-x * x / 2);
  const p = d * t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  return x >= 0 ? 1 - p : p;
}
export function phiInv(p) { // Acklam
  p = Math.min(Math.max(p, 1e-9), 1 - 1e-9);
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const q = p < 0.02425 ? Math.sqrt(-2 * Math.log(p)) : p > 1 - 0.02425 ? Math.sqrt(-2 * Math.log(1 - p)) : null;
  if (q !== null) {
    const x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    return p < 0.5 ? x : -x;
  }
  const r = p - 0.5, s = r * r;
  return (((((a[0] * s + a[1]) * s + a[2]) * s + a[3]) * s + a[4]) * s + a[5]) * r / (((((b[0] * s + b[1]) * s + b[2]) * s + b[3]) * s + b[4]) * s + 1);
}
export function nbAtLeast(k, mu, r) { // P(X >= k), X ~ NegBin(mean mu, size r)
  if (k <= 0) return 1;
  mu = Math.max(mu, 1e-3);
  const q = mu / (r + mu);
  let pmf = Math.pow(r / (r + mu), r), cdf = 0;
  for (let x = 0; x < k; x++) { cdf += pmf; pmf *= (x + r) / (x + 1) * q; }
  return Math.min(1, Math.max(0, 1 - cdf));
}
export function interp(x, xs, ys) {
  if (x <= xs[0]) return ys[0];
  for (let i = 1; i < xs.length; i++) if (x <= xs[i]) {
    const t = (x - xs[i - 1]) / ((xs[i] - xs[i - 1]) || 1);
    return ys[i - 1] + t * (ys[i] - ys[i - 1]);
  }
  return ys[ys.length - 1];
}
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const r2 = (x) => Math.round(x * 100) / 100;
const shiftOdds = (p, f) => (p * f) / (1 - p + p * f); // multiply the odds-ratio by f

// ---------------------------------------------------------------- learning from settled bets
export function learnFactors(history) {
  const acc = {};
  for (const b of history || []) {
    if (b.practice) continue;
    for (const l of b.legs || []) {
      if (l.hit !== 0 && l.hit !== 1) continue;
      const m = l.kind === 'team' ? (l.line > 0 ? 'margin' : 'h2h') : l.stat;
      const a = (acc[m] ||= { n: 0, hits: 0, exp: 0 });
      a.n++; a.hits += l.hit; a.exp += l.p0 ?? l.p;
    }
  }
  const K = 12, out = {};
  for (const [m, a] of Object.entries(acc)) out[m] = { ...a, f: r2(clamp((a.hits + K * 0.5) / (a.exp + K * 0.5), 0.7, 1.3) * 100) / 100 };
  return out;
}

// ---------------------------------------------------------------- probabilities
function hgaOf(model, home, away, vstate) {
  const st = model.state || {}, hga = model.params.rating.hga;
  const hs = st[home] || 'VIC', as = st[away] || 'VIC', v = vstate || hs;
  if (v === hs && v !== as) return hga;
  if (v === as && v !== hs) return -hga;
  return hga * 0.25;
}

export function gameProbs(model, g) {
  const P = model.params.rating, R = model.ratings;
  const exp = (R[g.home] || 0) - (R[g.away] || 0) + hgaOf(model, g.home, g.away, g.vstate);
  let pOwn = interp(phi(exp / P.sigma), model.calib.h2h[0], model.calib.h2h[1]);
  pOwn = clamp(pOwn, 0.03, 0.97);
  const pHome = g.cons != null ? 0.6 * pOwn + 0.4 * g.cons : pOwn;
  const sd = P.margin_sd || 34;
  return { pHome, pOwn, cons: g.cons, expMargin: sd * phiInv(pHome), sd };
}

function bookTeam(model, p, line) {
  const m = model.params.bookie_margin;
  const pb = 0.5 + 0.88 * (p - 0.5); // bookies sit a touch closer to 50/50 than our numbers
  return Math.max(1.01, 1 / Math.min(0.99, pb * (1 + (line > 0 ? m.line : m.h2h))));
}

export function teamLeg(model, g, gp, team, line, learn) {
  const home = team === g.home, opp = home ? g.away : g.home;
  const em = home ? gp.expMargin : -gp.expMargin;
  let p0 = line <= 0 ? (home ? gp.pHome : 1 - gp.pHome) : phi((em - line + 0.5) / gp.sd);
  p0 = clamp(p0, P_MIN, P_MAX);
  const f = learn[line > 0 ? 'margin' : 'h2h']?.f || 1;
  const p = clamp(shiftOdds(p0, f), P_MIN, P_MAX);
  return {
    id: `T|${g.key}|${team}|${line}`, kind: 'team', game: g.key, home: g.home, team, opp, line, stat: line > 0 ? 'margin' : 'h2h',
    label: line > 0 ? `${team} by ${line}+` : `${team} to win`, sub: `v ${opp}`,
    p: r2(p * 1000) / 1000, p0: r2(p0 * 1000) / 1000, fair: r2(1 / p), book: r2(bookTeam(model, p0, line)), av: 1,
  };
}

export function playerMu(model, pl, stat, opp, isHome) {
  const P = model.params, [s, l] = pl.m[stat];
  const of = model.opp?.[opp]?.[stat] ?? 1, hf = P.home_f[stat] || 1;
  return (P.w * s + (1 - P.w) * l) * (1 + P.ow * (of - 1)) * (isHome ? hf : 1 / hf);
}

export function playerLeg(model, g, pl, stat, line, learn) {
  const isHome = pl.t === g.home, opp = isHome ? g.away : g.home, r = model.params.nb_r[stat];
  const mu = playerMu(model, pl, stat, opp, isHome);
  const raw = nbAtLeast(line, mu, r);
  const cal = model.calib[stat];
  let p0 = clamp(interp(raw, cal[0], cal[1]), P_MIN, P_MAX);
  const f = learn[stat]?.f || 1;
  const p = clamp(shiftOdds(p0, f), P_MIN, P_MAX);
  const pn = clamp(nbAtLeast(line, pl.m[stat][2], r), P_MIN, P_MAX); // a simple last-10 average, like a lazy bookie
  const pb = 0.25 * pn + 0.75 * p0; // bookies are sharp: mostly our number, nudged toward a plain recent average
  const book = Math.max(1.01, 1 / Math.min(0.99, pb * (1 + model.params.bookie_margin.prop)));
  const short = stat === 'goals' && line === 1 ? 'to kick a goal' : `${line}+ ${stat}`;
  return {
    id: `P|${g.key}|${pl.n}|${stat}|${line}`, kind: 'player', game: g.key, player: pl.n, team: pl.t, opp, stat, line,
    label: `${pl.n} ${short}`, sub: `${pl.t} v ${opp} · avg ${r2(mu)} · last 5: ${(pl.l5?.[stat] || []).join(', ') || '—'}`,
    mu: r2(mu), p: r2(p * 1000) / 1000, p0: r2(p0 * 1000) / 1000, fair: r2(1 / p), book: r2(book), av: pl.av,
  };
}

// ---------------------------------------------------------------- the round's market
export function buildMarket(model, history, round = model.round) {
  const learn = learnFactors(history);
  const games = round.games;
  const byTeam = {};
  for (const g of games) { byTeam[g.home] = g; byTeam[g.away] = g; }
  const gps = {}, options = [], gameInfo = [];
  for (const g of games) {
    const gp = (gps[g.key] = gameProbs(model, g));
    gameInfo.push({ ...g, pHome: r2(gp.pHome * 1000) / 1000, pOwn: r2(gp.pOwn * 1000) / 1000, expMargin: Math.round(gp.expMargin) });
    for (const team of [g.home, g.away]) for (const line of TEAM_LINES) options.push(teamLeg(model, g, gp, team, line, learn));
  }
  const lines = model.lines;
  for (const pl of model.players) {
    const g = byTeam[pl.t];
    if (!g) continue;
    for (const stat of Object.keys(lines)) {
      for (const line of lines[stat]) {
        const leg = playerLeg(model, g, pl, stat, line, learn);
        if (leg.p > 0.02 && leg.p < 0.985) { leg.g = pl.g; options.push(leg); }
      }
    }
  }
  return { learn, games: gameInfo, gps, options, byTeam, round };
}

// ---------------------------------------------------------------- optimiser
// Choose N legs (one per player, one team result per game, a cap per game) whose bookie multi is as close as
// possible to the target while keeping the chance that ALL legs hit as high as possible.
function groupOf(o) { return o.kind === 'team' ? `T|${o.game}` : `P|${o.player}`; }

export function optimise(options, { legs: N, target, locked = [], perGame }) {
  const lockedIds = new Set(locked.map((l) => l.id));
  const lockedGroups = new Set(locked.map(groupOf));
  const pool = options.filter((o) => !lockedIds.has(o.id) && !lockedGroups.has(groupOf(o)));
  const need = Math.max(0, N - locked.length);
  const lnT = Math.log(target);
  const lockLnO = locked.reduce((s, l) => s + Math.log(l.book), 0), lockLnP = locked.reduce((s, l) => s + Math.log(l.p), 0);
  const games = new Set(options.map((o) => o.game));
  perGame = perGame || Math.max(3, Math.ceil(N / Math.max(1, games.size)) + 1);
  const lockCount = {};
  for (const l of locked) lockCount[l.game] = (lockCount[l.game] || 0) + 1;

  const groups = new Map();
  for (const o of pool) { const k = groupOf(o); if (!groups.has(k)) groups.set(k, []); groups.get(k).push(o); }
  const lp = (o) => Math.log(o.p), lo = (o) => Math.log(o.book);

  function pick(lambda) {
    const best = [];
    for (const [k, arr] of groups) {
      let b = null, bs = -Infinity;
      for (const o of arr) { const s = lp(o) + lambda * lo(o); if (s > bs) { bs = s; b = o; } }
      best.push([bs, b]);
    }
    best.sort((a, b) => b[0] - a[0]);
    const cnt = { ...lockCount }, out = [];
    for (const [, o] of best) {
      if (out.length >= need) break;
      if ((cnt[o.game] || 0) >= perGame) continue;
      cnt[o.game] = (cnt[o.game] || 0) + 1; out.push(o);
    }
    return out;
  }
  const tot = (sel) => sel.reduce((s, o) => s + lo(o), lockLnO);
  let sel = pick(0), lo_ = 0, hi = 0;
  if (tot(sel) < lnT && need > 0) {
    hi = 0.05;
    while (hi < 200 && tot(pick(hi)) < lnT) hi *= 2;
    for (let i = 0; i < 40; i++) { const mid = (lo_ + hi) / 2; if (tot(pick(mid)) < lnT) lo_ = mid; else hi = mid; }
    const a = pick(lo_), b = pick(hi);
    sel = Math.abs(tot(a) - lnT) < Math.abs(tot(b) - lnT) ? a : b;
  }

  // local search: single swaps that bring the multi nearer the target for the least loss of hit chance
  const TOL = Math.log(1.03);
  const obj = (s) => { const gap = Math.abs(tot(s) - lnT); return s.reduce((a, o) => a + lp(o), 0) - 25 * Math.max(0, gap - TOL); };
  let cur = obj(sel);
  for (let iter = 0; iter < 60; iter++) {
    let bestGain = 1e-9, bestMove = null;
    const usedGroups = new Set(sel.map(groupOf));
    const cnt = { ...lockCount }; for (const o of sel) cnt[o.game] = (cnt[o.game] || 0) + 1;
    for (let i = 0; i < sel.length; i++) {
      const out = sel[i], og = groupOf(out);
      for (const o of pool) {
        const g = groupOf(o);
        if (o === out || (g !== og && usedGroups.has(g))) continue;
        if (o.game !== out.game && (cnt[o.game] || 0) >= perGame) continue;
        const trial = sel.slice(); trial[i] = o;
        const gain = obj(trial) - cur;
        if (gain > bestGain) { bestGain = gain; bestMove = trial; }
      }
    }
    if (!bestMove) break;
    sel = bestMove; cur += bestGain;
  }
  const all = [...locked, ...sel];
  const maxSel = pick(1e6), maxOdds = Math.exp(tot(maxSel)), minOdds = Math.exp(tot(pick(0)));
  return { legs: all, lnP: lockLnP + sel.reduce((a, o) => a + lp(o), 0), minOdds, maxOdds };
}

// ---------------------------------------------------------------- plain-English requests
const norm = (s) => s.toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '').replace(/[’']/g, '').replace(/\s+/g, ' ').trim();
function findTeams(text) {
  const hits = [];
  for (const [team, al] of Object.entries(TEAM_ALIASES)) for (const a of al) {
    const re = new RegExp(`(^|[^a-z])${a.replace(/ /g, '\\s+')}(?=$|[^a-z])`, 'g');
    let m; while ((m = re.exec(text))) hits.push({ team, i: m.index + m[1].length, len: a.length });
  }
  hits.sort((a, b) => a.i - b.i || b.len - a.len);
  const out = []; let end = -1;
  for (const h of hits) if (h.i >= end) { out.push(h); end = h.i + h.len; }
  return out;
}

export function parseRequest(text, market, model) {
  const notes = [], locked = [], filter = { excludeTeams: new Set(), onlyStats: null, noTeam: false, onlyTeams: null };
  let legs = null, target = null;
  const t = norm(text || '').replace(/\b(?:round|rd)\s*\d{1,2}\b/g, ' ').replace(/\bnext (?:game|match)( only)?\b/g, ' ').replace(/\s+/g, ' ').trim();
  let m;
  if ((m = t.match(/(\d{1,2})\s*(?:-|\s)?\s*(?:legs?|leg multi|leg)/))) legs = clamp(+m[1], 1, 21);
  if ((m = t.match(/(?:\$|odds (?:of |around |about )?)(\d+(?:\.\d+)?)/) || t.match(/(\d+(?:\.\d+)?)\s*(?:x|times)\b/))) target = clamp(+m[1], 1.1, 5000);
  for (const mm of t.matchAll(/(?:avoid|no|not|without|exclude|except|nothing from)\s+(?:the\s+)?([a-z ]+?)(?=$|[,.;&]| and | but )/g)) {
    for (const h of findTeams(mm[1])) filter.excludeTeams.add(h.team);
    if (/team|result|winner|h2h|head/.test(mm[1])) filter.noTeam = true;
  }
  if ((m = t.match(/only\s+([a-z ,&]+?)(?=$|[.;]| legs?| for | in )/))) {
    const st = [...new Set(m[1].split(/[ ,&]+|and/).map((w) => STAT_WORDS[w]).filter(Boolean))];
    if (st.length) filter.onlyStats = new Set(st);
    const tm = findTeams(m[1]).map((h) => h.team);
    if (tm.length && !st.length) filter.onlyTeams = new Set(tm);
    if (/player/.test(m[1])) filter.noTeam = true;
  }
  if (/(players? only|player props only|no team bets)/.test(t)) filter.noTeam = true;

  // names of players in this round
  const roundPlayers = model.players.filter((p) => market.byTeam[p.t]);
  const full = new Map(), sur = new Map();
  for (const p of roundPlayers) {
    full.set(norm(p.n), p);
    const s = norm(p.n.split(' ').slice(1).join(' ')); if (!sur.has(s)) sur.set(s, []); sur.get(s).push(p);
  }
  const segs = t.split(/,|;|\n|&| and | plus | with | also /).map((s) => s.trim()).filter(Boolean);
  const optById = new Map(market.options.map((o) => [o.id, o]));
  const learn = market.learn;
  for (const seg of segs) {
    // ---- player stat
    let pl = null;
    for (const [n, p] of full) if (seg.includes(n)) { pl = p; break; }
    if (!pl) for (const [nick, n] of Object.entries(NICKNAMES)) if (new RegExp(`(^|[^a-z])${nick}(?=$|[^a-z])`).test(seg) && full.get(norm(n))) { pl = full.get(norm(n)); break; }
    if (!pl) for (const [s, arr] of sur) if (s.length > 2 && new RegExp(`(^|[^a-z])${s}(?=$|[^a-z])`).test(seg)) {
      if (arr.length > 1) {
        const first = seg.match(/([a-z]+)\s+/)?.[1];
        const sw = seg.split(' ').map((w) => STAT_WORDS[w]).find(Boolean) || (/goal/.test(seg) ? 'goals' : 'disposals');
        pl = arr.find((p) => norm(p.n).startsWith(first || '#')) || [...arr].sort((a, b) => b.m[sw][1] - a.m[sw][1])[0];
        notes.push(`"${s}" could be ${arr.map((p) => p.n).join(' or ')} — went with ${pl.n}. Use the full name to choose.`);
      }
      else pl = arr[0];
      if (pl) break;
    }
    if (pl) {
      let stat = null, line = null;
      if ((m = seg.match(/(\d+)\s*\+?\s*(?:or more\s+)?([a-z]+)/)) && STAT_WORDS[m[2]]) { line = +m[1]; stat = STAT_WORDS[m[2]]; }
      else if ((m = seg.match(/([a-z]+)\s*(?:over|at least|\>=?)\s*(\d+(?:\.5)?)/)) && STAT_WORDS[m[1]]) { stat = STAT_WORDS[m[1]]; line = Math.floor(+m[2]) + (m[2].includes('.5') ? 1 : 0); }
      else if (/(kick|boot|score)s? (a|one|1) goal|goal ?kicker|anytime|to kick a goal|a goal/.test(seg)) { stat = 'goals'; line = 1; }
      else if ((m = seg.match(/(?:kick|boot)s?\s+(\d+|two|three|four|five)\s+goals?/))) { stat = 'goals'; line = { two: 2, three: 3, four: 4, five: 5 }[m[1]] || +m[1]; }
      else { for (const w of seg.split(' ')) if (STAT_WORDS[w]) { stat = STAT_WORDS[w]; break; } }
      if (stat && line == null) { // "Bont disposals" → best line for him
        const cands = market.options.filter((o) => o.player === pl.n && o.stat === stat).sort((a, b) => Math.abs(a.p - 0.75) - Math.abs(b.p - 0.75));
        if (cands[0]) { locked.push(cands[0]); notes.push(`${pl.n}: picked ${cands[0].line}+ ${stat} (about 75% likely)`); }
        continue;
      }
      if (stat && line != null) {
        const id = `P|${market.byTeam[pl.t].key}|${pl.n}|${stat}|${line}`;
        locked.push(optById.get(id) || playerLeg(model, market.byTeam[pl.t], pl, stat, line, learn));
        continue;
      }
      notes.push(`Couldn't tell what you wanted for ${pl.n} — try e.g. "${pl.n.split(' ').pop()} 25+ disposals"`);
      continue;
    }
    // ---- team result
    const teams = findTeams(seg);
    if (!teams.length) continue;
    const team = teams[0].team, g = market.byTeam[team];
    if (!g) { notes.push(`${team} don't play in ${market.round.label}`); continue; }
    const isLose = /(lose|loses|lost|losing|beaten|go down|to fall)/.test(seg) && !/beat\s/.test(seg.slice(0, teams[0].i));
    const winTeam = isLose ? (team === g.home ? g.away : g.home) : team;
    if (/(win|beat|over|defeat|salute|lose|loses|lost|beaten|go down|h2h|head to head|by \d+)/.test(seg) || teams.length > 1 || seg.trim() === norm(team)) {
      if (!(/(win|beat|over|defeat|salute|lose|beaten|go down|h2h|by \d+)/.test(seg))) {
        if (teams.length === 1 && seg.replace(norm(team), '').trim().length > 0 && !/(avoid|no|not|without|exclude|except|only)/.test(seg)) continue;
        if (/(avoid|no|not|without|exclude|except|only)/.test(seg)) continue;
      }
      let line = 0;
      if ((m = seg.match(/by\s*(\d+)/))) line = clamp(+m[1], 1, 120);
      locked.push(teamLeg(model, g, market.gps[g.key], winTeam, line, learn));
    }
  }
  // tidy: one leg per player / per team-result group, as asked last wins
  const seen = new Map();
  for (const l of locked) seen.set(l.kind === 'team' ? `T|${l.game}` : `P|${l.player}|${l.stat}`, l);
  return { locked: [...seen.values()], filter, legs, target, notes };
}

// ---------------------------------------------------------------- which games
// Default: the next round still to be played. A request can ask ahead ("round 5") or for just the next game.
export function pickRound(model, text) {
  const rounds = model.rounds?.length ? model.rounds : [model.round];
  const t = norm(text || ''), notes = [];
  let r = rounds[0];
  const m = t.match(/\b(?:round|rd)\s*(\d{1,2})\b/);
  if (m) {
    const want = rounds.find((x) => x.round === +m[1]);
    if (want) r = want;
    else notes.push(`Round ${m[1]} isn't open yet (open rounds: ${rounds[0].round}–${rounds[rounds.length - 1].round}), so this uses ${r.label}.`);
  }
  if (/\bnext (?:game|match)\b/.test(t) && r.games.length) r = { ...r, label: `${r.label} · ${r.games[0].home} v ${r.games[0].away}`, games: [r.games[0]] };
  return { round: r, notes };
}

// ---------------------------------------------------------------- generate
export function generate(model, history, req) {
  const picked = pickRound(model, req.mode === 'custom' ? req.request : '');
  const market = buildMarket(model, history, picked.round);
  let legs = clamp(Math.round(+req.legs || 4), 1, 21), target = clamp(+req.target || 5, 1.1, 5000), parsed = null;
  let pool = market.options.filter((o) => o.kind === 'team' || (o.av >= 0.75 && o.g >= 8));
  if (req.mode === 'custom' && req.request) {
    parsed = parseRequest(req.request, market, model);
    if (parsed.legs) legs = parsed.legs;
    if (parsed.target) target = parsed.target;
    const f = parsed.filter;
    pool = pool.filter((o) => !f.excludeTeams.has(o.team) && !f.excludeTeams.has(o.opp) &&
      !(f.noTeam && o.kind === 'team') && !(f.onlyStats && o.kind === 'player' && !f.onlyStats.has(o.stat)) &&
      !(f.onlyTeams && !f.onlyTeams.has(o.team) && !f.onlyTeams.has(o.opp)));
    if (f.onlyStats && !/team|win|result/.test(norm(req.request))) pool = pool.filter((o) => o.kind === 'player');
    if (parsed.locked.length > legs) legs = Math.min(21, parsed.locked.length);
  }
  const locked = parsed?.locked || [];
  const res = optimise(pool, { legs, target, locked });
  const L = res.legs;
  const pAll = L.reduce((a, l) => a * l.p, 1);
  const fair = L.reduce((a, l) => a * l.fair, 1), book = L.reduce((a, l) => a * l.book, 1);
  const closeness = Math.max(0, 100 - Math.abs(Math.log(book / target)) / Math.log(target > 1.5 ? target : 1.5) * 100);
  const sameGame = {}; for (const l of L) sameGame[l.game] = (sameGame[l.game] || 0) + 1;
  const warnings = [...picked.notes, ...(parsed?.notes || [])];
  const allSet = locked.length >= legs;
  if (allSet) warnings.push('Every leg came from your request, so the target odds were not used.');
  if (L.length < legs) warnings.push(`Only ${L.length} legs fit the rules (one per player, one result per game, max per game).`);
  if (!allSet && book < target * 0.97 && Math.abs(Math.log(book / target)) > 0.03 && res.maxOdds < target) warnings.push(`${legs} legs can reach about ${r2(res.maxOdds)}x at most this round — add legs to go higher.`);
  if (!allSet && book > target * 1.03 && res.minOdds > target) warnings.push(`The safest ${legs}-leg multi is already ${r2(res.minOdds)}x — use fewer legs for lower odds.`);
  if (Object.values(sameGame).some((n) => n > 1)) warnings.push('Some legs share a game. Their results are linked, so bookies usually pay a bit less than the straight product on same-game legs.');
  const ev = pAll * book;
  return {
    round: { label: market.round.label, practice: !!market.round.practice, season: market.round.season, round: market.round.round, first: market.round.games[0]?.day },
    request: { mode: req.mode, legs, target, text: req.request || null },
    legs: L.map(({ g, ...l }) => ({ ...l, locked: locked.some((x) => x.id === l.id) || undefined, p: Math.round(l.p * 1000) / 1000, p0: Math.round(l.p0 * 1000) / 1000 })),
    multi: { p: pAll, fair: r2(fair), book: r2(book), target, closeness: Math.round(closeness), ev: r2(ev), legs: L.length, minOdds: r2(res.minOdds), maxOdds: r2(res.maxOdds) },
    warnings,
  };
}

// ---------------------------------------------------------------- settling
export function autoSettle(model, bets) {
  let changed = 0;
  for (const b of bets) {
    if (b.status !== 'pending') continue;
    for (const l of b.legs) {
      if (l.hit === 0 || l.hit === 1 || l.hit === 'void') continue;
      const r = model.results?.[l.game];
      if (!r) continue;
      if (l.kind === 'team') {
        const home = l.home || l.game.split('-')[2];
        const margin = l.team === home ? r.hs - r.as_ : r.as_ - r.hs;
        l.hit = margin >= Math.max(1, l.line) ? 1 : 0; l.actual = margin; changed++;
      } else if (r.p && Object.keys(r.p).length) {
        const row = r.p[l.player];
        if (!row) { l.hit = 'void'; changed++; continue; }
        const v = row[model.stats.indexOf(l.stat)];
        l.hit = v >= l.line ? 1 : 0; l.actual = v; changed++;
      }
    }
    finalise(b);
  }
  return changed;
}

export function finalise(b) {
  const live = b.legs.filter((l) => l.hit !== 'void');
  if (live.some((l) => l.hit === 0)) b.status = 'lost';
  else if (live.length && live.every((l) => l.hit === 1)) b.status = 'won';
  else if (!live.length && b.legs.length) b.status = 'void';
  else b.status = 'pending';
  if (b.status !== 'pending') b.settledAt ||= new Date().toISOString();
  return b;
}

export function reportCard(model, history) {
  const learn = learnFactors(history);
  const settled = (history || []).filter((b) => !b.practice && (b.status === 'won' || b.status === 'lost'));
  const exp = settled.reduce((a, b) => a + (b.multi?.p || 0), 0);
  return {
    built: model.built, dataThrough: model.data_through, tested: model.report.tested_on,
    stats: Object.fromEntries(Object.entries(model.report.stats).map(([s, v]) => [s, { n: v.n, model: v.brier_model, naive: v.brier_naive, edge: r2((1 - v.brier_model / v.brier_naive) * 1000) / 10 }])),
    h2h: model.report.h2h, learn,
    bets: { settled: settled.length, won: settled.filter((b) => b.status === 'won').length, expected: r2(exp) },
  };
}
