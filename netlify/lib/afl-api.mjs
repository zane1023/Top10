// Request handling for /api/afl-bets. Storage and model are injected so this can be tested without Netlify.
import { createHash, timingSafeEqual, randomBytes } from 'node:crypto';
import { generate, autoSettle, finalise, reportCard, buildMarket } from './afl-engine.mjs';

const DEFAULT_PIN_SHA256 = createHash('sha256').update('1144').digest(); // change with the BET_PIN env var
const FAILS = new Map(); // ip -> [timestamps]
const FAIL_WINDOW = 15 * 60 * 1000, FAIL_LIMIT = 8, KEEP = 1500;

const json = (status, body) => new Response(JSON.stringify(body), {
  status, headers: { 'content-type': 'application/json', 'cache-control': 'no-store', 'x-robots-tag': 'noindex' },
});

function pinOk(pin, envPin) {
  const want = envPin ? createHash('sha256').update(String(envPin)).digest() : DEFAULT_PIN_SHA256;
  const got = createHash('sha256').update(String(pin ?? '')).digest();
  return timingSafeEqual(want, got);
}

function clean(l) { // only keep known fields from a client-sent leg
  const pick = ['id', 'kind', 'game', 'home', 'player', 'team', 'opp', 'stat', 'line', 'label', 'sub', 'mu', 'p', 'p0', 'fair', 'book', 'av'];
  const o = {};
  for (const k of pick) if (l[k] !== undefined) o[k] = typeof l[k] === 'string' ? l[k].slice(0, 160) : l[k];
  for (const k of ['line', 'mu', 'p', 'p0', 'fair', 'book', 'av']) if (o[k] !== undefined) o[k] = Number(o[k]) || 0;
  o.hit = null;
  return o;
}

async function mutate(store, fn) {
  for (let i = 0; i < 6; i++) {
    const cur = (await store.read()) || { data: { bets: [] }, etag: null };
    const data = cur.data || { bets: [] };
    const out = await fn(data);
    if (out === false) return data;
    if (await store.write(data, cur.etag)) return data;
  }
  throw new Error('busy, try again');
}

export async function handle(req, { store, model, ip = 'x', envPin }) {
  if (req.method !== 'POST') return json(405, { error: 'POST only' });
  const now = Date.now();
  const fails = (FAILS.get(ip) || []).filter((t) => now - t < FAIL_WINDOW);
  if (fails.length >= FAIL_LIMIT) return json(429, { error: 'Too many wrong PINs. Try again in 15 minutes.' });
  let body;
  try { body = await req.json(); } catch { return json(400, { error: 'bad json' }); }
  if (!pinOk(body.pin, envPin)) { fails.push(now); FAILS.set(ip, fails); return json(401, { error: 'Wrong PIN' }); }
  FAILS.delete(ip);

  const read = async () => ((await store.read())?.data?.bets) || [];
  const a = body.action;

  if (a === 'login') return json(200, { ok: true });

  if (!model || model.stub || !Array.isArray(model.players)) {
    return json(503, { error: 'The model has not been built yet (the last deploy could not build it).', builtAt: model?.at || null, log: model?.log_tail || [] });
  }

  if (a === 'meta') {
    const bets = await read();
    const mk = buildMarket(model, bets);
    return json(200, {
      round: { ...model.round, games: mk.games }, built: model.built, dataThrough: model.data_through,
      rounds: (model.rounds || [model.round]).map((r) => ({ round: r.round, label: r.label, first: r.games[0]?.day, games: r.games.length })),
      report: reportCard(model, bets), players: model.players.length, sources: model.sources,
    });
  }

  if (a === 'generate') {
    const bets = await read();
    const out = generate(model, bets, {
      mode: body.mode === 'custom' ? 'custom' : 'best', legs: body.legs, target: body.target,
      request: typeof body.request === 'string' ? body.request.slice(0, 600) : '',
    });
    return json(200, out);
  }

  if (a === 'place') {
    const r = body.bet || {};
    const legs = Array.isArray(r.legs) ? r.legs.slice(0, 21).map(clean) : [];
    if (!legs.length) return json(400, { error: 'no legs' });
    const bet = {
      id: randomBytes(6).toString('hex'), placedAt: new Date().toISOString(), status: 'pending',
      practice: r.round ? !!r.round.practice : !!model.round.practice, round: String(r.round?.label || model.round.label).slice(0, 120),
      season: Number(r.round?.season) || model.round.season, roundNo: Number(r.round?.round) || model.round.round,
      mode: r.request?.mode === 'custom' ? 'custom' : 'best', request: (r.request?.text || '').slice(0, 600) || null,
      target: Number(r.request?.target) || null, stake: Math.max(0, Math.min(1e6, Number(body.stake) || 0)), note: String(body.note || '').slice(0, 200),
      legs,
      multi: { p: legs.reduce((x, l) => x * (l.p || 0), 1), fair: legs.reduce((x, l) => x * (l.fair || 1), 1), book: Number(body.book) || legs.reduce((x, l) => x * (l.book || 1), 1) },
      modelBuilt: model.built,
    };
    await mutate(store, (d) => { d.bets.unshift(bet); d.bets = d.bets.slice(0, KEEP); });
    return json(200, { ok: true, bet });
  }

  if (a === 'history' || a === 'autosettle') {
    let bets = await read();
    const probe = JSON.parse(JSON.stringify(bets));
    if (autoSettle(model, probe) > 0) {
      const d = await mutate(store, (d) => (autoSettle(model, d.bets) > 0 ? undefined : false));
      bets = d.bets;
    }
    return json(200, { bets, report: reportCard(model, bets) });
  }

  if (a === 'settle') {
    const d = await mutate(store, (d) => {
      const b = d.bets.find((x) => x.id === body.id);
      if (!b) return false;
      if (Array.isArray(body.legs)) for (const { i, hit } of body.legs) {
        if (b.legs[i]) b.legs[i].hit = hit === 1 || hit === 0 || hit === 'void' ? hit : null, b.legs[i].manual = true;
      }
      if (body.status === 'won') for (const l of b.legs) if (l.hit == null) l.hit = 1;
      if (body.status === 'pending') { for (const l of b.legs) { l.hit = null; delete l.actual; } b.settledAt = null; }
      finalise(b);
      if (body.status === 'lost' && b.status !== 'lost') { b.status = 'lost'; b.settledAt = new Date().toISOString(); } // a leg missed but not marked which
    });
    return json(200, { bets: d.bets, report: reportCard(model, d.bets) });
  }

  if (a === 'delete') {
    const d = await mutate(store, (d) => { const n = d.bets.length; d.bets = d.bets.filter((x) => x.id !== body.id); return n === d.bets.length ? false : undefined; });
    return json(200, { bets: d.bets, report: reportCard(model, d.bets) });
  }

  return json(400, { error: 'unknown action' });
}
