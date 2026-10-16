// Apollyon Invaders leaderboard: pure logic, storage is injected so it can be tested.
// One record per device. Names up to 8 characters, filtered server side.

import { createHmac, randomBytes, timingSafeEqual, createHash } from 'node:crypto';

export const NAME_MAX = 8;
// No score cap. Only limit is what a JS number can hold exactly (about 9 quadrillion).
export const MAX_SCORE = Number.MAX_SAFE_INTEGER;
export const TOKEN_MAX_AGE_MS = 7 * 24 * 60 * 60 * 1000;
export const MIN_RUN_SECONDS = 8;
export const BOARD_KEEP = 200;
export const TOP_SHOWN = 10;
export const RATE_WINDOW_MS = 10 * 60 * 1000;
export const RATE_LIMIT = 60;

// ---- name filter -------------------------------------------------------
// Contains-match: blocked anywhere in the name. Prefix-match: blocked only at the
// start, so ordinary names like GRAPE or ESSEX still work.
export const BLOCK_CONTAINS = ['nigg', 'niga', 'fuck', 'shit', '67', 'kkk', 'fag', 'gay', 'coon', 'pussy'];
export const BLOCK_PREFIX = ['rape', 'sex'];
const LEET = { '4': 'a', '@': 'a', '3': 'e', '1': 'i', '!': 'i', '0': 'o', '5': 's', '$': 's' };

export function normalizeForFilter(name) {
  let out = '';
  for (const ch of String(name).toLowerCase()) out += LEET[ch] || ch;
  return out.replace(/[^a-z0-9]/g, '');
}

export function isBlockedName(name) {
  const raw = String(name).toLowerCase().replace(/[^a-z0-9]/g, '');
  const leet = normalizeForFilter(name);
  for (const v of [raw, leet]) {
    for (const b of BLOCK_CONTAINS) if (v.includes(b)) return true;
    for (const b of BLOCK_PREFIX) if (v.startsWith(b)) return true;
  }
  return false;
}

// Returns {ok:true,name} or {ok:false,reason}
export function cleanName(input) {
  if (typeof input !== 'string') return { ok: false, reason: 'Enter a name' };
  // strip control chars and zero-width/bidi tricks, collapse whitespace
  let s = input.replace(/[\u0000-\u001f\u007f-\u009f​-‏‪-‮⁠-⁯﻿]/g, '')
    .replace(/\s+/g, ' ').trim();
  const chars = Array.from(s);
  if (chars.length < 1) return { ok: false, reason: 'Enter a name' };
  if (chars.length > NAME_MAX) return { ok: false, reason: 'Max ' + NAME_MAX + ' characters' };
  s = chars.join('');
  if (isBlockedName(s)) return { ok: false, reason: 'That name is not allowed' };
  return { ok: true, name: s };
}

// ---- tokens ------------------------------------------------------------
const b64u = (buf) => Buffer.from(buf).toString('base64url');

export function makeToken(secret, now) {
  const nonce = b64u(randomBytes(9));
  const body = now.toString(36) + '.' + nonce;
  const sig = b64u(createHmac('sha256', secret).update(body).digest()).slice(0, 22);
  return body + '.' + sig;
}

export function readToken(secret, token) {
  if (typeof token !== 'string' || token.length > 120) return null;
  const parts = token.split('.');
  if (parts.length !== 3) return null;
  const body = parts[0] + '.' + parts[1];
  const want = b64u(createHmac('sha256', secret).update(body).digest()).slice(0, 22);
  const a = Buffer.from(want), b = Buffer.from(parts[2]);
  if (a.length !== b.length || !timingSafeEqual(a, b)) return null;
  const issued = parseInt(parts[0], 36);
  if (!Number.isFinite(issued)) return null;
  return { issued, nonce: parts[1] };
}

// ---- helpers -----------------------------------------------------------
function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { 'content-type': 'application/json', 'cache-control': 'no-store' },
  });
}

function deviceKey(secret, deviceId) {
  return createHmac('sha256', secret).update('dev:' + deviceId).digest('base64url').slice(0, 24);
}

function validDeviceId(d) {
  return typeof d === 'string' && /^[A-Za-z0-9_-]{16,64}$/.test(d);
}

function ranked(entries) {
  return Object.entries(entries)
    .map(([k, v]) => ({ k, n: v.n, s: v.s, t: v.t }))
    .sort((a, b) => b.s - a.s || a.t - b.t);
}

// Names saved before the block list grew are masked when shown. RENAMES picks a
// specific display name for a stored name (lowercase letters and digits only).
// Any other masked name becomes "Pilot", or "Pilot 1", "Pilot 2"... by rank when
// there is more than one.
export const PLACEHOLDER_NAME = 'Pilot';
export const RENAMES = { niga: 'Black', kkk: 'Lawson' };

function displayNames(list) {
  const out = new Map();
  const fallback = [];
  for (const e of list) {
    if (!isBlockedName(e.n)) { out.set(e.k, e.n); continue; }
    const raw = String(e.n).toLowerCase().replace(/[^a-z0-9]/g, '');
    if (RENAMES[raw]) out.set(e.k, RENAMES[raw]);
    else fallback.push(e.k);
  }
  fallback.forEach((k, i) => out.set(k, fallback.length > 1 ? PLACEHOLDER_NAME + ' ' + (i + 1) : PLACEHOLDER_NAME));
  return out;
}

// Two names count as the same if they match ignoring case and spaces.
const nameKey = (n) => String(n).normalize('NFKC').toLowerCase().replace(/\s+/g, '');

// Older boards can hold two pilots with the same name. Keep the higher score and
// delete the rest. Names on the block list are left alone (they are masked instead).
function dupeKeys(entries) {
  const groups = new Map();
  for (const e of ranked(entries)) {
    if (isBlockedName(e.n)) continue;
    const g = nameKey(e.n);
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(e.k);
  }
  const drop = [];
  for (const keys of groups.values()) for (const k of keys.slice(1)) drop.push(k);
  return drop;
}

function dedupeEntries(entries) {
  const drop = dupeKeys(entries);
  for (const k of drop) delete entries[k];
  return drop.length;
}

function publicView(entries, myKey) {
  const list = ranked(entries);
  const names = displayNames(list);
  const top = list.slice(0, TOP_SHOWN).map((e) => ({ n: names.get(e.k), s: e.s, you: e.k === myKey }));
  let me = null;
  if (myKey) {
    const i = list.findIndex((e) => e.k === myKey);
    if (i >= 0) me = { rank: i + 1, n: names.get(list[i].k), s: list[i].s };
  }
  return { top, me, total: list.length };
}

// store interface: { read():{data,etag}|null, write(data, etag|null):Promise<boolean> }
async function mutateBoard(store, fn) {
  for (let attempt = 0; attempt < 6; attempt++) {
    const cur = await store.read();
    const board = cur && cur.data && cur.data.entries ? cur.data : { entries: {} };
    const result = fn(board);
    if (!result.changed) return { board, result };
    const ok = await store.write(board, cur ? cur.etag : null);
    if (ok) return { board, result };
  }
  throw new Error('board busy');
}

// ---- request handler ---------------------------------------------------
// ctx: { store, secret, now:number, ip:string, rate:Map }
export async function handle(req, ctx) {
  const url = new URL(req.url);
  const { store, secret, now } = ctx;

  // best effort rate limit, per function instance
  if (ctx.rate) {
    const rec = ctx.rate.get(ctx.ip) || { t: now, c: 0 };
    if (now - rec.t > RATE_WINDOW_MS) { rec.t = now; rec.c = 0; }
    rec.c++;
    ctx.rate.set(ctx.ip, rec);
    if (ctx.rate.size > 5000) ctx.rate.clear();
    if (rec.c > RATE_LIMIT) return json({ error: 'slow down' }, 429);
  }

  if (req.method === 'GET') {
    const d = url.searchParams.get('d');
    const myKey = validDeviceId(d) ? deviceKey(secret, d) : null;
    const cur = await store.read();
    const entries = cur && cur.data && cur.data.entries ? cur.data.entries : {};
    if (dupeKeys(entries).length) {
      const fixed = await mutateBoard(store, (b) => ({ changed: dedupeEntries(b.entries) > 0 }));
      return json(publicView(fixed.board.entries, myKey));
    }
    return json(publicView(entries, myKey));
  }

  if (req.method !== 'POST') return json({ error: 'method' }, 405);

  let body;
  try { body = await req.json(); } catch (e) { return json({ error: 'bad json' }, 400); }
  if (!body || typeof body !== 'object') return json({ error: 'bad body' }, 400);

  const action = url.searchParams.get('a');

  if (action === 'start') {
    return json({ token: makeToken(secret, now) });
  }

  if (action === 'submit') {
    if (!validDeviceId(body.device)) return json({ error: 'bad device' }, 400);
    const nm = cleanName(body.name);
    if (!nm.ok) return json({ error: nm.reason }, 422);

    const score = body.score;
    if (!Number.isSafeInteger(score) || score < 0 || score > MAX_SCORE) return json({ error: 'bad score' }, 422);

    const tok = readToken(secret, body.token);
    if (!tok) return json({ error: 'bad token' }, 403);
    const age = now - tok.issued;
    if (age < 0 || age > TOKEN_MAX_AGE_MS) return json({ error: 'expired' }, 403);
    const secs = Math.max(age / 1000, 1);
    if (score > 0 && secs < MIN_RUN_SECONDS) return json({ error: 'too fast' }, 422);

    const key = deviceKey(secret, body.device);
    const { board, result } = await mutateBoard(store, (b) => {
      const prev = b.entries[key];
      if (prev && prev.k === tok.nonce) return { changed: false, replay: true };
      // names must be unique: compare against what every other pilot is shown as
      if (!prev || prev.n !== nm.name) {
        const want = nameKey(nm.name);
        for (const [k, shown] of displayNames(ranked(b.entries))) {
          if (k !== key && nameKey(shown) === want) return { changed: false, taken: true };
        }
      }
      let changed = false;
      if (!prev) {
        if (score <= 0) return { changed: false, skipped: true };
        b.entries[key] = { n: nm.name, s: score, t: now, k: tok.nonce };
        changed = true;
      } else {
        if (score > prev.s) { prev.s = score; prev.t = now; changed = true; }
        if (prev.n !== nm.name) { prev.n = nm.name; changed = true; }
        if (changed) prev.k = tok.nonce;
      }
      // trim
      const keys = Object.keys(b.entries);
      if (keys.length > BOARD_KEEP) {
        const keep = new Set(ranked(b.entries).slice(0, BOARD_KEEP).map((e) => e.k));
        for (const k of keys) if (!keep.has(k)) delete b.entries[k];
      }
      return { changed };
    });
    if (result.replay) return json({ error: 'token used' }, 409);
    if (result.taken) return json({ error: 'That name is taken, pick a different name' }, 409);
    return json(publicView(board.entries, key));
  }

  return json({ error: 'unknown action' }, 400);
}

export function secretFromString(s) {
  return createHash('sha256').update(String(s)).digest();
}
