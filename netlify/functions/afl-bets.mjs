// Netlify Function (v2) for the private AFL HP multi page (bets.html).
// The model (netlify/data/model.json) is built on every Netlify deploy by tools/netlify_build.py and bundled in here.
// If that build failed, model.json is a small stub and the API reports it instead of crashing.
// Optional env vars: BET_PIN (replaces the default PIN).
import { getStore } from '@netlify/blobs';
import model from '../data/model.json';
import { handle } from '../lib/afl-api.mjs';

const store = () => getStore({ name: 'afl-hp-bets', consistency: 'strong' });

export default async (req, context) => {
  const s = store();
  const adapter = {
    async read() {
      const r = await s.getWithMetadata('bets', { type: 'json', consistency: 'strong' });
      return r ? { data: r.data, etag: r.etag } : null;
    },
    async write(data, etag) {
      const r = etag ? await s.setJSON('bets', data, { onlyIfMatch: etag }) : await s.setJSON('bets', data, { onlyIfNew: true });
      return !!r.modified;
    },
  };
  try {
    return await handle(req, { store: adapter, model, ip: context?.ip || req.headers.get('x-nf-client-connection-ip') || 'x', envPin: process.env.BET_PIN });
  } catch (e) {
    return new Response(JSON.stringify({ error: String(e?.message || e) }), { status: 500, headers: { 'content-type': 'application/json' } });
  }
};

export const config = { path: '/api/afl-bets' };
