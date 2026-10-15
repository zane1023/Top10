// Netlify Function (v2) for the Apollyon Invaders leaderboard.
// Needs @netlify/blobs (listed in package.json, Netlify installs it on deploy).
import { getStore } from '@netlify/blobs';
import { randomBytes } from 'node:crypto';
import { handle } from '../lib/lb-core.mjs';

const rate = new Map();
let cachedSecret = null;

const store = () => getStore({ name: 'apollyon-invaders', consistency: 'strong' });

async function getSecret(s) {
  if (cachedSecret) return cachedSecret;
  // Optional: set LB_SECRET in Netlify env vars. Otherwise one is generated and kept in the store.
  if (process.env.LB_SECRET) return (cachedSecret = process.env.LB_SECRET);
  const cur = await s.get('secret', { type: 'text' });
  if (cur) return (cachedSecret = cur);
  const fresh = randomBytes(32).toString('hex');
  await s.set('secret', fresh, { onlyIfNew: true });
  return (cachedSecret = (await s.get('secret', { type: 'text' })) || fresh);
}

export default async (req, context) => {
  const s = store();
  const secret = await getSecret(s);
  const adapter = {
    async read() {
      const r = await s.getWithMetadata('board', { type: 'json', consistency: 'strong' });
      return r ? { data: r.data, etag: r.etag } : null;
    },
    async write(data, etag) {
      const r = etag
        ? await s.setJSON('board', data, { onlyIfMatch: etag })
        : await s.setJSON('board', data, { onlyIfNew: true });
      return !!r.modified;
    },
  };
  try {
    return await handle(req, {
      store: adapter,
      secret,
      now: Date.now(),
      ip: (context && context.ip) || req.headers.get('x-nf-client-connection-ip') || 'unknown',
      rate,
    });
  } catch (e) {
    return new Response(JSON.stringify({ error: 'server' }), {
      status: 500,
      headers: { 'content-type': 'application/json' },
    });
  }
};

export const config = { path: '/api/lb' };
