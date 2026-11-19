#!/usr/bin/env python3
"""
AFL HP prediction model builder.

Builds netlify/data/model.json, which the AFL HP bets page (bets.html + /api/afl-bets) uses.
Run weekly by .github/workflows/afl-hp-weekly.yml; can also be run by hand:

    python tools/build_model.py                    # downloads everything it needs
    python tools/build_model.py --parquet file.parquet --offline

What it does
  1. Player-match history (AFL StatsPro feed via github.com/DanielTomaro13/AFL-Modelling).
  2. Current-season fixture, results and model-consensus tips from the Squiggle API.
  3. Any completed current-season games missing from the feed are scraped from AFL Tables.
  4. Team ratings: margin-based Elo, parameters fitted on 2016+ results (walk-forward).
  5. Player props (disposals, goals, marks, tackles, kicks, handballs): blended short/long form,
     opponent "conceded" factor, home factor, negative-binomial spread; every parameter is picked by
     walk-forward back-testing on the latest seasons, then probabilities are calibrated (isotonic)
     against what actually happened.
  6. Current squads = latest team played for + 2026->27 list changes (tools/moves_2027.txt).
"""
import argparse, io, json, math, os, re, sys, time, datetime as dt
from collections import defaultdict, deque
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PARQUET_URL = 'https://raw.githubusercontent.com/DanielTomaro13/AFL-Modelling/main/data/processed/player_match.parquet'
UA = {'User-Agent': 'AFL-HP personal model (github.com/zane1023/Top10)'}
STATS = ['disposals', 'goals', 'marks', 'tackles', 'kicks', 'handballs']
LINES = {'disposals': [10, 15, 20, 25, 30, 35, 40], 'goals': [1, 2, 3, 4], 'marks': [3, 5, 7, 9],
         'tackles': [2, 4, 6, 8], 'kicks': [8, 12, 16, 20, 25], 'handballs': [6, 10, 14, 18]}
TEAMMAP = {'Brisbane Lions': 'Brisbane', 'GWS GIANTS': 'GWS', 'Greater Western Sydney': 'GWS', 'Sydney Swans': 'Sydney',
           'West Coast Eagles': 'West Coast', 'Gold Coast SUNS': 'Gold Coast', 'Gold Coast Suns': 'Gold Coast',
           'Geelong Cats': 'Geelong', 'Adelaide Crows': 'Adelaide'}
STATE = {'Adelaide': 'SA', 'Port Adelaide': 'SA', 'Brisbane': 'QLD', 'Gold Coast': 'QLD', 'GWS': 'NSW', 'Sydney': 'NSW',
         'West Coast': 'WA', 'Fremantle': 'WA', 'Tasmania': 'TAS'}
VENUE_STATE = {'MCG': 'VIC', 'M.C.G.': 'VIC', 'Marvel Stadium': 'VIC', 'Docklands': 'VIC', 'GMHBA Stadium': 'VIC', 'Kardinia Park': 'VIC',
               'Adelaide Oval': 'SA', 'Optus Stadium': 'WA', 'Perth Stadium': 'WA', 'Gabba': 'QLD', 'People First Stadium': 'QLD', 'Carrara': 'QLD',
               'SCG': 'NSW', 'S.C.G.': 'NSW', 'ENGIE Stadium': 'NSW', 'Sydney Showground': 'NSW', 'Manuka Oval': 'ACT',
               'University of Tasmania Stadium': 'TAS', 'York Park': 'TAS', 'Ninja Stadium': 'TAS', 'Bellerive Oval': 'TAS',
               'Marrara Oval': 'NT', 'TIO Stadium': 'NT', 'Traeger Park': 'NT', 'Mars Stadium': 'VIC', 'Eureka Stadium': 'VIC',
               'Norwood Oval': 'SA', 'Barossa Park': 'SA', 'Hands Oval': 'WA', 'Cazalys Stadium': 'QLD', 'Riverway Stadium': 'QLD'}
AFLT_ID = {'Adelaide': '01', 'Carlton': '03', 'Collingwood': '04', 'Essendon': '05', 'Western Bulldogs': '07', 'Fremantle': '08',
           'Geelong': '09', 'Hawthorn': '10', 'Melbourne': '11', 'North Melbourne': '12', 'Port Adelaide': '13', 'Richmond': '14',
           'St Kilda': '15', 'Sydney': '16', 'West Coast': '18', 'Brisbane': '19', 'Gold Coast': '20', 'GWS': '21'}
STATE_ALL = set(['Adelaide', 'Brisbane', 'Carlton', 'Collingwood', 'Essendon', 'Fremantle', 'Geelong', 'Gold Coast', 'GWS', 'Hawthorn',
                 'Melbourne', 'North Melbourne', 'Port Adelaide', 'Richmond', 'St Kilda', 'Sydney', 'West Coast', 'Western Bulldogs', 'Tasmania'])
AFLT_COLS = {'KI': 'kicks', 'MK': 'marks', 'HB': 'handballs', 'DI': 'disposals', 'GL': 'goals', 'TK': 'tackles', '%P': 'tog'}

def log(*a): print('[build]', *a, flush=True)
def tn(x): return TEAMMAP.get(x, x)

def http_get(url, timeout=60):
    import urllib.request
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r: return r.read()

# ------------------------------------------------------------------ data
def load_parquet(path):
    try:
        return pd.read_parquet(path)
    except Exception as e:
        log('pandas parquet failed (%s); using bundled reader' % e)
        sys.path.insert(0, HERE)
        import pq_fallback
        names, cols, _ = pq_fallback.read(path)
        return pd.DataFrame({n: cols[n] for n in names})

def tidy(df):
    keep = ['season', 'round', 'date', 'venue', 'venue_state', 'is_home', 'player', 'team', 'opponent', 'team_score', 'opp_score',
            'timeOnGroundPercentage'] + STATS
    df = df[[c for c in keep if c in df.columns]].copy()
    for c in ('team', 'opponent'): df[c] = df[c].map(tn)
    if pd.api.types.is_numeric_dtype(df['date']):     # raw microseconds (bundled reader) vs a real datetime (pyarrow)
        df['date'] = pd.to_datetime(df['date'], unit='us', utc=True)
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_convert('Australia/Melbourne')
    df['day'] = df['date'].dt.strftime('%Y-%m-%d')
    for c in ['season', 'round', 'is_home', 'team_score', 'opp_score'] + STATS:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')   # plain numpy floats, whatever the reader gave
    df = df.dropna(subset=['player', 'team', 'disposals'])
    df['season'] = df.season.astype(int); df['round'] = df['round'].astype(int)
    return df

def squiggle(q):
    return json.loads(http_get('https://api.squiggle.com.au/?q=' + q))

def scrape_afltables(game, year, known_players):
    """Player stats for one completed game from AFL Tables. Returns list of row dicts or []."""
    h, a = game['home'], game['away']
    ids = sorted([AFLT_ID[h], AFLT_ID[a]])
    url = f"https://afltables.com/afl/stats/games/{year}/{ids[0]}{ids[1]}{game['day'].replace('-', '')}.html"
    try:
        tables = pd.read_html(io.StringIO(http_get(url).decode('latin-1')))
    except Exception as e:
        log('  afltables miss', url, e); return []
    stat_tables = []
    for t in tables:
        cols = [c[-1] if isinstance(c, tuple) else c for c in t.columns]
        top = ' '.join(str(c[0]) for c in t.columns if isinstance(c, tuple))
        if 'Player' in cols and 'DI' in cols and 'KI' in cols:
            t = t.copy(); t.columns = cols; stat_tables.append((top, t))
    if len(stat_tables) < 2: return []
    rows = []
    for idx, (top, t) in enumerate(stat_tables[:2]):
        team = h if (h.split()[0] in top) else a if (a.split()[0] in top) else (h if idx == 0 else a)
        opp = a if team == h else h
        ts, os_ = (game['hscore'], game['ascore']) if team == h else (game['ascore'], game['hscore'])
        pool = known_players.get(team, set())
        for _, r in t.iterrows():
            nm = str(r['Player'])
            if ',' not in nm or nm.startswith(('Rushed', 'Totals', 'Opposition')): continue
            sur, giv = [x.strip() for x in nm.split(',', 1)]
            full = f'{giv} {sur}'
            cand = [p for p in pool if p.lower().endswith(sur.lower()) and p[0].lower() == giv[0].lower()]
            player = cand[0] if len(cand) == 1 else full
            row = dict(season=year, round=game['round'], date=pd.Timestamp(game['day'], tz='Australia/Melbourne'), day=game['day'],
                       venue=game['venue'], venue_state=VENUE_STATE.get(game['venue']), is_home=int(team == h), player=player,
                       team=team, opponent=opp, team_score=ts, opp_score=os_)
            for k, v in AFLT_COLS.items():
                val = pd.to_numeric(r.get(k, 0), errors='coerce')
                row['timeOnGroundPercentage' if v == 'tog' else v] = 0 if pd.isna(val) else float(val)
            rows.append(row)
    log('  afltables ok', url, len(rows))
    return rows

# ------------------------------------------------------------------ team ratings
def team_games(df):
    g = df[df.is_home == 1].drop_duplicates(['season', 'round', 'team', 'opponent'])
    g = g[['season', 'round', 'date', 'venue_state', 'team', 'opponent', 'team_score', 'opp_score']].sort_values('date')
    return g.rename(columns={'team': 'home', 'opponent': 'away', 'team_score': 'hs', 'opp_score': 'aws'}).reset_index(drop=True)

def hga_of(home, away, vstate, hga):
    hs, as_ = STATE.get(home, 'VIC'), STATE.get(away, 'VIC')
    if vstate == hs and vstate != as_: return hga
    if vstate == as_ and vstate != hs: return -hga
    return hga * 0.25

def run_elo(g, k, carry, hga, sigma, start_eval):
    from scipy.stats import norm
    R = defaultdict(float); season = None; ll = []; preds = []; errs = []
    for r in g.itertuples():
        if r.season != season:
            for t in R: R[t] *= carry
            season = r.season
        exp = R[r.home] - R[r.away] + hga_of(r.home, r.away, r.venue_state, hga)
        if not (pd.isna(r.hs) or pd.isna(r.aws)):
            m = r.hs - r.aws
            if r.season >= start_eval:
                p = float(np.clip(norm.cdf(exp / sigma), 1e-4, 1 - 1e-4))
                y = 1.0 if m > 0 else 0.0 if m < 0 else 0.5
                ll.append(-(y * math.log(p) + (1 - y) * math.log(1 - p))); preds.append((p, y)); errs.append(m - exp)
            err = m - exp
            R[r.home] += k * err; R[r.away] -= k * err
    run_elo.margin_sd = float(np.std(errs)) if errs else 36.0
    return R, (np.mean(ll) if ll else 9), preds

def fit_ratings(g, start_eval):
    best = None
    for k in (0.03, 0.045, 0.06, 0.08):
        for carry in (0.4, 0.5, 0.6, 0.7):
            for hga in (4, 6, 8, 10):
                for sigma in (22, 25, 28, 32):
                    _, ll, _ = run_elo(g, k, carry, hga, sigma, start_eval)
                    if best is None or ll < best[0]: best = (ll, k, carry, hga, sigma)
    return dict(logloss=best[0], k=best[1], carry=best[2], hga=best[3], sigma=best[4])

# ------------------------------------------------------------------ calibration helpers
def pav(x, y, bins=20):
    """Isotonic fit on quantile bins; returns breakpoints (raw p) and calibrated p (monotone)."""
    x = np.asarray(x); y = np.asarray(y)
    if len(x) < 200: return [[0, 1], [0, 1]]
    qs = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(qs, x, side='right') - 1, 0, len(qs) - 2)
    bx, by, bw = [], [], []
    for b in range(len(qs) - 1):
        m = idx == b
        if m.sum() == 0: continue
        bx.append(x[m].mean()); by.append(y[m].mean()); bw.append(m.sum())
    vals, wts, xs = [], [], []
    for xi, yi, wi in zip(bx, by, bw):
        vals.append(yi); wts.append(wi); xs.append([xi])
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1]); w = wts[-2] + wts[-1]
            xs[-2] += xs[-1]; vals[-2] = v; wts[-2] = w; vals.pop(); wts.pop(); xs.pop()
    px, py = [0.0], [0.0]
    for v, xx in zip(vals, xs):
        px.append(float(np.mean(xx))); py.append(float(v))
    px.append(1.0); py.append(1.0)
    return [[round(v, 4) for v in px], [round(v, 4) for v in py]]

def apply_cal(p, cal):
    return np.interp(p, cal[0], cal[1])

# ------------------------------------------------------------------ player model
def player_features(df, hl_s, hl_l):
    df = df.sort_values(['player', 'date']).copy()
    gp = df.groupby('player', sort=False)
    for s in STATS:
        x = gp[s]
        df[f'{s}_S'] = x.transform(lambda v: v.ewm(halflife=hl_s).mean().shift(1))
        df[f'{s}_L'] = x.transform(lambda v: v.ewm(halflife=hl_l).mean().shift(1))
        df[f'{s}_N'] = x.transform(lambda v: v.rolling(10, min_periods=1).mean().shift(1))
    df['n_prior'] = gp.cumcount()
    return df

def opp_factors(df, window=22):
    """For each team-game: how much of each stat that team conceded vs league average, over its previous `window` games."""
    tg = df.groupby(['season', 'round', 'date', 'team', 'opponent'])[STATS].sum().reset_index()
    league = tg.groupby('season')[STATS].transform('mean')
    rel = tg[STATS] / league
    conc = tg[['season', 'round', 'date', 'opponent']].rename(columns={'opponent': 'def_team'}).copy()
    for s in STATS: conc[s] = rel[s].values
    conc = conc.sort_values('date')
    out = conc.copy()
    for s in STATS:
        out[s + '_f'] = conc.groupby('def_team')[s].transform(lambda v: v.rolling(window, min_periods=4).mean().shift(1))
    cur = {t: {s: float(g[s].tail(window).mean()) for s in STATS} for t, g in conc.groupby('def_team')}
    return out[['season', 'round', 'def_team'] + [s + '_f' for s in STATS]], cur

def nb_sf(k, mu, r):
    from scipy.stats import nbinom
    mu = np.maximum(mu, 1e-3)
    return nbinom.sf(np.asarray(k) - 1, r, r / (r + mu))

def nb_logpmf(x, mu, r):
    from scipy.stats import nbinom
    mu = np.maximum(mu, 1e-3)
    return nbinom.logpmf(x, r, r / (r + mu))

# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet'); ap.add_argument('--offline', action='store_true')
    ap.add_argument('--out', default=os.path.join(ROOT, 'netlify', 'data', 'model.json'))
    ap.add_argument('--today')
    ap.add_argument('--fixture', help='JSON file with Squiggle-style games (and optional tips) gathered by hand; used instead of calling Squiggle')
    ap.add_argument('--extra', help='CSV of extra player-match rows (same columns as the feed) for games the feed is missing')
    a = ap.parse_args()
    today = dt.date.fromisoformat(a.today) if a.today else dt.datetime.now(dt.timezone(dt.timedelta(hours=10))).date()
    t0 = time.time()

    # 1. history
    if a.parquet: path = a.parquet
    else:
        path = '/tmp/player_match.parquet'
        log('downloading', PARQUET_URL); open(path, 'wb').write(http_get(PARQUET_URL, 180))
    df = tidy(load_parquet(path))
    log('history rows', len(df), 'seasons', df.season.min(), df.season.max())

    # 2. current-season fixture/results (Squiggle)
    year = today.year
    sq_games, tips = [], {}
    if a.fixture:
        fx = json.load(open(a.fixture))
        sq_games = fx.get('games', [])
        tips = {int(k): v for k, v in fx.get('tips', {}).items()}
        if sq_games: year = max(int(g['year']) for g in sq_games)
        log('fixture file games', len(sq_games), 'for', year)
    elif not a.offline:
        try:
            for yr in (year, year + 1) if today.month >= 10 else (year,):
                js = squiggle(f'games;year={yr}')['games']
                if js: sq_games = js; year = yr if any(not gm.get('complete') for gm in js) or yr == today.year else year
            log('squiggle games', len(sq_games), 'for', year)
        except Exception as e:
            log('squiggle unavailable:', e)
    games = []
    for gm in sq_games:
        if tn(gm.get('hteam')) not in STATE_ALL or tn(gm.get('ateam')) not in STATE_ALL: continue   # finals not set yet
        try:
            games.append(dict(id=gm['id'], season=int(gm['year']), round=int(gm['round']), roundname=gm.get('roundname') or f"Round {gm['round']}",
                              home=tn(gm['hteam']), away=tn(gm['ateam']), day=str(gm['date'])[:10], time=str(gm['date'])[11:16],
                              venue=gm.get('venue'), complete=int(gm.get('complete') or 0),
                              hscore=gm.get('hscore'), ascore=gm.get('ascore')))
        except Exception: pass

    if a.extra:
        ex = pd.read_csv(a.extra)
        ex['date'] = pd.to_datetime(ex['day']).dt.tz_localize('Australia/Melbourne')
        df = pd.concat([df, ex], ignore_index=True); log('added', len(ex), 'rows from', a.extra)
    # 3. completed current-season games missing from the feed -> AFL Tables
    if games and not a.offline and not a.fixture:
        have = set(zip(df.season, df.day, df.team))
        known = df[df.season >= df.season.max() - 1].groupby('team').player.agg(set).to_dict()
        extra = []
        for gm in games:
            if gm['complete'] == 100 and (gm['season'], gm['day'], gm['home']) not in have and gm['home'] in AFLT_ID and gm['away'] in AFLT_ID:
                extra += scrape_afltables(gm, gm['season'], known); time.sleep(1.0)
        if extra:
            df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True); log('added', len(extra), 'rows from AFL Tables')

    df = df.sort_values('date').reset_index(drop=True)
    last_season = int(df.season.max())

    # 4. team ratings
    tg = team_games(df)
    rp = fit_ratings(tg, start_eval=2019)
    R, ll, preds = run_elo(tg, rp['k'], rp['carry'], rp['hga'], rp['sigma'], start_eval=last_season - 1)
    if year > last_season:   # new season not started: apply the off-season regression now
        for t in R: R[t] *= rp['carry']
    rp['margin_sd'] = round(run_elo.margin_sd, 1)
    h2h_cal = pav([p for p, y in preds], [y for p, y in preds], bins=10)
    log('ratings', rp, 'recent logloss', round(ll, 4))

    # 5. player model — walk-forward fit on the last two seasons
    of_hist, of_cur = opp_factors(df)
    best = None; ev0 = last_season - 1
    home_f = {}
    for hl_s in (2, 3, 4, 6):
        for hl_l in (30, 50, 80):
            f = player_features(df, hl_s, hl_l)
            f = f.merge(of_hist, left_on=['season', 'round', 'opponent'], right_on=['season', 'round', 'def_team'], how='left')
            ev = f[(f.season >= ev0) & (f.n_prior >= 5)]
            if not home_f:
                for s in STATS:
                    hm = ev[ev.is_home == 1][s].mean() / ev[f'{s}_L'][ev.is_home == 1].mean()
                    am = ev[ev.is_home == 0][s].mean() / ev[f'{s}_L'][ev.is_home == 0].mean()
                    home_f[s] = float(np.clip(hm / am, 0.9, 1.1)) ** 0.5
            for w in (0.6, 0.7, 0.8):
                for ow in (0.75, 1.0, 1.25):
                    err = 0
                    for s in STATS:
                        mu = (w * ev[f'{s}_S'] + (1 - w) * ev[f'{s}_L']) * (1 + ow * (ev[f'{s}_f'].fillna(1) - 1)) * np.where(ev.is_home == 1, home_f[s], 1 / home_f[s])
                        err += float(((ev[s] - mu) ** 2).mean() / max(ev[s].var(), 1e-6))
                    if best is None or err < best[0]: best = (err, hl_s, hl_l, w, ow)
    _, HLS, HLL, W, OW = best
    log('player params', dict(hl_s=HLS, hl_l=HLL, w=W, ow=OW), 'home factors', home_f)
    f = player_features(df, HLS, HLL)
    f = f.merge(of_hist, left_on=['season', 'round', 'opponent'], right_on=['season', 'round', 'def_team'], how='left')
    ev = f[(f.season >= ev0) & (f.n_prior >= 5)].copy()
    nb_r, calib, report = {}, {}, {'stats': {}}
    for s in STATS:
        mu = ((W * ev[f'{s}_S'] + (1 - W) * ev[f'{s}_L']) * (1 + OW * (ev[f'{s}_f'].fillna(1) - 1)) * np.where(ev.is_home == 1, home_f[s], 1 / home_f[s])).values
        x = ev[s].values
        rbest = max(((float(np.mean(nb_logpmf(x, mu, r))), r) for r in (2, 3, 5, 8, 12, 18, 25, 40, 80)))[1]
        nb_r[s] = rbest
        P, Y, PN, SE = [], [], [], []
        naive = ev[f'{s}_N'].values; sea = ev.season.values
        for L in LINES[s]:
            p = nb_sf(L, mu, rbest); pn = nb_sf(L, naive, rbest)
            m = (p > 0.03) & (p < 0.97)
            P += list(p[m]); Y += list((x[m] >= L).astype(float)); PN += list(pn[m]); SE += list(sea[m])
        P, Y, PN, SE = map(np.array, (P, Y, PN, SE))
        calib[s] = pav(P, Y)                       # final map uses everything
        tr, te = SE == ev0, SE > ev0               # honest score: calibrate on one season, test on the next
        cal_tr = pav(P[tr], Y[tr])
        P, Y, PN = P[te], Y[te], PN[te]
        Pc = apply_cal(P, cal_tr)
        rel = []
        for lo in np.arange(0, 1, 0.1):
            m = (Pc >= lo) & (Pc < lo + 0.1)
            if m.sum() >= 50: rel.append([round(lo + 0.05, 2), int(m.sum()), round(float(Y[m].mean()), 3)])
        report['stats'][s] = dict(n=int(len(Y)), brier_model=round(float(np.mean((Pc - Y) ** 2)), 4),
                                  brier_naive=round(float(np.mean((PN - Y) ** 2)), 4), reliability=rel)
    ph = np.array([p for p, y in preds]); yh = np.array([y for p, y in preds])
    report['h2h'] = dict(n=int(len(yh)), brier=round(float(np.mean((apply_cal(ph, h2h_cal) - yh) ** 2)), 4),
                         tipping=round(float(np.mean((ph > 0.5) == (yh > 0.5))), 3))
    report['tested_on'] = f'{last_season} season, out of sample (form from earlier games only; calibration learned on {ev0})'

    # 6. rounds to predict: the next round by default, later rounds for bets asked for in advance
    rnd, rounds = None, []
    upcoming = [g for g in games if g['complete'] < 100 and g['season'] >= year]
    if upcoming:
        s0 = min(g['season'] for g in upcoming)
        for i, r0 in enumerate(sorted({g['round'] for g in upcoming if g['season'] == s0})):
            rg = sorted([g for g in upcoming if g['season'] == s0 and g['round'] == r0], key=lambda g: (g['day'], g['time']))
            if i < 2 and not a.fixture:   # tipster consensus only exists for the next round or two
                try:
                    for t in squiggle(f"tips;year={s0};round={r0}")['tips']:
                        if t.get('hconfidence') is not None: tips.setdefault(int(t['gameid']), []).append(float(t['hconfidence']) / 100)
                except Exception as e: log('tips unavailable', r0, e)
            rn = str(rg[0]['roundname'] or '')
            rr = dict(season=s0, round=r0, label=f"{s0} {rn if rn.lower().startswith(('round', 'opening', 'qualifying', 'elimination', 'semi', 'preliminary', 'grand')) else f'Round {r0}'}",
                      practice=False, games=[])
            for g in rg:
                cons = tips.get(g['id']); vs = VENUE_STATE.get(g['venue'])
                rr['games'].append(dict(key=f"{g['season']}-{r0}-{g['home']}-{g['away']}", home=g['home'], away=g['away'], day=g['day'], time=g['time'],
                                        venue=g['venue'], vstate=vs, cons=round(float(np.mean(cons)), 3) if cons else None, ncons=len(cons or [])))
            rounds.append(rr)
        rnd = rounds[0]
    if not rnd:   # off-season: practise on the last completed home-and-away round
        lr = df[(df.season == last_season) & (df['round'] <= 24)]['round'].max()
        gg = tg[(tg.season == last_season) & (tg['round'] == lr)]
        rnd = dict(season=last_season, round=int(lr), label=f'{last_season} Round {lr} (practice — already played)', practice=True, games=[])
        for g in gg.itertuples():
            rnd['games'].append(dict(key=f'{last_season}-{lr}-{g.home}-{g.away}', home=g.home, away=g.away, day=str(g.date)[:10], time='',
                                     venue=None, vstate=g.venue_state, cons=None, ncons=0))
        rounds = [rnd]

    # practice rounds are a true test: form, ratings and availability use only games BEFORE that round
    if rnd['practice']:
        cutoff = min(g['day'] for g in rnd['games'])
        hist = df[df.day < cutoff]
        R, _, _ = run_elo(tg[tg.date.dt.strftime('%Y-%m-%d') < cutoff], rp['k'], rp['carry'], rp['hga'], rp['sigma'], start_eval=9999)
        _, of_cur = opp_factors(hist)
    else:
        hist = df
    cy = rnd['season']

    # 7. squads, form and availability going into that round
    latest = hist.sort_values('date').groupby('player').tail(1).set_index('player')
    cur_season_games = hist[hist.season == cy]
    moves = [l.strip().split('|') for l in open(os.path.join(HERE, 'moves_2027.txt')) if l.strip() and not l.startswith('#')]
    players = {}
    team_games_by = hist.groupby(['season', 'team'])['round'].nunique().to_dict()
    for p, r in latest.iterrows():
        if r.season < last_season: continue
        team = r.team
        players[p] = dict(t=team, last=r.day, s=int(r.season))
    if cy > last_season:
        for io_, team, p in moves:
            if io_ == 'OUT' and p in players and players[p]['t'] == team: players.pop(p)
        for io_, team, p in moves:
            if io_ == 'IN' and p in players: players[p]['t'] = team
    # form state AFTER every game (for upcoming predictions)
    out_players = []
    g_sorted = hist.sort_values('date')
    for p, gdf in g_sorted.groupby('player'):
        if p not in players: continue
        info = players[p]; n = len(gdf)
        if n < 3: continue
        m = {}
        for s in STATS:
            v = gdf[s].astype(float)
            m[s] = [round(float(v.ewm(halflife=HLS).mean().iloc[-1]), 2), round(float(v.ewm(halflife=HLL).mean().iloc[-1]), 2),
                    round(float(v.tail(10).mean()), 2)]
        recent = gdf[gdf.season >= info['s']]
        team_n = team_games_by.get((info['s'], gdf.team.iloc[-1]), 1)
        if cy == info['s'] and len(cur_season_games):
            team_last = cur_season_games[cur_season_games.team == info['t']].day.max()
            av = 1.0 if info['last'] == team_last else round(min(1.0, (recent.day >= sorted(cur_season_games[cur_season_games.team == info['t']].day.unique())[-4:][0]).sum() / 4) * 0.5, 2)
        else:
            av = round(min(1.0, len(recent) / max(team_n, 1)), 2)
        l5 = {s: [int(x) for x in gdf[s].tail(5)] for s in STATS}
        out_players.append(dict(n=p, t=info['t'], g=n, av=av, m=m, l5=l5, last=info['last']))
    log('players exported', len(out_players))

    # 8. results for auto-settling (current season + practice round)
    res = {}
    seasons_for_res = {year, rnd['season']}
    rs = df[df.season.isin(seasons_for_res)]
    for (s, r, day), gdf in rs.groupby(['season', 'round', 'day']):
        for (home, away), x in gdf.groupby(['team', 'opponent']):
            if x.is_home.iloc[0] != 1: continue
            key = f'{s}-{r}-{home}-{away}'
            both = gdf[gdf.team.isin([home, away])]
            res[key] = dict(hs=int(x.team_score.iloc[0]), as_=int(x.opp_score.iloc[0]),
                            p={pl: [int(v) for v in row] for pl, row in zip(both.player, both[STATS].fillna(0).values)})
    for g in games:   # results Squiggle has but no player stats yet (scores only)
        key = f"{g['season']}-{g['round']}-{g['home']}-{g['away']}"
        if g['complete'] == 100 and key not in res and g['hscore'] is not None:
            res[key] = dict(hs=int(g['hscore']), as_=int(g['ascore']), p={})

    model = dict(built=dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC'), data_through=str(df.day.max()),
                 season=year, stats=STATS, lines=LINES,
                 params=dict(rating=rp, hl_s=HLS, hl_l=HLL, w=W, ow=OW, home_f=home_f, nb_r=nb_r,
                             bookie_margin={'h2h': 0.05, 'line': 0.07, 'prop': 0.12}),
                 ratings={t: round(v, 2) for t, v in R.items()}, state=STATE, opp={t: {s: round(v, 3) for s, v in d.items()} for t, d in of_cur.items()},
                 calib=dict(**calib, h2h=h2h_cal), report=report, round=rnd, rounds=rounds, players=out_players, results=res,
                 sources=['AFL StatsPro feed via github.com/DanielTomaro13/AFL-Modelling', 'Squiggle API (fixture, results, tips)', 'AFL Tables (fill-in match stats)'])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w') as fh: json.dump(model, fh, separators=(',', ':'))
    log('wrote', a.out, round(os.path.getsize(a.out) / 1024), 'KB in', round(time.time() - t0), 's')

if __name__ == '__main__':
    main()
