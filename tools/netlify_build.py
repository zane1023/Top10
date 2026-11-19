#!/usr/bin/env python3
"""Netlify build step for the AFL HP page. Never fails the deploy.

1. installs the Python packages,
2. builds netlify/data/model.json with tools/build_model.py (written to a temp file first),
3. adds the hidden "timeline" link to index.html,
4. if anything goes wrong, writes a small stub model that says so (the site still publishes and the
   page shows what failed), and always writes build-status.txt in the site root with the log tail.
"""
import json, os, shutil, subprocess, sys, time, datetime as dt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'netlify', 'data')
MODEL = os.path.join(DATA, 'model.json')
TMP = os.path.join(DATA, 'model.building.json')
LOG = []

def say(*a):
    line = ' '.join(str(x) for x in a)
    LOG.append(line); print(line, flush=True)

def run(cmd, timeout):
    say('$', ' '.join(cmd))
    try:
        p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        out = p.stdout or ''
        for l in out.splitlines()[-60:]: say('  ' + l)
        return p.returncode == 0
    except subprocess.TimeoutExpired:
        say('  TIMED OUT after', timeout, 's'); return False
    except Exception as e:
        say('  could not run:', repr(e)); return False

def main():
    t0 = time.time()
    os.makedirs(DATA, exist_ok=True)
    say('python', sys.version.split()[0], 'on', sys.platform)
    req = os.path.join(ROOT, 'requirements.txt')
    ok = run([sys.executable, '-m', 'pip', 'install', '-q', '-r', req], 900)
    if not ok: ok = run([sys.executable, '-m', 'pip', 'install', '-q', '--user', '-r', req], 900)
    built = False
    if ok:
        built = run([sys.executable, os.path.join(ROOT, 'tools', 'build_model.py'), '--out', TMP], 1200) and os.path.exists(TMP)
        if built:
            try: json.load(open(TMP)); shutil.move(TMP, MODEL); say('model.json ready,', os.path.getsize(MODEL) // 1024, 'KB')
            except Exception as e: built = False; say('model file was not valid JSON:', repr(e))
    else:
        say('pip install failed, skipping the model build')
    if not built:
        if os.path.exists(TMP): os.remove(TMP)
        stub = {'stub': True, 'at': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC'), 'log_tail': LOG[-40:]}
        with open(MODEL, 'w') as fh: json.dump(stub, fh)
        say('wrote a stub model: the page will say the model is not built yet')
    run([sys.executable, os.path.join(ROOT, 'tools', 'link_timeline.py')], 60)
    say('done in', round(time.time() - t0), 's, model built:', built)
    with open(os.path.join(ROOT, 'build-status.txt'), 'w') as fh: fh.write('\n'.join(LOG) + '\n')

if __name__ == '__main__':
    try: main()
    except Exception as e:
        print('netlify_build crashed:', repr(e))
    sys.exit(0)
