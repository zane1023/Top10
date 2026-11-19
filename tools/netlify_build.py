#!/usr/bin/env python3
"""Netlify build step for the AFL HP page. Never fails the deploy.

1. gets the Python packages installed, trying several ways (Netlify's Python may refuse a plain
   `pip install` with "externally-managed-environment"),
2. builds netlify/data/model.json with tools/build_model.py (written to a temp file first),
3. adds the hidden "timeline" link to index.html,
4. if anything goes wrong, writes a small stub model that says so (the site still publishes and the
   page shows what failed), and always writes build-status.txt in the site root with the log.
"""
import json, os, shutil, subprocess, sys, time, datetime as dt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'netlify', 'data')
MODEL = os.path.join(DATA, 'model.json')
TMP = os.path.join(DATA, 'model.building.json')
DEPS = os.path.join(ROOT, '.pydeps')
REQ = os.path.join(ROOT, 'requirements.txt')
LOG = []

def say(*a):
    line = ' '.join(str(x) for x in a)
    LOG.append(line); print(line, flush=True)

def run(cmd, timeout, env=None):
    say('$', ' '.join(cmd))
    try:
        p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout,
                           env={**os.environ, **(env or {})})
        for l in (p.stdout or '').splitlines()[-40:]: say('  ' + l)
        say('  -> exit', p.returncode)
        return p.returncode == 0
    except subprocess.TimeoutExpired:
        say('  TIMED OUT after', timeout, 's'); return False
    except Exception as e:
        say('  could not run:', repr(e)); return False

def have_packages(py, env=None):
    return run([py, '-c', 'import pandas, numpy, scipy; print("packages ok", pandas.__version__, scipy.__version__)'], 120, env)

def get_python():
    """Returns (python executable, extra env) that can import the packages, or (None, None)."""
    py = sys.executable
    if have_packages(py): return py, {}
    pip = [py, '-m', 'pip', 'install', '-q', '--disable-pip-version-check', '-r', REQ]
    if run([py, '-m', 'pip', '--version'], 60) is False:
        run([py, '-m', 'ensurepip', '--upgrade', '--user'], 300)
    # 1. plain install  2. allowed on a "managed" system Python  3. into a local folder  4. a virtualenv
    if run(pip, 900) and have_packages(py): return py, {}
    if run(pip + ['--break-system-packages'], 900) and have_packages(py): return py, {}
    env = {'PYTHONPATH': DEPS}
    if run(pip + ['--target', DEPS], 900) and have_packages(py, env): return py, env
    venv = os.path.join(ROOT, '.venv')
    if run([py, '-m', 'venv', venv], 300):
        vpy = os.path.join(venv, 'bin', 'python')
        if run([vpy, '-m', 'pip', 'install', '-q', '-r', REQ], 900) and have_packages(vpy): return vpy, {}
    return None, None

def main():
    t0 = time.time()
    os.makedirs(DATA, exist_ok=True)
    say('python', sys.version.split()[0], 'at', sys.executable, 'on', sys.platform)
    py, env = get_python()
    built = False
    if py:
        built = run([py, os.path.join(ROOT, 'tools', 'build_model.py'), '--out', TMP], 1500, env) and os.path.exists(TMP)
        if built:
            try: json.load(open(TMP)); shutil.move(TMP, MODEL); say('model.json ready,', os.path.getsize(MODEL) // 1024, 'KB')
            except Exception as e: built = False; say('model file was not valid JSON:', repr(e))
    else:
        say('could not install the Python packages, skipping the model build')
    if not built:
        if os.path.exists(TMP): os.remove(TMP)
        stub = {'stub': True, 'at': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC'), 'log_tail': LOG[-40:]}
        with open(MODEL, 'w') as fh: json.dump(stub, fh)
        say('wrote a stub model: the page will say the model is not built yet')
    run([sys.executable, os.path.join(ROOT, 'tools', 'link_timeline.py')], 60)
    for d in (DEPS, os.path.join(ROOT, '.venv')):   # keep these out of the published site
        shutil.rmtree(d, ignore_errors=True)
    say('done in', round(time.time() - t0), 's, model built:', built)
    with open(os.path.join(ROOT, 'build-status.txt'), 'w') as fh: fh.write('\n'.join(LOG) + '\n')

if __name__ == '__main__':
    try: main()
    except Exception as e:
        print('netlify_build crashed:', repr(e))
    sys.exit(0)
