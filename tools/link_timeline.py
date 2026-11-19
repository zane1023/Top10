#!/usr/bin/env python3
"""Run in the Netlify build: turns the word "timeline" in the home page's brief section into the hidden link to /bets.html.
Safe to run any number of times; does nothing if the link is already there or the sentence has changed."""
import os
p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'index.html')
s = open(p, encoding='utf-8').read()
old = "You'll hear back with a quote and timeline.</p>"
new = "You'll hear back with a quote and <a href=\"/bets.html\" style=\"color:inherit;text-decoration:none;cursor:text\">timeline</a>.</p>"
if 'href="/bets.html"' in s: print('timeline link already present')
elif old in s: open(p, 'w', encoding='utf-8').write(s.replace(old, new, 1)); print('linked "timeline" to /bets.html')
else: print('WARNING: brief sentence not found; "timeline" link not added')
