"""Read-only publication checks; no model loading, network, training, or inference."""
from pathlib import Path
import ast
import hashlib
import json
import re
import sqlite3
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks = []
    def check(name, condition):
        checks.append((name, bool(condition)))
    for name in ['README.md', 'README.zh-CN.md']:
        text = (ROOT / name).read_text(encoding='utf-8')
        check(name + ': language links', '[English](README.md) | [简体中文](README.zh-CN.md)' in text)
        for link in re.findall(r'\]\(([^)]+)\)', text):
            if not re.match(r'https?://|#', link):
                check(name + ': ' + link, (ROOT / link.split('#')[0]).exists())
    for path in ROOT.rglob('*.md'):
        if '.git' in path.parts:
            continue
        for link in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            if not re.match(r'https?://|#', link):
                target = link.split('#')[0].split('?')[0]
                check('Markdown link ' + path.relative_to(ROOT).as_posix() + ': ' + link,
                      (path.parent / target).exists())
    for path in (ROOT / 'reports/assets').rglob('*.svg'):
        tree = ET.parse(path)
        check('SVG XML ' + path.relative_to(ROOT).as_posix(),
              tree.getroot().tag == '{http://www.w3.org/2000/svg}svg')
        check('SVG PNG alternative ' + path.stem, path.with_suffix('.png').is_file())
    for path in (ROOT / 'reports/assets').rglob('*.png'):
        check('PNG signature ' + path.relative_to(ROOT).as_posix(),
              path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
    counts = {'sft':1000, 'preference':500, 'dev':36, 'evaluation':60,
              'preference_eval':100, 'independent_test':24}
    for stem, expected in counts.items():
        path = ROOT / 'synthesis/dataset' / (stem + '.jsonl')
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        check('dataset count ' + stem, len(rows) == expected)
    for path in ROOT.rglob('*.py'):
        if any(p in {'.git', '.venv', '__pycache__'} for p in path.parts):
            continue
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path.relative_to(ROOT)))
    for path in ROOT.rglob('*.json'):
        if '.git' not in path.parts:
            json.loads(path.read_text(encoding='utf-8-sig'))
    for path in ROOT.rglob('*.sqlite'):
        with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
            check('sqlite ' + path.relative_to(ROOT).as_posix(), db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok')
    for lang in ['en', 'zh-CN']:
        check('PDF signature ' + lang, (ROOT / 'reports' / ('research-report.' + lang + '.pdf')).read_bytes().startswith(b'%PDF-'))
    manifest = ROOT / 'release-manifest.json'
    if manifest.exists():
        entries = json.loads(manifest.read_text(encoding='utf-8'))['files']
        for entry in entries:
            path = ROOT / entry['path']
            check('hash ' + entry['path'], path.is_file() and digest(path) == entry['sha256'])
    else:
        check('release manifest exists', False)
    failed = [name for name, ok in checks if not ok]
    print(json.dumps({'passed': not failed, 'checks': len(checks), 'failed': failed,
                      'scope': 'File consistency and saved artifact checks only; no model execution.'}, indent=2))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
