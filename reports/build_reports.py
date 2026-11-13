"""Build the two research reports from one source tree; run no model experiment."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', help='Existing Tectonic executable; defaults to PATH')
    parser.add_argument('--language', choices=['en', 'zh-CN', 'both'], default='both')
    args = parser.parse_args()
    engine = args.engine or shutil.which('tectonic')
    if not engine:
        parser.error('Install Tectonic or pass --engine. No dependency installer is started.')
    for language, folder, name in [
        ('en', 'en', 'research-report.en.pdf'),
        ('zh-CN', 'zh', 'research-report.zh-CN.pdf'),
    ]:
        if args.language not in ['both', language]:
            continue
        with tempfile.TemporaryDirectory(prefix='tutoring-report-') as stage:
            subprocess.run([engine, '-X', 'compile', '--untrusted', '--outdir', stage,
                            'main.tex'], cwd=HERE/'source'/folder, check=True)
            pdf = Path(stage)/'main.pdf'
            if not pdf.read_bytes().startswith(b'%PDF-'):
                raise RuntimeError('The compiler did not produce a PDF.')
            temporary = HERE/(name+'.tmp')
            shutil.copyfile(pdf, temporary)
            temporary.replace(HERE/name)
        print(HERE/name)


if __name__ == '__main__':
    main()
