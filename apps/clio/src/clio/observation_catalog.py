"""Render the configured Clio observations as a local HTML table."""
import argparse
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
from os import path
from pathlib import Path

import yaml

from clio.config import ClioObservations


STYLE = '''
    :root { color-scheme: light; font: 15px/1.5 system-ui, sans-serif; color: #182936; background: #f0f4f6; }
    body { margin: 0; }
    main { max-width: 62rem; margin: 0 auto; padding: clamp(1rem, 4vw, 3rem); }
    h1 { margin: 0; font-size: clamp(1.6rem, 4vw, 2.3rem); letter-spacing: -.035em; }
    p { margin: .5rem 0 1.25rem; color: #405665; }
    .scroll { overflow-x: auto; border: 1px solid #c8d6df; border-radius: .65rem; background: #fff; }
    .scroll:focus-visible { outline: 2px solid #176676; outline-offset: 3px; }
    table { width: 100%; min-width: 36rem; border-collapse: collapse; }
    th, td { padding: .72rem .9rem; text-align: left; vertical-align: top; }
    thead { color: #f7fbff; background: #193b4d; }
    thead th, .days { white-space: nowrap; }
    tbody tr + tr { border-top: 1px solid #e0e9ee; }
    tbody tr:nth-child(even) { background: #f6f9fa; }
    .days { font-variant-numeric: tabular-nums; }
    .mode + .mode { margin-top: .2rem; }
    .mode strong { display: inline-block; min-width: 5.2rem; font-size: .86rem; color: #405665; }
    ol { display: inline; margin: 0; padding: 0; list-style: none; }
    li { display: inline; }
    li + li::before { content: ' → '; color: #405665; }
    code { font-size: .89em; }
    #notice { color: #7e2828; }
    #notice:empty { display: none; }
'''

REFRESH = '''
<script>
  const current = document.querySelector('main');
  async function refresh() {
    try {
      const response = await fetch('/', {cache: 'no-store'});
      if (!response.ok) throw new Error('Configuration unavailable');
      const content = new DOMParser().parseFromString(await response.text(), 'text/html');
      const updated = content.querySelector('main');
      if (current.innerHTML !== updated.innerHTML) current.replaceChildren(...updated.childNodes);
      current.querySelector('#notice').textContent = '';
    } catch {
      current.querySelector('#notice').textContent = 'Configuration unavailable. Retrying in five seconds.';
    } finally {
      setTimeout(refresh, 5000);
    }
  }
  setTimeout(refresh, 5000);
</script>
'''


def read_catalog(config_path: Path) -> ClioObservations:
    project = yaml.safe_load(path.expandvars(config_path.read_text(encoding='utf-8')))
    if not isinstance(project, dict):
        raise ValueError('Expected a project configuration mapping')
    return ClioObservations.model_validate(project.get('clio'))


def render_sources(label: str, sources: list[str]) -> str:
    items = ''.join(f'<li><code>{escape(source)}</code></li>' for source in sources)
    values = f'<ol aria-label="{label} source priority">{items}</ol>' if items else '—'
    return f'<div class="mode"><strong>{label}</strong> {values}</div>'


def render_page(config: ClioObservations | None, *, live: bool = False) -> str:
    if config is None:
        content = '<p id="notice" role="status">Configuration unavailable. Retrying in five seconds.</p>'
    else:
        rows = []
        for metric, policy in config.observations.items():
            days = policy.backfill_days
            unit = 'day' if days == 1 else 'days'
            sources = render_sources('Live', policy.sources.live) + render_sources('Historical', policy.sources.historical)
            rows.append(f'<tr><th scope="row"><code>{escape(metric)}</code></th>'
                        f'<td class="days">{days} {unit}</td><td>{sources}</td></tr>')
        content = (
            '<p>Backfill time is the configured recovery window. Sources are listed in priority order.</p>'
            '<p id="notice" role="status"></p>'
            '<div class="scroll" tabindex="0" role="region" aria-label="Observation policies"><table>'
            '<thead><tr><th scope="col">Metric</th><th scope="col">Backfill time</th>'
            '<th scope="col">Sources</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
        )
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clio observations</title><style>{STYLE}</style></head>
<body><main><h1>Clio observations</h1>{content}</main>{REFRESH if live else ''}</body></html>
'''


def serve(config_path: Path, port: int) -> None:
    class CatalogHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != '/':
                self.send_error(404)
                return
            try:
                config = read_catalog(config_path)
            except (OSError, ValueError, yaml.YAMLError):
                config = None
            html = render_page(config, live=True).encode('utf-8')
            self.send_response(200 if config is not None else 503)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(html)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(html)

        def log_message(self, *args):
            pass

    with HTTPServer(('127.0.0.1', port), CatalogHandler) as server:
        print(f'Clio observations at http://127.0.0.1:{server.server_port}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/project.yaml'))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--port', type=int, help='Local viewer port (default: 8765)')
    mode.add_argument('--output', type=Path, help='Save a standalone HTML snapshot')
    args = parser.parse_args()
    try:
        if args.output:
            args.output.write_text(render_page(read_catalog(args.config)), encoding='utf-8')
        else:
            serve(args.config, 8765 if args.port is None else args.port)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
