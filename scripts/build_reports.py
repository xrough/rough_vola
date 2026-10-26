"""Render the executed public notebooks as portable HTML, without private data."""
from __future__ import annotations
import hashlib
from html import escape,unescape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
from urllib.parse import quote,unquote,urlsplit,urlunsplit

import nbformat
from nbconvert import HTMLExporter
from run_notebooks import ROOT,NAMES

EXPECTED_FIGURES = dict(zip(NAMES, (2, 1, 1), strict=True))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Links(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False);self.parts=[]
    def tag(self):
        text=self.get_starttag_text()
        def replace(match):
            url=unescape(match.group(3));parsed=urlsplit(url)
            if parsed.scheme or parsed.netloc or not parsed.path:return match.group(0)
            target=(ROOT/'notebooks'/unquote(parsed.path)).resolve()
            if not target.is_relative_to(ROOT) or not target.exists():raise ValueError('Broken or external local report link')
            if target.name in NAMES:path=target.with_suffix('.html').name
            else:path=quote(Path(os.path.relpath(target,ROOT/'reports')).as_posix(),safe='/-._~')
            value=urlunsplit(('','',path,parsed.query,parsed.fragment))
            return match.group(1)+match.group(2)+escape(value,quote=True)+match.group(2)
        self.parts.append(re.sub(r'((?:href|src)\s*=\s*)([\"\'])(.*?)\2',replace,text,flags=re.I|re.S))
    def handle_starttag(self,*args):self.tag()
    def handle_startendtag(self,*args):self.tag()
    def handle_endtag(self,tag):self.parts.append(f'</{tag}>')
    def handle_data(self,data):self.parts.append(data)
    def handle_entityref(self,name):self.parts.append(f'&{name};')
    def handle_charref(self,name):self.parts.append(f'&#{name};')
    def handle_comment(self,data):self.parts.append(f'<!--{data}-->')
    def handle_decl(self,data):self.parts.append(f'<!{data}>')


def main():
    destination=ROOT/'reports';destination.mkdir(exist_ok=True)
    stats=[]
    for name in NAMES:
        path=ROOT/'notebooks'/name;n=nbformat.read(path,4)
        if not n.metadata.get('public_execution',{}).get('kernel_pinned_to_selected_interpreter'):
            raise ValueError('Execute the public notebooks before building reports')
        code=[c for c in n.cells if c.cell_type=='code' and c.source.strip()]
        if any(c.execution_count is None or any(o.output_type=='error' for o in c.get('outputs',[])) for c in code):
            raise ValueError('Unexecuted or failed notebook cell')
        figures=sum(any(mime.startswith('image/') for mime in o.get('data',{}))
            for c in code for o in c.get('outputs',[]))
        if figures != EXPECTED_FIGURES[name]:
            raise ValueError(f'Unexpected embedded figure count in {name}: {figures}')
        exporter=HTMLExporter(template_name='lab');exporter.exclude_input=True
        body,_=exporter.from_notebook_node(n);parser=Links();parser.feed(body);body=''.join(parser.parts)
        nav='<nav style="padding:1em 2em;border-bottom:1px solid #ccd"><a href="index.html">Research report index</a> · Current synthetic demos and explicitly archived aggregate findings</nav>'
        body=re.sub(r'(<body[^>]*>)',lambda m:m.group(1)+nav,body,count=1)
        (destination/Path(name).with_suffix('.html').name).write_text(body)
        stats.append({'notebook':name,'code_cells':len(code),'error_outputs':0,
            'embedded_figures':figures,
            'elapsed_seconds':n.metadata.public_execution.elapsed_seconds})
    items=''.join(f'<li><a href="{Path(name).with_suffix(".html").name}">{escape(name.removesuffix(".ipynb").replace("_"," "))}</a></li>' for name in NAMES)
    (destination/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Rough volatility research</title>'
        '<body style="max-width:900px;margin:3em auto;font:17px/1.6 system-ui;padding:0 1em"><h1>Pricing, calibration and the SPY research handoff</h1>'
        '<p>The first two reports execute small current-source synthetic examples. The third audits optional input presence and displays archived aggregate pilot evidence; it does not refit SPY.</p>'
        f'<ol>{items}</ol><p><a href="research_findings.md">Research findings</a> · <a href="spy_pilot.md">SPY assumptions and limitations</a> · '
        '<a href="aggregate_findings/manifest.json">Aggregate provenance scope</a></p>'
        '<p>No raw vendor data, observed contracts, private paths or historical Monte Carlo payloads are distributed. Full local provenance is separate. '
        'Failed checks remain visible; these reports do not establish general rough-American pricing or hedging readiness.</p></body></html>')
    paths=[*(ROOT/'notebooks').glob('*.ipynb'),*(destination/'aggregate_findings').glob('*'),
        *(destination).glob('*.html'),*(destination).glob('*.md'),ROOT/'scripts/run_notebooks.py',
        ROOT/'scripts/public_notebook_support.py',ROOT/'scripts/export_public_findings.py',ROOT/'scripts/build_reports.py']
    manifest={'version':'public_report_v1','synthetic_demos_executed':True,'market_data_recomputed':False,
        'historical_campaigns_recomputed':False,'full_private_provenance_distributed':False,
        'notebooks':stats,'files':{str(p.relative_to(ROOT)):sha(p) for p in sorted(paths) if p.is_file()}}
    (destination/'public_report_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Built three portable HTML reports; {len(manifest["files"])} public artifact fingerprints.')


if __name__=='__main__':main()
