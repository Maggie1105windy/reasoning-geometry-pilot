from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET
import json

out = Path(__file__).parent / 'sources_text'
out.mkdir(parents=True, exist_ok=True)
paths = list(Path(r'D:\2026本基\立项申请书').glob('*.docx')) + [
    Path(r'D:\2026本基\7月文献调研与研究思路计划.docx'),
    Path(r'D:\2026本基\7月文献调研与研究思路计划v1.0.docx'),
    Path(r'D:\2026本基\3.24 idea.docx'),
    Path(r'D:\2026本基\0707汇报.docx'),
    Path(r'D:\2026本基\0801todolist.docx'),
]
ns = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
for i,path in enumerate(paths):
    with ZipFile(path) as z:
        root = ET.fromstring(z.read('word/document.xml'))
    lines = ['SOURCE: '+str(path)]
    for j,p in enumerate(root.findall('.//w:p',ns)):
        t=''.join(p.itertext()) if False else ''.join(x.text or '' for x in p.findall('.//w:t',ns))
        if t.strip(): lines.append(f'P{j}: {t}')
    target=out/f'{i:02d}_{path.stem}.txt'
    target.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'source':str(path),'text':str(target),'paragraphs':len(lines)-1},ensure_ascii=False))
