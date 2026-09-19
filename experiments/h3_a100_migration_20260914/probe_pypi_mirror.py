"""Read-only mirror availability and bounded transfer probe."""
import html.parser
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urljoin

lock = Path('/cache/zhonghao/h3/a100_v1/cuda_environment.lock.txt').read_text()
pins = dict(re.findall(r'^([\w.-]+)==([^\s\\]+)', lock, re.M))
print(json.dumps({'packages': len(pins), 'core': {x: pins[x] for x in ('torch', 'vllm', 'transformers', 'diffusers')} }), flush=True)

class Links(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.links += [v for k, v in attrs if k == 'href']

for base in ('https://mirrors.huaweicloud.com/repository/pypi/simple/', 'https://pypi.tuna.tsinghua.edu.cn/simple/'):
    url = base + 'torch/'
    try:
        page = subprocess.check_output(['curl', '-fsSL', '--max-time', '15', url], text=True)
        parser = Links()
        parser.feed(page)
        links = [urljoin(url, x) for x in parser.links if
                 f'torch-{pins["torch"]}-cp312-cp312-manylinux' in x and 'x86_64' in x]
        if not links:
            print(json.dumps({'mirror': base, 'exact_torch_available': False}), flush=True)
            continue
        result = subprocess.run(['curl', '-fsSL', '--max-time', '12', '--range', '0-4194303',
                                 '-o', '/dev/null', '-w', '%{http_code} %{size_download} %{speed_download}', links[0].split('#')[0]],
                                text=True, capture_output=True)
        print(json.dumps({'mirror': base, 'torch_url': links[0], 'probe': result.stdout,
                          'returncode': result.returncode, 'stderr': result.stderr}), flush=True)
    except Exception as exc:
        print(json.dumps({'mirror': base, 'error': repr(exc)}), flush=True)
