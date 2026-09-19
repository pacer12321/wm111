from __future__ import annotations

import ast
from array import array
import json
from pathlib import Path
import struct
import sys
import zipfile


HERE = Path(__file__).resolve().parent
OUT = Path(
    r"C:\Users\DZH\.codex\visualizations\2026\08\28\01a04765-754a-7f70-be4d-570ebbecffa7\h3-base-attention-global-average.html"
)


def load_matrix(path: Path, key: str):
    with zipfile.ZipFile(path) as archive:
        raw = archive.read(f"{key}.npy")
    if raw[:6] != b"\x93NUMPY":
        raise ValueError("not an NPY payload")
    if raw[6] == 1:
        header_len = struct.unpack_from("<H", raw, 8)[0]
        offset = 10
    else:
        header_len = struct.unpack_from("<I", raw, 8)[0]
        offset = 12
    header = ast.literal_eval(raw[offset : offset + header_len].decode("latin1"))
    if header["descr"] != "<f4" or header["fortran_order"]:
        raise ValueError(f"unsupported NPY layout: {header}")
    values = array("f")
    values.frombytes(raw[offset + header_len :])
    if sys.byteorder != "little":
        values.byteswap()
    rows, cols = header["shape"]
    return [
        [round(float(values[r * cols + c]), 8) for c in range(cols)]
        for r in range(rows)
    ]


def main():
    summary = json.loads((HERE / "summary.json").read_text())
    matrix = load_matrix(HERE / "attention_maps.npz", "temporal_conditional")
    same = summary["mean_same_frame_fraction_within_source_attention"]
    uniform = summary["uniform_same_frame_fraction"]
    top = summary["same_frame_is_argmax_fraction"]
    offset = summary["mean_expected_absolute_frame_offset"]
    metrics = json.dumps(
        {
            "same": f"{100 * same:.2f}%",
            "uniform": f"{100 * uniform:.2f}%",
            "lift": f"{same / uniform:.2f}×",
            "top": f"{100 * top:.2f}%（4/37）",
            "offset": f"{offset:.2f}",
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )
    data = json.dumps(matrix, separators=(",", ":"))

    fragment = f'''<div id="h3-global-attention-map">
  <h2>Ref2VA 原始底座全局平均时间注意力 / Ref2VA base global mean temporal attention</h2>
  <div class="text-muted text-small">56 heads × 50 DiT layers × 49 denoising forwards · 每个目标帧采样 16 个空间 query / 16 sampled spatial queries per target frame · 概率级算术平均 / probability-level arithmetic mean · 以注意力已落到 source video 为条件 / conditioned on source-video attention</div>
  <div class="hm-plot" id="hm-global-plot"></div>
  <div class="hm-metrics tabular-nums" id="hm-global-metrics"></div>
  <div class="hm-legend text-small" aria-label="共享颜色刻度 / Color scale">
    <span>0</span><span class="hm-ramp" aria-hidden="true"></span><span id="hm-global-max" class="tabular-nums"></span>
    <span class="text-muted">逐目标帧归一化；颜色在第99百分位封顶 / row-normalized; color capped at the 99th percentile</span>
  </div>
  <div class="hm-selection" id="hm-global-selection" aria-live="polite"></div>
  <div class="tooltip" id="hm-global-tooltip" role="tooltip" hidden></div>
</div>
<style>
#h3-global-attention-map {{ position: relative; width: 100%; color: var(--foreground); }}
#h3-global-attention-map h2 {{ margin: 0 0 0.25rem; }}
#h3-global-attention-map .hm-plot {{ width: 100%; max-width: 700px; margin-top: 0.5rem; }}
#h3-global-attention-map .hm-plot svg {{ display: block; width: 100%; }}
#h3-global-attention-map .hm-frame {{ fill: none; stroke: var(--border); stroke-width: 1; }}
#h3-global-attention-map .hm-cell {{ fill: var(--viz-series-1); cursor: crosshair; }}
#h3-global-attention-map .hm-cell.is-diagonal {{ stroke: var(--foreground); stroke-width: 0.8; }}
#h3-global-attention-map .hm-cell.is-selected {{ stroke: var(--viz-series-2); stroke-width: 2; }}
#h3-global-attention-map .hm-axis {{ fill: var(--foreground); }}
#h3-global-attention-map .hm-gridline {{ stroke: var(--border); stroke-width: 0.6; opacity: 0.55; }}
#h3-global-attention-map .hm-metrics {{ margin-top: 0.4rem; color: var(--foreground); }}
#h3-global-attention-map .hm-metrics span {{ display: inline-block; margin-right: 0.9rem; margin-bottom: 0.25rem; }}
#h3-global-attention-map .hm-legend {{ display: flex; align-items: center; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.6rem; }}
#h3-global-attention-map .hm-ramp {{ width: 10rem; height: 0.65rem; background: linear-gradient(90deg, color-mix(in srgb, var(--viz-series-1) 5%, transparent), var(--viz-series-1)); }}
#h3-global-attention-map .hm-selection {{ margin-top: 0.65rem; min-height: 1.5rem; }}
#h3-global-attention-map .tooltip {{ position: absolute; z-index: 5; pointer-events: none; background: var(--popover); color: var(--popover-foreground); border: 1px solid var(--border); padding: 0.4rem 0.55rem; max-width: 19rem; }}
</style>
<script>
(() => {{
  const root = document.getElementById('h3-global-attention-map');
  const matrix = {data};
  const metrics = {metrics};
  const ns = 'http://www.w3.org/2000/svg';
  const frames = 37;
  const ticks = [0, 9, 18, 27, 36];
  const sorted = matrix.flat().slice().sort((a,b) => a-b);
  const vmax = sorted[Math.floor(0.99 * (sorted.length - 1))];
  const tooltip = root.querySelector('#hm-global-tooltip');
  const selected = {{ target: 18, source: 18 }};
  root.querySelector('#hm-global-max').textContent = (100 * vmax).toFixed(2) + '%+';
  root.querySelector('#hm-global-metrics').innerHTML = `<span>对应帧占 source attention / same-frame share <strong>${{metrics.same}}</strong></span><span>均匀基线 / uniform <strong>${{metrics.uniform}}</strong></span><span>对角线提升 / diagonal lift <strong>${{metrics.lift}}</strong></span><span>对应帧为 Top-1 / same-frame is Top-1 <strong>${{metrics.top}}</strong></span><span>平均跨帧距离 / mean |Δt| <strong>${{metrics.offset}} 帧 / frames</strong></span>`;

  function el(name, attrs = {{}}, text = '') {{
    const node = document.createElementNS(ns, name);
    Object.entries(attrs).forEach(([key,value]) => node.setAttribute(key, String(value)));
    if (text) node.textContent = text;
    return node;
  }}

  function updateSelection() {{
    const value = matrix[selected.target][selected.source];
    root.querySelector('#hm-global-selection').innerHTML = `<strong>已选：目标帧 T${{selected.target}} → 源帧 S${{selected.source}} / Selected T${{selected.target}} → S${{selected.source}}</strong> · 占该目标帧 source attention 的 ${{(100*value).toFixed(3)}}% / of source attention`;
    root.querySelectorAll('.hm-cell').forEach(cell => {{
      cell.classList.toggle('is-selected', Number(cell.dataset.target) === selected.target && Number(cell.dataset.source) === selected.source);
    }});
  }}

  function draw() {{
    const container = root.querySelector('#hm-global-plot');
    container.replaceChildren();
    const width = Math.max(310, Math.floor(container.getBoundingClientRect().width));
    const height = Math.max(420, Math.min(690, Math.floor(width * 0.98)));
    const margin = {{ left: 64, right: 18, top: 12, bottom: 62 }};
    const side = Math.min(width - margin.left - margin.right, height - margin.top - margin.bottom);
    const x0 = margin.left + Math.max(0, (width - margin.left - margin.right - side) / 2);
    const y0 = margin.top;
    const cell = side / frames;
    const svg = el('svg', {{ viewBox: `0 0 ${{width}} ${{height}}`, role: 'img', 'aria-label': 'Ref2VA全局平均时间注意力热力图。行是目标帧，列是源帧。 / Ref2VA global mean temporal attention heatmap. Rows are target frames and columns are source frames.' }});
    svg.appendChild(el('title', {{}}, 'Ref2VA全局平均时间注意力 / Ref2VA global mean temporal attention'));
    svg.appendChild(el('desc', {{}}, '黑色对角线框表示目标帧与源帧索引相同；颜色表示在source attention中的占比 / Outlined diagonal cells are same-frame pairs; color is the share within source attention.'));
    ticks.forEach(t => {{
      const x = x0 + (t + 0.5) * cell;
      const y = y0 + (t + 0.5) * cell;
      svg.appendChild(el('line', {{ x1:x, y1:y0, x2:x, y2:y0+side, class:'hm-gridline' }}));
      svg.appendChild(el('line', {{ x1:x0, y1:y, x2:x0+side, y2:y, class:'hm-gridline' }}));
    }});
    for (let t = 0; t < frames; t++) {{
      for (let s = 0; s < frames; s++) {{
        const value = matrix[t][s];
        const opacity = Math.min(1, 0.04 + 0.96 * Math.sqrt(value / vmax));
        const rect = el('rect', {{ x:x0+s*cell, y:y0+t*cell, width:cell+0.08, height:cell+0.08, 'fill-opacity':opacity, class:`hm-cell${{t===s ? ' is-diagonal' : ''}}`, 'aria-label':`目标帧 ${{t}} 到源帧 ${{s}} / target ${{t}} to source ${{s}}: ${{(100*value).toFixed(3)}} percent` }});
        rect.dataset.target = t;
        rect.dataset.source = s;
        rect.addEventListener('pointerenter', event => {{
          tooltip.hidden = false;
          tooltip.textContent = `目标 T${{t}} → 源 S${{s}} / Target T${{t}} → Source S${{s}} · ${{(100*value).toFixed(3)}}%`;
          const box = root.getBoundingClientRect();
          tooltip.style.left = Math.min(event.clientX - box.left + 10, box.width - 245) + 'px';
          tooltip.style.top = Math.max(0, event.clientY - box.top - 42) + 'px';
        }});
        rect.addEventListener('pointerleave', () => {{ tooltip.hidden = true; }});
        rect.addEventListener('click', () => {{ selected.target=t; selected.source=s; updateSelection(); }});
        svg.appendChild(rect);
      }}
    }}
    svg.appendChild(el('rect', {{ x:x0, y:y0, width:side, height:side, class:'hm-frame', 'data-chart-frame':'' }}));
    ticks.forEach(t => {{
      svg.appendChild(el('text', {{ x:x0+(t+0.5)*cell, y:y0+side+22, 'text-anchor':'middle', class:'hm-axis text-small' }}, String(t)));
      svg.appendChild(el('text', {{ x:x0-10, y:y0+(t+0.5)*cell+4, 'text-anchor':'end', class:'hm-axis text-small' }}, String(t)));
    }});
    svg.appendChild(el('text', {{ x:x0+side/2, y:y0+side+52, 'text-anchor':'middle', class:'hm-axis axis-title', 'data-axis':'x' }}, '源帧索引 / Source frame index'));
    svg.appendChild(el('text', {{ x:15, y:y0+side/2, 'text-anchor':'middle', class:'hm-axis axis-title', 'data-axis':'y', transform:`rotate(-90 15 ${{y0+side/2}})` }}, '目标帧索引 / Target frame index'));
    container.appendChild(svg);
    updateSelection();
  }}

  let frame;
  const observer = new ResizeObserver(() => {{ cancelAnimationFrame(frame); frame=requestAnimationFrame(draw); }});
  observer.observe(root);
  draw();
}})();
</script>
'''
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(fragment, encoding="utf-8")
    print(OUT)
    print(OUT.stat().st_size)


if __name__ == "__main__":
    main()
