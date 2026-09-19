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
    r"C:\Users\DZH\.codex\visualizations\2026\08\28\01a04765-754a-7f70-be4d-570ebbecffa7\h3-base-attention-first-last.html"
)


def load_npy(npz_path: Path, key: str):
    with zipfile.ZipFile(npz_path) as archive:
        raw = archive.read(f"{key}.npy")
    if raw[:6] != b"\x93NUMPY":
        raise ValueError("not an NPY payload")
    major = raw[6]
    if major == 1:
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
    if len(values) != rows * cols:
        raise ValueError("NPY payload length mismatch")
    return [
        [round(float(values[r * cols + c]), 8) for c in range(cols)]
        for r in range(rows)
    ]


def pct(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}%"


def main():
    first_summary = json.loads((HERE / "summary_step_00.json").read_text())
    last_summary = json.loads((HERE / "summary_step_48.json").read_text())
    first = load_npy(HERE / "attention_maps_step_00.npz", "temporal_conditional")
    last = load_npy(HERE / "attention_maps_step_48.npz", "temporal_conditional")
    data = json.dumps({"first": first, "last": last}, separators=(",", ":"))

    def metrics(summary):
        return {
            "sourceMass": pct(summary["mean_total_attention_mass_to_source"]),
            "sameFrame": pct(summary["mean_same_frame_fraction_within_source_attention"]),
            "sameFrameTop": f'{round(summary["same_frame_is_argmax_fraction"] * 37)}/37',
            "exact": pct(summary["mean_exact_same_position_fraction_within_all_source_attention"], 3),
            "neighbor": pct(summary["mean_3x3_same_frame_neighborhood_fraction_within_all_source_attention"], 3),
            "offset": f'{summary["mean_expected_absolute_frame_offset"]:.2f}',
        }

    meta = json.dumps(
        {
            "first": metrics(first_summary),
            "last": metrics(last_summary),
            "uniformFrame": pct(first_summary["mean_same_frame_fraction_under_uniform_source_attention"]),
            "uniformExact": pct(first_summary["mean_exact_position_fraction_under_uniform_source_attention"], 3),
        },
        separators=(",", ":"),
    )

    fragment = f'''<div id="h3-base-attn-first-last">
  <h2>Ref2VA 原始底座稠密注意力：首次与末次 DiT 前向 / Ref2VA base dense attention: first vs last DiT forward</h2>
  <div class="text-muted text-small">第 24 层 / Layer 24 · 56 个头取平均 / 56 heads averaged · 每个目标帧 16 个空间查询 / 16 spatial queries per target frame · QK 归一化与 RoPE 后 / post-QK-norm and post-RoPE · 无 VDN、无 selector、未跨 timestep 平均 / no VDN, selector, or timestep averaging</div>
  <div class="hm-grid">
    <section aria-labelledby="hm-first-title">
      <h3 id="hm-first-title">首次前向 / First forward · step 0</h3>
      <div class="hm-plot" id="hm-first"></div>
      <div class="hm-metrics tabular-nums" id="hm-first-metrics"></div>
    </section>
    <section aria-labelledby="hm-last-title">
      <h3 id="hm-last-title">末次前向 / Last forward · step 48</h3>
      <div class="hm-plot" id="hm-last"></div>
      <div class="hm-metrics tabular-nums" id="hm-last-metrics"></div>
    </section>
  </div>
  <div class="hm-legend text-small" aria-label="共享颜色刻度 / Shared color scale">
    <span>0</span><span class="hm-ramp" aria-hidden="true"></span><span id="hm-scale-max" class="tabular-nums"></span>
    <span class="text-muted">在源视频内部逐行归一化；两图共享第 99 百分位颜色上限 / row-normalized within source; shared 99th-percentile scale</span>
  </div>
  <div class="hm-selection" id="hm-selection" aria-live="polite"></div>
  <div class="tooltip" id="hm-tooltip" role="tooltip" hidden></div>
</div>
<style>
#h3-base-attn-first-last {{ position: relative; width: 100%; color: var(--foreground); }}
#h3-base-attn-first-last h2 {{ margin: 0 0 0.25rem; }}
#h3-base-attn-first-last h3 {{ margin: 1rem 0 0.25rem; }}
#h3-base-attn-first-last .hm-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1.25rem; }}
#h3-base-attn-first-last .hm-plot {{ width: 100%; min-width: 0; }}
#h3-base-attn-first-last .hm-plot svg {{ display: block; width: 100%; }}
#h3-base-attn-first-last .hm-frame {{ fill: none; stroke: var(--border); stroke-width: 1; }}
#h3-base-attn-first-last .hm-cell {{ fill: var(--viz-series-1); cursor: crosshair; }}
#h3-base-attn-first-last .hm-cell.is-diagonal {{ stroke: var(--foreground); stroke-width: 0.7; }}
#h3-base-attn-first-last .hm-cell.is-selected {{ stroke: var(--viz-series-2); stroke-width: 2; }}
#h3-base-attn-first-last .hm-axis {{ fill: var(--foreground); }}
#h3-base-attn-first-last .hm-gridline {{ stroke: var(--border); stroke-width: 0.6; opacity: 0.55; }}
#h3-base-attn-first-last .hm-metrics {{ margin-top: 0.25rem; color: var(--foreground); }}
#h3-base-attn-first-last .hm-metrics span {{ display: inline-block; margin-right: 0.75rem; margin-bottom: 0.2rem; }}
#h3-base-attn-first-last .hm-legend {{ display: flex; align-items: center; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.8rem; }}
#h3-base-attn-first-last .hm-ramp {{ width: 9rem; height: 0.65rem; background: linear-gradient(90deg, color-mix(in srgb, var(--viz-series-1) 5%, transparent), var(--viz-series-1)); }}
#h3-base-attn-first-last .hm-selection {{ margin-top: 0.65rem; min-height: 1.5rem; }}
#h3-base-attn-first-last .tooltip {{ position: absolute; z-index: 5; pointer-events: none; background: var(--popover); color: var(--popover-foreground); border: 1px solid var(--border); padding: 0.4rem 0.55rem; max-width: 18rem; }}
@media (max-width: 700px) {{
  #h3-base-attn-first-last .hm-grid {{ grid-template-columns: 1fr; gap: 0.25rem; }}
}}
</style>
<script>
(() => {{
  const root = document.getElementById('h3-base-attn-first-last');
  const matrices = {data};
  const meta = {meta};
  const ns = 'http://www.w3.org/2000/svg';
  const frames = 37;
  const ticks = [0, 9, 18, 27, 36];
  const all = matrices.first.flat().concat(matrices.last.flat()).slice().sort((a,b) => a-b);
  const vmax = all[Math.floor(0.99 * (all.length - 1))];
  const tooltip = root.querySelector('#hm-tooltip');
  const selected = {{ target: 18, source: 18 }};

  root.querySelector('#hm-scale-max').textContent = (100 * vmax).toFixed(2) + '%+';

  function el(name, attrs = {{}}, text = '') {{
    const node = document.createElementNS(ns, name);
    Object.entries(attrs).forEach(([k,v]) => node.setAttribute(k, String(v)));
    if (text) node.textContent = text;
    return node;
  }}

  function updateSelection() {{
    const a = matrices.first[selected.target][selected.source];
    const b = matrices.last[selected.target][selected.source];
    root.querySelector('#hm-selection').innerHTML = `<strong>已选：目标帧 T${{selected.target}} → 源帧 S${{selected.source}} / Selected T${{selected.target}} → S${{selected.source}}</strong> · 首次 / first ${{(100*a).toFixed(3)}}% · 末次 / last ${{(100*b).toFixed(3)}}% · 变化 / Δ ${{(100*(b-a)).toFixed(3)}} 个百分点 / percentage points`;
    root.querySelectorAll('.hm-cell').forEach(cell => {{
      cell.classList.toggle('is-selected', Number(cell.dataset.target) === selected.target && Number(cell.dataset.source) === selected.source);
    }});
  }}

  function metricText(which) {{
    const m = meta[which];
    return `<span>全部注意力 → 源视频 / all attention → source <strong>${{m.sourceMass}}</strong></span><span>对应帧占源注意力 / same-frame share of source <strong>${{m.sameFrame}}</strong>（成为最强源帧 / top: ${{m.sameFrameTop}}）</span><span>源注意力中的精确同位置 / exact same position within source <strong>${{m.exact}}</strong></span><span>源注意力中的同帧 3×3 邻域 / same-frame 3×3 within source <strong>${{m.neighbor}}</strong></span><span>平均时间帧差 / mean |Δt| <strong>${{m.offset}} 帧 / frames</strong></span>`;
  }}

  function draw(which, containerId) {{
    const container = root.querySelector(containerId);
    const matrix = matrices[which];
    container.replaceChildren();
    const width = Math.max(310, Math.floor(container.getBoundingClientRect().width));
    const height = Math.max(390, Math.min(580, Math.floor(width * 0.92)));
    const margin = {{ left: 58, right: 16, top: 12, bottom: 58 }};
    const side = Math.min(width - margin.left - margin.right, height - margin.top - margin.bottom);
    const x0 = margin.left + Math.max(0, (width - margin.left - margin.right - side) / 2);
    const y0 = margin.top;
    const cell = side / frames;
    const stageLabel = which === 'first' ? '首次 / First' : '末次 / Last';
    const svg = el('svg', {{ viewBox: `0 0 ${{width}} ${{height}}`, role: 'img', 'aria-label': `${{stageLabel}} DiT 前向时间注意力热力图 / temporal attention heatmap. 行是目标帧，列是源帧 / Rows are target frames and columns are source frames.` }});
    svg.appendChild(el('title', {{}}, `${{stageLabel}} DiT 前向时间注意力 / temporal attention`));
    svg.appendChild(el('desc', {{}}, '在源视频 token 内逐行归一化；带描边的对角线表示目标帧与源帧索引相同 / Row-normalized among source-video tokens; the outlined diagonal marks same-frame target-source pairs.'));

    ticks.forEach(t => {{
      const x = x0 + (t + 0.5) * cell;
      const y = y0 + (t + 0.5) * cell;
      svg.appendChild(el('line', {{ x1: x, y1: y0, x2: x, y2: y0 + side, class: 'hm-gridline' }}));
      svg.appendChild(el('line', {{ x1: x0, y1: y, x2: x0 + side, y2: y, class: 'hm-gridline' }}));
    }});

    for (let t = 0; t < frames; t++) {{
      for (let s = 0; s < frames; s++) {{
        const value = matrix[t][s];
        const opacity = Math.min(1, 0.04 + 0.96 * Math.sqrt(value / vmax));
        const rect = el('rect', {{
          x: x0 + s * cell,
          y: y0 + t * cell,
          width: cell + 0.08,
          height: cell + 0.08,
          'fill-opacity': opacity,
          class: `hm-cell${{t === s ? ' is-diagonal' : ''}}`,
          'aria-label': `目标帧 ${{t}} 到源帧 ${{s}} / target frame ${{t}} to source frame ${{s}}: ${{(100*value).toFixed(3)}} percent`
        }});
        rect.dataset.target = t;
        rect.dataset.source = s;
        rect.addEventListener('pointerenter', event => {{
          tooltip.hidden = false;
          tooltip.textContent = `目标 T${{t}} → 源 S${{s}} / Target T${{t}} → Source S${{s}} · 占源注意力 ${{(100*value).toFixed(3)}}% / of source attention`;
          const box = root.getBoundingClientRect();
          tooltip.style.left = Math.min(event.clientX - box.left + 10, box.width - 230) + 'px';
          tooltip.style.top = Math.max(0, event.clientY - box.top - 42) + 'px';
        }});
        rect.addEventListener('pointerleave', () => {{ tooltip.hidden = true; }});
        rect.addEventListener('click', () => {{ selected.target = t; selected.source = s; updateSelection(); }});
        svg.appendChild(rect);
      }}
    }}

    svg.appendChild(el('rect', {{ x: x0, y: y0, width: side, height: side, class: 'hm-frame', 'data-chart-frame': '' }}));
    ticks.forEach(t => {{
      svg.appendChild(el('text', {{ x: x0 + (t + 0.5) * cell, y: y0 + side + 20, 'text-anchor': 'middle', class: 'hm-axis text-small' }}, String(t)));
      svg.appendChild(el('text', {{ x: x0 - 10, y: y0 + (t + 0.5) * cell + 4, 'text-anchor': 'end', class: 'hm-axis text-small' }}, String(t)));
    }});
    svg.appendChild(el('text', {{ x: x0 + side / 2, y: y0 + side + 48, 'text-anchor': 'middle', class: 'hm-axis axis-title', 'data-axis': 'x' }}, '源帧索引 / Source frame index'));
    const yLabel = el('text', {{ x: 14, y: y0 + side / 2, 'text-anchor': 'middle', class: 'hm-axis axis-title', 'data-axis': 'y', transform: `rotate(-90 14 ${{y0 + side/2}})` }}, '目标帧索引 / Target frame index');
    svg.appendChild(yLabel);
    container.appendChild(svg);
  }}

  function redraw() {{
    draw('first', '#hm-first');
    draw('last', '#hm-last');
    updateSelection();
  }}

  root.querySelector('#hm-first-metrics').innerHTML = metricText('first');
  root.querySelector('#hm-last-metrics').innerHTML = metricText('last');
  let frame;
  const observer = new ResizeObserver(() => {{ cancelAnimationFrame(frame); frame = requestAnimationFrame(redraw); }});
  observer.observe(root);
  redraw();
}})();
</script>
'''
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(fragment, encoding="utf-8")
    print(OUT)
    print(OUT.stat().st_size)


if __name__ == "__main__":
    main()
