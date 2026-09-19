"""Build the inline five-timestep attention visualization fragment."""

from __future__ import annotations

import argparse
from pathlib import Path


TEMPLATE = r'''<div id="h3-ts-attention-v1">
  <h2>Ref2VA 的 T→S 注意力如何随去噪阶段变化</h2>
  <div class="viz-row text-small text-muted" aria-label="Experiment configuration">
    <span>固定 DiT layer 24</span><span>56 heads 全平均</span><span>每个 target 帧采样 16 个空间 query</span><span>只统计 T→S，并在 source 内归一化</span>
  </div>
  <div class="heatmap-grid" id="h3-ts-grid"></div>
  <div class="legend-row text-small">
    <span>低 attention</span><span class="attention-gradient" aria-hidden="true"></span><span>高 attention（五图统一上限：99.5 percentile）</span>
    <span class="diagonal-key" aria-hidden="true"></span><span>对应帧参考线</span>
  </div>
  <div class="sr-only" id="h3-ts-description">五张热图分别展示去噪 step 0、12、24、36、48 时，target frame 对 source frame 的条件注意力概率。所有图使用相同颜色尺度。</div>
  <div class="tooltip" id="h3-ts-tooltip" role="tooltip" hidden></div>
</div>
<style>
  #h3-ts-attention-v1 { position: relative; width: 100%; color: var(--foreground); }
  #h3-ts-attention-v1 h2 { margin: 0 0 8px; }
  #h3-ts-attention-v1 .viz-row { gap: 14px; margin-bottom: 14px; }
  #h3-ts-attention-v1 .heatmap-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; width: 100%; }
  #h3-ts-attention-v1 .heatmap-panel { min-width: 0; }
  #h3-ts-attention-v1 .panel-title { font-weight: 500; margin-bottom: 2px; }
  #h3-ts-attention-v1 .panel-metrics { color: var(--muted-foreground); margin-bottom: 4px; min-height: 34px; }
  #h3-ts-attention-v1 .chart-host { width: 100%; }
  #h3-ts-attention-v1 .chart-host svg { display: block; width: 100%; }
  #h3-ts-attention-v1 .axis text, #h3-ts-attention-v1 .axis-title { fill: var(--foreground); font-size: 12px; }
  #h3-ts-attention-v1 .axis path, #h3-ts-attention-v1 .axis line { stroke: var(--border); }
  #h3-ts-attention-v1 rect[data-chart-frame] { fill: none; stroke: var(--border); }
  #h3-ts-attention-v1 .legend-row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-top: 12px; }
  #h3-ts-attention-v1 .attention-gradient { width: 130px; height: 10px; background: linear-gradient(90deg, color-mix(in srgb, var(--viz-series-1) 4%, transparent), var(--viz-series-1)); }
  #h3-ts-attention-v1 .diagonal-key { width: 24px; border-top: 2px solid var(--foreground); margin-left: 8px; }
  #h3-ts-attention-v1 .tooltip { position: absolute; pointer-events: none; z-index: 10; background: var(--popover); color: var(--popover-foreground); padding: 6px 8px; border-radius: 6px; box-shadow: 0 4px 16px color-mix(in srgb, var(--foreground) 18%, transparent); }
  @media (max-width: 850px) { #h3-ts-attention-v1 .heatmap-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
  @media (max-width: 480px) { #h3-ts-attention-v1 .heatmap-grid { grid-template-columns: 1fr; } }
</style>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(() => {
  const root = document.getElementById('h3-ts-attention-v1');
  const payload = __PAYLOAD__;
  const grid = root.querySelector('#h3-ts-grid');
  const tooltip = root.querySelector('#h3-ts-tooltip');
  const fmtPct = d3.format('.1%');
  const fmt = d3.format('.3f');
  const panels = payload.records.map(record => {
    const section = document.createElement('section');
    section.className = 'heatmap-panel';
    section.innerHTML = `<div class="panel-title">Step ${record.step}</div>
      <div class="panel-metrics text-small">同帧 ${fmtPct(record.same_frame_fraction)} · Top-1 ${fmtPct(record.same_frame_top1_fraction)}<br>±2帧 ${fmtPct(record.within_2_fraction)} · E|Δt| ${record.expected_absolute_frame_offset.toFixed(2)}</div>
      <div class="chart-host"></div>`;
    grid.appendChild(section);
    return { section, host: section.querySelector('.chart-host'), record };
  });

  function draw({host, record}) {
    const width = Math.max(180, Math.floor(host.getBoundingClientRect().width));
    const height = width + 18;
    const margin = {top: 7, right: 8, bottom: 38, left: 43};
    const inner = Math.min(width - margin.left - margin.right, height - margin.top - margin.bottom);
    host.replaceChildren();
    const svg = d3.select(host).append('svg')
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('role', 'img')
      .attr('aria-describedby', 'h3-ts-description');
    svg.append('title').text(`Denoising step ${record.step} target-to-source temporal attention heatmap`);
    svg.append('desc').text(`Same-frame fraction ${fmtPct(record.same_frame_fraction)}, same-frame Top-1 ${fmtPct(record.same_frame_top1_fraction)}, expected absolute frame offset ${record.expected_absolute_frame_offset.toFixed(2)}.`);
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
    const n = payload.frames;
    const cell = inner / n;
    const flat = [];
    record.matrix.forEach((row, target) => row.forEach((value, source) => flat.push({target, source, value})));
    g.append('rect').attr('data-chart-frame', '').attr('width', inner).attr('height', inner);
    g.selectAll('rect.attention-cell').data(flat).join('rect')
      .attr('class', 'attention-cell')
      .attr('x', d => d.source * cell)
      .attr('y', d => d.target * cell)
      .attr('width', cell + 0.12)
      .attr('height', cell + 0.12)
      .attr('fill', 'var(--viz-series-1)')
      .attr('fill-opacity', d => 0.025 + 0.975 * Math.sqrt(Math.min(1, d.value / payload.shared_color_ceiling_p995)))
      .on('pointerenter pointermove', (event, d) => {
        tooltip.hidden = false;
        tooltip.textContent = `T${d.target} → S${d.source}: ${fmt(d.value)}`;
        const box = root.getBoundingClientRect();
        tooltip.style.left = `${event.clientX - box.left + 12}px`;
        tooltip.style.top = `${event.clientY - box.top + 12}px`;
      })
      .on('pointerleave', () => { tooltip.hidden = true; });
    g.append('line')
      .attr('x1', cell / 2).attr('y1', cell / 2)
      .attr('x2', inner - cell / 2).attr('y2', inner - cell / 2)
      .attr('stroke', 'var(--foreground)').attr('stroke-width', 1.2).attr('pointer-events', 'none');
    const scale = d3.scaleLinear().domain([0, n - 1]).range([cell / 2, inner - cell / 2]);
    const ticks = [0, 9, 18, 27, 36];
    g.append('g').attr('class', 'axis').attr('transform', `translate(0,${inner})`).call(d3.axisBottom(scale).tickValues(ticks).tickSizeOuter(0));
    g.append('g').attr('class', 'axis').call(d3.axisLeft(scale).tickValues(ticks).tickSizeOuter(0));
    svg.append('text').attr('class', 'axis-title').attr('x', margin.left + inner / 2).attr('y', height - 3).attr('text-anchor', 'middle').text('Source frame / S帧');
    svg.append('text').attr('class', 'axis-title').attr('transform', `translate(12,${margin.top + inner / 2}) rotate(-90)`).attr('text-anchor', 'middle').text('Target frame / T帧');
  }
  const redraw = () => panels.forEach(draw);
  redraw();
  let lastWidth = Math.floor(grid.getBoundingClientRect().width);
  new ResizeObserver(entries => {
    const width = Math.floor(entries[0].contentRect.width);
    if (width !== lastWidth) {
      lastWidth = width;
      requestAnimationFrame(redraw);
    }
  }).observe(grid);
})();
</script>'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = args.input.read_text()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(TEMPLATE.replace("__PAYLOAD__", payload), encoding="utf-8")


if __name__ == "__main__":
    main()
