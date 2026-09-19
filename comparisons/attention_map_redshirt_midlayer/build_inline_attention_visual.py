import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
NPZ_PATH = ROOT / "attention_maps.npz"
SUMMARY_PATH = ROOT / "summary.json"
OUT_PATH = Path(
    r"C:\Users\DZH\.codex\visualizations\2026\08\28\01a04765-754a-7f70-be4d-570ebbecffa7\h3-attention-map.html"
)


def quantize(array: np.ndarray, scale: float) -> list:
    normalized = np.clip(array / max(scale, 1e-12), 0.0, 1.0)
    return np.rint(normalized * 255.0).astype(np.uint8).tolist()


data = np.load(NPZ_PATH)
summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

temporal = data["temporal_conditional"].astype(np.float64)
representatives = data["representative_frames"].astype(int).tolist()
top_source = summary["top_source_frame_by_target"]
roi = data["roi_attention"].astype(np.float64)

spatial = []
for idx, target_frame in enumerate(representatives):
    top_frame = int(top_source[target_frame])
    same_map = roi[idx, target_frame]
    top_map = roi[idx, top_frame]
    spatial.append(
        {
            "target": target_frame,
            "topSource": top_frame,
            "sameScale": float(same_map.max()),
            "topScale": float(top_map.max()),
            "same": quantize(same_map, float(same_map.max())),
            "top": quantize(top_map, float(top_map.max())),
        }
    )

payload = {
    "temporal": quantize(temporal, float(temporal.max())),
    "temporalScale": float(temporal.max()),
    "spatial": spatial,
    "roi": data["roi_yx"].astype(int).tolist(),
    "metrics": {
        "sourceMass": summary["mean_total_attention_mass_to_source"],
        "sameFrame": summary["mean_same_frame_fraction_within_source_attention"],
        "uniformFrame": summary["mean_same_frame_fraction_under_uniform_source_attention"],
        "top1": summary["same_frame_is_argmax_fraction"],
        "meanOffset": summary["mean_expected_absolute_frame_offset"],
        "exactPosition": summary[
            "mean_exact_same_position_fraction_within_all_source_attention"
        ],
        "uniformPosition": summary[
            "mean_exact_position_fraction_under_uniform_source_attention"
        ],
        "neighbor": summary[
            "mean_3x3_same_frame_neighborhood_fraction_within_all_source_attention"
        ],
        "uniformNeighbor": summary[
            "mean_3x3_fraction_upper_bound_under_uniform_source_attention"
        ],
    },
}

template = r'''
<div id="h3-attention-map">
  <style>
    #h3-attention-map { color: var(--foreground); width: 100%; }
    #h3-attention-map .am-title { margin: 0 0 4px; }
    #h3-attention-map .am-subtitle { margin: 0 0 14px; color: var(--muted-foreground); }
    #h3-attention-map .am-layout { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(260px, .75fr); gap: 24px; align-items: start; }
    #h3-attention-map .am-panel { min-width: 0; }
    #h3-attention-map .am-panel h3 { margin: 0 0 8px; }
    #h3-attention-map .am-chart { width: 100%; min-height: 320px; }
    #h3-attention-map .am-spatial { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    #h3-attention-map .am-spatial-chart { width: 100%; min-width: 0; }
    #h3-attention-map .am-caption { margin: 8px 0 0; color: var(--muted-foreground); }
    #h3-attention-map .am-callout { margin: 10px 0 0; padding-left: 12px; border-left: 3px solid var(--viz-series-2); }
    #h3-attention-map .am-controls { margin: 0 0 10px; }
    #h3-attention-map .am-axis text, #h3-attention-map .am-label { fill: var(--foreground); font-size: 12px; }
    #h3-attention-map .am-axis path, #h3-attention-map .am-axis line { stroke: var(--border); }
    #h3-attention-map .am-frame { fill: none; stroke: var(--border); }
    #h3-attention-map .am-diagonal { fill: none; stroke: var(--foreground); stroke-width: 1.4; }
    #h3-attention-map .am-roi { fill: none; stroke: var(--foreground); stroke-width: 2; }
    #h3-attention-map .am-tooltip { position: absolute; pointer-events: none; display: none; z-index: 20; background: var(--popover); color: var(--popover-foreground); border: 1px solid var(--border); padding: 6px 8px; border-radius: 6px; }
    #h3-attention-map .am-relative { position: relative; }
    @media (max-width: 700px) {
      #h3-attention-map .am-layout { grid-template-columns: 1fr; }
      #h3-attention-map .am-spatial { grid-template-columns: 1fr; }
      #h3-attention-map .am-chart { min-height: 300px; }
    }
  </style>

  <h2 class="am-title">Ref2VA + VDN 的真实 T→S 注意力</h2>
  <p class="am-subtitle">第 25/50 个 DiT block，第一次去噪前向；Q/K 已经过 QK-norm 与 RoPE。每个 target 帧采样 16 个空间 query、汇总 56 个 head。</p>

  <div class="am-layout">
    <section class="am-panel" aria-labelledby="am-temporal-title">
      <h3 id="am-temporal-title">时间注意力：target 帧 → source 帧</h3>
      <div class="am-relative">
        <div id="am-temporal" class="am-chart"></div>
        <div id="am-tooltip" class="am-tooltip text-small" role="tooltip"></div>
      </div>
      <p class="am-caption text-small">每一行在 source 帧内归一化；白框对角线代表“对应帧”。越亮表示该 target 帧分给该 source 帧的注意力越高。</p>
      <p id="am-temporal-result" class="am-callout"></p>
    </section>

    <section class="am-panel" aria-labelledby="am-spatial-title">
      <h3 id="am-spatial-title">空间注意力：一个人物附近 query 看向哪里</h3>
      <div class="viz-controls am-controls" aria-label="选择 target 帧">
        <label class="form-label" for="am-target-select">Target 帧</label>
        <select id="am-target-select" class="form-select"></select>
      </div>
      <div class="am-spatial">
        <div class="am-spatial-chart">
          <div id="am-same-map"></div>
          <p id="am-same-label" class="am-caption text-small"></p>
        </div>
        <div class="am-spatial-chart">
          <div id="am-top-map"></div>
          <p id="am-top-label" class="am-caption text-small"></p>
        </div>
      </div>
      <p class="am-caption text-small">两个空间面板各自归一化，只用于看形状；白框标出同一空间坐标。帧之间的绝对权重应以左侧时间图为准。</p>
      <p id="am-spatial-result" class="am-callout"></p>
    </section>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
  <script>
    (() => {
      const root = document.getElementById('h3-attention-map');
      const DATA = __PAYLOAD__;
      const fg = () => getComputedStyle(root).getPropertyValue('--foreground').trim();
      const series = () => getComputedStyle(root).getPropertyValue('--viz-series-1').trim();
      const border = () => getComputedStyle(root).getPropertyValue('--border').trim();
      const pct = value => `${(value * 100).toFixed(2)}%`;
      const ratio = (a, b) => `${(a / b).toFixed(1)}×`;
      const tooltip = d3.select(root.querySelector('#am-tooltip'));

      root.querySelector('#am-temporal-result').textContent =
        `结论：同帧只占 source 注意力的 ${pct(DATA.metrics.sameFrame)}，均匀基线为 ${pct(DATA.metrics.uniformFrame)}；37 个 target 帧中，同帧成为最大权重帧的次数为 0。注意力主要落在首尾 anchor 帧，而不是沿对角线集中。`;
      root.querySelector('#am-spatial-result').textContent =
        `同帧同位置的绝对份额只有 ${pct(DATA.metrics.exactPosition)}，但约为均匀基线的 ${ratio(DATA.metrics.exactPosition, DATA.metrics.uniformPosition)}；说明存在位置偏好，却远未形成“只看对应位置”的集中交互。`;

      const select = root.querySelector('#am-target-select');
      DATA.spatial.forEach((item, index) => {
        const option = document.createElement('option');
        option.value = String(index);
        option.textContent = `T = ${item.target}`;
        select.appendChild(option);
      });

      function drawTemporal() {
        const host = root.querySelector('#am-temporal');
        const width = Math.max(320, host.getBoundingClientRect().width || 520);
        const height = Math.min(620, Math.max(360, width * 0.86));
        const margin = { top: 12, right: 18, bottom: 54, left: 64 };
        const iw = width - margin.left - margin.right;
        const ih = height - margin.top - margin.bottom;
        d3.select(host).selectAll('*').remove();
        const svg = d3.select(host).append('svg')
          .attr('viewBox', `0 0 ${width} ${height}`)
          .attr('width', '100%')
          .attr('height', height)
          .attr('role', 'img')
          .attr('aria-label', 'Target frame to source frame attention heatmap');
        svg.append('title').text('Target frame to source frame attention');
        svg.append('desc').text('The diagonal is not dominant; attention favors the first and last source frames.');
        const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
        const x = d3.scaleBand().domain(d3.range(37)).range([0, iw]).padding(0.02);
        const y = d3.scaleBand().domain(d3.range(37)).range([0, ih]).padding(0.02);
        const cells = [];
        DATA.temporal.forEach((row, target) => row.forEach((value, sourceFrame) => cells.push({target, sourceFrame, value})));
        g.selectAll('rect.am-cell').data(cells).join('rect')
          .attr('class', 'am-cell')
          .attr('x', d => x(d.sourceFrame)).attr('y', d => y(d.target))
          .attr('width', x.bandwidth()).attr('height', y.bandwidth())
          .attr('fill', series())
          .attr('fill-opacity', d => 0.04 + 0.96 * Math.pow(d.value / 255, 0.55))
          .on('pointermove', (event, d) => {
            const rect = root.getBoundingClientRect();
            tooltip.style('display', 'block')
              .style('left', `${event.clientX - rect.left + 12}px`)
              .style('top', `${event.clientY - rect.top + 12}px`)
              .text(`T${d.target} → S${d.sourceFrame}: ${pct((d.value / 255) * DATA.temporalScale)}`);
          })
          .on('pointerleave', () => tooltip.style('display', 'none'));
        g.selectAll('rect.am-diagonal').data(d3.range(37)).join('rect')
          .attr('class', 'am-diagonal')
          .attr('x', d => x(d)).attr('y', d => y(d))
          .attr('width', x.bandwidth()).attr('height', y.bandwidth());
        g.append('rect').attr('class', 'am-frame').attr('width', iw).attr('height', ih);
        const ticks = [0, 9, 18, 27, 36];
        g.append('g').attr('class', 'am-axis').attr('transform', `translate(0,${ih})`)
          .call(d3.axisBottom(x).tickValues(ticks).tickSizeOuter(0));
        g.append('g').attr('class', 'am-axis')
          .call(d3.axisLeft(y).tickValues(ticks).tickSizeOuter(0));
        svg.append('text').attr('class', 'am-label').attr('data-axis', 'x')
          .attr('x', margin.left + iw / 2).attr('y', height - 10).attr('text-anchor', 'middle').text('Source frame index');
        svg.append('text').attr('class', 'am-label').attr('data-axis', 'y')
          .attr('transform', `translate(15,${margin.top + ih / 2}) rotate(-90)`).attr('text-anchor', 'middle').text('Target frame index');
      }

      function drawSpatialMap(hostSelector, values, title, roi) {
        const host = root.querySelector(hostSelector);
        const width = Math.max(250, host.getBoundingClientRect().width || 300);
        const height = Math.max(190, width * 24 / 42 + 38);
        d3.select(host).selectAll('*').remove();
        const svg = d3.select(host).append('svg')
          .attr('viewBox', `0 0 ${width} ${height}`)
          .attr('width', '100%').attr('height', height)
          .attr('role', 'img').attr('aria-label', title);
        svg.append('title').text(title);
        const margin = {top: 25, right: 6, bottom: 6, left: 6};
        const iw = width - margin.left - margin.right;
        const ih = height - margin.top - margin.bottom;
        const x = d3.scaleBand().domain(d3.range(42)).range([0, iw]).padding(0.015);
        const y = d3.scaleBand().domain(d3.range(24)).range([0, ih]).padding(0.015);
        const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
        const cells = [];
        values.forEach((row, yy) => row.forEach((value, xx) => cells.push({yy, xx, value})));
        g.selectAll('rect.am-space-cell').data(cells).join('rect')
          .attr('class', 'am-space-cell')
          .attr('x', d => x(d.xx)).attr('y', d => y(d.yy))
          .attr('width', x.bandwidth()).attr('height', y.bandwidth())
          .attr('fill', series())
          .attr('fill-opacity', d => 0.035 + 0.965 * Math.pow(d.value / 255, 0.55));
        g.append('rect').attr('class', 'am-roi')
          .attr('x', x(roi[1])).attr('y', y(roi[0]))
          .attr('width', x.bandwidth()).attr('height', y.bandwidth());
        g.append('rect').attr('class', 'am-frame').attr('width', iw).attr('height', ih);
        svg.append('text').attr('class', 'am-label').attr('x', width / 2).attr('y', 16)
          .attr('text-anchor', 'middle').text(title);
      }

      function drawSelectedSpatial() {
        const item = DATA.spatial[Number(select.value || 0)];
        drawSpatialMap('#am-same-map', item.same, `T${item.target} → S${item.target}（同帧）`, DATA.roi);
        drawSpatialMap('#am-top-map', item.top, `T${item.target} → S${item.topSource}（最高帧）`, DATA.roi);
        root.querySelector('#am-same-label').textContent = `对应 source 帧 S=${item.target}`;
        root.querySelector('#am-top-label').textContent = `实际最高 source 帧 S=${item.topSource}`;
      }

      select.addEventListener('change', drawSelectedSpatial);
      let resizeTimer;
      const redraw = () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => { drawTemporal(); drawSelectedSpatial(); }, 80);
      };
      new ResizeObserver(redraw).observe(root);
      drawTemporal();
      drawSelectedSpatial();
    })();
  </script>
</div>
'''

html = template.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
OUT_PATH.write_text(html, encoding="utf-8")
print(OUT_PATH)
print(OUT_PATH.stat().st_size)
