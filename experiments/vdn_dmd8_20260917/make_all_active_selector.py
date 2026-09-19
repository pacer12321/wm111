from pathlib import Path

import torch


source = Path("/cache/zhonghao/h3/dmd8_b_skip_20260917/selector_dmd8_latent/fixed_selector_payload.pt")
destination = Path("/cache/zhonghao/h3/dmd8_b_skip_20260917/selector_dmd8_all_active/fixed_selector_payload.pt")
destination.parent.mkdir(parents=True, exist_ok=False)
payload = torch.load(source, map_location="cpu", weights_only=True)
payload["active_target_mask"] = torch.ones_like(payload["active_target_mask"], dtype=torch.bool)
payload["metric"] = "all_active_interleaved_full_compute_control"
payload["active_ratio"] = 1.0
payload["selector_budget_control"] = "no token skip; all forwards full refresh"
torch.save(payload, destination)
print(destination)
