"""Extract order/shapes/dtypes from an already constructed REAL meta DiT.

This module intentionally does not initialize distributed state or load H3.
The integration harness must construct ONLY MiniMaxH3DiTModel under a meta
device context with TP=1/SP=2 configuration, not the complete pipeline.
"""


def extract_meta_model_metadata(model, *, dit_tp=1, require_openvdn=True):
    if dit_tp != 1:
        raise ValueError("Current H3 Stage-B adapter requires TP=1")
    if type(model).__name__ != "MiniMaxH3DiTModel":
        raise ValueError("Require actual MiniMaxH3DiTModel")
    if require_openvdn and not getattr(model.arch, "openvdn_enabled", False):
        raise ValueError("Require actual OpenVDN-enabled MiniMaxH3DiTModel")
    if not require_openvdn and getattr(model.arch, "openvdn_enabled", False):
        raise ValueError("Base-only prepared loading requires OpenVDN-disabled MiniMaxH3DiTModel")
    if len(model.blocks) != 50 or len(model.token_refiner.blocks) != 2:
        raise ValueError("Require full 50+2 block model")
    names = {}
    for name, _ in model.named_parameters():
        names[name] = "parameter"
    entries, nonpersistent = [], []

    def describe(name, tensor, kind):
        if tensor.device.type != "meta":
            raise ValueError(f"Unexpected real tensor allocation: {name}/{tensor.device}")
        dtype = {"torch.bfloat16": "BF16", "torch.float32": "F32"}.get(str(tensor.dtype))
        if dtype is None:
            raise ValueError(f"Unsupported model dtype {name}/{tensor.dtype}")
        return dict(name=name, shape=list(tensor.shape), dtype=dtype, kind=kind)

    # Global params-then-buffers preserves the relative order DLO sees
    # within each main block. Non-persistent buffers are recorded separately
    # because they are not checkpoint entries but DLO also offloads them.
    for name, tensor in model.named_parameters():
        entries.append(describe(name, tensor, "parameter"))
    for name, tensor in model.named_buffers():
        owner = model
        path = name.split(".")
        for component in path[:-1]:
            owner = getattr(owner, component)
        item = describe(name, tensor, "buffer")
        if path[-1] in owner._non_persistent_buffers_set:
            nonpersistent.append(item)
        else:
            entries.append(item)
    return dict(device="meta", dit_tp=dit_tp, parameters_and_persistent_buffers=entries,
                nonpersistent_buffers_requiring_reconstruction=nonpersistent,
                order_source="real_named_parameters_then_named_buffers_not_checkpoint_order")
