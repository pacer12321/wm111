"""Generate, do not apply, an identical opt-in B/D pipeline bootstrap patch."""
import argparse
import difflib
from pathlib import Path

BOOTSTRAP = '''

# Isolated prepared-storage experiment bootstrap; no attention/model edits.
import os as _h3_prepared_os
if _h3_prepared_os.environ.get("ZHONGHAO_H3_PREPARED_OFFLOAD", "0") != "0":
    import sys as _h3_prepared_sys
    from h3_prepared_integration import install_for_pipeline_module as _h3_prepared_install
    _h3_prepared_install(_h3_prepared_sys.modules[__name__])
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = args.source.read_text(encoding="utf-8")
    if "_h3_prepared_install" in original:
        raise ValueError("Source already contains prepared bootstrap")
    if "class MiniMaxH3Pipeline" not in original:
        raise ValueError("Not an H3 pipeline source")
    if args.output.exists():
        raise FileExistsError(args.output)
    target = "vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py"
    patch = "".join(difflib.unified_diff(original.splitlines(keepends=True),
                                       (original.rstrip() + "\n" + BOOTSTRAP).splitlines(keepends=True),
                                       fromfile="a/" + target, tofile="b/" + target))
    args.output.write_text(patch, encoding="utf-8", newline="\n")
    print("Generated isolated-candidate patch only; source not modified: " + str(args.output))


if __name__ == "__main__":
    main()
