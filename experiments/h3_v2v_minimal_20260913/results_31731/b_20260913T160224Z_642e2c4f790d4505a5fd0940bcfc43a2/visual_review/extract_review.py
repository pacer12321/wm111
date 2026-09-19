"""Read-only source/A/B inputs; local CPU four-column temporal-order review.

The reverse column uses source stills indexed 123-i only. It is not generated
ground truth or a new video. Reuse SHA-pinned A extraction helper functions,
never its main function, so the previous review and production stay untouched.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
RUN_NAME = "b_20260913T160224Z_642e2c4f790d4505a5fd0940bcfc43a2"
RESULTS = Path("C:/Users/DZH/OneDrive/Documents/ChatGPT/wm111/experiments/h3_v2v_minimal_20260913/results_31731")
RUN_B = RESULTS / RUN_NAME
RUN_A = RESULTS / "a_20260913T153514Z_91c6c0f98ba84c7c828202d5934176cd"
HELPER = RUN_A / "visual_review/extract_review.py"
HELPER_SHA = "b641e849c6aa903c7fb5494d32884fca7b2e2ecea4279ac5fc8673192a4e6fe5"
INDICES = (0, 15, 30, 45, 60, 75, 90, 105, 123)
SOURCE_SHA = "4fb53ba396d3695eb3bb4eadb0b9fe64dc88592e1c8baf3f6cab59a43f66b0a2"
A_SHA = "17b83ee2196302e9523e3e54aa239e3d6e7c8d5070c83d7c066e4f8cf06b38d7"
B_SHA = "c15f08b33a5419ee704350133afd7441386738e11836aa7c605e71ae481c75cc"
RESULT_SHA = "be970bc20378ca2b4e54c776ccfe16922108024b5ebeeb5e296e15c353411b3c"
REQUEST_SHA = "2ff23c11e1c0510ae0eb126e8e7e1bd0e45e92eb5d143b3a1b9b48aa38ab43d9"
PROMPT = "Reverse the temporal order of the entire reference clip: play all of its actions backward in time. Keep the same two people, their appearance, lighting, camera framing, and background."
SOURCES = {
    "source": (RUN_A / "source.mp4", SOURCE_SHA, 1527004, (1280, 720), False),
    "A8": (RUN_A / "a_50step.mp4", A_SHA, 4271598, (1344, 768), True),
    "B8": (RUN_B / "b_50step.mp4", B_SHA, 4657820, (1344, 768), True),
}
KINDS = ("source", "source_reverse_reference", "A8", "B8")


def digest(path):
    if not path.is_file() or path.is_symlink() or path.resolve() != path.absolute():
        raise RuntimeError(f"Expected regular non-symlink file: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_helpers():
    if digest(HELPER) != HELPER_SHA:
        raise RuntimeError("A extraction helper changed")
    spec = importlib.util.spec_from_file_location("readonly_reverse_A_extraction_helpers", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.INDICES != INDICES or module.SOURCE_SHA != SOURCE_SHA or module.A_SHA != A_SHA:
        raise RuntimeError("Unexpected A helper constants")
    return module


def make_contact(indices, images):
    width, height, header, label = 448, 256, 61, 41
    sheet = Image.new("RGB", (4 * width, header + len(indices) * (height + label)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
    titles = ("SOURCE forward", "SOURCE reversed-index reference", "A8 dense | 50 requested steps", "B8 global source | 50 requested steps")
    for col, title in enumerate(titles):
        draw.text((col * width + 7, 6), title, fill="black", font=font)
    draw.text((width + 7, 29), "Diagnostic stills only; NOT generated GT", fill="darkred", font=font)
    for row, index in enumerate(indices):
        y = header + row * (height + label)
        refs = (("source", index), ("source", 123-index), ("A8", index), ("B8", index))
        for col, (kind, input_index) in enumerate(refs):
            draw.text((col * width + 7, y + 3), f"row i={index} | source index={input_index}" if col == 1
                      else f"frame {input_index} | PTS {input_index/24:.3f}s", fill="black", font=font)
            if col == 1:
                draw.text((col * width + 7, y + 22), f"source PTS {input_index/24:.3f}s", fill="black", font=font)
            fitted = ImageOps.contain(images[kind][input_index], (width, height), Image.Resampling.LANCZOS)
            draw.rectangle((col * width, y + label, (col + 1) * width - 1, y + label + height - 1), fill="black")
            sheet.paste(fitted, (col * width + (width-fitted.width)//2, y + label + (height-fitted.height)//2))
    return sheet


def main():
    if HERE != RUN_B / "visual_review" or HERE.is_symlink() or RUN_B.is_symlink():
        raise RuntimeError("Unexpected private B review directory")
    helper = load_helpers()
    contacts = ["source_forward_reversed_A8_B8_contact.jpg"] + [f"source_forward_reversed_A8_B8_contact_part_{n}.jpg" for n in (1, 2, 3)]
    outputs = [HERE / helper.png_name(kind, i) for kind in KINDS for i in INDICES]
    outputs += [HERE / name for name in contacts] + [HERE / "extraction_record.json"]
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise RuntimeError("Refusing to overwrite review artifacts")
    result_path, request_path = RUN_B / "result.json", RUN_B / "request.json"
    if digest(result_path) != RESULT_SHA or digest(request_path) != REQUEST_SHA:
        raise RuntimeError("Formal B request/result identity mismatch")
    result, request = json.loads(result_path.read_bytes()), json.loads(request_path.read_bytes())
    if (result.get("success") is not True or result.get("requested_steps") != 50
            or result.get("http_code") != "200" or result.get("curl_returncode") != 0
            or result.get("content_type") != "video/mp4" or result.get("output_sha256") != B_SHA
            or result.get("output_video") != f"/cache/zhonghao/h3/b_model_trial/01234567/explicit_reverse_couple_124/B/runs/{RUN_NAME}/output/b_50step.mp4"
            or result.get("ffprobe", {}).get("verified") is not True):
        raise RuntimeError("Unexpected formal B result identity/specification")
    if (request.get("run_id") != "642e2c4f790d4505a5fd0940bcfc43a2"
            or request.get("sample_id") != "explicit_reverse_couple_124"
            or request.get("source_sha256") != SOURCE_SHA or request.get("requested_steps") != 50
            or request.get("fields", {}).get("prompt") != PROMPT):
        raise RuntimeError("Unexpected formal B request identity/prompt")
    for kind, (path, sha, size, _, _) in SOURCES.items():
        if digest(path) != sha or path.stat().st_size != size:
            raise RuntimeError(f"Input identity/size mismatch: {kind}")
    records, images = {}, {}
    for kind, spec in SOURCES.items():
        records[kind], images[kind] = helper.decode(kind, spec)
    for kind, (path, sha, size, _, _) in SOURCES.items():
        if digest(path) != sha or path.stat().st_size != size:
            raise RuntimeError(f"Input changed during decode: {kind}")
    if digest(result_path) != RESULT_SHA or digest(request_path) != REQUEST_SHA or digest(HELPER) != HELPER_SHA:
        raise RuntimeError("Request/result/helper changed during decode")
    for kind in KINDS:
        for i in INDICES:
            input_kind, input_index = ("source", 123-i) if kind == "source_reverse_reference" else (kind, i)
            helper.save_image(images[input_kind][input_index], HERE / helper.png_name(kind, i), "PNG")
    helper.save_image(make_contact(INDICES, images), HERE / contacts[0], "JPEG", quality=94)
    for part in range(3):
        helper.save_image(make_contact(INDICES[part*3:part*3+3], images), HERE / contacts[part+1], "JPEG", quality=94)
    record = dict(run=RUN_NAME, videos=records, output_row_indices=list(INDICES),
                  reversed_source_indices=[123-i for i in INDICES], prompt=PROMPT,
                  result_sha256=RESULT_SHA, request_sha256=REQUEST_SHA,
                  readonly_helper_path=str(HELPER), readonly_helper_sha256=HELPER_SHA,
                  readonly_helper_functions_used=["decode", "png_name", "save_image"],
                  helper_main_invoked=False,
                  contacts=[str(HERE / name) for name in contacts], native_png_count=36,
                  source_reverse_reference_is_generated_ground_truth=False,
                  reverse_reference_is_new_video=False,
                  native_aspect_ratios_preserved=True, contact_resizing="Aspect-preserving contain with black letterbox",
                  all_input_hashes_checked_before_and_after_decode=True,
                  new_model_calls=False, remote_operations=False, input_files_modified=False,
                  previous_A_review_modified=False, continuous_playback=False, audio_reviewed=False,
                  limits=["Reverse source stills are a diagnostic index-order reference, not generated GT.",
                          "No claim of pixelwise or camera alignment between different-sized input/output videos.",
                          "Full decoding and matching PTS do not establish visual quality or editing success.",
                          "Only selected frames will be visually inspected; no continuous playback or audio review here."])
    with (HERE / "extraction_record.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps({"run": RUN_NAME, "all_sha_verified": True,
                      "decoded_frames": {kind: value["decoded_frames"] for kind, value in records.items()},
                      "contacts": record["contacts"]}))


if __name__ == "__main__":
    main()
