"""NLF (Sarandi et al., NeurIPS 2024) in its own process.

The released TorchScript needs torch >= 2.4; the project's venv is pinned at 2.1, so this
file runs under `models/pretrained/nlf/venv` and imports nothing from the project. It
talks to `tryon.body.regressor.NLF` over stdin / stdout: each message is an 8-byte
little-endian length and a pickle.

  request   {"frame": HxWx3 uint8 BGR, "box": [x, y, w, h] or None,
             "K": 3x3 float intrinsics}
  reply     {"rotvec": (55, 3), "betas": (10,), "joints3d": (J, 3) in the
             camera frame (metres)} for the most confident person, or None

The model is stored in fp16 and slow on CPU, so the casts are rewritten to fp32 once and
cached next to the weights. With a box handed in its YOLO detector is skipped.

The rewrite edits the archive's TorchScript source (`_patch_source`): the fp16 casts are
literal dtype 5 arguments. The older graph rewrite (`_rewrite_dtype_casts`) inlines every
call and grows the saved code ~30x (slow to load); it stays as the fallback.

On Apple silicon the image backbone runs on the GPU (`_backbone_on_mps`): one more
source edit moves the crop batch to MPS and the features back; heads and body-model fit
stay on the CPU, where many small ops are fast. Answers move by <0.5 mm in the joints.
"""

from __future__ import annotations

import pickle
import struct
import sys
from pathlib import Path


def _all_nodes(block):
    for node in block.nodes():
        yield node
        for inner in node.blocks():
            yield from _all_nodes(inner)


def _argument_names(node):
    try:
        schema = node.schema()
    except RuntimeError:
        return []
    if schema == "(no schema)" or "(" not in schema:
        return []
    args = schema[schema.index("(") + 1 : schema.rindex(") ->")]
    return [a.strip().split("=")[0].split(" ")[-1] for a in args.split(", ") if a.strip() not in ("*", "")]


def _rewrite_dtype_casts(module, mapping, device="cpu"):
    """{5: 6}: every hard-coded fp16 cast becomes fp32 (and fp16 tensor constants become
    fp32); traced "cuda" device constants become `device`."""
    import torch

    seen, alive = set(), []

    def visit(mod):
        for name in mod._c._method_names():
            graph = mod._c._get_method(name).graph
            if id(graph) in seen:
                continue
            seen.add(id(graph))
            alive.append(graph)
            torch._C._jit_pass_inline(graph)
            for node in graph.findAllNodes("prim::Constant", True):
                if node.output().type().kind() == "DeviceObjType" and node.s("value").startswith("cuda"):
                    node.s_("value", device)
                if 5 in mapping and node.output().type().kind() == "TensorType" and node.t("value").dtype == torch.float16:
                    node.t_("value", node.t("value").float())
            for node in list(_all_nodes(graph.block())):
                names = _argument_names(node)
                for i, value in enumerate(node.inputs()):
                    producer = value.node()
                    if i < len(names) and names[i] == "dtype" and producer.kind() == "prim::Constant" and value.type().kind() == "IntType" and producer.i("value") in mapping:
                        constant = graph.create("prim::Constant")
                        constant.i_("value", mapping[producer.i("value")])
                        constant.output().setType(torch._C.IntType.get())
                        constant.insertBefore(node)
                        node.replaceInput(i, constant.output())
        for child in mod.children():
            visit(child)

    visit(module)


# (pattern, replacement) on the archive's code: every fp16 cast in NLF-S
# 0.2.2, and the "cuda" device of its (disabled) autocast blocks
_SOURCE_EDITS = (
    (r"torch\.to\((\w+), 5\)", r"torch.to(\1, 6)"),
    (r'\.__init__\("cuda", ', '.__init__("cpu",  '),
    (r"\.transpose_weights\(([^()]*), 5, \)", r".transpose_weights(\1, 6, )"),
)
_EXPECTED_EDITS = 8

# Test-time augmentation: crops of the person box, each slightly rotated
# and scaled, averaged by NLF itself (`num_aug`); 1 = one crop.
NUM_AUG = 1

# the backbone's calls in nlf_model.py: crops to MPS, features back
_BACKBONE_EDIT = (
    r"= \(backbone\)\.forward\(image, \)",
    '= torch.to((backbone).forward(torch.to(image, torch.device("mps"), 6, False, False, None), ), '
    'torch.device("cpu"), 6, False, False, None)',
)
_EXPECTED_BACKBONE_EDITS = 2


def _patch_source(weights, out):
    """The released archive with its fp16 casts edited to fp32 in the TorchScript source,
    loaded, weights made fp32 and saved to `out`. Returns False if the archive is not the
    one the edits were written for."""
    import io
    import re
    import zipfile

    import torch

    patched, edits = io.BytesIO(), 0
    with zipfile.ZipFile(weights) as source, zipfile.ZipFile(patched, "w", zipfile.ZIP_STORED) as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename.endswith(".py") and "/code/" in item.filename:
                text = data.decode()
                for pattern, replacement in _SOURCE_EDITS:
                    text, count = re.subn(pattern, replacement, text)
                    edits += count
                data = text.encode()
            target.writestr(item, data)
    if edits != _EXPECTED_EDITS:
        return False
    patched.seek(0)
    model = torch.jit.load(patched, map_location="cpu").eval()
    model.float()
    torch.jit.save(model, str(out))
    return True


def _backbone_on_mps(patched, out):
    """`patched` (the fp32 archive) with the backbone edit, saved to `out`.
    False if the archive is not the one the edit was written for."""
    import io
    import re
    import zipfile

    import torch

    edited, edits = io.BytesIO(), 0
    with zipfile.ZipFile(patched) as source, zipfile.ZipFile(edited, "w", zipfile.ZIP_STORED) as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename.endswith("/nlf_model.py"):
                text, edits = re.subn(*_BACKBONE_EDIT, data.decode())
                data = text.encode()
            target.writestr(item, data)
    if edits != _EXPECTED_BACKBONE_EDITS:
        return False
    edited.seek(0)
    torch.jit.save(torch.jit.load(edited, map_location="cpu"), str(out))
    return True


def _warm_up(model):
    """A call on a blank frame, so the first person does not pay for first-call allocations."""
    import numpy as np

    frame = np.zeros((480, 360, 3), np.uint8)
    request = {"frame": frame, "box": np.array([60, 40, 240, 400], np.float32),
               "K": np.array([[400, 0, 180], [0, 400, 240], [0, 0, 1]], np.float32)}
    answer(model, request)


def load(weights, device="cpu"):
    import torch
    import torchvision  # noqa: F401  registers the ops the TorchScript uses

    weights = Path(weights)
    if device == "cuda":
        # as released: fp16 on an NVIDIA GPU is what it was built for
        return torch.jit.load(str(weights), map_location="cuda").eval()
    patched = weights.with_name(weights.stem + "_fp32_cpu_src.torchscript")
    if not patched.exists():
        try:
            ok = _patch_source(weights, patched)
            if ok:
                _warm_up(torch.jit.load(str(patched), map_location="cpu").eval())  # it runs, in fp32
        except Exception as error:  # not the archive the edits were written for
            print(f"nlf_worker: source patch failed ({error}); graph rewrite instead", file=sys.stderr)
            ok = False
        if not ok and patched.exists():
            patched.unlink()
    if patched.exists():
        if device == "mps":
            on_mps = weights.with_name(weights.stem + "_fp32_mps_backbone.torchscript")
            try:
                if on_mps.exists() or _backbone_on_mps(patched, on_mps):
                    model = torch.jit.load(str(on_mps), map_location="cpu").eval()
                    model.crop_model.backbone.to("mps")
                    _warm_up(model)  # fails here, not on the first person, if MPS cannot run it
                    return model
            except Exception as error:
                print(f"nlf_worker: backbone on MPS failed ({error}); CPU instead", file=sys.stderr)
        return torch.jit.load(str(patched), map_location="cpu").eval()
    cached = weights.with_name(weights.stem + "_fp32_cpu.torchscript")
    if not cached.exists():
        model = torch.jit.load(str(weights), map_location="cpu").eval()
        _rewrite_dtype_casts(model, {5: 6}, "cpu")
        model.float()
        torch.jit.save(model, str(cached))  # the edits only take effect once re-serialised
    return torch.jit.load(str(cached), map_location="cpu").eval()


def _read(stream):
    header = stream.read(8)
    if len(header) < 8:
        return None
    return pickle.loads(stream.read(struct.unpack("<Q", header)[0]))


def _write(stream, value):
    data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack("<Q", len(data)) + data)
    stream.flush()


def answer(model, request):
    import cv2
    import numpy as np
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # inputs; the MPS edit moves its own
    frame = cv2.cvtColor(request["frame"], cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(frame.transpose(2, 0, 1).copy())[None].to(device)
    k = torch.as_tensor(np.asarray(request["K"], np.float32))[None].to(device)
    # Without TorchScript's graph optimisation: its first calls on a new process took
    # ~10 s (profiling and fusing); unoptimised answers are bit-identical.
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        if request.get("box") is not None:
            box = torch.as_tensor(np.asarray(request["box"], np.float32)).reshape(1, 4).to(device)
            pred = model.estimate_smpl_batched(x, [box], intrinsic_matrix=k, model_name="smplx", num_aug=NUM_AUG)
            index = 0
        else:
            pred = model.detect_smpl_batched(x, intrinsic_matrix=k, model_name="smplx")
            if len(pred["pose"][0]) == 0:
                return None
            boxes = pred["boxes"][0].float().cpu().numpy()
            index = int(np.argmax(boxes[:, 4])) if boxes.shape[1] > 4 else 0
    return {
        "rotvec": pred["pose"][0][index].float().cpu().numpy().reshape(-1, 3),
        "betas": pred["betas"][0][index].float().cpu().numpy().reshape(-1)[:10],
        "joints3d": pred["joints3d"][0][index].float().cpu().numpy(),
    }


def main():
    import torch

    torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    device = sys.argv[3] if len(sys.argv) > 3 else "cpu"
    global NUM_AUG
    NUM_AUG = int(sys.argv[4]) if len(sys.argv) > 4 else NUM_AUG
    model = load(sys.argv[1], "cuda" if torch.cuda.is_available() else device)
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr  # nothing but replies on the pipe
    _warm_up(model)
    _write(stdout, "ready")
    while True:
        request = _read(stdin)
        if request is None:
            return
        try:
            _write(stdout, answer(model, request))
        except Exception as error:  # the caller treats it as "no person"
            print(f"nlf_worker: {error}", file=sys.stderr)
            _write(stdout, None)


if __name__ == "__main__":
    main()
