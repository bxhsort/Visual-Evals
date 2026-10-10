"""Single source of truth for benchmark output-root derivation.

Each ``gen/**/launch.sh`` computes the directory its generated images and score
files land in from the sampling knobs (a ``build_output_root`` / ``build_output_folder``
heredoc). That logic is duplicated in the launcher and in ``inference.py``; a result
collector needs to recompute the *same* path to find the scores. This module is the one
place that derivation lives, kept byte-for-byte compatible with the launcher heredocs.

t2i (geneval, dpgbench, longtextbench):
    eval_output[cfg<g>]_<size>[_ts<g>][_noshift][_<cfg_norm>]
it2i (imgedit, geditbench):
    eval_output_edit[_<size>]_txtcfg<txt>_imgcfg<img>[_ts<g>][_noshift][_<cfg_renorm>]

The *results root* returned by :func:`derive_output_root` is the directory the scorer
writes into — for t2i geneval/dpgbench ``<base>/<root>/<model>/<step>``, longtextbench
appends ``/<subdir>``, and it2i imgedit/geditbench use ``<base>/<folder>/<model>/<step>``
(the parent of the per-subdir image dir, matching ``RESULT_ROOT_DIR`` in those launchers).

Usable as a library (``derive_output_root(...)``) or a CLI:

    python output_paths.py --bench geneval --output-base /abs/outputs \
        --model-name m --step-name s --image-size 512 --cfg-scale 4.0 \
        --timestep-shift 3.0 --cfg-norm global
"""

from __future__ import annotations

import argparse
import math
import os

# bench -> task family + default OUTPUT_SUBDIR + whether the subdir is part of the
# results root. Mirrors the launchers' OUTPUT_SUBDIR defaults.
BENCHMARKS = {
    "geneval": {"family": "t2i", "subdir": None, "subdir_in_root": False},
    "geneval2": {"family": "t2i", "subdir": "geneval2", "subdir_in_root": True},
    "dpgbench": {"family": "t2i", "subdir": None, "subdir_in_root": False},
    "longtextbench": {"family": "t2i", "subdir": "longtextbench_{mode}", "subdir_in_root": True},
    "imgedit": {"family": "it2i", "subdir": "imgedit", "subdir_in_root": False},
    "geditbench": {"family": "it2i", "subdir": "gedit_bench", "subdir_in_root": False},
    "cvtg_2k": {"family": "t2i", "subdir": None, "subdir_in_root": False},
    "tiifbench": {"family": "t2i", "subdir": None, "subdir_in_root": False},
    "oneigbench": {"family": "t2i", "subdir": None, "subdir_in_root": False},
    "u1_5bench": {"family": "t2i", "subdir": "u1_5bench_{mode}", "subdir_in_root": True},
    "qwenimagebench": {"family": "t2i", "subdir": "qwenimagebench_{mode}", "subdir_in_root": True},
    "wise": {"family": "t2i", "subdir": None, "subdir_in_root": False},
    "bizgeneval": {"family": "t2i", "subdir": None, "subdir_in_root": False},
    "igbench": {"family": "t2i", "subdir": "igbench", "subdir_in_root": True},
    "rise": {"family": "it2i", "subdir": "rise_{mode}", "subdir_in_root": False},
}


def _parse_square_image_size(raw_value) -> int:
    """Replicates parse_square_image_size() in the it2i launchers."""
    raw_value = str(raw_value).strip().lower()
    if "x" in raw_value:
        width_str, height_str = raw_value.split("x", 1)
    elif "*" in raw_value:
        width_str, height_str = raw_value.split("*", 1)
    else:
        width_str = raw_value
        height_str = raw_value

    if not width_str.isdigit() or not height_str.isdigit():
        raise ValueError(f"image_size must be a positive integer or square spec, got {raw_value!r}")

    width = int(width_str)
    height = int(height_str)
    if width <= 0 or height <= 0:
        raise ValueError(f"image_size must be positive, got {raw_value!r}")
    if width != height:
        raise ValueError(f"image_size must be square, got {raw_value!r}")
    return width


def derive_t2i_root(image_size, cfg_scale, timestep_shift, cfg_norm) -> str:
    """Single source of truth for the t2i output-root name. The launchers and inference.py
    both call this (no duplicated heredocs). Unified scheme — every setting always in the name:

        eval_output_t2i_cfg{cfg:g}_{size}[_ts{ts:g}][_noshift][_{cfg_norm}]
        default -> eval_output_t2i_cfg4_1024_ts3_global
    """
    image_size = int(float(image_size))
    cfg_scale = float(cfg_scale)
    timestep_shift = float(timestep_shift)
    cfg_norm = str(cfg_norm)

    output_root = f"eval_output_t2i_cfg{cfg_scale:g}_{image_size}"
    if not math.isclose(timestep_shift, 1.0):
        output_root = f"{output_root}_ts{timestep_shift:g}"
    if timestep_shift < 1.0:
        output_root = f"{output_root}_noshift"
    if cfg_norm != "none":
        output_root = f"{output_root}_{cfg_norm}"

    return output_root


def derive_it2i_folder(txt_cfg, img_cfg, cfg_renorm, timestep_shift, image_size) -> str:
    """Replicates build_output_folder() in the it2i launchers (imgedit/geditbench).

    ``txt_cfg`` / ``img_cfg`` are kept verbatim as strings (the launcher passes them
    through unformatted), so "4.0" stays "4.0" — do NOT float-format them.
    """
    txt_cfg = str(txt_cfg)
    img_cfg = str(img_cfg)
    cfg_renorm = str(cfg_renorm)
    timestep_shift = float(timestep_shift)
    image_size = _parse_square_image_size(image_size)

    # Unified scheme — prefix eval_output_it2i_, size ALWAYS included:
    #   eval_output_it2i_{size}_txtcfg{txt}_imgcfg{img}[_ts{ts:g}][_noshift][_{cfg_renorm}]
    #   default -> eval_output_it2i_1024_txtcfg4_imgcfg1_ts3_global
    output_folder = f"eval_output_it2i_{image_size}_txtcfg{txt_cfg}_imgcfg{img_cfg}"
    if not math.isclose(timestep_shift, 1.0):
        output_folder = f"{output_folder}_ts{timestep_shift:g}"
    if timestep_shift < 1.0:
        output_folder = f"{output_folder}_noshift"
    if cfg_renorm != "none":
        output_folder = f"{output_folder}_{cfg_renorm}"

    return output_folder


def derive_output_root(
    bench,
    output_base,
    model_name="model",
    step_name="step",
    image_size=512,
    # t2i knobs
    cfg_scale=4.0,
    cfg_norm="global",
    # it2i knobs
    txt_cfg="4.0",
    img_cfg="1.0",
    cfg_renorm="global",
    # shared
    timestep_shift=3.0,
    # longtextbench
    mode="en",
    output_subdir=None,
):
    """Return the directory the scorer writes into for ``bench``.

    The defaults match the tuned defaults baked into the launchers / docs/BENCHMARK_ENV.md.
    """
    if bench not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {bench!r}; known: {sorted(BENCHMARKS)}")
    spec = BENCHMARKS[bench]

    if spec["family"] == "t2i":
        root_name = derive_t2i_root(image_size, cfg_scale, timestep_shift, cfg_norm)
    else:
        root_name = derive_it2i_folder(txt_cfg, img_cfg, cfg_renorm, timestep_shift, image_size)

    path = os.path.join(output_base, root_name, model_name, step_name)

    if spec["subdir_in_root"]:
        subdir = output_subdir
        if subdir is None and spec["subdir"] is not None:
            subdir = spec["subdir"].format(mode=mode)
        if subdir:
            path = os.path.join(path, subdir)

    return path


def _main() -> None:
    ap = argparse.ArgumentParser(description="Derive a benchmark's output/results root.")
    ap.add_argument("--bench", required=True, choices=sorted(BENCHMARKS))
    ap.add_argument("--output-base", required=True)
    ap.add_argument("--model-name", default="model")
    ap.add_argument("--step-name", default="step")
    ap.add_argument("--image-size", default="512")
    ap.add_argument("--cfg-scale", default="4.0")
    ap.add_argument("--cfg-norm", default="global")
    ap.add_argument("--txt-cfg", default="4.0")
    ap.add_argument("--img-cfg", default="1.0")
    ap.add_argument("--cfg-renorm", default="global")
    ap.add_argument("--timestep-shift", default="3.0")
    ap.add_argument("--mode", default="en")
    ap.add_argument("--output-subdir", default=None)
    args = ap.parse_args()

    print(
        derive_output_root(
            bench=args.bench,
            output_base=args.output_base,
            model_name=args.model_name,
            step_name=args.step_name,
            image_size=args.image_size,
            cfg_scale=args.cfg_scale,
            cfg_norm=args.cfg_norm,
            txt_cfg=args.txt_cfg,
            img_cfg=args.img_cfg,
            cfg_renorm=args.cfg_renorm,
            timestep_shift=args.timestep_shift,
            mode=args.mode,
            output_subdir=args.output_subdir,
        )
    )


if __name__ == "__main__":
    _main()
