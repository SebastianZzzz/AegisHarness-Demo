#!/usr/bin/env python3
import os
import io
import sys
import time
import math
import json
import random
import string
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

try:
    import click
except Exception:
    print("This script requires the 'click' package. Install via: pip install click")
    sys.exit(1)

try:
    from PIL import Image
except Exception:
    print("This script requires the 'Pillow' package. Install via: pip install Pillow")
    sys.exit(1)


# ----------------------------
# Utility and data structures
# ----------------------------

SUBSAMPLING_MAP = {
    "4:4:4": 0,
    "4:2:2": 1,
    "4:2:0": 2,
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def human_size(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def sanitize_filename(name: str) -> str:
    # Basic sanitization to avoid problematic filenames
    valid = "-_.()abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    out = []
    for ch in name:
        if ch in valid:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def temp_atomic_path(target_path: str) -> str:
    # Create a temp path in the same directory for atomic rename
    dirn = os.path.dirname(os.path.abspath(target_path))
    base = os.path.basename(target_path)
    rand_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return os.path.join(dirn, f".{base}.{rand_suffix}.tmp")


def encode_size_in_bytes(img: Image.Image,
                         quality: int,
                         subsampling: int,
                         progressive: bool,
                         exif: Optional[bytes],
                         icc_profile: Optional[bytes],
                         keep_exif: bool,
                         keep_icc: bool) -> int:
    # Encode to in-memory buffer to estimate size (no disk I/O)
    buf = io.BytesIO()
    save_kwargs: Dict[str, Any] = {
        "format": "JPEG",
        "quality": int(quality),
        "subsampling": int(subsampling),
        "progressive": bool(progressive),
        "optimize": True,
        "quality_layers": None,
    }
    if keep_exif:
        save_kwargs["exif"] = exif
    if keep_icc:
        save_kwargs["icc_profile"] = icc_profile

    # Ensure mode is RGB for JPEG
    if img.mode != "RGB":
        img_to_save = img.convert("RGB")
    else:
        img_to_save = img

    img_to_save.save(buf, **save_kwargs)
    return buf.tell()


def save_jpeg_atomic(img: Image.Image,
                     final_path: str,
                     quality: int,
                     subsampling: int,
                     progressive: bool,
                     exif: Optional[bytes],
                     icc_profile: Optional[bytes],
                     keep_exif: bool,
                     keep_icc: bool) -> None:
    # Write to temp file then atomically replace
    tmp_path = temp_atomic_path(final_path)
    try:
        if img.mode != "RGB":
            img_to_save = img.convert("RGB")
        else:
            img_to_save = img

        save_kwargs: Dict[str, Any] = {
            "format": "JPEG",
            "quality": int(quality),
            "subsampling": int(subsampling),
            "progressive": bool(progressive),
            "optimize": True,
        }
        if keep_exif:
            save_kwargs["exif"] = exif
        if keep_icc:
            save_kwargs["icc_profile"] = icc_profile

        img_to_save.save(tmp_path, **save_kwargs)

        # Atomic rename
        os.replace(tmp_path, final_path)
    except Exception:
        # Cleanup temp on failure
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise


# ----------------------------
# Core processing logic
# ----------------------------

@dataclass
class CompressionConfig:
    input_path: str
    output_path: str
    target_size: Optional[int] = None
    quality_min: int = 10
    quality_max: int = 95
    subsampling: int = 2  # 0:4:4:4, 1:4:2:2, 2:4:2:0
    progressive: bool = False
    keep_exif: bool = True
    keep_icc: bool = True
    downscale: bool = False
    downscale_factor: float = 0.9
    threads: int = 4
    overwrite: bool = False
    preserve_structure: bool = True
    verbose: bool = False


def _quality_encoded_size(img: Image.Image,
                          quality: int,
                          cfg: CompressionConfig,
                          exif: Optional[bytes],
                          icc: Optional[bytes]) -> int:
    return encode_size_in_bytes(
        img,
        quality=quality,
        subsampling=cfg.subsampling,
        progressive=cfg.progressive,
        exif=exif,
        icc_profile=icc,
        keep_exif=cfg.keep_exif,
        keep_icc=cfg.keep_icc,
    )


def _process_image_task(input_path: str, out_path: str, cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    cfg = CompressionConfig(**cfg_dict)
    result: Dict[str, Any] = {
        "input_path": input_path,
        "output_path": out_path,
        "success": False,
        "error": None,
        "original_size": None,
        "final_size": None,
        "quality_used": None,
        "downscaled": False,
        "duration_ms": 0,
    }

    try:
        if not os.path.exists(input_path) or not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Ensure output directory exists
        ensure_dir(os.path.dirname(out_path))

        original_size = os.path.getsize(input_path)
        result["original_size"] = original_size

        with Image.open(input_path) as img:
            # Best effort to read metadata
            exif = img.info.get("exif") if cfg.keep_exif else None
            icc = img.info.get("icc_profile") if cfg.keep_icc else None

            # Start with possibly downscaled image loop
            current_img = img
            downscaled = False
            final_quality = None
            final_size = None

            # Normalize output path naming if needed to avoid overwrites
            # Already resolved by outer function; this loop handles only writing
            # Execute a loop: try to reach target_size via binary search, else downscale and retry
            max_downscale_steps = 5 if cfg.downscale else 0
            scale_iter = 0

            while True:
                # If target_size is provided, try to reach it via quality search
                target = cfg.target_size
                if target is not None:
                    low = cfg.quality_min
                    high = cfg.quality_max
                    best_q = None
                    best_size = None

                    while low <= high:
                        mid = (low + high) // 2
                        size = _quality_encoded_size(
                            current_img, mid, cfg, exif, icc
                        )
                        if size <= target:
                            best_q = mid
                            best_size = size
                            low = mid + 1
                        else:
                            high = mid - 1

                    if best_q is not None:
                        final_quality = best_q
                        final_size = best_size
                        break  # found acceptable quality
                    else:
                        # Not possible at current resolution; try downscale if allowed
                        if cfg.downscale and scale_iter < max_downscale_steps:
                            w, h = current_img.size
                            new_w = max(1, int(w * cfg.downscale_factor))
                            new_h = max(1, int(h * cfg.downscale_factor))
                            if new_w < 1 or new_h < 1:
                                break  # cannot downscale further
                            current_img = current_img.resize((new_w, new_h), resample=Image.LANCZOS)
                            downscaled = True
                            scale_iter += 1
                            continue  # retry with smaller image
                        else:
                            # If cannot downscale, try with maximum quality as fallback
                            final_quality = cfg.quality_max
                            final_size = _quality_encoded_size(current_img, final_quality, cfg, exif, icc)
                            break
                else:
                    # No target specified: just encode with max quality (or use provided size)
                    final_quality = cfg.quality_max
                    final_size = _quality_encoded_size(current_img, final_quality, cfg, exif, icc)
                    break

            # After loop, we have final_quality and maybe downscaled flag
            result["quality_used"] = int(final_quality) if final_quality is not None else None
            result["downscaled"] = bool(downscaled)

            # Prepare final save
            if final_quality is None:
                final_quality = cfg.quality_max

            # Build final output path with existing constraints
            # If overwrite is False and output exists, choose a new path by suffixing
            final_out_path = out_path
            if os.path.exists(final_out_path) and not cfg.overwrite:
                base, ext = os.path.splitext(out_path)
                i = 1
                while True:
                    candidate = f"{base}__{i}{ext}"
                    if not os.path.exists(candidate):
                        final_out_path = candidate
                        break
                    i += 1
            # Compute final exif/icc to save if needed
            final_exif = exif if cfg.keep_exif else None
            final_icc = icc if cfg.keep_icc else None

            save_start = time.time()
            save_jpeg_atomic(current_img, final_out_path, int(final_quality),
                             int(cfg.subsampling), bool(cfg.progressive),
                             final_exif, final_icc,
                             cfg.keep_exif, cfg.keep_icc)
            save_end = time.time()

            final_size = os.path.getsize(final_out_path)
            result["final_size"] = int(final_size)
            result["output_path"] = final_out_path
            result["duration_ms"] = int((save_end - save_start) * 1000.0)
            result["success"] = True

    except Exception as e:
        result["error"] = str(e)
        result["success"] = False

    return result


# ----------------------------
# CLI and orchestration
# ----------------------------

def _collect_image_paths(input_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".JPG", ".JPEG"}
    paths = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if any(f.endswith(ext) for ext in exts):
                paths.append(os.path.join(root, f))
    return paths


@click.group()
def cli():
    """JPEG Batch Compressor (CLI)"""
    pass


@cli.command("compress-single")
@click.option("--input-path", type=click.Path(exists=True, dir_okay=False, file_okay=True), required=True,
              help="Path to the input JPEG image.")
@click.option("--output-dir", type=click.Path(file_okay=False, dir_okay=True), required=True,
              help="Directory to save the compressed JPEG.")
@click.option("--target-size", type=int, default=None,
              help="Target file size in bytes. If set, tool will try to adapt quality to reach this size.")
@click.option("--quality-min", type=int, default=10, help="Minimum quality for adaptive encoding.")
@click.option("--quality-max", type=int, default=95, help="Maximum quality for adaptive encoding.")
@click.option("--subsampling", type=click.Choice(list(SUBSAMPLING_MAP.keys()) + ["4:4:4", "4:2:2", "4:2:0"], default="4:2:0",
              help="JPEG subsampling.")
@click.option("--progressive/--no-progressive", default=False, help="Use progressive JPEG encoding.")
@click.option("--keep-exif/--drop-exif", default=True, help="Keep EXIF metadata.")
@click.option("--keep-icc/--drop-icc", default=True, help="Keep ICC color profile.")
@click.option("--downscale/--no-downscale", default=False, help="Enable progressive downscaling when target size cannot be reached by quality.")
@click.option("--downscale-factor", type=float, default=0.9, help="Factor by which to downscale when enabled (per step).")
@click.option("--threads", type=int, default=4, help="Number of worker threads for processing.")
@click.option("--overwrite/--no-overwrite", default=False, help="Overwrite existing files if present.")
@click.option("--verbose/--no-verbose", default=False, help="Verbose logging.")
def compress_single(input_path, output_dir, target_size, quality_min, quality_max,
                    subsampling, progressive, keep_exif, keep_icc, downscale, downscale_factor,
                    threads, overwrite, verbose):
    """Compress a single JPEG image with adaptive quality to target size."""
    subs = SUBSAMPLING_MAP.get(subsampling, 2)

    cfg = CompressionConfig(
        input_path=input_path,
        output_path=os.path.join(output_dir, os.path.basename(input_path)),
        target_size=target_size,
        quality_min=quality_min,
        quality_max=quality_max,
        subsampling=subs,
        progressive=progressive,
        keep_exif=keep_exif,
        keep_icc=keep_icc,
        downscale=downscale,
        downscale_factor=downscale_factor,
        threads=max(1, int(threads)),
        overwrite=overwrite,
        verbose=verbose,
    )

    # Prepare output path with directory creation
    ensure_dir(output_dir)
    input_dir = os.path.dirname(input_path)
    rel_out = os.path.basename(input_path)
    output_path = os.path.join(output_dir, rel_out)

    # Resolve in case of overwrite policy
    if os.path.exists(output_path) and not cfg.overwrite:
        base, ext = os.path.splitext(output_path)
        i = 1
        while True:
            cand = f"{base}__{i}{ext}"
            if not os.path.exists(cand):
                output_path = cand
                break
            i += 1

    cfg.output_path = output_path  # final dest
    cfg_dict = cfg.__dict__

    # Run single task (no multiprocessing)
    try:
        res = _process_image_task(input_path, output_path, cfg_dict)
        if res.get("success"):
            click.echo(f"OK: {input_path} -> {res['output_path']} "
                       f"({human_size(res['original_size'])} -> {human_size(res['final_size'])}, "
                       f"quality={res['quality_used']}, time={res['duration_ms']}ms)")
        else:
            click.echo(f"ERROR: {input_path} -> {res.get('error')}")
    except Exception as e:
        click.echo(f"FATAL ERROR: {e}")


@cli.command("batch")
@click.option("--input-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True,
              help="Input directory to recursively process JPEG files.")
@click.option("--output-dir", type=click.Path(file_okay=False, dir_okay=True), required=True,
              help="Output directory for processed images. Will preserve subdirectory structure.")
@click.option("--target-size", type=int, default=None,
              help="Target file size in bytes for each image.")
@click.option("--quality-min", type=int, default=10, help="Minimum quality for adaptive encoding.")
@click.option("--quality-max", type=int, default=95, help="Maximum quality for adaptive encoding.")
@click.option("--subsampling", type=click.Choice(list(SUBSAMPLING_MAP.keys()) + ["4:4:4", "4:2:2", "4:2:0"], default="4:2:0",
              help="JPEG subsampling.")
@click.option("--progressive/--no-progressive", default=False, help="Use progressive JPEG encoding.")
@click.option("--keep-exif/--drop-exif", default=True, help="Keep EXIF metadata.")
@click.option("--keep-icc/--drop-icc", default=True, help="Keep ICC color profile.")
@click.option("--downscale/--no-downscale", default=False, help="Enable downscaling if target size can't be reached by quality.")
@click.option("--downscale-factor", type=float, default=0.9, help="Downscale factor per step when enabled.")
@click.option("--threads", type=int, default=4, help="Maximum number of parallel workers.")
@click.option("--overwrite/--no-overwrite", default=False, help="Overwrite existing files if present.")
@click.option("--recursive/--no-recursive", default=True, help="Process subdirectories recursively.")
@click.option("--verbose/--no-verbose", default=False, help="Verbose logging.")
def batch(input_dir, output_dir, target_size, quality_min, quality_max,
          subsampling, progressive, keep_exif, keep_icc, downscale, downscale_factor,
          threads, overwrite, recursive, verbose):
    """Batch compress JPEGs in a directory with multithreading."""
    subs = SUBSAMPLING_MAP.get(subsampling, 2)

    cfg = {
        "target_size": target_size,
        "quality_min": quality_min,
        "quality_max": quality_max,
        "subsampling": subs,
        "progressive": progressive,
        "keep_exif": keep_exif,
        "keep_icc": keep_icc,
        "downscale": downscale,
        "downscale_factor": downscale_factor,
        "threads": max(1, int(threads)),
        "overwrite": overwrite,
        "preserve_structure": True,
        "verbose": verbose
    }

    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    ensure_dir(output_dir)
    images = _collect_image_paths(input_dir)

    if not images:
        click.echo("No JPEG images found in the input directory.")
        return

    total = len(images)
    processed = 0
    successes = 0
    failures = 0
    start_time = time.time()

    # Prepare tasks
    tasks = []
    for path in images:
        rel_path = os.path.relpath(path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        # ensure target directory exists
        out_dir = os.path.dirname(out_path)
        ensure_dir(out_dir)
        if os.path.exists(out_path) and not cfg["overwrite"]:
            base, ext = os.path.splitext(out_path)
            i = 1
            while True:
                cand = f"{base}__{i}{ext}"
                if not os.path.exists(cand):
                    out_path = cand
                    break
                i += 1
        tasks.append((path, out_path, cfg))

    # Process with multiprocessing
    max_workers = max(1, cfg["threads"])
    # We will use the multiprocessing-like approach but keep code simple and safe
    # Use ThreadPool if PIL operations release GIL poorly; we use ProcessPool for CPU-bound
    from concurrent.futures import ProcessPoolExecutor, as_completed

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_process_image_task, path, out_path, cfg): (path, out_path)
            for (path, out_path, cfg) in [(t[0], t[1], t[2]) for t in tasks]
        }
        for fut in as_completed(future_map):
            res = fut.result()
            results.append(res)

    # Summarize
    for res in results:
        processed += 1
        if res.get("success"):
            successes += 1
        else:
            failures += 1

    duration = int((time.time() - start_time) * 1000)
    click.echo("Batch summary:")
    click.echo(f"  Total: {total}, Processed: {processed}, Successes: {successes}, Failures: {failures}")
    click.echo(f"  Time: {duration} ms")
    if verbose:
        for r in results:
            if not r.get("success"):
                click.echo(f"  [ERROR] {r.get('input_path')} -> {r.get('error')}")


# ----------------------------
# Entry point
# ----------------------------

def main():
    cli()


if __name__ == "__main__":
    main()