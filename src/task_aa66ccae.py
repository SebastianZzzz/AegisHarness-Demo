import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple

try:
    from PIL import Image, ImageOps
except Exception as e:
    sys.stderr.write(f"ERROR: Pillow is required to run this script: {e}\n")
    sys.exit(2)

# Optional progress bar (tqdm)
_TQDM_AVAILABLE = False
try:
    from tqdm import tqdm  # type: ignore
    _TQDM_AVAILABLE = True
except Exception:
    _TQDM_AVAILABLE = False

ALLOWED_EXTS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp"
}


@dataclass
class Options:
    input_path: Path
    output_dir: Optional[Path]
    recursive: bool
    force_format: Optional[str]
    method: str  # 'L' or 'LA'
    max_size: Optional[int]
    preserve_metadata: bool
    overwrite: bool
    suffix: str
    apply_orientation: bool
    verbose: int
    show_progress: bool


def _setup_logging(verbose: int) -> None:
    if verbose >= 2:
        level = logging.DEBUG
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def is_image_file(p: Path) -> bool:
    if not p.is_file():
        return False
    if p.suffix.lower() in ALLOWED_EXTS:
        return True
    try:
        with Image.open(p) as im:
            im.verify()
        return True
    except Exception:
        return False


def enumerate_images(input_path: Path, recursive: bool) -> Generator[Path, None, None]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if input_path.is_file():
        if is_image_file(input_path):
            yield input_path
        return
    if input_path.is_dir():
        if recursive:
            for p in input_path.rglob("*"):
                if p.is_file() and is_image_file(p):
                    yield p
        else:
            for p in input_path.iterdir():
                if p.is_file() and is_image_file(p):
                    yield p
    else:
        raise FileNotFoundError(f"Input path is not a file or directory: {input_path}")


def build_output_path(
    input_path: Path,
    output_dir: Optional[Path],
    suffix: str,
    force_format: Optional[str],
) -> Path:
    out_dir = Path(output_dir) if output_dir else input_path.parent
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    ext = (force_format or input_path.suffix[1:]).lower()
    out_name = f"{stem}{suffix}.{ext}"
    return out_dir / out_name


def _load_image_preserving_paths(path: Path) -> Tuple[Image.Image, Optional[bytes], Optional[bytes]]:
    """
    Open the image safely and return a PIL Image object along with its EXIF bytes and ICC profile if available.
    Performs a verify() pass to catch corrupted files before processing.
    Returns the loaded image (reopened) and its exif bytes and icc profile bytes (or None).
    """
    try:
        # First pass: verify integrity
        with Image.open(path) as im:
            im.verify()

        # Second pass: actual loading
        with Image.open(path) as im:
            exif_bytes = im.info.get("exif")
            icc_profile = im.info.get("icc_profile")
            return im.copy(), exif_bytes, icc_profile
    except Exception as e:
        raise ValueError(f"Failed to load image '{path}': {e}") from e


def apply_orientation_if_needed(img: Image.Image, apply_orientation: bool) -> Image.Image:
    if apply_orientation:
        try:
            return ImageOps.exif_transpose(img)
        except Exception:
            # If something goes wrong, return the original image
            return img
    return img


def _resize_image(img: Image.Image, max_size: Optional[int]) -> Image.Image:
    if not max_size or max_size <= 0:
        return img
    w, h = img.size
    max_dim = max(w, h)
    if max_dim <= max_size:
        return img
    scale = max_size / float(max_dim)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), resample=Image.LANCZOS)


def grayscale_image(img: Image.Image, method: str, preserve_alpha: bool) -> Image.Image:
    method = method.upper()
    if method not in ("L", "LA"):
        raise ValueError(f"Unsupported grayscale method: {method}")

    if method == "L":
        if preserve_alpha and img.mode == "RGBA":
            rgb = img.convert("RGB")
            gray = rgb.convert("L")
            a = img.getchannel("A")
            return Image.merge("LA", (gray, a))
        else:
            return img.convert("L")
    else:  # LA
        # Ensure we have an alpha channel to preserve
        if img.mode == "RGBA":
            return img.convert("LA")
        else:
            # Promote to RGBA, then to LA (alpha 255)
            return img.convert("RGBA").convert("LA")


def process_image(input_path: Path, options: Options) -> Tuple[bool, str]:
    """
    Process a single image. Returns (success, message)
    """
    try:
        out_path = build_output_path(input_path, options.output_dir, options.suffix, options.force_format)
        if out_path.exists() and not options.overwrite:
            logging.info(f"SKIP: Output exists and overwrite disabled -> {out_path}")
            return True, f"skipped (exists): {out_path}"

        # Load with verification and preserve metadata if requested
        img, exif_bytes, icc_profile = _load_image_preserving_paths(input_path)

        # Orientation handling
        img = apply_orientation_if_needed(img, options.apply_orientation)

        # Resize if requested
        img = _resize_image(img, options.max_size)

        # Grayscale conversion
        img = grayscale_image(img, options.method, options.preserve_metadata)

        # Prepare save parameters
        save_kwargs = {}
        if options.preserve_metadata:
            if exif_bytes:
                save_kwargs["exif"] = exif_bytes
            if icc_profile:
                save_kwargs["icc_profile"] = icc_profile

        # Save
        if options.force_format:
            img.save(out_path, format=options.force_format.upper(), **save_kwargs)
        else:
            img.save(out_path, **save_kwargs)

        logging.info(f"OK: {input_path} -> {out_path}")
        return True, f"saved: {out_path}"
    except Exception as e:
        logging.error(f"FAILED: {input_path} -> error: {e}")
        return False, f"failed: {e}"


def collect_stats(results: list) -> dict:
    total = len(results)
    successes = sum(1 for ok, _ in results if ok)
    failures = total - successes
    return {
        "total": total,
        "successes": successes,
        "failures": failures,
    }


def display_summary(stats: dict, duration_seconds: float) -> None:
    logging.info("Processing Summary:")
    logging.info(f"  Total images: {stats['total']}")
    logging.info(f"  Successful: {stats['successes']}")
    logging.info(f"  Failed: {stats['failures']}")
    logging.info(f"  Time elapsed: {duration_seconds:.2f}s")


def parse_args(argv: Optional[list] = None) -> Tuple[Options, Optional[Path], Optional[bool]]:
    parser = argparse.ArgumentParser(
        prog="grayscale.py",
        description="Convert images to grayscale using Pillow with optional metadata preservation."
    )

    parser.add_argument("-i", "--input", dest="input", required=True, help="Path to an image file or directory")
    parser.add_argument("-o", "--output", dest="output", help="Output directory (optional). Defaults to input dir with suffix.")
    parser.add_argument("-r", "--recursive", dest="recursive", action="store_true", help="Process directories recursively")
    parser.add_argument("-f", "--force-format", dest="force_format", help="Force output format (e.g., png, jpeg).")
    parser.add_argument("-m", "--method", dest="method", choices=["L", "LA"], default="L", help="Grayscale method: L or LA (default: L)")
    parser.add_argument("-s", "--maximize-size", dest="max_size", type=int, default=None, help="Maximum dimension (width/height) to resize to")
    parser.add_argument("--preserve-metadata", dest="preserve_metadata", action="store_true", default=True, help="Preserve EXIF and ICC data (default: true)")
    parser.add_argument("--no-preserve-metadata", dest="preserve_metadata", action="store_false", help="Do not preserve metadata")
    parser.add_argument("--overwrite", dest="overwrite", action="store_true", default=False, help="Overwrite existing outputs")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false", help="Do not overwrite existing outputs (default)")
    parser.add_argument("-u", "--suffix", dest="suffix", default="_gray", help="Suffix to append to output filenames (default: _gray)")
    parser.add_argument("--apply-orientation", dest="apply_orientation", action="store_true", default=True, help="Apply EXIF orientation before processing")
    parser.add_argument("--no-apply-orientation", dest="apply_orientation", action="store_false", help="Do not apply EXIF orientation before processing")
    parser.add_argument("-v", "--verbose", dest="verbose", action="count", default=0, help="Increase verbosity (can be repeated)")
    parser.add_argument("--progress", dest="progress", action="store_true", default=False, help="Show progress bar during batch processing")
    parser.add_argument("--no-progress", dest="progress", action="store_false", help="Hide progress bar during batch processing")

    ns = parser.parse_args(argv)

    input_path = Path(ns.input).expanduser().resolve()

    if ns.output:
        output_dir = Path(ns.output).expanduser().resolve()
    else:
        output_dir = None

    options = Options(
        input_path=input_path,
        output_dir=output_dir,
        recursive=bool(ns.recursive),
        force_format=ns.force_format,
        method=ns.method,
        max_size=ns.max_size,
        preserve_metadata=bool(ns.preserve_metadata),
        overwrite=bool(ns.overwrite),
        suffix=ns.suffix,
        apply_orientation=bool(ns.apply_orientation),
        verbose=int(ns.verbose),
        show_progress=bool(ns.progress),
    )

    show_progress_default = _TQDM_AVAILABLE
    if options.show_progress is False:
        show_progress_default = False
    elif options.show_progress is True:
        show_progress_default = True
    # return parsed options and a placeholder for potential tests
    return options, input_path, show_progress_default


def main(argv: Optional[list] = None) -> int:
    options, input_path, _ = parse_args(argv)

    _setup_logging(options.verbose)

    start_time = time.time()

    try:
        image_paths = list(enumerate_images(options.input_path, options.recursive))
    except Exception as e:
        logging.error(f"Input enumeration failed: {e}")
        return 2

    if not image_paths:
        logging.warning("No image files found to process.")
        return 0

    total_count = len(image_paths)
    results = []
    if options.show_progress and _TQDM_AVAILABLE:
        iterator = tqdm(image_paths, total=total_count, desc="Converting")
        it = iterator
    else:
        it = image_paths

    # Process images
    for p in it:
        ok, msg = process_image(p, options)
        results.append((ok, msg))

    duration = time.time() - start_time
    stats = collect_stats(results)
    display_summary(stats, duration)

    # Return code: 0 if all success, 1 if some failed
    return 0 if stats["failures"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())