#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Iterator

try:
    from tqdm import tqdm  # type: ignore
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False
    tqdm = None  # type: ignore

from PIL import Image, ImageOps

# Supported image extensions
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


def enumerate_images(input_path: Path, recursive: bool) -> Iterator[Path]:
    if input_path.is_dir():
        if recursive:
            for root, _, files in os.walk(input_path):
                for name in files:
                    p = Path(root) / name
                    if is_image_file(p):
                        yield p
        else:
            for p in input_path.iterdir():
                if p.is_file() and is_image_file(p):
                    yield p
    elif input_path.is_file():
        if is_image_file(input_path):
            yield input_path
    else:
        return


def build_output_path(input_path: Path, output_dir: Optional[Path], suffix: str,
                      force_format: Optional[str]) -> Path:
    input_path = input_path.resolve()
    if output_dir is None:
        out_dir = input_path.parent
    else:
        out_dir = output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    base = input_path.stem
    if force_format:
        ext_out = f".{force_format.lower().lstrip('.')}"
    else:
        ext_out = input_path.suffix
    out_filename = f"{base}{suffix}{ext_out}"
    return out_dir / out_filename


def grayscale_image(img: Image.Image, method: str) -> Image.Image:
    """
    Convert an image to grayscale based on method.
    method: 'L' or 'LA'
    - 'L': convert to single-channel luminance (alpha dropped)
    - 'LA': convert to luminance while preserving alpha channel (if present),
            otherwise adds a fully opaque alpha
    """
    method = method.upper()
    if method == 'L':
        return img.convert('L')
    elif method == 'LA':
        gray = img.convert('L')
        bands = img.getbands()
        if 'A' in bands:
            a = img.getchannel('A')
        else:
            a = Image.new('L', img.size, 255)
        return Image.merge('LA', (gray, a))
    else:
        raise ValueError(f"Unsupported grayscale method: {method}")


def safe_open(input_path: Path) -> Image.Image:
    """
    Open an image safely and verify integrity.
    Returns a Pillow Image object.
    """
    with Image.open(input_path) as im:
        try:
            im.verify()  # validate integrity
        except Exception as e:
            raise e
    # Reopen for actual processing
    return Image.open(input_path)


def process_image(input_path: Path,
                  method: str,
                  apply_orientation: bool,
                  max_size: Optional[int]) -> Image.Image:
    """
    Open and process a single image: orientation (optional), resize (optional),
    then grayscale according to method.
    Returns the processed PIL Image (not yet saved).
    """
    with Image.open(input_path) as img:
        if apply_orientation:
            img = ImageOps.exif_transpose(img)

        if max_size is not None and max_size > 0:
            w, h = img.size
            if w > max_size or h > max_size:
                scale = min(max_size / float(w), max_size / float(h))
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                img = img.resize((new_w, new_h), resample=Image.LANCZOS)

        processed = grayscale_image(img, method)
        if processed is None:
            raise RuntimeError("Failed to convert image to grayscale.")
        # If orientation applied, ensure we still have the proper data
        return processed.copy()  # return an independent image


def read_exif_and_icc(img: Image.Image) -> tuple[Optional[bytes], Optional[bytes]]:
    exif = img.info.get('exif')
    icc = img.info.get('icc_profile')
    return exif, icc


def ensure_parent_dir(p: Path):
    p_parent = p.parent
    p_parent.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Convert images to grayscale using Pillow. Supports batch processing, "
                    "optional resizing, and metadata preservation."
    )
    parser.add_argument('-i', '--input', required=True,
                        help='Path to an image file or a directory containing images.')
    parser.add_argument('-o', '--output', required=False,
                        help='Output directory. If omitted, outputs are saved next to inputs with suffix.')
    parser.add_argument('-r', '--recursive', action='store_true',
                        help='Process directories recursively (default: false).')
    parser.add_argument('-f', '--force-format', required=False,
                        help='Force output format (e.g., png, jpeg).')
    parser.add_argument('-m', '--method', choices=['L', 'LA'], default='L',
                        help="Grayscale method: 'L' (default) or 'LA' to preserve alpha.")
    parser.add_argument('-s', '--maximize-size', type=int, default=None,
                        help='Maximum width/height for processed images (maintains aspect ratio).')
    parser.add_argument('--preserve-metadata', dest='preserve_metadata', action='store_true',
                        help='Preserve EXIF and ICC metadata when saving (default).')
    parser.add_argument('--no-preserve-metadata', dest='preserve_metadata', action='store_false',
                        help='Do not preserve EXIF/ICC metadata when saving.')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help='Overwrite existing output files.')
    parser.add_argument('--no-overwrite', dest='overwrite', action='store_false',
                        help='Do not overwrite existing output files (default).')
    parser.add_argument('-u', '--suffix', default='_gray',
                        help='Suffix to append to output filenames (default "_gray").')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress informational logs.')
    parser.add_argument('--verbose', action='store_true',
                        help='Increase logging verbosity for debugging.')
    parser.add_argument('--progress', dest='progress', action='store_true',
                        help='Show a progress bar during batch processing.')
    parser.add_argument('--no-progress', dest='progress', action='store_false',
                        help='Do not show a progress bar during batch processing.')
    parser.set_defaults(progress=HAS_TQDM)

    args = parser.parse_args()

    # Logging configuration
    if args.verbose:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.WARNING
    else:
        log_level = logging.INFO
    logging.basicConfig(level=log_level,
                        format='[%(levelname)s] %(message)s')

    input_path = Path(args.input)
    if not input_path.exists():
        logging.error("Input path does not exist: %s", input_path)
        sys.exit(2)

    # Resolve output directory
    output_dir = None
    if args.output:
        output_dir = Path(args.output).expanduser().resolve()
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.error("Cannot create output directory: %s (%s)", output_dir, e)
            sys.exit(2)

    # Validate input type
    if not (input_path.is_file() or input_path.is_dir()):
        logging.error("Input path is not a file or directory: %s", input_path)
        sys.exit(2)

    # Enumerate images
    image_paths = list(enumerate_images(input_path, args.recursive))
    if not image_paths:
        logging.info("No supported image files found in the given path.")
        sys.exit(0)

    total = len(image_paths)
    logging.info("Found %d image(s) to process.", total)

    successes = 0
    failures = 0
    start_time = time.time()

    # Progress wrapper
    iterator = enumerate_images(input_path, args.recursive)
    if args.progress and HAS_TQDM:
        iterator = tqdm(image_paths, desc="Processing images", unit="file")

    # Process each image
    for in_path in image_paths:
        try:
            in_path = Path(in_path)
            # Open and verify (robust against corrupt files)
            with Image.open(in_path) as _tmp:
                _ = _tmp.verify()
            processed: Image.Image = process_image(
                in_path,
                method=args.method,
                apply_orientation=True,  # always apply orientation to ensure consistent grayscale output
                max_size=args.maximize_size
            )

            # Prepare output path
            out_path = build_output_path(
                in_path,
                output_dir,
                args.suffix,
                args.force_format
            )

            if out_path.exists() and not args.overwrite:
                logging.warning("Output exists and overwrite disabled: %s", out_path)
                failures += 1
                continue

            # Metadata handling
            exif, icc = (None, None)  # default
            # Read metadata from the source image (only if preservation requested)
            with Image.open(in_path) as src_img:
                if args.preserve_metadata:
                    exif, icc = read_exif_and_icc(src_img)

            # If saving to JPEG and method == 'LA', alpha cannot be stored; convert to 'L'
            save_as = None
            if args.force_format:
                save_as = args.force_format.upper()
            else:
                save_ext = out_path.suffix.lower().lstrip('.')
                save_as = save_ext.upper()

            if save_as in ('JPEG', 'JPG') and args.method == 'LA':
                processed = processed.convert('L')

            ensure_parent_dir(out_path)

            save_kw = {}
            if args.preserve_metadata:
                if exif is not None:
                    save_kw['exif'] = exif
                if icc is not None:
                    save_kw['icc_profile'] = icc

            # Save with or without explicit format to rely on extension
            if args.force_format:
                processed.save(out_path, format=args.force_format.upper(), **save_kw)
            else:
                processed.save(out_path, **save_kw)

            logging.info("Saved: %s", out_path)
            successes += 1
        except KeyboardInterrupt:
            logging.info("Interrupted by user.")
            break
        except Exception as e:
            logging.error("Failed processing '%s': %s", in_path, e)
            failures += 1

    elapsed = time.time() - start_time
    summary = {
        'total': total,
        'successes': successes,
        'failures': failures,
        'elapsed_sec': round(elapsed, 2)
    }

    print("\nProcessing complete.")
    print(f"Total: {summary['total']}, Successes: {summary['successes']}, "
          f"Failures: {summary['failures']}, Time: {summary['elapsed_sec']}s")


if __name__ == "__main__":
    main()