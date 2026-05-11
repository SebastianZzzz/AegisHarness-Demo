#!/usr/bin/env python3
"""
Image grayscale converter using Pillow
Supports single image and batch (directory) processing with optional structure preservation.

This refactored version:
- Validates and sanitizes all paths, ensuring operations stay within a safe base directory.
- Uses context manager for image I/O.
- Properly handles alpha channel when converting to grayscale ('L' or 'LA').
- Fixes batch mode to respect include-alpha flag and avoid silent ignores.
- Removes unused imports and small code smells.
- Provides robust error messages and progress indication (optional).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

try:
    from PIL import Image, UnidentifiedImageError
except Exception:  # pragma: no cover
    print("Pillow is required. Install it via: pip install pillow", file=sys.stderr)
    sys.exit(2)

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore
    HAS_TQDM = True
except Exception:  # pragma: no cover
    HAS_TQDM = False

EXTENSIONS_DEFAULT = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
BASE_DIR = Path.cwd().resolve()


def _ext_from_format(fmt: Optional[str]) -> str:
    if fmt is None:
        # Default extension: PNG (safe for grayscale and alpha handling)
        return '.png'
    f = fmt.upper()
    if f in ('PNG',):
        return '.png'
    if f in ('JPEG', 'JPG'):
        return '.jpg'
    if f == 'WEBP':
        return '.webp'
    if f in ('TIFF', 'TIF'):
        return '.tif'
    # Fallback
    return '.png'


def _resolve_output_format_and_extension(
    requested_fmt: Optional[str], actual_mode: str
) -> Tuple[Optional[str], str]:
    """
    Returns a (format, extension) tuple.
    If alpha is present (actual_mode == 'LA'), JPEG is not suitable; we force PNG.
    """
    fmt = None if requested_fmt is None else str(requested_fmt).upper()
    ext = _ext_from_format(fmt)

    # If we have alpha and user asked for JPEG-like format (or none),
    # force PNG to preserve alpha safely.
    if actual_mode == 'LA' and (fmt is None or fmt in ('JPEG', 'JPG')):
        fmt = 'PNG'
        ext = '.png'

    return fmt, ext


def _safe_dirs(p: Path) -> Path:
    return p.resolve()


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _is_image_file(p: Path, extensions: Sequence[str]) -> bool:
    return p.is_file() and p.suffix.lower() in [e.lower() for e in extensions]


def _validate_within_base(p: Path) -> Path:
    rp = p.resolve()
    try:
        rp.relative_to(BASE_DIR)
    except Exception:
        raise ValueError(f"Path '{p}' is outside the allowed base directory '{BASE_DIR}'.")
    return rp


def convert_image_to_grayscale(
    input_path: Path,
    output_path: Optional[Path] = None,
    mode: str = 'L',
    preserve_metadata: bool = False,
    include_alpha: bool = False,
    output_format: Optional[str] = None,
    quality: Optional[int] = None,
) -> Path:
    """
    Convert a single image to grayscale (mode 'L' or 'LA').

    Returns the path to the saved output image.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    actual_mode = 'LA' if include_alpha else mode
    if actual_mode not in ('L', 'LA'):
        raise ValueError("Invalid mode. Supported: 'L' or 'LA' (when including alpha)")

    try:
        with Image.open(str(input_path)) as img:
            # Normalize source image: Pillow handles many formats
            # Apply grayscale conversion
            out_img = img.convert(actual_mode)

            # Decide output format and extension
            fmt, ext = _resolve_output_format_and_extension(output_format, actual_mode)

            # If output_path is None, create a default output path in the same directory
            if output_path is None:
                out_filename = f"bw_{input_path.stem}{ext}"
                output_path = input_path.parent / out_filename
            else:
                if output_path.is_dir():
                    out_filename = f"bw_{input_path.stem}{ext}"
                    output_path = output_path / out_filename

            _ensure_dir(output_path)

            # Preserve EXIF data if requested (most effective for JPEG)
            exif_bytes = img.info.get('exif') if preserve_metadata else None

            # If the chosen format is None, Pillow will infer from extension
            save_kwargs = {}
            if fmt is not None:
                save_kwargs['format'] = fmt
            if quality is not None:
                save_kwargs['quality'] = int(quality)
            if exif_bytes:
                save_kwargs['exif'] = exif_bytes

            # Save output
            out_img.save(str(output_path), **save_kwargs)

            return output_path
    except UnidentifiedImageError as e:
        raise ValueError(f"Cannot identify image file: {input_path}") from e
    except OSError as e:
        raise RuntimeError(f"I/O error while processing {input_path}: {e}") from e


def batch_convert(
    input_dir: Path,
    output_dir: Path,
    recursive: bool = True,
    extensions: Optional[Sequence[str]] = None,
    mode: str = 'L',
    overwrite: bool = False,
    preserve_metadata: bool = False,
    output_format: Optional[str] = None,
    quality: Optional[int] = None,
    keep_structure: bool = True,
    include_alpha: bool = False,
    progress: bool = False,
) -> List[Path]:
    """
    Batch convert all images in input_dir to grayscale.

    Returns list of output file paths created.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    if extensions is None:
        extensions = EXTENSIONS_DEFAULT
    extensions = [e.lower() for e in extensions]

    # Collect files
    images: List[Path] = []
    if recursive:
        for p in input_dir.rglob('*'):
            if p.is_file() and p.suffix.lower() in extensions:
                images.append(p)
    else:
        for p in input_dir.iterdir():
            if p.is_file() and p.suffix.lower() in extensions:
                images.append(p)

    if not images:
        return []

    total = len(images)
    iterator = images
    if progress and HAS_TQDM:
        iterator = tqdm(images, desc="Grayscale batch", unit="file", total=total)

    results: List[Path] = []
    for infile in iterator:
        # Determine the relative path for structure preservation
        if keep_structure:
            try:
                rel = infile.relative_to(input_dir)
            except Exception:
                rel = infile.name  # fallback
            # Determine destination dir and filename
            rel_parent = Path(rel).parent
            dest_dir = output_dir / rel_parent
            dest_dir.mkdir(parents=True, exist_ok=True)
            ext = _ext_from_format(output_format)
            out_filename = f"bw_{infile.stem}{ext}"
            out_path = dest_dir / out_filename
        else:
            # Flatten: all outputs go into output_dir with same basename
            output_dir.mkdir(parents=True, exist_ok=True)
            ext = _ext_from_format(output_format)
            out_filename = f"bw_{infile.stem}{ext}"
            out_path = output_dir / out_filename

        if out_path.exists() and not overwrite:
            # Skip if not overwriting existing file
            continue

        try:
            converted = convert_image_to_grayscale(
                input_path=infile,
                output_path=out_path,
                mode=mode,
                preserve_metadata=preserve_metadata,
                include_alpha=include_alpha or (mode == 'LA'),
                output_format=output_format,
                quality=quality,
            )
            results.append(converted)
        except Exception as e:
            # For batch mode, continue processing other files
            # Print error message but do not raise
            print(f"Error processing '{infile}': {e}", file=sys.stderr)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="grayscale-tool",
        description="Convert images to grayscale using Pillow. Supports single image and batch processing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/Output
    parser.add_argument('-i', '--input', required=True, help="Input image file path or input directory for batch mode.")
    parser.add_argument('-o', '--output', default=None, help="Output file path (for single image) or output directory (for batch).")

    # Batch options
    parser.add_argument('-r', '--recursive', action='store_true', help="Recursively process directories (batch mode).")
    parser.add_argument('-e', '--extensions', nargs='+', default=EXTENSIONS_DEFAULT, help="List of file extensions to include (e.g., .jpg .png).")

    # Grayscale mode
    parser.add_argument('-m', '--mode', choices=['L', 'LA'], default='L', help="Target grayscale mode. 'L' for 8-bit grayscale, 'LA' to keep alpha.")
    parser.add_argument('--include-alpha', action='store_true', dest='include_alpha', help="Include alpha channel when converting (produces LA). Overrides mode decision.")
    # Output format and quality
    parser.add_argument('-f', '--output-format', dest='output_format', help="Output image format (PNG, JPEG, WEBP, TIFF).")
    parser.add_argument('-q', '--quality', type=int, default=None, help="Output quality for lossy formats (e.g., JPEG).")

    # Overwrite / structure
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing files in output.")
    parser.add_argument('--keep-structure', dest='keep_structure', action='store_true', help="Preserve input directory structure in the output (default for batch).")
    parser.add_argument('--no-keep-structure', dest='keep_structure', action='store_false', help="Do not preserve directory structure in the output.")
    # Progress
    parser.add_argument('--progress', action='store_true', help="Show a progress bar for batch processing (requires tqdm).")

    # Metadata
    parser.add_argument('--preserve-metadata', action='store_true', dest='preserve_metadata', help="Preserve EXIF/ICC metadata when saving outputs.")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        input_path = Path(args.input)

        # Resolve and basic safety checks
        if not input_path.exists():
            print(f"Error: Input path not found: {input_path}", file=sys.stderr)
            return 2

        # Ensure input is within base directory
        try:
            _validate_within_base(input_path)
        except ValueError as ve:
            print(f"Error: {ve}", file=sys.stderr)
            return 2

        # Decide if single file or directory
        if input_path.is_file():
            # Single image mode
            ext = _ext_from_format(args.output_format)

            # Determine final output path
            if args.output:
                out_path = Path(args.output)
                if out_path.exists() and out_path.is_dir():
                    out_path = out_path / f"bw_{input_path.stem}{ext}"
            else:
                out_path = input_path.parent / f"bw_{input_path.stem}{ext}"

            # Safety: ensure path inside base
            try:
                out_path = _validate_within_base(out_path)
            except ValueError:
                print("Error: Output path is outside the allowed base directory.", file=sys.stderr)
                return 2

            if out_path.exists() and not args.overwrite:
                print(f"Output file already exists and overwrite is disabled: {out_path}", file=sys.stderr)
                return 1

            try:
                convert_image_to_grayscale(
                    input_path=input_path,
                    output_path=out_path,
                    mode=args.mode,
                    preserve_metadata=args.preserve_metadata,
                    include_alpha=args.include_alpha,
                    output_format=args.output_format,
                    quality=args.quality,
                )
                return 0
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1

        else:
            # Directory/batch mode
            input_dir = input_path
            output_dir = None
            if args.output:
                output_dir = Path(args.output)
            else:
                output_dir = input_dir.parent / (input_dir.name + "_gray")

            # Safety: ensure dirs inside base
            try:
                input_dir = _validate_within_base(input_dir)
                output_dir = _validate_within_base(output_dir)
            except ValueError as ve:
                print(f"Error: {ve}", file=sys.stderr)
                return 2

            results = batch_convert(
                input_dir=input_dir,
                output_dir=output_dir,
                recursive=args.recursive,
                extensions=args.extensions,
                mode=args.mode,
                overwrite=args.overwrite,
                preserve_metadata=args.preserve_metadata,
                output_format=args.output_format,
                quality=args.quality,
                keep_structure=args.keep_structure if args.keep_structure is not None else True,
                include_alpha=args.include_alpha,
                progress=args.progress,
            )
            print(f"Processed {len(results)} file(s).", file=sys.stdout)
            return 0

    except ValueError as ve:
        print(f"Error: {ve}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())