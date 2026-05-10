#!/usr/bin/env python3
"""
Pillow-based Image Grayscale Converter (CLI + Core)

Minimal, runnable skeleton implementing:
- Single image grayscale with optional alpha preservation
- Batch processing for directories
- EXIF orientation correction
- Decompression bomb guarding
- Safe input/output validation
- Deterministic output naming with suffix (e.g., _gray)
"""

import argparse
import logging
import os
from typing import Optional, Tuple

try:
    from PIL import Image, ImageOps
    from PIL import UnidentifiedImageError, DecompressionBombError
except Exception:  # pragma: no cover
    raise SystemExit("Pillow is required. Install with 'pip install pillow'.")


# Exceptions
class InputValidationError(Exception):
    pass


class ImageProcessingError(Exception):
    pass


class OutputWriteError(Exception):
    pass


# Constants and helpers
_SUFFIX_DEFAULT = "_gray"

_FORMAT_EXT_MAP = {
    'JPEG': '.jpg',
    'JPG': '.jpg',
    'PNG': '.png',
    'TIFF': '.tiff',
    'WEBP': '.webp',
    'BMP': '.bmp',
    'GIF': '.gif',
}


def _ext_for_format(fmt: Optional[str], default_ext: str) -> str:
    if fmt:
        fmt_u = fmt.upper()
        # Normalize common aliases
        if fmt_u in ('JPEG', 'JPG'):
            return _FORMAT_EXT_MAP['JPEG']
        if fmt_u in _FORMAT_EXT_MAP:
            return _FORMAT_EXT_MAP[fmt_u]
        # Fallback: derive from format string
        return '.' + fmt_u.lower()
    return default_ext


def _guess_format_from_path(path: str) -> Optional[str]:
    _, ext = os.path.splitext(path)
    if not ext:
        return None
    ext = ext.lower()
    if ext in ('.jpg', '.jpeg'):
        return 'JPEG'
    if ext == '.png':
        return 'PNG'
    if ext in ('.tif', '.tiff'):
        return 'TIFF'
    if ext == '.webp':
        return 'WEBP'
    if ext == '.bmp':
        return 'BMP'
    if ext == '.gif':
        return 'GIF'
    return None


def _image_has_alpha(im: Image.Image) -> bool:
    if im.mode in ('RGBA', 'LA', 'PA'):
        return True
    if im.mode == 'P':
        return im.info.get('transparency') is not None
    return False


def _convert_frame_to_grayscale(frame: Image.Image, preserve_alpha: bool) -> Image.Image:
    if not _image_has_alpha(frame):
        return frame.convert('L')
    # Ensure RGBA for consistent alpha handling
    rgba = frame.convert('RGBA')
    gray_l = rgba.convert('L')
    if preserve_alpha:
        alpha = rgba.getchannel('A')
        return Image.merge('RGBA', (gray_l, gray_l, gray_l, alpha))
    else:
        return gray_l


def single_image_to_grayscale(image: Image.Image, preserve_alpha: bool) -> Image.Image:
    """
    Convert a single image (possibly animated) to grayscale.
    - If animated, process only the first frame for this minimal version.
    - Preserve alpha if requested.
    """
    if getattr(image, "is_animated", False) and image.is_animated:
        try:
            image.seek(0)
            frame = image.copy()
        except Exception:
            frame = image.copy()
        return _convert_frame_to_grayscale(frame, preserve_alpha)
    else:
        return _convert_frame_to_grayscale(image, preserve_alpha)


def _load_image(path: str) -> Image.Image:
    if not path:
        raise InputValidationError("Input path is empty.")
    if not os.path.exists(path):
        raise InputValidationError(f"Input path does not exist: {path}")
    if not os.path.isfile(path):
        raise InputValidationError(f"Input path is not a file: {path}")

    try:
        with Image.open(path) as img:
            # Correct orientation if needed
            img = ImageOps.exif_transpose(img)
            # Copy to detach from file handle
            return img.copy()
    except DecompressionBombError as e:
        raise InputValidationError("Image too large or decompression-bomb detected.") from e
    except UnidentifiedImageError as e:
        raise InputValidationError("Cannot identify the image file (unsupported/invalid).") from e
    except (OSError, ValueError) as e:
        raise ImageProcessingError("Failed to load or process image.") from e


def _save_image(image: Image.Image, output_path: str, fmt: Optional[str] = None, **save_params) -> None:
    if not output_path:
        raise OutputWriteError("Output path is empty.")
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # If user specified a format, ensure we don't end up with a mismatch extension
    if fmt:
        fmtu = fmt.upper()
        if fmtu in ('JPEG', 'JPG') and image.mode in ('RGBA', 'LA'):
            # JPEG cannot handle alpha; drop alpha
            image = image.convert('RGB')
        try:
            image.save(output_path, format=fmtu, **save_params)
        except (OSError, ValueError) as e:
            raise OutputWriteError("Failed to save output image with specified format.") from e
        return

    # Infer format from extension if available
    try:
        image.save(output_path, **save_params)
    except (OSError, ValueError) as e:
        raise OutputWriteError("Failed to save output image.") from e


def determine_output_path_for_file(in_path: str, out_arg: Optional[str], suffix: str, fmt: Optional[str]) -> str:
    basename = os.path.basename(in_path)
    name, _ext = os.path.splitext(basename)

    ext_out = _ext_for_format(fmt, _ext if _ext else '')
    if out_arg is None:
        out_dir = os.path.dirname(in_path)
        return os.path.join(out_dir, f"{name}{suffix}{ext_out}")
    if os.path.isdir(out_arg):
        return os.path.join(out_arg, f"{name}{suffix}{ext_out}")
    # Out arg is treated as a specific file path
    return out_arg


def is_image_file_path(path: str) -> bool:
    if not path:
        return False
    if not os.path.isfile(path):
        return False
    # Basic check based on extension; more robust check is done by PIL during load
    allowed_ext = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.webp', '.bmp', '.gif'}
    _, ext = os.path.splitext(path)
    return ext.lower() in allowed_ext


# CLI
_USAGE = "pillow-gray - simple grayscale converter for images (single or batch)."


def _configure_logging(level_str: Optional[str], quiet: bool) -> None:
    if quiet:
        log_level = logging.WARNING
    else:
        if level_str:
            level_map = {
                'DEBUG': logging.DEBUG,
                'INFO': logging.INFO,
                'WARNING': logging.WARNING,
                'ERROR': logging.ERROR,
            }
            log_level = level_map.get(level_str.upper(), logging.INFO)
        else:
            log_level = logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )


def _process_single(input_path: str, output_arg: Optional[str], preserve_alpha: bool, fmt: Optional[str],
                    overwrite: bool) -> Tuple[str, Optional[str]]:
    # Load
    img = _load_image(input_path)

    # Convert
    grayscale = single_image_to_grayscale(img, preserve_alpha)

    # Determine output path
    out_path = determine_output_path_for_file(input_path, output_arg, _SUFFIX_DEFAULT, fmt)

    # Overwrite guard
    if os.path.exists(out_path) and not overwrite:
        raise OutputWriteError(f"Output file already exists: {out_path}. Use --overwrite to override.")

    # Save
    _save_image(grayscale, out_path, fmt)

    return out_path, None


def _process_batch(input_dir: str, output_dir: Optional[str], preserve_alpha: bool, fmt: Optional[str],
                   overwrite: bool) -> int:
    """
    Process all image files under input_dir (non-recursive for simplicity).
    Returns number of successfully processed files.
    """
    if not os.path.isdir(input_dir):
        raise InputValidationError(f"Batch mode requires a directory path. Got: {input_dir}")

    count = 0
    for root, _, files in os.walk(input_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            if not is_image_file_path(fpath):
                continue
            try:
                img = _load_image(fpath)
                grayscale = single_image_to_grayscale(img, preserve_alpha)
                out_path = determine_output_path_for_file(
                    fpath, output_dir, _SUFFIX_DEFAULT, fmt
                )
                if os.path.exists(out_path) and not overwrite:
                    logging.warning("Skipped (exists): %s", out_path)
                    continue
                _save_image(grayscale, out_path, fmt)
                logging.info("Processed: %s -> %s", fpath, out_path)
                count += 1
            except (InputValidationError, ImageProcessingError, OutputWriteError) as e:
                logging.error("Failed to process %s: %s", fpath, e)
    return count


def main():
    parser = argparse.ArgumentParser(prog="pillow-gray", description="Convert images to grayscale with optional alpha preservation.")
    parser.add_argument('-i', '--input', required=True, help="Input file or directory.")
    parser.add_argument('-o', '--output', required=False, help="Output file or directory. If omitted for single input, outputs next to input with _gray suffix.")
    parser.add_argument('-p', '--preserve-alpha', action='store_true', help="Preserve alpha channel (RGBA).")
    parser.add_argument('-f', '--format', dest='fmt', help="Output format (JPEG, PNG, TIFF, WEBP, GIF, BMP). If omitted, keeps input format.")
    parser.add_argument('-w', '--overwrite', action='store_true', help="Overwrite existing output files.")
    parser.add_argument('--batch', action='store_true', help="Enable batch processing for directories (recursively).")
    parser.add_argument('--log-level', default=None, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help="Logging level.")
    parser.add_argument('--quiet', action='store_true', help="Suppress non-critical output (quiet).")

    args = parser.parse_args()

    _configure_logging(args.log_level, args.quiet)

    input_path = os.path.abspath(args.input)
    output_arg = os.path.abspath(args.output) if args.output else None
    preserve_alpha = bool(args.preserve_alpha)
    fmt = args.fmt
    overwrite = bool(args.overwrite)

    # Normalize batch decision
    batch_mode = bool(args.batch) or os.path.isdir(input_path)

    try:
        if batch_mode and os.path.isdir(input_path):
            logging.info("Starting batch processing: %s", input_path)
            # If output is a file, not a directory, that's invalid in batch mode
            if output_arg and not os.path.isdir(output_arg):
                logging.warning("Output path is not a directory for batch processing; using as base file/directory will be interpreted as directory.")
            processed = _process_batch(input_path, output_arg, preserve_alpha, fmt, overwrite)
            logging.info("Batch processing complete. Files processed: %d", processed)
        else:
            # Single file processing
            if not os.path.exists(input_path) or not os.path.isfile(input_path):
                raise InputValidationError(f"Input file does not exist: {input_path}")

            # Determine output path for a single file
            if output_arg is None:
                # default to input directory with suffix
                base, ext = os.path.splitext(os.path.basename(input_path))
                ext_out = _ext_for_format(fmt, ext)
                output_path = os.path.join(os.path.dirname(input_path), f"{base}{_SUFFIX_DEFAULT}{ext_out}")
            elif os.path.isdir(output_arg):
                base, ext = os.path.splitext(os.path.basename(input_path))
                ext_out = _ext_for_format(fmt, ext)
                output_path = os.path.join(output_arg, f"{base}{_SUFFIX_DEFAULT}{ext_out}")
            else:
                output_path = output_arg

            if os.path.exists(output_path) and not overwrite:
                raise OutputWriteError(f"Output file already exists: {output_path}. Use --overwrite to override.")

            # Load, convert, save
            img = _load_image(input_path)
            grayscale = single_image_to_grayscale(img, preserve_alpha)
            _save_image(grayscale, output_path, fmt)
            logging.info("Processed: %s -> %s", input_path, output_path)

    except InputValidationError as e:
        logging.error("Input validation error: %s", e)
        raise SystemExit(2)
    except ImageProcessingError as e:
        logging.error("Image processing error: %s", e)
        raise SystemExit(3)
    except OutputWriteError as e:
        logging.error("Output write error: %s", e)
        raise SystemExit(4)
    except Exception as e:
        logging.exception("Unexpected error: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()