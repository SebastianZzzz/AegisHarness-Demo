#!/usr/bin/env python3
# Pillow-based grayscale CLI skeleton with optional alpha preservation
import os
import sys
import glob
import argparse
import logging
from typing import Optional

from PIL import Image, ImageOps, UnidentifiedImageError

# Exceptions
class GrayscaleError(Exception):
    pass

class InputValidationError(GrayscaleError):
    pass

class ImageProcessingError(GrayscaleError):
    pass

class OutputWriteError(GrayscaleError):
    pass

# Constants
SUPPORTED_INPUT_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}
DEFAULT_OUTPUT_SUFFIX = "_gray"
DEFAULT_OUTPUT_FORMAT = None  # None means infer from path

def _infer_format_from_path(path: str) -> Optional[str]:
    ext = os.path.splitext(path)[1].lower()
    mapping = {
        '.jpg': 'JPEG',
        '.jpeg': 'JPEG',
        '.png': 'PNG',
        '.gif': 'GIF',
        '.tiff': 'TIFF',
        '.tif': 'TIFF',
        '.webp': 'WEBP',
        '.bmp': 'BMP',
    }
    return mapping.get(ext)

def _ensure_dir_for_path(path: str) -> None:
    dirn = os.path.dirname(os.path.abspath(path))
    if dirn and not os.path.exists(dirn):
        os.makedirs(dirn, exist_ok=True)

def load_image(path: str) -> Image.Image:
    if not path:
        raise InputValidationError("Input path is empty.")
    if not os.path.exists(path):
        raise InputValidationError(f"Input path does not exist: {path}")
    try:
        with Image.open(path) as img:
            # Apply EXIF orientation if present
            img = ImageOps.exif_transpose(img)
            # If animated, attempt to grab first frame for a safe fallback
            if getattr(img, "is_animated", False):
                try:
                    img.seek(0)
                    frame = img.copy()
                    img = frame
                except Exception:
                    pass
            return img.copy()
    except Image.DecompressionBombError as e:
        raise ImageProcessingError(f"Decompression bomb detected for {path}") from e
    except UnidentifiedImageError as e:
        raise InputValidationError(f"Unsupported or invalid image file: {path}") from e
    except OSError as e:
        raise ImageProcessingError(f"Cannot load image file: {path}") from e

def single_image_to_grayscale(image: Image.Image, preserve_alpha: bool) -> Image.Image:
    # If the image is animated, keep first frame (fallback for initial version)
    if getattr(image, "is_animated", False):
        try:
            image.seek(0)
            image = image.copy()
        except Exception:
            pass

    bands = image.getbands()
    has_alpha = 'A' in bands
    if has_alpha:
        grayscale_l = image.convert('L')
        alpha = image.getchannel('A')
        if preserve_alpha:
            if image.mode == 'RGBA':
                return Image.merge('RGBA', (grayscale_l, grayscale_l, grayscale_l, alpha))
            elif image.mode == 'LA':
                # LA means L + A channels; keep L as grayscale and preserve A
                return Image.merge('LA', (grayscale_l, alpha))
            else:
                # Fallback: return grayscale without guaranteed alpha preservation
                return grayscale_l
        else:
            return grayscale_l
    else:
        # No alpha channel; straightforward grayscale
        return image.convert('L')

def determine_output_path(input_path: str, output_path: Optional[str], suffix: str = DEFAULT_OUTPUT_SUFFIX, output_format: Optional[str] = None) -> str:
    # If no explicit output path, derive from input path
    if not output_path:
        base, ext = os.path.splitext(input_path)
        dirn = os.path.dirname(input_path)
        name = os.path.basename(base)
        out_ext = ext
        out_name = f"{name}{suffix}{out_ext}"
        return os.path.join(dirn, out_name)

    # If output_path is a directory, place output inside it with modified name
    if os.path.isdir(output_path) or output_path.endswith(os.sep) or output_path.endswith('/'):
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        base, ext = os.path.splitext(os.path.basename(input_path))
        out_ext = ext
        out_name = f"{base}{suffix}{out_ext}"
        return os.path.join(output_path, out_name)

    # Otherwise treat as a file path
    return output_path

def save_image(image: Image.Image, path: str, fmt: Optional[str] = None, overwrite: bool = False, **save_params) -> None:
    if os.path.exists(path) and not overwrite:
        raise OutputWriteError(f"Output file already exists: {path}")
    _ensure_dir_for_path(path)

    fmt_to_use = fmt or _infer_format_from_path(path)
    if fmt_to_use is None:
        fmt_to_use = None  # Let Pillow infer from file extension if possible

    # JPEG cannot store alpha; convert if needed
    if fmt_to_use in ('JPEG', 'JPG') and image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')

    try:
        image.save(path, format=fmt_to_use, **save_params)
    except Exception as e:
        raise OutputWriteError(f"Failed to write output image to {path}: {e}") from e

def process_single(input_path: str, output_path: Optional[str], preserve_alpha: bool, out_format: Optional[str], overwrite: bool) -> str:
    img = load_image(input_path)
    gray = single_image_to_grayscale(img, preserve_alpha)
    out_path = determine_output_path(input_path, output_path, suffix=DEFAULT_OUTPUT_SUFFIX, output_format=out_format)
    save_image(gray, out_path, fmt=out_format, overwrite=overwrite)
    return out_path

def collect_input_paths(input_path: str) -> list:
    if not os.path.exists(input_path):
        raise InputValidationError(f"Input path does not exist: {input_path}")
    if os.path.isdir(input_path):
        results = []
        for ext in SUPPORTED_INPUT_EXTS:
            results.extend(glob.glob(os.path.join(input_path, f"**/*{ext}"), recursive=True))
        results = sorted(set(results))
        return [p for p in results if os.path.isfile(p)]
    else:
        return [input_path]

def main():
    parser = argparse.ArgumentParser(prog="pillow-gray", description="Convert images to grayscale with optional alpha preservation.")
    parser.add_argument('-i', '--input', required=True, help='Input image file or directory for batch processing.')
    parser.add_argument('-o', '--output', required=False, help='Output file or directory. If omitted, outputs are placed beside inputs with _gray suffix.')
    parser.add_argument('-p', '--preserve-alpha', action='store_true', help='Preserve alpha channel when converting to grayscale.')
    parser.add_argument('--batch', '--recursive', action='store_true', help='Enable batch processing for directories.')
    parser.add_argument('-f', '--format', dest='format', choices=['JPEG','PNG','TIFF','WEBP','BMP'], help='Output image format.')
    parser.add_argument('-w', '--overwrite', action='store_true', help='Overwrite existing output files.')
    parser.add_argument('--quiet', action='store_true', help='Suppress non-error messages.')
    parser.add_argument('--verbose', action='store_true', help='Increase output verbosity.')
    args = parser.parse_args()

    # Simple logging configuration
    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG
    if args.quiet:
        log_level = logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    input_path = args.input
    output_path = args.output
    preserve_alpha = bool(args.preserve_alpha)
    out_format = args.format
    overwrite = bool(args.overwrite)

    try:
        input_paths = collect_input_paths(input_path)
        # If batch processing (multiple inputs or explicit directory), process in loop
        if len(input_paths) > 1 or (output_path and (os.path.isdir(output_path) or output_path.endswith(os.sep) or output_path.endswith('/'))):
            logging.info("Starting batch processing...")
            processed = 0
            for idx, in_p in enumerate(input_paths, start=1):
                try:
                    result_path = process_single(in_p, output_path, preserve_alpha, out_format, overwrite)
                    logging.info(f"[{idx}/{len(input_paths)}] -> {result_path}")
                    processed += 1
                except GrayscaleError as ge:
                    logging.error(f"Error processing {in_p}: {ge}")
            logging.info(f"Batch finished. {processed}/{len(input_paths)} images processed.")
        else:
            logging.info("Processing single image...")
            result_path = process_single(input_path, output_path, preserve_alpha, out_format, overwrite)
            logging.info(f"Wrote: {result_path}")
    except InputValidationError as e:
        logging.error(str(e))
        sys.exit(2)
    except ImageProcessingError as e:
        logging.error(str(e))
        sys.exit(3)
    except OutputWriteError as e:
        logging.error(str(e))
        sys.exit(4)
    except GrayscaleError as e:
        logging.error(str(e))
        sys.exit(5)
    except Exception as e:
        logging.exception("Unexpected error: %s", e)
        sys.exit(99)

if __name__ == "__main__":
    main()