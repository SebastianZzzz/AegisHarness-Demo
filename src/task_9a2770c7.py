import os
import sys
import argparse
import logging
from typing import Optional, List, Tuple
from PIL import Image, ImageOps, UnidentifiedImageError

# Exceptions
class InputValidationError(Exception):
    pass

class ImageProcessingError(Exception):
    pass

class OutputWriteError(Exception):
    pass

# Constants
SUPPORTED_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.webp', '.bmp', '.gif'
}
DEFAULT_SUFFIX = "_gray"
DEFAULT_MAX_PIXELS = 10000 * 10000  # ~100 MP, decompression bomb guard


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _infer_format_from_path(path: str, explicit_format: Optional[str]) -> Optional[str]:
    if explicit_format:
        return explicit_format.upper()
    # Infer from extension
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext in ('.jpg', '.jpeg'):
        return 'JPEG'
    if ext == '.png':
        return 'PNG'
    if ext == '.webp':
        return 'WEBP'
    if ext in ('.tiff', '.tif'):
        return 'TIFF'
    if ext == '.bmp':
        return 'BMP'
    if ext == '.gif':
        return 'GIF'
    return None


def load_image(path: str) -> Image.Image:
    if not path:
        raise InputValidationError("Input path is empty.")
    if not os.path.exists(path):
        raise InputValidationError(f"Input path does not exist: {path}")
    if not os.path.isfile(path):
        raise InputValidationError(f"Input path is not a file: {path}")

    try:
        with Image.open(path) as img:
            # Apply orientation correction as early as possible
            img = ImageOps.exif_transpose(img)
            # Load to avoid lazy IO and to check size against bombs
            img.load()
            image = img.copy()  # work with a separate instance
    except UnidentifiedImageError as e:
        raise ImageProcessingError(f"Cannot identify image: {path}. Reason: {e}") from e
    except (OSError, ValueError) as e:
        raise ImageProcessingError(f"Failed to load image: {path}. Reason: {e}") from e

    # Simple decompression bomb guard
    w, h = image.size
    if w * h > DEFAULT_MAX_PIXELS:
        raise ImageProcessingError(f"Image too large: {w}x{h} pixels exceeds limit.")

    return image


def single_image_to_grayscale(image: Image.Image, preserve_alpha: bool) -> Image.Image:
    """
    Convert a single image to grayscale.
    - If image has no alpha: return L mode image.
    - If image has alpha and preserve_alpha is True: return RGBA with R/G/B = grayscale and original alpha.
    - If image has alpha and preserve_alpha is False: return L mode grayscale.
    - For palettes (mode 'P'): convert to RGBA first to preserve alpha information when requested.
    """
    if image is None:
        raise ValueError("Input image is None.")

    # Ensure we can access alpha information
    if image.mode == 'P':
        image = image.convert('RGBA')  # palette with possible transparency

    has_alpha = image.mode in ('RGBA', 'LA')
    grayscale = image.convert('L')

    if not has_alpha or not preserve_alpha:
        return grayscale

    # Preserve alpha by merging channels into RGBA
    # Use the alpha channel from the original image
    try:
        alpha = image.split()[-1]  # last channel is alpha in RGBA/LA
    except Exception:
        # Fallback: if something unexpected, return grayscale without alpha
        return grayscale

    return Image.merge('RGBA', (grayscale, grayscale, grayscale, alpha))


def save_image(image: Image.Image, path: str, format: Optional[str] = None, **save_params) -> None:
    if image is None:
        raise ValueError("Cannot save a None image.")

    _ensure_dir(path)
    try:
        img_format = _infer_format_from_path(path, format)
        if img_format:
            image.save(path, format=img_format, **save_params)
        else:
            image.save(path, **save_params)
    except (OSError, ValueError) as e:
        raise OutputWriteError(f"Failed to write image to {path}. Reason: {e}") from e


def determine_output_path(input_path: str, output_dir: Optional[str], suffix: str, out_format: Optional[str]) -> str:
    base_name = os.path.basename(input_path)
    name, _ = os.path.splitext(base_name)
    ext = os.path.splitext(input_path)[1] if not out_format else f".{out_format}"
    if not ext.startswith('.'):
        ext = f".{ext}"
    if output_dir:
        _ensure_dir(output_dir)
        return os.path.join(output_dir, f"{name}{suffix}{ext}")
    # Default: same directory as input
    dirn = os.path.dirname(os.path.abspath(input_path))
    return os.path.join(dirn, f"{name}{suffix}{ext}")


def collect_image_paths(input_dir: str, recursive: bool) -> List[str]:
    image_paths = []
    if not os.path.isdir(input_dir):
        return image_paths
    if recursive:
        for root, _, files in os.walk(input_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    image_paths.append(os.path.join(root, f))
    else:
        for f in os.listdir(input_dir):
            full = os.path.join(input_dir, f)
            if os.path.isfile(full):
                ext = os.path.splitext(f)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    image_paths.append(full)
    return image_paths


def process_single(input_path: str, output_path: Optional[str], preserve_alpha: bool,
                   out_format: Optional[str], overwrite: bool, logger: logging.Logger) -> Optional[str]:
    logger.info("Processing file: %s", input_path)
    image = load_image(input_path)
    result = single_image_to_grayscale(image, preserve_alpha)

    if output_path is None:
        output_path = determine_output_path(input_path, None, DEFAULT_SUFFIX, out_format)

    if os.path.exists(output_path) and not overwrite:
        logger.warning("Output exists and overwrite is disabled: %s", output_path)
        return None

    save_image(result, output_path, format=out_format)
    logger.info("Saved grayscale image to: %s", output_path)
    return output_path


def process_batch(input_dir: str, output_dir: Optional[str], suffix: str, preserve_alpha: bool,
                  out_format: Optional[str], overwrite: bool, recursive: bool,
                  logger: logging.Logger) -> List[str]:
    results: List[str] = []
    paths = collect_image_paths(input_dir, recursive)
    if not paths:
        logger.warning("No input images found in directory: %s", input_dir)
        return results

    if output_dir is None:
        output_dir = input_dir
    _ensure_dir(output_dir)

    for in_path in paths:
        out_path = determine_output_path(in_path, output_dir, suffix, out_format)
        if os.path.exists(out_path) and not overwrite:
            logger.info("Skipping existing file (no overwrite): %s", out_path)
            continue
        try:
            image = load_image(in_path)
            result = single_image_to_grayscale(image, preserve_alpha)
            save_image(result, out_path, format=out_format)
            logger.info("Processed: %s -> %s", in_path, out_path)
            results.append(out_path)
        except (InputValidationError, ImageProcessingError, OutputWriteError) as e:
            logger.error("Failed to process %s: %s", in_path, e)
            continue
    return results


def setup_logger(level: int) -> logging.Logger:
    logger = logging.getLogger("pillow-gray")
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert images to grayscale using Pillow. Supports optional alpha preservation and batch processing."
    )
    parser.add_argument("-i", "--input", required=True, help="Input file path or directory containing images.")
    parser.add_argument("-o", "--output", required=False,
                        help="Output file path or directory. If processing a directory and --output is a directory, outputs go there.")
    parser.add_argument("-p", "--preserve-alpha", action="store_true",
                        help="Preserve alpha channel when converting to grayscale.")
    parser.add_argument("--batch", action="store_true", help="Process a directory of images (batch mode).")
    parser.add_argument("--recursive", action="store_true", help="Recursively process subdirectories in batch mode.")
    parser.add_argument("-f", "--format", dest="format", required=False,
                        help="Output image format (e.g., jpeg, png). If omitted, inferred from output extension or input.")
    parser.add_argument("--overwrite", "-w", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode; minimal logs.")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode; detailed logs.")
    return parser.parse_args()


def main():
    args = parse_args()
    # Logging level
    if args.quiet:
        log_level = logging.WARNING
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logger = setup_logger(log_level)

    input_path = args.input
    output_path = args.output
    preserve_alpha = bool(args.preserve_alpha)
    batch_mode = bool(args.batch)
    recursive = bool(args.recursive)
    out_format = (args.format or None)
    overwrite = bool(args.overwrite)

    # Validate input path
    if not os.path.exists(input_path):
        logger.error("Input path does not exist: %s", input_path)
        sys.exit(2)

    try:
        if batch_mode:
            if not os.path.isdir(input_path):
                raise InputValidationError("Batch mode requires input to be a directory.")
            if output_path and not os.path.isdir(output_path):
                # If an output path is provided and exists as a file, that's odd; create if possible
                if os.path.exists(output_path) and not os.path.isdir(output_path):
                    raise InputValidationError("Output path provided for batch mode must be a directory when processing multiple files.")
                # If not exists, we'll try to create as directory later
            results = process_batch(input_path, output_path, DEFAULT_SUFFIX, preserve_alpha,
                                    out_format, overwrite, recursive, logger)
            logger.info("Batch processing finished. %d files written.", len(results))
        else:
            # Single image mode
            if os.path.isdir(input_path):
                raise InputValidationError("Single-file mode requires a file input, not a directory. Use --batch for directories.")
            # Determine explicit output for single file
            if output_path:
                if os.path.isdir(output_path):
                    # Output is a directory; put a file next to input with suffix
                    out_path = determine_output_path(input_path, output_path, DEFAULT_SUFFIX, out_format)
                else:
                    out_path = output_path
            else:
                out_path = determine_output_path(input_path, None, DEFAULT_SUFFIX, out_format)

            processed = process_single(input_path, out_path, preserve_alpha, out_format, overwrite, logger)
            if processed:
                logger.info("Output written to: %s", processed)
            else:
                logger.info("Output skipped due to existing file and no overwrite.")
    except (InputValidationError, ImageProcessingError, OutputWriteError) as e:
        logger.error("Error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()