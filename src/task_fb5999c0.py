import os
import sys
import shutil
import tempfile
from typing import Optional, Tuple, List
from PIL import Image, ImageChops

# Optional numpy for accelerated grayscale (luma) computations
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False

# ------------------------------------------------------------
# Defaults and configuration
# ------------------------------------------------------------

class Defaults:
    METHOD: str = 'luma'          # 'luma', 'average', 'lum'
    PRESERVE_ALPHA: bool = False
    PRESERVE_METADATA: bool = True
    OUTPUT_FORMAT: Optional[str] = None  # e.g. 'png', 'jpg'. If None, use input ext.
    SUFFIX: str = '_gray'
    MAX_PIXELS: int = 25_000_000     # max pixels before downscale
    VERBOSITY: int = 1
    OUTPUT_QUIET: bool = False

# Supported image extensions
SUPPORTED_EXTS = {
    '.png', '.jpg', '.jpeg', '.webp', '.tiff', '.tif', '.bmp', '.gif', '.jp2'
}

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def is_supported_image_path(path: str) -> bool:
    if not path:
        return False
    _, ext = os.path.splitext(path)
    return ext.lower() in SUPPORTED_EXTS

def ensure_dir_exists(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if not d:
        return
    os.makedirs(d, exist_ok=True)

def derive_output_path(input_path: str,
                       input_root: str,
                       output_root: Optional[str],
                       suffix: str = Defaults.SUFFIX,
                       ext: Optional[str] = None) -> str:
    """
    Derive an output path preserving relative structure from input_root.
    If output_root is None, outputs are placed next to input files.
    """
    rel = os.path.relpath(input_path, input_root)
    rel_dir = os.path.dirname(rel)
    base = os.path.basename(rel)
    name, _ = os.path.splitext(base)
    out_ext = ext if ext else os.path.splitext(base)[1]
    if not out_ext.startswith('.'):
        out_ext = f'.{out_ext}'

    out_name = f"{name}{suffix}{out_ext}"
    if output_root:
        out_dir = os.path.join(output_root, rel_dir)
    else:
        out_dir = os.path.join(input_root, rel_dir)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, out_name)

def downscale_if_large(image: Image.Image, max_pixels: int) -> Image.Image:
    w, h = image.size
    total = w * h
    if total <= max_pixels:
        return image
    scale = (max_pixels / total) ** 0.5
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return image.resize((new_w, new_h), Image.LANCZOS)

def _weighted_luma_numpy(rgb_image: Image.Image) -> Image.Image:
    arr = np.asarray(rgb_image, dtype=np.float32)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    lum = (0.299 * r + 0.587 * g + 0.114 * b).astype('uint8')
    return Image.fromarray(lum, mode='L')

def _weighted_luma_fallback(r: Image.Image, g: Image.Image, b: Image.Image) -> Image.Image:
    # Fallback using per-channel weighted point operations
    r2 = r.point(lambda p: int(0.299 * p))
    g2 = g.point(lambda p: int(0.587 * p))
    b2 = b.point(lambda p: int(0.114 * p))
    tmp = ImageChops.add(r2, g2)
    lum = ImageChops.add(tmp, b2)
    return lum

def _average_fallback(r: Image.Image, g: Image.Image, b: Image.Image) -> Image.Image:
    r2 = r.point(lambda p: p // 3)
    g2 = g.point(lambda p: p // 3)
    b2 = b.point(lambda p: p // 3)
    tmp = ImageChops.add(r2, g2)
    lum = ImageChops.add(tmp, b2)
    return lum

# ------------------------------------------------------------
# Core image processing
# ------------------------------------------------------------

def grayscale_image(image: Image.Image,
                    method: str = 'luma',
                    preserve_alpha: bool = False) -> Image.Image:
    """
    Convert an image to grayscale using the specified method.
    If preserve_alpha is True and the input has an alpha channel, the output will be RGBA
    with grayscale RGB channels and preserved alpha channel.
    Supported methods: 'luma' (weighted by 0.299/0.587/0.114), 'average', 'lum'/'luminance'
    """
    method = method.lower()
    if method not in {'luma', 'average', 'lum', 'luminance'}:
        raise ValueError(f"Unsupported grayscale method: {method}")

    has_alpha = 'A' in image.getbands()

    if preserve_alpha and has_alpha:
        # Work in RGBA, preserve alpha
        rgba = image.convert('RGBA')
        r, g, b, a = rgba.split()

        if method in {'lum', 'luminance', 'luma'}:
            if NUMPY_AVAILABLE:
                lum = _weighted_luma_numpy(rgba.convert('RGB'))
            else:
                lum = _weighted_luma_fallback(r, g, b)
        else:  # 'average'
            lum = _average_fallback(r, g, b)

        out = Image.merge('RGBA', (lum, lum, lum, a))
        return out
    else:
        rgb = image.convert('RGB')
        r, g, b = rgb.split()

        if method in {'lum', 'luminance', 'luma'}:
            if NUMPY_AVAILABLE:
                lum = _weighted_luma_numpy(rgb)
            else:
                lum = _weighted_luma_fallback(r, g, b)
        else:  # 'average'
            lum = _average_fallback(r, g, b)

        # Return single-channel grayscale
        return lum

# ------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------

def load_image(path: str) -> Image.Image:
    """
    Safely load an image from disk. Returns a Pillow Image in memory.
    Uses a copy to ensure the file can be closed.
    """
    try:
        with Image.open(path) as img:
            img.load()
            return img.copy()
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {path}")
    except OSError as e:
        raise OSError(f"Cannot load image '{path}': {e}")

def save_image(image: Image.Image,
               path: str,
               format: Optional[str] = None,
               preserve_metadata: bool = True) -> None:
    """
    Save an image atomically to disk.
    - Writes to a temporary file in the same directory and then renames.
    - If preserve_metadata is True and image.info contains 'exif', it's preserved.
    - If format is None, Pillow infers from the extension.
    """
    ensure_dir_exists(path)
    exif = image.info.get('exif') if preserve_metadata else None
    # For atomic write, use temporary file
    dirn = os.path.dirname(os.path.abspath(path))
    base = os.path.basename(path)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=base + '.', dir=dirn)
    os.close(tmp_fd)

    try:
        if format:
            image.save(tmp_path, format=format, exif=exif)
        else:
            image.save(tmp_path, exif=exif)
        os.replace(tmp_path, path)
    except Exception as e:
        # Cleanup temp file on failure
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise IOError(f"Failed to save image to '{path}': {e}")

# ------------------------------------------------------------
# Simple validators and CLI helpers
# ------------------------------------------------------------

def _format_path_handling_input(input_path: str) -> Tuple[str, bool]:
    """
    Return (path, is_dir)
    """
    if not input_path:
        raise ValueError("Input path is empty.")
    is_dir = os.path.isdir(input_path)
    if not is_dir and not os.path.isfile(input_path):
        raise ValueError(f"Input path does not exist: {input_path}")
    return input_path, is_dir

def _collect_input_files(input_path: str) -> List[str]:
    """
    If input_path is a directory, collect all supported image files recursively.
    If input_path is a file, return [input_path].
    """
    path, is_dir = _format_path_handling_input(input_path)
    if is_dir:
        files = []
        for root, _, filenames in os.walk(path):
            for name in filenames:
                full = os.path.join(root, name)
                if is_supported_image_path(full):
                    files.append(full)
        return sorted(files)
    else:
        return [path]

def _natural_output_path(input_file: str,
                       input_root: str,
                       output_root: Optional[str],
                       suffix: str = Defaults.SUFFIX,
                       ext: Optional[str] = None) -> str:
    if output_root and os.path.isdir(output_root):
        return derive_output_path(input_file, input_root, output_root, suffix, ext)
    else:
        # If output_root is None or not a dir, attempt to use input_root (file's dir)
        return derive_output_path(input_file, input_root, None, suffix, ext)

def _ensure_output_path_for_file(in_path: str,
                               out_path: Optional[str],
                               suffix: str,
                               out_ext: Optional[str]) -> str:
    if out_path:
        if os.path.isdir(out_path):
            # derive a path in the directory
            in_root = os.path.dirname(in_path)
            return derive_output_path(in_path, in_root, out_path, suffix, out_ext)
        else:
            # direct file path
            return out_path
    else:
        # default: same dir as input, with suffix
        in_root = os.path.dirname(in_path)
        return derive_output_path(in_path, in_root, None, suffix, out_ext)

# ------------------------------------------------------------
# Testing utility (optional CLI switch)
# ------------------------------------------------------------

def run_internal_tests():
    """
    Very small internal smoke tests to ensure core grayscale functionality works.
    Intended to be run via CLI flag --run-tests.
    """
    from io import BytesIO
    errors = 0

    # Create a simple red image
    img = Image.new('RGB', (100, 100), color=(255, 0, 0))
    try:
        out = grayscale_image(img, method='luma', preserve_alpha=False)
        assert out.mode == 'L'
        assert out.size == img.size
    except Exception as e:
        print(f"Test 1 failed: {e}")
        errors += 1

    # Test average
    try:
        out = grayscale_image(img, method='average', preserve_alpha=False)
        assert out.mode == 'L'
        assert out.size == img.size
    except Exception as e:
        print(f"Test 2 failed: {e}")
        errors += 1

    # Test alpha preservation
    img_rgba = Image.new('RGBA', (80, 60), color=(0, 128, 255, 128))
    try:
        out = grayscale_image(img_rgba, method='luma', preserve_alpha=True)
        assert out.mode == 'RGBA'
        assert out.size == img_rgba.size
    except Exception as e:
        print(f"Test 3 failed: {e}")
        errors += 1

    # Test with numpy fallback path (if numpy not available)
    if not NUMPY_AVAILABLE:
        try:
            out = grayscale_image(img, method='luma', preserve_alpha=False)
            assert out.mode == 'L'
        except Exception as e:
            print(f"Test 4 failed: {e}")
            errors += 1

    if errors == 0:
        print("All internal tests passed.")
    else:
        print(f"{errors} internal test(s) failed.")

# ------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------

def _parse_args(argv: List[str]):
    import argparse

    parser = argparse.ArgumentParser(
        prog="grayscale_tool",
        description="Convert images to grayscale using Pillow with optional alpha and metadata handling."
    )

    parser.add_argument("input_path",
                        help="Input image file or directory containing images.")
    parser.add_argument("--output-path", dest="output_path",
                        help="Output file or directory. If a directory is given, outputs are placed there with suffix.")
    parser.add_argument("--method", dest="method",
                        default=Defaults.METHOD,
                        choices=['luma','average','lum','luminance'],
                        help="Grayscale method: luma (weights), average, lum/luminance (same as Pillow's L).")
    parser.add_argument("--preserve-alpha", dest="preserve_alpha",
                        action="store_true", default=Defaults.PRESERVE_ALPHA,
                        help="Preserve alpha channel when input has alpha (output will be RGBA).")
    parser.add_argument("--no-preserve-alpha", dest="preserve_alpha",
                        action="store_false", help="Do not preserve alpha (output is grayscale without alpha).")
    parser.add_argument("--preserve-metadata", dest="preserve_metadata",
                        action="store_true", default=Defaults.PRESERVE_METADATA,
                        help="Preserve image metadata (EXIF/IPTC).")
    parser.add_argument("--no-preserve-metadata", dest="preserve_metadata",
                        action="store_false", help="Do not preserve metadata.")
    parser.add_argument("--output-format", dest="output_format",
                        default=Defaults.OUTPUT_FORMAT,
                        help="Output format extension without dot (e.g., png, jpg). If omitted, uses input extension.")
    parser.add_argument("--suffix", dest="suffix",
                        default=Defaults.SUFFIX,
                        help="Suffix to append to output filenames (default: _gray).")
    parser.add_argument("--batch", dest="batch", action="store_true",
                        help="Explicitly enable batch processing (for directories).")
    parser.add_argument("--max-pixels", dest="max_pixels", type=int,
                        default=Defaults.MAX_PIXELS,
                        help="Maximum number of pixels allowed per image. Images larger will be downscaled.")
    parser.add_argument("--quiet", dest="quiet", action="store_true",
                        help="Minimal output.")
    parser.add_argument("--verbose", dest="verbose", action="store_true",
                        help="Verbose output.")
    parser.add_argument("--run-tests", dest="run_tests", action="store_true",
                        help="Run internal grayscale function tests and exit.")

    return parser.parse_args(argv)

def _log(message: str, verbose: int = 1, quiet: bool = False, level: int = 1) -> None:
    if quiet:
        return
    if verbose >= level:
        print(message)

def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    args = _parse_args(argv)

    # Run internal tests if requested
    if getattr(args, 'run_tests', False):
        run_internal_tests()
        return 0

    input_path = args.input_path
    out_root = args.output_path
    method = args.method if args.method else Defaults.METHOD
    preserve_alpha = bool(args.preserve_alpha)
    preserve_metadata = bool(args.preserve_metadata)
    output_format = args.output_format
    suffix = args.suffix if args.suffix else Defaults.SUFFIX
    max_pixels = int(args.max_pixels) if args.max_pixels else Defaults.MAX_PIXELS
    verbose = 1 if args.verbose else 0
    quiet = bool(args.quiet)

    _log("Starting grayscale conversion...", verbose=verbose, quiet=quiet, level=1)

    # Gather input files
    try:
        input_path_resolved = os.path.abspath(input_path)
        is_dir = os.path.isdir(input_path_resolved)
        if not is_dir and not os.path.isfile(input_path_resolved):
            print(f"Input path not found: {input_path}")
            return 2
        # If input is directory, collect all supported images
        files_to_process = _collect_input_files(input_path_resolved) if is_dir else [input_path_resolved]
        if not files_to_process:
            print("No supported image files found in input path.")
            return 0
    except Exception as e:
        print(f"Error validating input: {e}")
        return 2

    total = len(files_to_process)
    processed = 0

    # For each input file, determine output path and process
    for in_file in files_to_process:
        try:
            # Downscale if needed
            img = load_image(in_file)
            img = downscale_if_large(img, max_pixels)

            # Compute output path
            if out_root:
                # If output path is a directory, derive path inside it
                if os.path.isdir(out_root):
                    out_path = derive_output_path(in_file, os.path.dirname(in_file),
                                                  out_root, suffix, output_format)
                else:
                    # If output_path is a file, and there is only one input, use it
                    if total == 1:
                        out_path = os.path.abspath(out_root)
                    else:
                        # When multiple inputs and a file path is given, refuse gracefully
                        print("Error: When processing multiple files, --output-path must be a directory.")
                        return 3
            else:
                out_path = derive_output_path(in_file, os.path.dirname(in_file), None, suffix, output_format)

            # Ensure extension and format
            if output_format:
                ext = '.' + output_format if not output_format.startswith('.') else output_format
            else:
                ext = os.path.splitext(out_path)[1]

            if not out_path.endswith(ext):
                out_path = os.path.splitext(out_path)[0] + ext

            _log(f"Processing: {in_file} -> {out_path}", verbose=1, quiet=quiet, level=1)

            # Process
            gray = grayscale_image(img, method=method, preserve_alpha=preserve_alpha)

            # Save
            save_image(gray, out_path, format=(output_format if output_format else None),
                       preserve_metadata=preserve_metadata)

            processed += 1
            if Defaults.VERVOSITY if hasattr(Defaults, 'VERBOSITY') else True:
                pass
            if (total > 0) and not quiet:
                pct = int((processed / total) * 100)
                _log(f"Progress: {processed}/{total} ({pct}%)", verbose=1, quiet=quiet, level=1)
        except Exception as e:
            print(f"Error processing '{in_file}': {e}")
            # Continue with next file
            continue

    _log(f"Finished processing {processed} file(s).", verbose=1, quiet=quiet, level=1)
    return 0

# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(130)