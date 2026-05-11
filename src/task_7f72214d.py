import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict

from PIL import Image

# Global configuration
MAX_PIXELS = 20_000_000  # safety cap to avoid huge memory usage
SUPPORTED_EXTS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp', '.avif'
}


def setup_logging(verbose_level: int, quiet: bool) -> None:
    if quiet:
        level = logging.WARNING
    else:
        # 0 -> INFO, 1+ -> DEBUG for verbose
        level = logging.DEBUG if verbose_level > 0 else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        stream=sys.stdout
    )


def is_within_base(path: Path, base_dir: Optional[Path]) -> bool:
    if base_dir is None:
        return True
    try:
        path.resolve().relative_to(base_dir.resolve())
        return True
    except Exception:
        return False


def derive_output_path(input_path: Path, suffix: str = "_gray", override_ext: Optional[str] = None) -> Path:
    # Determine extension
    ext = override_ext if override_ext else input_path.suffix
    if not ext:
        ext = ".png"  # fallback
    if not ext.startswith("."):
        ext = "." + ext
    return input_path.parent / (input_path.stem + suffix + ext)


def derive_output_path_for_single(input_path: Path,
                                output_arg: Optional[Path],
                                suffix: str = "_gray",
                                format_override: Optional[str] = None) -> Path:
    """
    Determine the final output path for a single-file operation.
    - If output_arg is None: derive a file in the input's directory.
    - If output_arg is an existing directory: place derived file inside it.
    - If output_arg is a file (has suffix): use it as the exact output path.
    - If output_arg is a non-existing path with no suffix: treat as directory and place derived file inside it.
    """
    derived = derive_output_path(input_path, suffix, format_override)

    if output_arg is None:
        return derived

    # If output_arg exists
    if output_arg.exists():
        if output_arg.is_dir():
            return output_arg / derived.name
        else:
            return output_arg

    # If not existing yet
    if output_arg.suffix:
        # Treat as explicit file path
        return output_arg
    else:
        # Treat as directory
        return output_arg / derived.name


def validate_input_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Input path is not a file: {path}")
    # Basic attempt to open as image to validate
    try:
        with Image.open(path) as img:
            img.verify()  # may raise
    except Exception as e:
        raise ValueError(f"Input file is not a readable image: {path}") from e


def load_and_preprocess(input_path: Path, max_pixels: int = MAX_PIXELS) -> Tuple[Image.Image, Optional[bytes]]:
    """
    Loads image, validates and downscales if needed to keep memory usage reasonable.
    Returns the PIL Image and optional EXIF data (bytes) if available.
    """
    exif_bytes: Optional[bytes] = None
    with Image.open(input_path) as img:
        exif_bytes = img.info.get('exif')
        img_copy = img.copy()  # detach from file handle

    w, h = img_copy.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        logging.info(f"Downscaling image from ({w}x{h}) to ({new_w}x{new_h}) to fit memory limits.")
        img_copy = img_copy.resize((new_w, new_h), Image.LANCZOS)

    return img_copy, exif_bytes


def grayscale(image: Image.Image) -> Image.Image:
    if image.mode == 'L':
        return image
    return image.convert('L')


def save_image(image: Image.Image,
               path: Path,
               format_override: Optional[str] = None,
               exif: Optional[bytes] = None,
               base_dir: Optional[Path] = None) -> Path:
    # Ensure safe base directory
    if base_dir is not None:
        if not is_within_base(path, base_dir):
            raise PermissionError(f"Output path is outside the allowed base directory: {path} (base: {base_dir})")

    path.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs: Dict[str, object] = {}
    if format_override is not None:
        save_kwargs['format'] = format_override
    if exif is not None:
        save_kwargs['exif'] = exif

    try:
        image.save(path, **save_kwargs)
    except Exception as e:
        raise IOError(f"Failed to save image to {path}: {e}") from e
    return path


def process_single(input_path: Path,
                   output_path: Path,
                   preserve_exif: bool,
                   format_override: Optional[str],
                   base_dir: Optional[Path]) -> Tuple[bool, str]:
    try:
        img, exif = load_and_preprocess(input_path)
        gray = grayscale(img)

        exif_to_use = exif if preserve_exif else None
        final_path = save_image(gray, output_path, format_override, exif_to_use, base_dir)

        return True, f"Processed: {input_path} -> {final_path}"
    except Exception as e:
        logging.error(f"Failed to process {input_path}: {e}")
        return False, f"Error processing {input_path}: {e}"


def discover_inputs(directory: Path, recursive: bool, exclude_dir: Optional[Path] = None) -> List[Path]:
    if recursive:
        candidates = directory.rglob('*')
    else:
        candidates = directory.glob('*')
    imgs: List[Path] = []
    for p in candidates:
        if exclude_dir is not None:
            try:
                excl = exclude_dir.resolve()
                pr = p.resolve()
                if pr == excl or excl in pr.parents:
                    continue
            except Exception:
                pass
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            imgs.append(p)
    return sorted(imgs)


def process_batch(batch_dir: Path,
                  output_dir: Path,
                  recursive: bool,
                  preserve_exif: bool,
                  format_override: Optional[str],
                  base_dir: Optional[Path],
                  suffix: str = "_gray",
                  summary_json: bool = False) -> int:
    inputs = discover_inputs(batch_dir, recursive, exclude_dir=output_dir)
    total = len(inputs)
    if total == 0:
        logging.info("No input images found in batch directory.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, str]] = []
    success_count = 0
    for idx, in_path in enumerate(inputs, start=1):
        out_path = derive_output_path_for_single(in_path, output_dir, suffix=suffix, format_override=format_override)

        # safety: ensure within base_dir if specified
        try:
            ok, msg = process_single(in_path, out_path, preserve_exif, format_override, base_dir)
        except Exception as e:
            ok = False
            msg = f"Error processing {in_path}: {e}"

        results.append({
            "input": str(in_path),
            "output": str(out_path),
            "status": "success" if ok else "error",
            "message": msg
        })
        if ok:
            success_count += 1
        logging.info(f"[{idx}/{total}] {msg}")

    # Summary output
    if summary_json:
        summary = {
            "total": total,
            "success": success_count,
            "failed": total - success_count,
            "details": results
        }
        print(json.dumps(summary, indent=2))
    else:
        logging.info(f"Batch finished: {success_count}/{total} succeeded.")
        for r in results:
            if r["status"] == "success":
                logging.info(f"OK: {r['input']} -> {r['output']}")
            else:
                logging.error(f"ERR: {r['input']} -> {r['message']}")

    return 0 if total == success_count else 2


def main():
    parser = argparse.ArgumentParser(
        description="Convert an image to grayscale using Pillow with optional EXIF preservation and batch support."
    )

    # Modes
    parser.add_argument('-i', '--input', dest='input_path', type=str,
                        help='Path to input image (single-file mode)')
    parser.add_argument('-o', '--output', dest='output_path', type=str,
                        help='Output path. In batch mode, this is interpreted as an output directory.')
    parser.add_argument('-B', '--base-dir', dest='base_dir', type=str,
                        help='Base directory to constrain output paths (security).')

    group_preserve = parser.add_mutually_exclusive_group()
    group_preserve.add_argument('--preserve-exif', dest='preserve_exif', action='store_true',
                                help='Preserve EXIF metadata in the output (default: False)')
    group_preserve.add_argument('--no-preserve-exif', dest='preserve_exif', action='store_false',
                                help='Do not preserve EXIF metadata (default: False)')
    parser.set_defaults(preserve_exif=False)

    batch_group = parser.add_mutually_exclusive_group()
    batch_group.add_argument('-b', '--batch', dest='batch_dir', type=str,
                             help='Batch mode: process all images in the given directory')
    # If both --input and --batch provided, we prioritize batch mode as per design
    parser.add_argument('-R', '--recursive', action='store_true', help='In batch mode, process directories recursively')
    parser.add_argument('-J', '--summary-json', dest='summary_json', action='store_true',
                        help='Output batch summary as JSON (implies batch mode)')

    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='Increase verbosity (can be used multiple times)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Quiet mode; minimal output')

    parser.add_argument('-f', '--format', dest='format', type=str,
                        help='Output image format override (e.g., PNG, JPEG)')

    parser.add_argument('--version', action='store_true', help='Print version information and exit')

    args = parser.parse_args()

    if args.version:
        print("grayscale_tool (Pillow) version 0.1.0")
        return 0

    # Setup logging
    setup_logging(args.verbose, args.quiet)

    # Determine mode
    base_dir = Path(args.base_dir) if args.base_dir else None
    if args.batch_dir:
        batch_dir = Path(args.batch_dir)
        if not batch_dir.exists():
            logging.error(f"Batch directory does not exist: {batch_dir}")
            return 1
        if not batch_dir.is_dir():
            logging.error(f"Batch path is not a directory: {batch_dir}")
            return 1

        # Determine output directory
        if args.output_path:
            output_dir = Path(args.output_path)
        else:
            output_dir = batch_dir / "gray"
        # Ensure all exterior constraints
        if not is_within_base(output_dir, base_dir):
            logging.error("Output directory is outside the allowed base directory.")
            return 1

        # Process batch
        return process_batch(
            batch_dir=batch_dir,
            output_dir=output_dir,
            recursive=args.recursive,
            preserve_exif=args.preserve_exif,
            format_override=args.format,
            base_dir=base_dir,
            suffix="_gray",
            summary_json=args.summary_json
        )

    else:
        # Single file mode
        if not args.input_path:
            logging.error("No input path provided. Use --input for single mode or --batch for batch mode.")
            return 1

        input_path = Path(args.input_path)
        try:
            validate_input_path(input_path)
        except Exception as e:
            logging.error(str(e))
            return 1

        # Resolve output path
        output_path = derive_output_path_for_single(input_path, Path(args.output_path) if args.output_path else None,
                                                    suffix="_gray", format_override=args.format)

        # Ensure safety
        if base_dir is not None and not is_within_base(output_path, base_dir):
            logging.error("Output path is outside the allowed base directory.")
            return 1

        ok, msg = process_single(input_path, output_path, args.preserve_exif, args.format, base_dir)
        if ok:
            logging.info(msg)
            return 0
        else:
            logging.error(msg)
            return 2


if __name__ == "__main__":
    sys.exit(main())