#!/usr/bin/env python3
"""
LBM Relighting Script - Cross-platform Image Relighting Tool
Process images from input folder and save relighted versions to output folder

Usage Examples:
    python relighting.py                                                   # Default settings
    python relighting.py --num_inference_steps 20                         # Custom quality
    python relighting.py --input /path/to/images --output /path/to/results
    python relighting.py --input ./my_images --output ./results --num_inference_steps 10
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from PIL import Image

from lbm.inference import evaluate, get_model

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def get_device():
    """Detect available device (CUDA GPU, MPS GPU for Mac, or CPU)"""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"✓ GPU detected: {gpu_name}")
        return device
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        logger.info("✓ GPU detected: Apple Metal Performance Shaders (Mac)")
        return device
    else:
        logger.info("⚠ No GPU detected, using CPU (slower processing)")
        return "cpu"


SCRIPT_DIR = Path(__file__).parent.absolute()
DEVICE = get_device()
TORCH_DTYPE = torch.bfloat16 if DEVICE != "cpu" else torch.float32


def setup_directories(input_path, output_dir):
    """Create necessary directories"""
    # Only create output directory (input can be file or folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = SCRIPT_DIR / "ckpts" / "relighting"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def load_model(ckpt_dir, torch_dtype=None):
    """Load the relighting model from HF Hub or local cache"""
    if torch_dtype is None:
        torch_dtype = TORCH_DTYPE
    
    if (ckpt_dir / "config.json").exists():
        logger.info(f"Loading relighting model from cache: {ckpt_dir}")
        model = get_model(str(ckpt_dir), torch_dtype=torch_dtype, device=DEVICE)
    else:
        logger.info("Downloading relighting model from Hugging Face Hub (5.0GB)...")
        logger.info("This may take a few minutes on first run...")
        model = get_model(
            "jasperai/LBM_relighting",
            save_dir=str(ckpt_dir),
            torch_dtype=torch_dtype,
            device=DEVICE,
        )
    return model


def process_images(input_dir, output_dir, num_inference_steps=1):
    """Process images - supports folder or single file"""
    ckpt_dir = setup_directories(input_dir, output_dir)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"LBM Relighting - Processing Images")
    logger.info(f"{'='*60}")
    logger.info(f"Device: {DEVICE.upper()}")
    logger.info(f"Inference steps: {num_inference_steps}")
    logger.info(f"Input:  {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"{'='*60}\n")

    # Get image files
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    image_files = []
    
    if input_dir.is_file():
        # Single image file
        if input_dir.suffix.lower() in image_extensions:
            image_files = [input_dir]
        else:
            logger.error(f"❌ Not a supported image format: {input_dir}")
            return
    elif input_dir.is_dir():
        # Folder - get all images
        for ext in image_extensions:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))
        # Remove duplicates
        image_files = list(set(image_files))
        image_files.sort()
    else:
        logger.error(f"❌ Path not found: {input_dir}")
        return

    if not image_files:
        logger.warning(f"❌ No images found in {input_dir}")
        return

    logger.info(f"Found {len(image_files)} image(s) to process\n")

    # Load model once
    logger.info("Loading model...")
    model = load_model(ckpt_dir)
    logger.info("✓ Model loaded successfully!\n")

    # Process each image
    success_count = 0
    for idx, image_path in enumerate(image_files, 1):
        try:
            logger.info(f"[{idx}/{len(image_files)}] Processing: {image_path.name}")

            # Load image
            source_image = Image.open(image_path).convert("RGB")
            w, h = source_image.size
            logger.info(f"         Input size: {w}x{h}")

            # Generate relighted image
            logger.info(f"         Relighting...")
            output_image = evaluate(model, source_image, num_sampling_steps=num_inference_steps)

            # Save output
            output_name = image_path.stem + "_relighted.jpg"
            output_path = output_dir / output_name
            output_image.save(output_path)
            logger.info(f"         ✓ Saved: {output_name}\n")
            success_count += 1

        except Exception as e:
            logger.error(f"         ✗ Error: {str(e)}\n")
            continue

    # Summary
    logger.info(f"{'='*60}")
    logger.info(f"Processing Complete!")
    logger.info(f"Successfully processed: {success_count}/{len(image_files)} images")
    logger.info(f"Output folder: {output_dir}")
    logger.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="LBM Relighting - Process images with custom input/output paths",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  Process all images in a folder:
    python relighting.py
    python relighting.py --input ./my_images --output ./results

  Process a SINGLE image:
    python relighting.py --input ./input/photo.jpg --output ./results
    python relighting.py --input C:\\Users\\me\\Pictures\\vacation.png --output C:\\Users\\me\\output
    python relighting.py --input ~/photo.jpg --output ~/results

  Custom quality:
    python relighting.py --input ./images --output ./results --num_inference_steps 20

  Mac/Linux:
    python relighting.py --input ~/Pictures/photos --output ~/Pictures/relighted
    python relighting.py --input ~/myphoto.jpg --output ~/output

  Kaggle:
    python relighting.py --input /kaggle/input/photo.jpg --output /kaggle/working
        """
    )
    
    parser.add_argument(
        "--input",
        type=Path,
        default=SCRIPT_DIR / "input",
        help="Input: folder (process all images) OR single image file (default: ./input)"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "output",
        help="Output folder path (default: ./output)"
    )
    
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=1,
        help="Number of inference steps. More = better quality but slower (default: 1, max: 50)"
    )

    args = parser.parse_args()
    
    # Convert to absolute paths
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    
    # Process images
    process_images(input_path, output_dir, num_inference_steps=args.num_inference_steps)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
