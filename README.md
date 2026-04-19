# ICAO_RE

Minimal LBM image relighting project.

## What it does

This project runs a relighting model on one or more images and saves the relighted output images.

## Files

- `relighting.py` - main script
- `input/` - default input images directory
- `output/` - default output images directory
- `ckpts/relighting/` - model cache directory

## Usage

### Run on Windows

```powershell
cd "C:\Users\amine\OneDrive\Desktop\TAHA BIKKI LAB\ICAO_RE"
.venv\Scripts\activate
python relighting.py --input .\input --output .\output --num_inference_steps 10
```

### Run on macOS / Linux

```bash
cd "/path/to/ICAO_RE"
source .venv/bin/activate
python relighting.py --input ./input --output ./output --num_inference_steps 10
```

### Run on Kaggle

```bash
python relighting.py --input /kaggle/input/your_images --output /kaggle/working --num_inference_steps 10
```

### Process a single image file

```bash
python relighting.py --input ./input/in2.png --output ./output --num_inference_steps 10
```

## Notes

- `--input` accepts either:
  - a folder (`./input`) to process all images inside
  - a single image file path (`./input/in2.png`)
- `--output` must be a folder path.
- The first run downloads the model automatically into `ckpts/relighting/`.
- The script detects GPU automatically when PyTorch is installed with CUDA support.
- If GPU is not available or PyTorch is CPU-only, it falls back to CPU.

## Environment

Install required packages in the virtual environment first. Example:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If the repo does not include `requirements.txt`, install the project dependencies manually or use the existing environment.

## Git push

If you want to publish this repository on GitHub:

```bash
git init
git add .
git commit -m "Initial relighting project"
git branch -M master
git remote add origin https://github.com/tahabikki/ICAO_RE.git
git push -u origin master
```
