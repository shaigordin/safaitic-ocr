# Safaitic OCR - Web Demo Documentation

This directory contains the web-based interactive demo for exploring VLM analysis results.

## Files

- `index.html` - Main demo page with pagination controls
- `data/*.json` - Evaluation results for each model (50 inscriptions, 150 prompts each)
- `inscription_images.zip` - Compressed images for all evaluated inscriptions (612 MB)

## Image Setup

The inscription images are not included directly in the repository due to their size. Instead, they are provided as a compressed zip file.

### Option 1: Extract Images Locally (Recommended for Development)

If you have the images locally at `../data/examples/BES15/`:

```bash
# Create symlink from docs/images to the local images
cd docs
ln -s ../data/examples/BES15 images
```

Then update `index.html` line ~620 to use local images:

```javascript
// Change from placeholder to:
const imgPath = `../images/${siglum}/im${String(i).padStart(7, '0')}.jpg`;
```

### Option 2: Use GitHub Pages with Extracted Images

For GitHub Pages deployment:

1. Extract the zip file:
```bash
cd docs
unzip inscription_images.zip -d images/
```

2. Update `.gitignore` to allow images in docs/images:
```
# In .gitignore, add exception:
!docs/images/
!docs/images/**/*.jpg
```

3. Commit and push the images (warning: 612 MB will be added to repo)

### Option 3: Use GitHub Releases (Recommended for GitHub Pages)

Keep images separate from the main repository:

1. Create a GitHub release
2. Upload `inscription_images.zip` as a release asset
3. Users can download and extract locally
4. Add instructions to the main README

### Option 4: Use External CDN

Host images on a CDN or external service and update the image URLs in `index.html`.

## Features

### Pagination

- Navigate between inscriptions using Previous/Next buttons
- Keyboard shortcuts: Arrow Left/Right keys
- Page counter shows current position (e.g., "1 of 50")
- Automatic scroll to top when changing pages

### Filtering

- **All Inscriptions**: Show all results
- **Successful Only**: Filter to inscriptions with at least one successful prompt
- **Errors Only**: Filter to inscriptions with at least one error

### Search

- Search by inscription siglum (e.g., "BES15 1")
- Real-time filtering as you type

### Model Comparison

Use the model selector to switch between different VLM results:

- **Qwen2.5-VL-7B**: 98.3% success rate (20 inscriptions evaluated)
- **Qwen2-VL-2B**: 100% success rate (smallest model, full evaluation)
- **Qwen2-VL-7B**: 98.0% success rate (full evaluation)
- **Idefics3-8B**: 100% success rate (full evaluation)
- **Pixtral-12B**: 100% success rate (largest model, full evaluation)

## Development

To run locally:

```bash
# Using Python's built-in HTTP server
cd docs
python3 -m http.server 8000

# Then open: http://localhost:8000
```

## Data Format

Each JSON file contains:

```json
{
  "metadata": {
    "timestamp": "YYYYMMDD_HHMMSS",
    "model_name": "model-name",
    "num_inscriptions": 50,
    "num_prompts": 150
  },
  "results": [
    {
      "inscription_siglum": "BES15 1",
      "num_images": 15,
      "ground_truth": {
        "transliteration": "...",
        "translation": "..."
      },
      "prompts": [
        {
          "prompt_name": "description",
          "prompt": "...",
          "response": "...",
          "duration_seconds": 0.5,
          "success": true,
          "error": null
        }
      ]
    }
  ]
}
```

## Image Organization

Images are organized by inscription siglum:

```
images/
  BES15 1/
    im0037246.jpg
    im0037250.jpg
    ...
  BES15 2/
    im0038113.jpg
  BES15 3/
    ...
```

Each inscription may have 1-15 high-resolution images showing different angles and lighting conditions.
