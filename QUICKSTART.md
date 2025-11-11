# Quick Start Guide

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Run VLM Analysis (FREE!)

```bash
# Test mode: 10 inscriptions
python generate_results.py

# All inscriptions
python generate_results.py --all
```

## 3. View Results

```bash
# Option 1: Open directly
open docs/index.html

# Option 2: Use local server
python -m http.server 8000 --directory docs
# Visit: http://localhost:8000
```

## 4. Deploy to GitHub Pages

```bash
git add .
git commit -m "Add VLM analysis results"
git push
```

Then enable GitHub Pages in repository settings:
- Settings → Pages
- Source: Deploy from branch `main`
- Folder: `/docs`

Your site will be live at: `https://YOUR_USERNAME.github.io/safaitic-ocr/`

## VLM Options

**FREE Gradio Spaces** (default):
- LLaVA OneVision (recommended)
- Qwen3-VL
- No API keys needed!

**Local Ollama** (optional):
```bash
ollama pull llama3.2-vision
```

## Python API

```python
from src import GradioSpaceVLM, SafaiticPrompts, get_inscription_data, load_metadata

# Load data
df = load_metadata("metadata/BES15.csv")
inscription = get_inscription_data(df, "data", "BES15 1", load_images=True)

# Initialize VLM (FREE!)
vlm = GradioSpaceVLM()

# Analyze
prompt = SafaiticPrompts.transliteration_attempt()
result = vlm.analyze_image(inscription.images[0], prompt)
print(result['response'])
```

## Troubleshooting

**Slow first request?**
- Gradio Spaces have cold start (30-60s first time)
- Subsequent requests are faster

**Connection errors?**
- Check internet connection
- Try: `python generate_results.py --space qwen3-vl`

**Need help?**
- See full README.md
- Check docs/ folder
- Open an issue on GitHub
