# Safaitic OCR - VLM Testing Pipeline

A comprehensive pipeline for testing Vision Language Models (VLMs) on ancient Safaitic inscriptions. Features **FREE VLM inference** via HuggingFace Gradio Spaces and a web application for viewing results.

## Overview

This toolkit enables researchers to:
- Test VLM performance on script recognition, transliteration, and translation using **FREE** Gradio Spaces
- Compare model outputs against scholarly ground truth from OCIANA database
- Run systematic batch evaluations across multiple inscriptions
- View results in an interactive web application
- Collaborate with non-coding scholars through Jupyter notebooks

## Features

- **FREE VLM Inference**: Uses HuggingFace Gradio Spaces (no API keys, no costs)
- **Multiple VLM Options**: LLaVA OneVision, Qwen3-VL, or local Ollama models
- **Web Application**: Beautiful results viewer deployed on GitHub Pages
- **Modular Architecture**: Reusable Python modules for VLM interfaces and evaluation
- **Batch Processing**: Systematic evaluation across inscription datasets
- **Ground Truth Comparison**: Scholarly transliterations and translations from OCIANA

## Project Structure

```
safaitic-ocr/
├── data/
│   └── examples/           # Inscription images organized by siglum
├── metadata/
│   └── BES15.csv          # Ground truth transliterations and translations
├── src/
│   ├── __init__.py
│   ├── utils.py           # Data loading and image processing
│   ├── vlm_interface.py   # Llama 3.2 Vision API wrapper
│   ├── prompt_templates.py # Safaitic-specific prompts
│   └── evaluator.py       # Evaluation metrics
├── notebooks/
│   ├── 01_setup_and_explore.ipynb
│   ├── 02_single_image_test.ipynb
│   └── 03_batch_evaluation.ipynb
├── results/               # Evaluation outputs (created on first run)
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

1. **Python 3.10+**
2. **HuggingFace Gradio Client** (installed automatically)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/shaigordin/safaitic-ocr.git
cd safaitic-ocr
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

That's it! No API keys, no additional setup required for FREE VLM inference.

### Optional: Local Ollama Setup

If you want to use local models instead of Gradio Spaces:

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull a vision model:
```bash
ollama pull llama3.2-vision
```

## Quick Start

### Generate VLM Analysis Results

Run batch analysis using FREE Gradio Spaces:

```bash
# Test mode: Analyze 10 inscriptions
python generate_results.py

# Analyze all inscriptions with images
python generate_results.py --all

# Analyze specific number
python generate_results.py --count 50
```

This will:
1. Connect to HuggingFace Gradio Space (LLaVA OneVision by default)
2. Analyze inscriptions with multiple prompts
3. Save results to `docs/data/latest.json`
4. Display progress and summary

### View Results

Open the web application:

```bash
# Open in your browser
open docs/index.html

# Or use Python's built-in server
python -m http.server 8000 --directory docs
# Then visit: http://localhost:8000
```

The web app displays:
- All analyzed inscriptions
- Original OCIANA images
- Ground truth transliterations/translations
- VLM analysis results for each prompt
- Search and filter capabilities

### Deploy to GitHub Pages

1. Push to GitHub:
```bash
git add .
git commit -m "Add VLM analysis results"
git push
```

2. Enable GitHub Pages:
   - Go to repository Settings → Pages
   - Source: Deploy from branch `main`
   - Folder: `/docs`
   - Save

3. Visit: `https://shaigordin.github.io/safaitic-ocr/`

3. Visit: `https://shaigordin.github.io/safaitic-ocr/`

### Python API Usage

Use the pipeline programmatically:

```python
from src import (
    load_metadata,
    get_inscription_data,
    GradioSpaceVLM,
    SafaiticPrompts
)

# Load inscription data
df = load_metadata("metadata/BES15.csv")
inscription = get_inscription_data(df, "data", "BES15 1", load_images=True)

# Initialize FREE VLM (Gradio Space)
vlm = GradioSpaceVLM(space_id="llava-onevision")

# Or use local Ollama
# from src import LlamaVision
# vlm = LlamaVision(model_name="llama3.2-vision")

# Analyze with a prompt
prompt = SafaiticPrompts.transliteration_attempt()
result = vlm.analyze_image(inscription.images[0], prompt)

print(f"Response: {result['response']}")
```

### Available VLM Options

**FREE Gradio Spaces** (recommended):
```python
# LLaVA OneVision (recommended for ancient scripts)
vlm = GradioSpaceVLM(space_id="llava-onevision")

# Qwen3-VL (official, very popular)
vlm = GradioSpaceVLM(space_id="qwen3-vl")

# Or use direct space path
vlm = GradioSpaceVLM(space_id="your-username/your-space")
```

**Local Ollama** (requires installation):
```python
vlm = LlamaVision(model_name="llama3.2-vision")
```

## Dataset

The project currently includes the **BES15** corpus:
- **157 unique inscriptions**
- **437+ high-quality images**
- Ground truth transliterations in scholarly transcription
- English translations
- Provenance information from Jordan

Data sourced from the [OCIANA database](https://ociana.osu.edu/).

## Prompt Templates

Built-in templates for different analysis tasks:

| Template | Purpose |
|----------|---------|
| `basic_description` | General image description |
| `script_identification` | Identify writing system |
| `character_recognition` | Recognize individual characters |
| `transliteration_attempt` | Convert script to Latin |
| `translation_attempt` | Interpret meaning |
| `comparative_analysis` | Multi-image synthesis |
| `condition_assessment` | Evaluate preservation |
| `structured_json` | Machine-readable output |

## Evaluation Metrics

The pipeline computes:

- **Character-level**: Edit distance, similarity ratio
- **Word-level**: Word accuracy, sequence alignment
- **Semantic**: Keyword precision/recall, F1 score
- **Script identification**: Correct classification rate
- **Overall score**: Weighted composite metric

## Results

Results are saved in `results/` directory with:
- Timestamp-stamped JSON files with full details
- Summary CSV files for spreadsheet analysis
- Visualizations and aggregate statistics

## Example Results Structure

```json
{
  "metadata": {
    "timestamp": "20251104_143022",
    "model": "llama3.2-vision",
    "num_inscriptions": 10,
    "prompt_template": "transliteration_attempt"
  },
  "summary": {
    "avg_overall_score": 0.342,
    "avg_character_similarity": 0.298,
    "avg_word_accuracy": 0.156
  },
  "results": [...]
}
```

## Extending the Pipeline

### Adding New VLMs

Create a new interface in `src/vlm_interface.py`:

```python
class GPT4Vision:
    def __init__(self, api_key):
        # Implementation
    
    def analyze_image(self, image, prompt):
        # Implementation
```

### Custom Prompts

```python
custom_prompt = SafaiticPrompts.custom_prompt(
    task="Identify personal names in this inscription",
    context="Focus on genealogical patterns (X son of Y)",
    output_format="List format"
)
```

### Custom Evaluation Metrics

Extend `InscriptionEvaluator` class with domain-specific metrics.

## Troubleshooting

### Gradio Space Issues

**"Space is loading" or slow first request:**
- Gradio Spaces use Zero GPU and sleep when inactive
- First request after sleep takes 30-60 seconds (cold start)
- Subsequent requests are faster

**Connection errors:**
1. Check your internet connection
2. Visit the space URL directly: `https://huggingface.co/spaces/kavaliha/llava-onevision`
3. Try alternative space: `python generate_results.py --space qwen3-vl`

**Rate limiting:**
- Gradio Spaces have fair use policies
- If rate limited, wait a few minutes or try different space

### Local Ollama Issues

**Ollama not responding:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
pkill ollama && ollama serve
```

**Model not found:**
```bash
# Pull the model
ollama pull llama3.2-vision

# Verify it's available
ollama list
```

### Data Issues

**Images not loading:**
- Ensure image files exist in `data/examples/[inscription_folder]/`
- Check file permissions
- Verify folder names match inscription sigla exactly

### Notebook Issues

**Jupyter kernel issues:**
```bash
# Install kernel
python -m ipykernel install --user --name=safaitic-ocr

# Select kernel in Jupyter: Kernel > Change Kernel > safaitic-ocr
```

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{safaitic_ocr_2025,
  title={Safaitic OCR: VLM Testing Pipeline for Ancient Arabian Scripts},
  author={[Your Name]},
  year={2025},
  url={https://github.com/shaigordin/safaitic-ocr}
}
```

## Data Sources

- OCIANA (Online Corpus of the Inscriptions of Ancient North Arabia): https://ociana.osu.edu/
- Al-Jallad, Ahmad. *An Outline of the Grammar of the Safaitic Inscriptions*. Brill, 2015.

## License

[Specify your license]

## Contributing

Contributions welcome! Areas for development:
- Additional VLM integrations
- Improved evaluation metrics
- Enhanced web application features
- Support for other ancient scripts
- Fine-tuning experiments

## Acknowledgments

- **OCIANA** database team for providing ground truth data
- **HuggingFace** for free Gradio Spaces infrastructure
- **Ollama** project for local VLM support
- Safaitic epigraphy research community
