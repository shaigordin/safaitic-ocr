# Safaitic OCR - VLM Evaluation for Ancient Inscriptions

**Preliminary research demonstrating Vision Language Model capabilities on Safaitic inscriptions** - a crucial first step toward developing grounded OCR for digital scholarly editions of ancient Arabian scripts.

## 🎯 Project Goals

This project evaluates **state-of-the-art VLMs** on Safaitic inscriptions to:

1. **Demonstrate current capabilities** of general-purpose VLMs on specialized ancient scripts
2. **Identify limitations** requiring fine-tuned grounded OCR approaches
3. **Establish baseline metrics** for future fine-tuning and dataset creation
4. **Show research potential** for AI-assisted digital scholarly editions

### About Safaitic

Safaitic is an ancient Arabian script used by nomads in modern-day Syria, Jordan, and Saudi Arabia (1st century BC - 4th century AD). Key challenges:
- **28 consonantal glyphs** - no vowels written
- **No word division** - continuous text carved in any direction
- **Rocky surfaces** - weathered, low-contrast inscriptions
- **Complex paleography** - 4+ script variants with experimental forms

Learn more: [OCIANA Safaitic Database](https://ociana.osu.edu/scripts_safaitic)

## 🚀 Three Approaches to VLM Analysis

### 1. **Local Mac Inference** (MLX-VLM) - *Recommended for Development*

Fast, private inference using Apple Silicon with [mlx-vlm](https://github.com/Blaizzy/mlx-vlm):

```bash
# Install MLX-VLM
pip install mlx-vlm

# Analyze inscriptions locally
python analyze_mlx.py --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit --count 10
```

**Advantages**: Free, private, fast on Mac, supports 200+ models
**Models**: Qwen2.5-VL (2B-72B), SmolVLM (256M-2B), Idefics3, Pixtral, Moondream

### 2. **Serverless Batch Processing** (HF Jobs + UV Scripts) - *Best for Scale*

Process all 1,401 inscriptions with zero GPU costs using [uv-scripts/ocr](https://huggingface.co/datasets/uv-scripts/ocr):

```bash
# Batch OCR on HuggingFace Jobs (no local GPU needed)
hf jobs uv run --flavor a100-large \
  https://huggingface.co/datasets/uv-scripts/ocr/raw/main/deepseek-ocr-vllm.py \
  shaigordin/safaitic-inscriptions \
  shaigordin/safaitic-ocr-results \
  --max-samples 1401
```

**Advantages**: Scalable, cost-effective, production OCR models
**Models**: DeepSeek-OCR (3B), Nanonets-OCR2 (3.7B), olmOCR2 (8B), RolmOCR (7B)

### 3. **Local Ollama** - *Fallback Option*

Local server with vision models:

```bash
ollama pull llama3.2-vision
python generate_results.py --count 50
```

**Advantages**: Fully local, no cloud dependency
**Limitations**: Frequent timeouts (300s), 44% success rate in testing

## 📊 Current Results

We've tested with **mlx-vlm (Qwen2.5-VL)** and **Ollama (llama3.2-vision)** on small samples:
- ✅ **Script identification**: Recognizes "ancient inscriptions", "rock surface", "text"
- ⚠️ **Transliteration**: Cannot read Safaitic letters (expected - not in training data)
- ⚠️ **Complex prompts**: Timeouts on detailed transliteration requests
- 📈 **Success rate**: ~60-100% depending on prompt complexity and model

**Key Finding**: General VLMs understand *context* (ancient, rock-carved) but lack *script knowledge* → **Fine-tuning required**

## 📁 Project Structure

```
safaitic-ocr/
├── data/
│   └── examples/           # 1,401 inscriptions with images (BES15 corpus)
├── metadata/
│   └── BES15.csv          # Ground truth from OCIANA database
├── src/
│   ├── utils.py           # Data loading utilities
│   ├── vlm_interface.py   # Ollama interface (legacy)
│   ├── mlx_vlm_interface.py    # NEW: MLX-VLM interface
│   ├── prompt_templates.py     # Safaitic-specific prompts
│   └── evaluator.py       # Evaluation metrics
├── notebooks/
│   ├── 01_setup_and_explore.ipynb
│   ├── 02_single_image_test.ipynb
│   ├── 03_batch_evaluation.ipynb
│   └── 04_preliminary_results_analysis.ipynb  # NEW
├── docs/
│   ├── index.html         # Results visualization
│   ├── data/latest.json   # Analysis results
│   └── future_work.md     # NEW: Grounded OCR roadmap
├── analyze_mlx.py         # NEW: MLX-VLM batch analysis
├── uv_batch_analysis.py   # NEW: HF Jobs UV script
├── generate_results.py    # Legacy Ollama script
└── requirements.txt
```

## 🏁 Quick Start

### Method 1: Local Mac Inference (MLX-VLM) - Recommended

**Requirements**: Apple Silicon Mac (M1/M2/M3/M4)

1. Clone and setup:
```bash
git clone https://github.com/shaigordin/safaitic-ocr.git
cd safaitic-ocr
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install mlx-vlm
```

2. Run analysis:
```bash
# Analyze 10 inscriptions with Qwen2.5-VL (7B, 4-bit quantized)
python analyze_mlx.py --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit --count 10 --verbose

# Try SmolVLM (2B, very fast)
python analyze_mlx.py --model mlx-community/SmolVLM-Instruct --count 10

# Analyze all inscriptions
python analyze_mlx.py --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit --all
```

**Available Models** (200+ on HuggingFace):
- `mlx-community/Qwen2.5-VL-7B-Instruct-4bit` (recommended, 4.5GB)
- `mlx-community/SmolVLM-Instruct` (fastest, 1.2GB)
- `mlx-community/Idefics3-8B-Llama3-4bit` (good, 4.8GB)
- `mlx-community/pixtral-12b-4bit` (large, 7GB)

Browse all: https://huggingface.co/mlx-community

### Method 2: Serverless Batch (HF Jobs) - Best for Scale

**Requirements**: HuggingFace account (free), HF CLI

1. Setup HF CLI:
```bash
pip install huggingface_hub[cli,hf_transfer]
huggingface-cli login  # Paste your HF token
```

2. Prepare your dataset:
```bash
# Upload inscription images to HuggingFace dataset
# (or use existing dataset)
```

3. Run serverless batch OCR:
```bash
# Process with DeepSeek-OCR on A100 GPU
hf jobs uv run --flavor a100-large \
  https://raw.githubusercontent.com/shaigordin/safaitic-ocr/main/uv_batch_analysis.py \
  shaigordin/safaitic-inscriptions \
  shaigordin/safaitic-ocr-results \
  --max-samples 1401

# Or use Nanonets-OCR2 model
hf jobs uv run --flavor a100-large \
  https://huggingface.co/datasets/uv-scripts/ocr/raw/main/nanonets-ocr2-vllm.py \
  shaigordin/safaitic-inscriptions \
  shaigordin/safaitic-ocr-results
```

**Supported Flavors**:
- `a100-large`: 1x A100 GPU (best performance)
- `l4x4`: 4x L4 GPUs (cost-effective)
- Check pricing: https://huggingface.co/docs/hub/jobs

### Method 3: Local Ollama (Fallback)

**Requirements**: Ollama installed

1. Install Ollama from [ollama.ai](https://ollama.ai)

2. Pull and run:
```bash
ollama pull llama3.2-vision
python generate_results.py --count 10
```

**Note**: Ollama had 44% success rate in our tests (frequent timeouts). Use MLX-VLM for better results.

## 📖 Detailed Usage

### MLX-VLM: Command-Line Options

```bash
python analyze_mlx.py \
  --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit \
  --count 10 \              # Number of inscriptions to analyze
  --all \                   # Or analyze all inscriptions
  --verbose                 # Show detailed progress

# Results saved to: docs/data/mlx_results_YYYYMMDD_HHMMSS.json
```

### UV Scripts: Batch Processing

```bash
# Basic usage
hf jobs uv run --flavor GPU_TYPE SCRIPT INPUT_DATASET OUTPUT_DATASET [OPTIONS]

# Example with custom parameters
hf jobs uv run --flavor a100-large \
  uv_batch_analysis.py \
  shaigordin/safaitic-inscriptions \
  shaigordin/safaitic-results-deepseek \
  --max-samples 100 \
  --model deepseek-ai/deepseek-ocr-3b-sft

# Monitor job
hf jobs status JOB_ID

# View logs
hf jobs logs JOB_ID
```

**Available OCR Models** (via uv-scripts):
- `deepseek-ai/deepseek-ocr-3b-sft` (3B, fast)
- `Nanonets/nanonets-ocr2` (3.7B, best accuracy)
- `openbmb/olmOCR2-base` (8B, research quality)
- `AliD/RolmOCR` (7B, experimental)

### Legacy Gradio Spaces

```bash
# Still works but slower than MLX-VLM
python generate_results.py --count 10
```

## 📊 Viewing Results

### Web Application

Open the interactive results viewer:

```bash
# Local file
open docs/index.html

# Or with local server
python -m http.server 8000 --directory docs
# Visit: http://localhost:8000
```

The web app displays:
- All analyzed inscriptions with images
- Ground truth transliterations/translations from OCIANA
- VLM responses for each prompt type
- Comparative analysis across models
- Search and filter capabilities

### Compare Results

```python
# Load and compare MLX vs Ollama results
import json

with open("docs/data/mlx_results_20250104.json") as f:
    mlx_results = json.load(f)

with open("docs/data/ollama_results_20250104.json") as f:
    ollama_results = json.load(f)

# Compare success rates
mlx_success = mlx_results["metadata"]["success_count"] / mlx_results["metadata"]["total_count"]
ollama_success = ollama_results["metadata"]["success_count"] / ollama_results["metadata"]["total_count"]

print(f"MLX-VLM success rate: {mlx_success:.1%}")
print(f"Ollama success rate: {ollama_success:.1%}")
```

### Deploy to GitHub Pages

1. Push results:
```bash
git add docs/data/*.json
git commit -m "Add VLM analysis results"
git push
```

2. Enable GitHub Pages:
   - Repository Settings → Pages
   - Source: `main` branch
   - Folder: `/docs`
   - Save

3. View online: `https://[username].github.io/safaitic-ocr/`

## 🐍 Python API

### MLX-VLM Usage

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image

# Load model (first time downloads ~4.5GB)
model_path = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
model, processor = load(model_path)
config = load_config(model_path)

# Prepare image and prompt
image = Image.open("data/examples/BES15 1/BES15_1_01.jpg")
prompt = "Describe the script visible in this ancient inscription."

# Format prompt with model's chat template
formatted_prompt = apply_chat_template(
    processor, config, prompt, num_images=1
)

# Generate response
output = generate(
    model,
    processor,
    formatted_prompt,
    image=[image],
    max_tokens=500,
    temperature=0.7
)

print(output)
```

### Using Safaitic-Specific Prompts

```python
from src.prompt_templates import SafaiticPrompts
from src.utils import load_metadata, get_inscription_data

# Load inscription data
df = load_metadata("metadata/BES15.csv")
inscription = get_inscription_data(df, "data", "BES15 1", load_images=True)

# Try different prompt types
prompts = {
    "description": SafaiticPrompts.basic_description(),
    "script_id": SafaiticPrompts.script_identification(),
    "transliteration": SafaiticPrompts.transliteration_attempt()
}

for prompt_type, prompt in prompts.items():
    formatted = apply_chat_template(processor, config, prompt, num_images=1)
    result = generate(model, processor, formatted, image=[inscription.images[0]])
    print(f"\n{prompt_type}:\n{result}\n")
```

### Ollama Python Interface (Legacy)

```python
from src.vlm_interface import LlamaVision
from src.prompt_templates import SafaiticPrompts

# Initialize Ollama client
vlm = LlamaVision(model_name="llama3.2-vision")

# Check model availability
if vlm.check_availability():
    # Analyze image
    prompt = SafaiticPrompts.script_identification()
    result = vlm.analyze_image(inscription.images[0], prompt)
    
    print(f"Response: {result['response']}")
    print(f"Duration: {result['duration']:.2f}s")
else:
    print("Ollama not available")
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

## 🎓 Future Work

This preliminary evaluation is the foundation for a larger **grounded OCR project** for Safaitic digital scholarly editions. See [docs/future_work.md](docs/future_work.md) for detailed roadmap covering:

- **Phase 2**: Creating grounded annotation dataset (character-level bounding boxes)
- **Phase 3**: Fine-tuning VLMs for Safaitic-specific OCR
- **Phase 4**: Production system for digital scholarly editions
- **Collaboration**: Funding opportunities and academic partnerships

**Key insight**: General VLMs provide context awareness but need fine-tuning with grounded annotations to actually read Safaitic script.

## 🤝 Contributing

Contributions welcome! Priority areas:

### Immediate (Phase 1 - Current)

- [ ] Test MLX-VLM with different models (Qwen2.5-VL 2B/32B, Idefics3, Pixtral)
- [ ] Create comparative analysis notebook (MLX vs Ollama vs HF Jobs results)
- [ ] Enhance web visualization with model comparison view
- [ ] Add error analysis dashboard (why do VLMs fail?)
- [ ] Test UV scripts with different OCR models

### Near-term (Phase 2 - Dataset Creation)

- [ ] Prototype annotation interface (Label Studio or custom)
- [ ] Annotate "gold standard" 100 inscriptions
- [ ] Develop synthetic data augmentation pipeline
- [ ] Create annotation guidelines for Safaitic epigraphy students
- [ ] Build annotation quality validation tools

### Long-term (Phase 3-4 - Fine-tuning & Production)

- [ ] Fine-tune Florence-2 on Safaitic grounded dataset
- [ ] Implement assisted transcription workflow for scholars
- [ ] Integrate with OCIANA database
- [ ] Support TEI EpiDoc export for scholarly editions
- [ ] Deploy production API for research community

**How to contribute**:

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Make changes and test
4. Commit: `git commit -m "Add your feature"`
5. Push: `git push origin feature/your-feature`
6. Open Pull Request

## 📧 Contact & Collaboration

Interested in Safaitic grounded OCR research? Looking for collaborators:

- 🎓 **Epigraphy experts** for annotation and validation
- 💻 **ML engineers** for fine-tuning and optimization  
- 💰 **Funding partners** for dataset creation and research
- 🏛️ **Institutions** with Safaitic collections or expertise

**Repository**: [github.com/shaigordin/safaitic-ocr](https://github.com/shaigordin/safaitic-ocr)

## 📚 Citation

If you use this project in your research:

```bibtex
@software{safaitic_ocr_2025,
  title={Safaitic OCR: Vision Language Model Evaluation for Ancient Arabian Scripts},
  author={Gordin, Shai},
  year={2025},
  url={https://github.com/shaigordin/safaitic-ocr},
  note={Preliminary evaluation for grounded OCR research}
}
```

## 📖 Data Sources

- **OCIANA** (Ohio State University): [ociana.osu.edu](https://ociana.osu.edu/)
  - Ground truth transliterations and translations
  - BES15 corpus (157 inscriptions, 437+ images)
- **Al-Jallad, Ahmad** (2015): *An Outline of the Grammar of the Safaitic Inscriptions*. Brill.
  - Linguistic and paleographic reference

## 📄 License

[Specify your license - e.g., MIT, CC BY-SA 4.0]

## 🙏 Acknowledgments

- **OCIANA Team** (Ohio State University) for open-access ground truth data
- **HuggingFace** for MLX-VLM library and serverless jobs infrastructure
- **Blaizzy** for [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) library
- **UV Scripts Team** for [uv-scripts/ocr](https://huggingface.co/datasets/uv-scripts/ocr) patterns
- **Ollama Project** for local VLM support
- **Safaitic epigraphy research community** for domain expertise

---

**Status**: Phase 1 (Preliminary Evaluation) - demonstrating VLM capabilities and limitations to support grounded OCR project proposal

**Next steps**: Test MLX-VLM performance, create comparative analysis, prepare grant proposals for Phase 2 (dataset creation)
