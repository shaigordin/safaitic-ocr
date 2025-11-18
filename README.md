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
# Safaitic OCR — Quick Start

This repository contains the Safaitic VLM evaluation and a small interactive demo located in `docs/`.

Quick pointers:

- Demo (browse results): `docs/index.html` (open locally or via a simple HTTP server)
- Evaluation JSON files: `docs/data/` (per-model results for 50 inscriptions)
- If you need the full release notes or changelog, they have been moved to `archive/`.

If you want the minimal development workflow:

```bash
git clone https://github.com/shaigordin/safaitic-ocr.git
cd safaitic-ocr
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the demo locally
cd docs
python3 -m http.server 8000
# Open http://localhost:8000
```

For image deployment we recommend hosting the images on GitHub Releases or an external CDN; see `archive/GITHUB_RELEASE_GUIDE.md` for the archived, detailed instructions.

If you want me to also (optionally) add image filenames into the `docs/data/*.json` files so the demo can load OCIANA-hosted images reliably, say so and I will add that as the next step.

––––
Light and functional README — let me know what extra link or short section you want here.
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

## 🎓 Next Steps: Grounded OCR Project

This comprehensive 5-model evaluation validates the **grounded OCR approach**:

### Phase 2: Dataset Creation (Next)
- Annotate 500-1,000 inscriptions with character-level bounding boxes
- Partner with expert epigraphers (Oxford, Leiden, Chicago)
- Release open-access dataset on HuggingFace

### Phase 3: Fine-tuning
- Fine-tune Qwen2.5-VL-7B or Qwen2-VL-2B on annotated dataset
- Target: >80% character-level accuracy
- LoRA/QLoRA for efficient training on Apple Silicon

### Phase 4: Production System
- Deploy OCR API and web interface
- Integrate with OCIANA database
- TEI EpiDoc export for scholarly editions

📄 **See [docs/future_work.md](docs/future_work.md) for complete roadmap**

**Key Insight**: VLMs have strong vision + context understanding but lack letter-level knowledge. Grounded OCR with bounding boxes fills this specific gap.

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
