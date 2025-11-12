# Future Work: Grounded OCR for Safaitic Digital Scholarly Editions

This document outlines the path from preliminary VLM evaluation to a production grounded OCR system for Safaitic inscriptions.

## 📊 Current State (Phase 1: Evaluation)

### What We've Accomplished

✅ **Evaluated general-purpose VLMs** on Safaitic inscriptions using three approaches:
- Local Mac inference (MLX-VLM with Qwen2.5-VL, SmolVLM)
- Serverless batch processing (HF Jobs with DeepSeek-OCR, Nanonets-OCR2)
- Local Ollama (llama3.2-vision as fallback)

✅ **Demonstrated baseline capabilities**:
- Script identification: VLMs recognize "ancient inscriptions", "rock surface"
- Context understanding: Understand archaeological/historical context
- Format preservation: Can output structured results

✅ **Identified critical limitations**:
- **Cannot read Safaitic letters** - not in training data
- **No grounding** - cannot localize characters in images
- **No script-specific knowledge** - missing paleographic understanding

### Key Findings

| Capability | Status | Evidence |
|------------|--------|----------|
| Image understanding | ✅ Good | Recognizes rocks, carvings, ancient context |
| Script identification | ⚠️ Generic | Identifies "ancient script" but not Safaitic |
| Letter recognition | ❌ Failed | Cannot read ʾ b g d h w z ḥ ṭ y k l... |
| Transliteration | ❌ Failed | Generates plausible-looking but incorrect text |
| Grounding | ❌ Missing | No bounding boxes or character localization |

**Conclusion**: General VLMs provide *context awareness* but require **fine-tuning with grounded OCR** for actual script reading.

---

## 🎯 Phase 2: Dataset Creation for Grounded OCR

### What is Grounded OCR?

Grounded OCR combines:
1. **Character detection** - Bounding boxes around each letter
2. **Character recognition** - Identifying which Safaitic letter (ʾ, b, g, d...)
3. **Sequence understanding** - Reading direction and word boundaries
4. **Context preservation** - Layout, surface condition, paleographic features

Unlike general OCR, grounded OCR:
- Provides **verifiable outputs** with spatial coordinates
- Enables **correction workflows** for scholars
- Supports **partial reading** of damaged inscriptions
- Maintains **provenance** of each character identification

### Dataset Requirements

Based on successful grounded OCR projects like [Kosmos-2.5](https://arxiv.org/abs/2309.11419) and [Florence-2](https://arxiv.org/abs/2311.06242), we need:

#### 1. **Annotated Images** (~500-1,000 inscriptions minimum)

```json
{
  "inscription_siglum": "BES15 1",
  "image_path": "data/examples/BES15 1/BES15_1_01.jpg",
  "ground_truth_transliteration": "l ʿbd h-ʾlh",
  "annotations": [
    {
      "character": "l",
      "bbox": [123, 456, 145, 489],
      "confidence": 1.0,
      "notes": "clear, well-preserved"
    },
    {
      "character": "ʿ",
      "bbox": [156, 458, 178, 487],
      "confidence": 0.9,
      "notes": "slightly weathered"
    }
    // ... more characters
  ],
  "reading_direction": "right-to-left",
  "script_variant": 1,
  "surface_condition": "good"
}
```

#### 2. **Annotation Tools**

Options for creating grounded annotations:

**A. Label Studio** (Open Source)
```yaml
# label-studio-config.xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="bbox" toName="image">
    <Label value="ʾ" background="red"/>
    <Label value="b" background="green"/>
    <Label value="g" background="blue"/>
    <!-- All 28 Safaitic letters -->
  </RectangleLabels>
  <TextArea name="transliteration" toName="image"/>
  <Choices name="direction" toName="image">
    <Choice value="right-to-left"/>
    <Choice value="left-to-right"/>
    <Choice value="boustrophedon"/>
  </Choices>
</View>
```

**B. CVAT** (Computer Vision Annotation Tool)
- Better for sequential labeling
- Supports polygon annotations (for irregular letter shapes)
- Team collaboration features

**C. Custom Annotation Interface**
- Build on top of [Gradio](https://gradio.app/) or [Streamlit](https://streamlit.io/)
- Integrate with OCIANA database for ground truth validation
- Add Safaitic-specific features (direction indicators, variant markers)

#### 3. **Synthetic Data Augmentation**

To supplement manual annotations:

```python
# Pseudo-code for augmentation pipeline
def augment_safaitic_image(image, annotations):
    """
    Create variations preserving spatial relationships
    """
    augmentations = [
        RandomRotation(degrees=(-5, 5)),
        RandomBrightness(factor=(0.8, 1.2)),
        RandomContrast(factor=(0.8, 1.2)),
        WeatheringSimulation(),  # Simulate erosion
        ShadowVariation(),       # Different lighting
        NoiseAddition(type='gaussian')
    ]
    
    # Apply augmentations while updating bboxes
    aug_image, aug_annotations = apply_with_bbox_transform(
        image, annotations, augmentations
    )
    
    return aug_image, aug_annotations
```

### Collaborative Annotation Strategy

**Phase A: Expert Annotation (Critical)**
- **Team**: 2-3 Safaitic epigraphy experts
- **Goal**: Annotate 100 "gold standard" inscriptions
- **Time**: ~40 hours per expert (80-120 total hours)
- **Output**: High-confidence training examples

**Phase B: Student Annotation (Expansion)**
- **Team**: Graduate students in Semitic epigraphy
- **Goal**: Annotate 400 additional inscriptions
- **Training**: Using Phase A examples + annotation guidelines
- **Quality**: Expert review on 20% sample

**Phase C: Model-Assisted Annotation (Scaling)**
- **Process**: Use early fine-tuned model to pre-annotate
- **Goal**: Annotate remaining 500 inscriptions
- **Human role**: Correct model predictions (faster than from-scratch)

### Estimated Timeline & Resources

| Phase | Duration | Personnel | Output |
|-------|----------|-----------|--------|
| Dataset Design | 2 weeks | 1 researcher | Annotation schema, tools setup |
| Phase A (Gold) | 4 weeks | 2-3 experts | 100 annotated inscriptions |
| Phase B (Expansion) | 8 weeks | 3-4 students | 400 annotated inscriptions |
| Phase C (Scaling) | 6 weeks | 2 students + model | 500 annotated inscriptions |
| **Total** | **~5 months** | **Mixed team** | **1,000 grounded annotations** |

---

## 🔬 Phase 3: Fine-Tuning Grounded VLM

### Model Selection

Based on recent grounded OCR research:

| Model | Size | Grounding | Pros | Cons |
|-------|------|-----------|------|------|
| **Florence-2** | 0.2B-0.7B | ✅ Native | Fast, efficient, proven OCR | May need more training |
| **Kosmos-2.5** | 1.3B | ✅ Native | Purpose-built for text | Larger model |
| **Qwen2.5-VL** | 2B-32B | ⚠️ Via fine-tuning | SOTA general VLM | Requires grounding training |
| **SmolVLM** | 256M-2B | ⚠️ Via fine-tuning | Tiny, efficient | Less powerful |

**Recommended**: Start with **Florence-2-large** (0.7B)
- Proven success on document OCR
- Native grounding capabilities
- Efficient for production deployment
- Can fine-tune on single GPU

### Fine-Tuning Approach

#### 1. **Using MLX-VLM for Apple Silicon**

```bash
# Fine-tune Florence-2 on Safaitic grounded dataset
python -m mlx_vlm.lora \\
  --model-path mlx-community/Florence-2-large-ft \\
  --dataset shaigordin/safaitic-grounded-ocr \\
  --epochs 10 \\
  --batch-size 4 \\
  --learning-rate 1e-4 \\
  --lora-rank 16 \\
  --output-path safaitic-florence2-adapters
```

#### 2. **Using HuggingFace Transformers**

```python
from transformers import AutoModelForVision2Seq, TrainingArguments, Trainer
from datasets import load_dataset

# Load pre-trained Florence-2
model = AutoModelForVision2Seq.from_pretrained(
    "microsoft/Florence-2-large",
    trust_remote_code=True
)

# Load Safaitic dataset
dataset = load_dataset("shaigordin/safaitic-grounded-ocr")

# Training arguments
training_args = TrainingArguments(
    output_dir="./safaitic-florence2",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    learning_rate=1e-4,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    bf16=True,  # Mixed precision
    gradient_checkpointing=True
)

# Fine-tune
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"]
)

trainer.train()
```

#### 3. **Evaluation Metrics**

```python
def evaluate_grounded_ocr(predictions, ground_truth):
    """
    Comprehensive evaluation for grounded Safaitic OCR
    """
    metrics = {
        # Character-level
        "char_precision": character_detection_precision(predictions, ground_truth),
        "char_recall": character_detection_recall(predictions, ground_truth),
        "char_f1": character_detection_f1(predictions, ground_truth),
        
        # Spatial accuracy
        "iou_mean": mean_bbox_iou(predictions, ground_truth),
        "iou_threshold_0.5": bbox_accuracy_at_threshold(predictions, ground_truth, 0.5),
        
        # Sequence accuracy
        "transliteration_cer": character_error_rate(pred_seq, gt_seq),
        "transliteration_wer": word_error_rate(pred_seq, gt_seq),
        
        # Safaitic-specific
        "direction_accuracy": reading_direction_accuracy(predictions, ground_truth),
        "variant_confusion": script_variant_confusion_matrix(predictions, ground_truth)
    }
    
    return metrics
```

### Expected Results

Based on similar fine-tuning projects:

| Metric | Baseline (General VLM) | After Fine-tuning | Target |
|--------|------------------------|-------------------|---------|
| Character Detection | 0% | 85-90% | >90% |
| Character Recognition | 0% | 75-85% | >85% |
| Transliteration CER | N/A | 10-15% | <10% |
| Reading Direction | ~50% (random) | >95% | >98% |

---

## 🏗️ Phase 4: Production System for Digital Editions

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Web Interface                          │
│  (Upload inscription photos, view results, corrections)  │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│              API Gateway (FastAPI)                       │
│  • Authentication  • Rate limiting  • Version control    │
└────────────────┬────────────────────────────────────────┘
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
┌──────────────┐    ┌──────────────────┐
│ Grounded OCR │    │ Human Review     │
│  (Florence-2)│◄───┤  Workflow        │
└──────┬───────┘    └──────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Digital Scholarly Edition Database  │
│  • Transliterations with confidence  │
│  • Bounding box coordinates          │
│  • Paleographic notes                │
│  • Expert corrections                │
│  • Version history                   │
└─────────────────────────────────────┘
```

### Key Features

#### 1. **Assisted Transcription Interface**

```python
# Workflow for scholars
def assisted_transcription(inscription_image):
    """
    AI-assisted workflow for creating scholarly editions
    """
    # Step 1: Automated grounded OCR
    initial_ocr = grounded_model.predict(inscription_image)
    # Returns: characters with bboxes + confidence scores
    
    # Step 2: Present to scholar
    review_interface = create_review_interface(
        image=inscription_image,
        predictions=initial_ocr,
        confidence_threshold=0.7  # Flag low-confidence predictions
    )
    
    # Step 3: Scholar corrections
    corrections = scholar.review_and_correct(review_interface)
    
    # Step 4: Update model (active learning)
    if corrections:
        model.learn_from_corrections(corrections)
    
    # Step 5: Generate scholarly output
    edition = create_digital_edition(
        image=inscription_image,
        final_transcription=corrections or initial_ocr,
        metadata={
            "transcriber": scholar.name,
            "confidence_mean": np.mean([c.confidence for c in corrections]),
            "ai_assisted": True,
            "model_version": model.version
        }
    )
    
    return edition
```

#### 2. **Quality Assurance**

- **Confidence scoring**: Flag characters <70% confidence for review
- **Peer review**: 2 experts review high-impact inscriptions
- **Consistency checking**: Compare against known Safaitic patterns
- **Version control**: Track all changes with scholarly attribution

#### 3. **Export Formats**

Support multiple scholarly edition standards:

```python
EXPORT_FORMATS = {
    "epidoc": generate_epidoc_xml,      # TEI EpiDoc standard
    "json-ld": generate_jsonld,         # Linked Open Data
    "csv": generate_csv_export,         # Simple tabular
    "ociana": generate_ociana_format,   # OCIANA database format
    "latex": generate_latex_edition     # Print publication
}
```

---

## 📚 References & Resources

### Key Papers on Grounded OCR

1. **Kosmos-2.5** (Microsoft, 2023): "A Multimodal Literate Model"
   - https://arxiv.org/abs/2309.11419
   - Pioneering work on grounded text generation

2. **Florence-2** (Microsoft, 2023): "Advancing a Unified Representation for Vision Tasks"
   - https://arxiv.org/abs/2311.06242
   - State-of-the-art grounded OCR model

3. **TrOCR** (Microsoft, 2021): "Transformer-based Optical Character Recognition"
   - https://arxiv.org/abs/2109.10282
   - Foundation for modern OCR approaches

### Safaitic Epigraphy Resources

1. **Al-Jallad (2015)**: *An Outline of the Grammar of the Safaitic Inscriptions*
   - Comprehensive linguistic reference

2. **OCIANA Database**: https://ociana.osu.edu/
   - Primary source for ground truth data

3. **DASI** (Database of Ancient Semitic Inscriptions): http://dasi.cnr.it/
   - Broader context of Ancient Arabian scripts

### Technical Resources

- **MLX-VLM**: https://github.com/Blaizzy/mlx-vlm
- **UV Scripts for OCR**: https://huggingface.co/datasets/uv-scripts/ocr
- **HuggingFace Datasets**: https://huggingface.co/docs/datasets/
- **Label Studio**: https://labelstud.io/

---

## 🤝 Contact & Collaboration

Interested in collaborating on Safaitic grounded OCR? Reach out:

- **GitHub**: [shaigordin/safaitic-ocr](https://github.com/shaigordin/safaitic-ocr)
- **Email**: [Your contact email]
- **HuggingFace**: [shaigordin](https://huggingface.co/shaigordin)

We're looking for:
- 🎓 Safaitic epigraphy experts for annotation
- 💻 ML engineers for model fine-tuning
- 📊 Funding opportunities and grant collaborators
- 🌍 Institutional partnerships

**Let's build the future of digital scholarly editions together!**
