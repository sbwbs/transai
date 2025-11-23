# How to Add Your Own Glossaries & Term Memory

This guide walks you through the exact steps to provide your own Excel glossaries or term memory files to TransAI.

---

## Option 1: Quick Start (Easiest)

### Step 1: Prepare Your Excel File

Create an Excel file with **at minimum 2 columns**:
- One column with **Korean terms** (any column name)
- One column with **English terms** (any column name)

**Example Excel structure:**

```
Sheet1:
┌─────────────────┬──────────────────┐
│ Korean          │ English          │
├─────────────────┼──────────────────┤
│ 임상시험        │ clinical trial    │
│ 피험자          │ study subject    │
│ 부작용          │ adverse event    │
│ 약효            │ efficacy         │
└─────────────────┴──────────────────┘
```

**File format options:**
- `.xlsx` (Excel) - Recommended
- `.csv` (Comma-separated)
- `.json` (JSON format)

### Step 2: Place File in Data Directory

```bash
# Copy your glossary to the data directory
cp your_glossary.xlsx /Users/won.suh/Project/transai/src/data/glossaries/

# Directory structure:
/Users/won.suh/Project/transai/src/data/glossaries/
├── your_glossary.xlsx
├── medical_terms.xlsx
└── clinical_protocols.json
```

### Step 3: Use in Pipeline

```python
from src.glossary.glossary_loader import GlossaryLoader

# Load your glossary with auto-detected column mapping
loader = GlossaryLoader()

glossary_terms = loader.load_glossary(
    file_path="src/data/glossaries/your_glossary.xlsx",
    file_type="excel"
)

print(f"Loaded {len(glossary_terms)} terms")

# Use in pipeline
from src.production_pipeline_batch_enhanced import EnhancedBatchPipeline

pipeline = EnhancedBatchPipeline(glossary=glossary_terms)
results = pipeline.run_enhanced_batch_pipeline(input_file="input.xlsx")
```

---

## Option 2: Configuration-Based (Recommended for Production)

### Step 1: Create Configuration File

Create `glossary_config.yaml` in your project root:

```yaml
version: "1.0"
default_language_pair: "ko-en"

sources:
  - name: "My Glossary"
    type: "excel"
    path: "src/data/glossaries/my_glossary.xlsx"
    mapping:
      korean: "Korean"      # Your Korean column name
      english: "English"    # Your English column name
    confidence_score: 0.90
    enabled: true
```

### Step 2: Load Using Configuration

```python
from src.glossary.glossary_loader import GlossaryLoader

# Load all glossaries from config
loader = GlossaryLoader(config_path="glossary_config.yaml")
all_terms, stats = loader.load_all_glossaries()

print(f"Loaded {stats['total_terms']} terms from {len(stats['sources'])} sources")
```

### Step 3: Automate in Pipeline Initialization

```python
from src.production_pipeline_batch_enhanced import EnhancedBatchPipeline

# Pipeline will automatically use glossary from config
pipeline = EnhancedBatchPipeline()

# Or override with specific config
pipeline = EnhancedBatchPipeline(glossary_config="custom_glossary_config.yaml")
```

---

## Excel File Format Details

### Minimal Format (2 columns)

```
┌─────────────────┬──────────────────┐
│ Korean          │ English          │
├─────────────────┼──────────────────┤
│ 임상시험        │ clinical trial    │
│ 피험자          │ study subject    │
└─────────────────┴──────────────────┘
```

### Standard Format (4 columns)

```
┌─────────────────┬──────────────────┬──────────┬───────────┐
│ Korean          │ English          │ Category │ Frequency │
├─────────────────┼──────────────────┼──────────┼───────────┤
│ 임상시험        │ clinical trial    │ clinical │ high      │
│ 피험자          │ study subject    │ clinical │ high      │
│ 부작용          │ adverse event    │ safety   │ medium    │
└─────────────────┴──────────────────┴──────────┴───────────┘
```

### Advanced Format (with alternatives)

```
┌─────────────────┬──────────────────┬──────────┬───────────────────────────┐
│ Korean          │ English          │ Category │ Alternatives              │
├─────────────────┼──────────────────┼──────────┼───────────────────────────┤
│ 임상시험        │ clinical trial    │ clinical │ 임상연구|임상조사         │
│ 피험자          │ study subject    │ clinical │ 시험대상자|참여자         │
│ 부작용          │ adverse event    │ safety   │ 이상반응|부정반응         │
└─────────────────┴──────────────────┴──────────┴───────────────────────────┘
```

**Note:** Alternatives are separated by `|` (pipe character)

---

## Column Mapping Guide

The glossary loader can work with **any column names**. Just specify the mapping:

### Auto-Detection (Easiest)

If your Excel has clear Korean and English columns, the loader will auto-detect:

```python
loader.load_glossary(
    file_path="my_glossary.xlsx",
    file_type="excel"
    # mapping=None  # Auto-detects Korean/English columns
)
```

### Explicit Mapping

```python
loader.load_glossary(
    file_path="my_glossary.xlsx",
    file_type="excel",
    mapping={
        "korean": "Korean Term",     # Your column name
        "english": "English Term",   # Your column name
        "category": "Domain",        # Optional
        "frequency": "Usage Count"   # Optional
    }
)
```

### Configuration Mapping

```yaml
sources:
  - name: "My Terms"
    type: "excel"
    path: "my_glossary.xlsx"
    mapping:
      korean: "한글 용어"          # Korean column
      english: "영문 용어"         # English column
      category: "분류"             # Category column
      frequency: "빈도"            # Frequency column
```

---

## JSON Format

### Basic JSON Structure

```json
[
  {
    "korean": "임상시험",
    "english": "clinical trial",
    "category": "clinical",
    "frequency": "high"
  },
  {
    "korean": "피험자",
    "english": "study subject",
    "category": "clinical",
    "frequency": "high",
    "alternatives": {
      "korean": ["시험대상자"],
      "english": ["trial participant"]
    }
  }
]
```

### JSON with Metadata

```json
{
  "glossary_metadata": {
    "version": "1.0",
    "language_pair": "ko-en",
    "created_date": "2025-11-23",
    "domain": "clinical_trials",
    "source": "my_organization"
  },
  "terms": [
    {
      "korean": "임상시험",
      "english": "clinical trial",
      "category": "clinical"
    }
  ]
}
```

---

## CSV Format

### Standard CSV

```csv
korean,english,category,frequency
임상시험,clinical trial,clinical,high
피험자,study subject,clinical,high
부작용,adverse event,safety,medium
```

### CSV with Custom Column Names

```csv
Korean Term,English Term,Domain,Usage
임상시험,clinical trial,clinical,high
피험자,study subject,clinical,high
```

Then specify mapping in config:
```yaml
mapping:
  korean: "Korean Term"
  english: "English Term"
  category: "Domain"
  frequency: "Usage"
```

---

## Practical Examples

### Example 1: Medical Device Terms

**File:** `medical_devices.xlsx`

```
┌──────────────────┬──────────────────────┬──────────┐
│ KO Term          │ EN Term              │ Domain   │
├──────────────────┼──────────────────────┼──────────┤
│ 의료기기         │ medical device       │ device   │
│ 생체적합성       │ biocompatibility     │ device   │
│ 성능시험         │ performance testing  │ device   │
│ 안전성평가       │ safety assessment    │ device   │
└──────────────────┴──────────────────────┴──────────┘
```

**Load with:**
```python
loader.load_glossary(
    file_path="medical_devices.xlsx",
    file_type="excel",
    mapping={
        "korean": "KO Term",
        "english": "EN Term",
        "category": "Domain"
    },
    confidence_score=0.95,  # High confidence for internal terms
    source_name="Medical Devices"
)
```

### Example 2: Regulatory Terms

**File:** `regulatory_terms.json`

```json
{
  "glossary_metadata": {
    "domain": "regulatory",
    "source": "FDA_guidelines"
  },
  "terms": [
    {
      "korean": "규제 승인",
      "english": "regulatory approval",
      "category": "regulatory"
    },
    {
      "korean": "임상시험승인",
      "english": "IND|Investigational New Drug",
      "category": "regulatory",
      "alternatives": {
        "korean": ["IND 승인"],
        "english": []
      }
    }
  ]
}
```

**Load with:**
```python
loader.load_glossary(
    file_path="regulatory_terms.json",
    file_type="json",
    confidence_score=0.98,  # Regulatory terms - highest confidence
    source_name="FDA Guidelines"
)
```

### Example 3: Multi-Source Configuration

**File:** `glossary_config.yaml`

```yaml
version: "1.0"

sources:
  # Internal medical device terms
  - name: "Medical Devices"
    type: "excel"
    path: "data/glossaries/medical_devices.xlsx"
    mapping:
      korean: "Korean"
      english: "English"
      category: "Domain"
    confidence_score: 0.95
    enabled: true

  # Regulatory standards
  - name: "FDA Regulatory"
    type: "json"
    path: "data/glossaries/fda_regulatory.json"
    confidence_score: 0.98
    enabled: true

  # Clinical trial terms
  - name: "Clinical Trials"
    type: "csv"
    path: "data/glossaries/clinical_trials.csv"
    mapping:
      korean: "Korean Term"
      english: "English Term"
    confidence_score: 0.90
    enabled: true

  # User-provided terms (can disable if needed)
  - name: "User Custom"
    type: "excel"
    path: "data/glossaries/user_custom.xlsx"
    confidence_score: 0.80
    enabled: true
```

**Load all with:**
```python
loader = GlossaryLoader(config_path="glossary_config.yaml")
all_terms, stats = loader.load_all_glossaries()

print(stats)
# Output:
# {
#   'sources': {
#     'Medical Devices': 245,
#     'FDA Regulatory': 189,
#     'Clinical Trials': 312,
#     'User Custom': 45
#   },
#   'total_terms': 791
# }
```

---

## Term Memory / Session-Based Glossary

### What is Term Memory?

Term Memory tracks translations during document processing to ensure consistency:

```
Segment 1: "임상시험" → "clinical trial" (from glossary)
Segment 2: "임상시험 결과" → Uses same "clinical trial" (from memory)
Segment 3: "임상시험 참여자" → Uses same "clinical trial" (from memory)
```

### Enable Term Memory

```python
from src.production_pipeline_batch_enhanced import EnhancedBatchPipeline

pipeline = EnhancedBatchPipeline(
    glossary=glossary_terms,
    use_valkey=True  # Enable persistent term memory caching
)

# Process document
results = pipeline.run_enhanced_batch_pipeline(
    input_file="document.xlsx"
)

# Term memory automatically maintains consistency across segments
# Cached in Valkey for <1ms lookups
```

### Access Learned Terms

```python
from src.memory.session_manager import SessionManager
from src.memory.valkey_manager import ValkeyManager

valkey = ValkeyManager()
session = SessionManager(valkey)

# Get terms learned during session
learned_terms = valkey.get_learned_terms(session_id="your_session_id")

# Output:
# [
#   {'korean': '임상시험', 'english': 'clinical trial', 'confidence': 0.95},
#   {'korean': '피험자', 'english': 'study subject', 'confidence': 0.92}
# ]
```

### Save Learned Terms as Glossary

```python
loader = GlossaryLoader()

# Get learned terms from session
learned_terms = valkey.get_learned_terms(session_id)

# Save as new glossary for future use
loader.save_glossary(
    terms=learned_terms,
    output_file="learned_terms.json",
    format="json"
)

# Now use as source in future configs
```

---

## Validation & Testing

### Validate Your Glossary

```python
loader = GlossaryLoader()

# Load glossary
terms = loader.load_glossary(
    file_path="my_glossary.xlsx",
    file_type="excel"
)

# Check for issues
print(f"Loaded {len(terms)} terms")

# Show sample
for term in terms[:5]:
    print(f"  {term['korean']} → {term['english']}")

# Check for missing translations
for term in terms:
    if not term['korean'] or not term['english']:
        print(f"⚠️ Missing translation: {term}")
```

### Test with Sample Translation

```python
from src.glossary.glossary_search import GlossarySearchEngine

search_engine = GlossarySearchEngine()
search_engine.load_terms(terms)

# Test search
results = search_engine.search("임상시험")

print(f"Found {len(results)} results:")
for result in results:
    print(f"  {result['korean']} → {result['english']} (score: {result['confidence']})")
```

---

## Troubleshooting

### Issue: "Column not found" error

**Solution:** Check your column names match exactly

```python
import pandas as pd

# Read your Excel to see actual column names
df = pd.read_excel("my_glossary.xlsx")
print(df.columns)
# Output: Index(['Korean', 'English', 'Domain'], dtype='object')

# Use exact names in mapping
mapping = {
    "korean": "Korean",    # Must match exactly
    "english": "English"
}
```

### Issue: Terms not being found during translation

**Causes:**
1. Korean text has extra spaces: `" 임상시험 "` vs `"임상시험"`
2. Column mapping is wrong
3. Glossary file not found

**Solution:**
```python
# Enable debug logging
loader = GlossaryLoader(log_level="DEBUG")

# Check loaded terms
terms = loader.load_glossary(file_path="glossary.xlsx")
print(f"Loaded {len(terms)} terms")
print(terms[0])  # Check first term structure
```

### Issue: Special characters not working

**Solution:** Ensure file is saved as UTF-8

```python
# Save Excel with UTF-8 encoding
df.to_excel("glossary.xlsx", index=False, encoding='utf-8')

# Or for CSV
df.to_csv("glossary.csv", index=False, encoding='utf-8')
```

---

## Best Practices

✅ **DO:**
- Use UTF-8 encoding for all files
- Include domain/category information for better matching
- Keep Korean and English terms consistent
- Use alternatives (|) for common synonyms
- Version your glossaries
- Test before production use

❌ **DON'T:**
- Mix multiple translations for same term (use alternatives instead)
- Include empty rows
- Use special characters in column names
- Store translations with extra whitespace
- Hardcode paths in code (use config files instead)

---

## Quick Reference

```python
# Quick load from Excel (auto-detect columns)
from src.glossary.glossary_loader import GlossaryLoader
loader = GlossaryLoader()
terms = loader.load_glossary("glossary.xlsx", "excel")

# Load with config (recommended)
loader = GlossaryLoader(config_path="glossary_config.yaml")
all_terms, stats = loader.load_all_glossaries()

# Use in pipeline
from src.production_pipeline_batch_enhanced import EnhancedBatchPipeline
pipeline = EnhancedBatchPipeline(glossary=terms)
results = pipeline.run_enhanced_batch_pipeline("input.xlsx")

# Save learned terms
loader.save_glossary(terms, "output.json", format="json")
```

---

## Need Help?

- See [GLOSSARY_SYSTEM.md](GLOSSARY_SYSTEM.md) for detailed documentation
- Check [glossary_config.example.yaml](../src/glossary/glossary_config.example.yaml) for configuration examples
- Run `python -m src.glossary.glossary_loader glossary_config.yaml` to test your config
