# Glossary System Documentation

## Overview

The TransAI glossary system manages medical terminology translation across Korean-English language pairs. It's designed to be flexible, scalable, and domain-agnostic while providing high-quality term consistency across all translations.

**Key Features:**
- Multi-source glossary loading (Excel, JSON, CSV)
- Flexible column mapping (no hardcoded field names)
- Fuzzy and exact matching algorithms
- Caching with Valkey/Redis for performance
- Support for alternative terminology
- Configurable glossary sources

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────┐
│         Glossary Configuration (YAML/JSON)              │
│  - Source definitions                                   │
│  - Column mappings                                       │
│  - Deduplication rules                                   │
└────────────────┬────────────────────────────────────────┘
                 │
        ┌────────▼────────┐
        │ GlossaryLoader  │
        │ (Generic)       │
        └────────┬────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
┌───▼──┐  ┌──────▼──┐  ┌────▼─────┐
│Excel │  │  JSON   │  │   CSV    │
│Files │  │ Files   │  │  Files   │
└──────┘  └─────────┘  └──────────┘
    │            │            │
    └────────────┼────────────┘
                 │
        ┌────────▼──────────┐
        │ Glossary Terms    │
        │ (Normalized)      │
        └────────┬──────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
┌───▼──────┐  ┌─▼────────┐  ┌▼──────────┐
│  Search  │  │  Caching │  │ Pipeline  │
│  Engine  │  │ (Valkey) │  │ Integration
└──────────┘  └──────────┘  └───────────┘
```

### Three-Tier Caching

```
Tier 1: Valkey/Redis Cache
├── Persistent glossary caching
├── O(1) lookups (<1ms)
└── Shared across sessions

Tier 2: Session Memory
├── Document-level term tracking
├── Consistency maintenance
└── Progress tracking

Tier 3: Search Index
├── Fast pattern matching
├── Fuzzy matching algorithms
└── Relevance scoring
```

---

## Configuration

### Glossary Configuration File

Create a `glossary_config.yaml` in your project:

```yaml
# Glossary Configuration
version: "1.0"
default_language_pair: "ko-en"

# Glossary sources to load
sources:
  - name: "Coding Form"
    type: "excel"
    path: "data/glossaries/coding_form.xlsx"
    sheet: "Terms"
    mapping:
      korean: "KR_SOC_27.0"      # Column name for Korean terms
      english: "SOC_27.0"        # Column name for English terms
      category: "Category"       # Optional
    confidence_score: 0.90
    enabled: true

  - name: "Clinical Trials"
    type: "excel"
    path: "data/glossaries/clinical_trials.xlsx"
    sheet: "Clinical"
    mapping:
      korean: "KO"
      english: "EN"
      category: "Domain"
    confidence_score: 0.95
    alternatives_separator: "|"  # For "term1|term2" format
    enabled: true

  - name: "Custom Domain"
    type: "json"
    path: "data/glossaries/custom_domain.json"
    confidence_score: 0.85
    enabled: true

  - name: "User Glossary"
    type: "csv"
    path: "data/glossaries/user_terms.csv"
    mapping:
      korean: "korean_term"
      english: "english_term"
    confidence_score: 0.80
    enabled: true

# Search settings
search:
  min_term_length: 2
  fuzzy_threshold: 0.60
  enable_preprocessing: true
  cache_enabled: true
  cache_ttl: 3600  # seconds

# Deduplication rules
deduplication:
  enabled: true
  keep_first: true  # Keep first occurrence or highest score
  merge_alternatives: true

# Output settings
output:
  default_format: "json"  # json, excel, csv
  include_metadata: true
  include_alternatives: true
```

### JSON Configuration Format

Alternatively, use `glossary_config.json`:

```json
{
  "version": "1.0",
  "default_language_pair": "ko-en",
  "sources": [
    {
      "name": "Coding Form",
      "type": "excel",
      "path": "data/glossaries/coding_form.xlsx",
      "sheet": "Terms",
      "mapping": {
        "korean": "KR_SOC_27.0",
        "english": "SOC_27.0",
        "category": "Category"
      },
      "confidence_score": 0.90,
      "enabled": true
    }
  ],
  "search": {
    "min_term_length": 2,
    "fuzzy_threshold": 0.60,
    "cache_enabled": true
  }
}
```

---

## Usage

### Basic Glossary Loading

```python
from src.glossary.glossary_loader import GlossaryLoader

# Initialize with default config
loader = GlossaryLoader()
glossary_terms, stats = loader.load_all_glossaries()

# Or load specific config file
loader = GlossaryLoader(config_path="path/to/glossary_config.yaml")
glossary_terms, stats = loader.load_all_glossaries()

print(f"Loaded {stats['total_terms']} terms")
print(f"Sources: {stats['sources_loaded']}")
```

### Load Single Glossary Source

```python
loader = GlossaryLoader()

# Load specific Excel file with custom column mapping
excel_terms = loader.load_glossary(
    file_path="data/glossaries/my_terms.xlsx",
    file_type="excel",
    sheet_name="Sheet1",
    mapping={
        "korean": "Korean",
        "english": "English",
        "category": "Domain"
    }
)

# Load JSON glossary
json_terms = loader.load_glossary(
    file_path="data/glossaries/medical_terms.json",
    file_type="json"
)

# Load CSV glossary
csv_terms = loader.load_glossary(
    file_path="data/glossaries/custom_terms.csv",
    file_type="csv",
    mapping={
        "korean": "ko_term",
        "english": "en_term"
    }
)
```

### Search Glossary Terms

```python
from src.glossary.glossary_search import GlossarySearchEngine

# Initialize search engine
search_engine = GlossarySearchEngine(
    min_term_length=2,
    fuzzy_threshold=0.60
)

# Load glossary into search engine
search_engine.load_terms(glossary_terms)

# Search for Korean term
results = search_engine.search("임상시험")
# Returns:
# [
#   {
#     'korean': '임상시험',
#     'english': 'clinical trial',
#     'confidence': 0.95,
#     'match_type': 'exact',
#     'source': 'Clinical_Trials'
#   },
#   {
#     'korean': '임상시험',
#     'english': 'clinical study',
#     'confidence': 0.90,
#     'match_type': 'exact',
#     'source': 'Clinical_Trials_Alternatives'
#   }
# ]

# Advanced search with options
results = search_engine.search(
    term="임상시험",
    language="ko",
    include_alternatives=True,
    min_confidence=0.80
)
```

### Use in Translation Pipeline

```python
from src.production_pipeline_batch_enhanced import EnhancedBatchPipeline

# Pipeline auto-loads glossary from config
pipeline = EnhancedBatchPipeline(
    model_name="Owl",
    glossary_config="path/to/glossary_config.yaml"
)

# Or programmatically
glossary_terms, stats = loader.load_all_glossaries()
pipeline = EnhancedBatchPipeline(glossary=glossary_terms)

# Translate with glossary support
results = pipeline.run_enhanced_batch_pipeline(
    input_file="documents/input.xlsx"
)
```

---

## Input File Formats

### Excel Format

**Standard Column Names (auto-detected):**
```
korean | english | category | frequency | source
```

**Custom Column Mapping:**
```python
# If your Excel has different column names, specify mapping
mapping = {
    "korean": "KR_Term",
    "english": "EN_Term",
    "category": "Domain",
    "frequency": "Usage_Frequency"
}
```

**Example Excel Structure:**
```
Sheet: "Clinical Terms"

| KR_Term      | EN_Term           | Domain   | Usage_Frequency |
|---|---|---|---|
| 임상시험     | clinical trial    | clinical | high            |
| 피험자       | study subject     | clinical | high            |
| 부작용       | adverse event     | safety   | medium          |
```

### JSON Format

**Standard Structure:**
```json
{
  "glossary_metadata": {
    "version": "1.0",
    "language_pair": "ko-en",
    "created_date": "2025-11-23",
    "source": "custom_domain"
  },
  "terms": [
    {
      "korean": "임상시험",
      "english": "clinical trial",
      "category": "clinical",
      "frequency": "high",
      "context": "의약품 또는 의료기기의 효능과 안전성을 평가하는 시험",
      "alternatives": {
        "korean": ["임상연구"],
        "english": ["clinical study"]
      }
    }
  ],
  "abbreviations": [
    {
      "korean": "ICH",
      "english": "International Council for Harmonisation",
      "context": "Regulatory standards"
    }
  ]
}
```

### CSV Format

**Standard Structure:**
```csv
korean,english,category,frequency,source
임상시험,clinical trial,clinical,high,trials
피험자,study subject,clinical,high,trials
부작용,adverse event,safety,medium,trials
```

---

## Output Formats

### Excel Output

The glossary loader can export to Excel with multiple sheets:

```python
loader = GlossaryLoader()
glossary_terms, stats = loader.load_all_glossaries()

# Save to Excel
loader.save_glossary(
    terms=glossary_terms,
    output_file="output/glossary.xlsx",
    format="excel",
    include_metadata=True
)
```

**Output Structure:**
```
Sheet 1: Combined_Glossary
├── korean | english | category | frequency | source | confidence

Sheet 2: Abbreviations (if any)
├── korean | english | context

Sheet 3: Statistics
├── Total Terms | Sources Loaded | Categories | etc.
```

### JSON Output

```python
loader.save_glossary(
    terms=glossary_terms,
    output_file="output/glossary.json",
    format="json"
)
```

### CSV Output

```python
loader.save_glossary(
    terms=glossary_terms,
    output_file="output/glossary.csv",
    format="csv"
)
```

---

## Common Use Cases

### Use Case 1: Load Multiple Glossaries from Different Domains

```python
from src.glossary.glossary_loader import GlossaryLoader

loader = GlossaryLoader(config_path="glossary_config.yaml")

# Config specifies 3 domains: clinical, device, regulatory
glossary_terms, stats = loader.load_all_glossaries()

print(f"Loaded from {len(stats['sources'])} sources:")
for source, count in stats['sources'].items():
    print(f"  - {source}: {count} terms")
```

### Use Case 2: Add Custom Terms to Existing Glossary

```python
# Load existing glossary
loader = GlossaryLoader()
glossary_terms, stats = loader.load_all_glossaries()

# Add custom terms
custom_terms = [
    {
        'korean': '사용자 정의 용어',
        'english': 'user defined term',
        'category': 'custom',
        'source': 'user_input',
        'confidence_score': 0.80
    }
]

# Combine
all_terms = glossary_terms + custom_terms

# Save combined glossary
loader.save_glossary(
    terms=all_terms,
    output_file="combined_glossary.json"
)
```

### Use Case 3: Create Domain-Specific Glossary

```python
loader = GlossaryLoader()

# Load only clinical domain
clinical_terms = loader.load_glossary(
    file_path="data/glossaries/clinical.xlsx",
    mapping={'korean': 'Korean', 'english': 'English'}
)

# Filter for specific domain
clinical_only = [t for t in clinical_terms if t.get('category') == 'clinical']

# Save domain-specific glossary
loader.save_glossary(
    terms=clinical_only,
    output_file="clinical_glossary.json"
)
```

### Use Case 4: Merge Multiple External Glossaries

```python
loader = GlossaryLoader()

# Load from multiple sources
sources = [
    "data/glossaries/source1.xlsx",
    "data/glossaries/source2.json",
    "data/glossaries/source3.csv"
]

all_terms = []
for source_path in sources:
    terms = loader.load_glossary(file_path=source_path)
    all_terms.extend(terms)

# Deduplicate
merged_glossary = loader.deduplicate_terms(all_terms)

# Save merged
loader.save_glossary(
    terms=merged_glossary,
    output_file="merged_glossary.json"
)
```

---

## Glossary Term Structure

### Standard Term Object

```python
{
    'korean': str,                      # Required: Korean term
    'english': str,                     # Required: English term
    'category': str,                    # Optional: Domain category
    'source': str,                      # Optional: Where it came from
    'confidence_score': float,          # Optional: 0.0-1.0 confidence
    'frequency': str,                   # Optional: 'high', 'medium', 'low'
    'context': str,                     # Optional: Usage context
    'alternatives': {                   # Optional: Alternative terms
        'korean': [str, ...],
        'english': [str, ...]
    }
}
```

### Example Terms

```python
# Simple term
{
    'korean': '임상시험',
    'english': 'clinical trial',
    'source': 'clinical_trials'
}

# Term with alternatives
{
    'korean': '임상시험',
    'english': 'clinical trial',
    'category': 'clinical',
    'source': 'clinical_trials',
    'confidence_score': 0.95,
    'frequency': 'high',
    'alternatives': {
        'korean': ['임상연구', '임상시험'],
        'english': ['clinical study', 'clinical investigation']
    }
}
```

---

## Best Practices

### 1. Organize Glossaries by Domain
```
data/glossaries/
├── clinical/
│   ├── clinical_trials.xlsx
│   └── protocols.json
├── device/
│   ├── medical_devices.xlsx
│   └── biocompatibility.csv
└── regulatory/
    ├── fda_terms.xlsx
    └── ich_gcp.json
```

### 2. Use Configuration Files
- Keep glossary sources in config files, not hardcoded
- Version control your configs
- Easy to switch glossaries for different projects

### 3. Version Your Glossaries
```json
{
  "glossary_metadata": {
    "version": "2.0",
    "created_date": "2025-11-23",
    "updated_date": "2025-11-23",
    "previous_version": "1.0"
  }
}
```

### 4. Maintain Confidence Scores
- Clinical terms: 0.90-0.95
- Device terms: 0.85-0.90
- User/custom terms: 0.70-0.85
- Automatically lower confidence for fuzzy matches

### 5. Regular Deduplication
```python
# Check for duplicates periodically
loader = GlossaryLoader()
all_terms = loader.load_all_glossaries()

duplicates = loader.find_duplicates(all_terms)
print(f"Found {len(duplicates)} duplicate terms")

# Merge duplicates
merged = loader.deduplicate_terms(all_terms)
```

### 6. Document Your Glossaries
```json
{
  "glossary_metadata": {
    "version": "1.0",
    "description": "Clinical trial terminology for Phase 2 MVP",
    "language_pair": "ko-en",
    "domain": "clinical_trials",
    "created_by": "translation_team",
    "created_date": "2025-11-23",
    "coverage": "95% of clinical trial documents",
    "maintenance": "Updated monthly",
    "notes": "Follows ICH GCP guidelines"
  }
}
```

---

## Troubleshooting

### Issue: "Column not found" error

**Solution:** Check your column mapping matches actual Excel columns

```python
# First, inspect available columns
import pandas as pd
df = pd.read_excel("glossary.xlsx")
print(df.columns)

# Then use correct names in mapping
mapping = {
    "korean": "actual_korean_column_name",
    "english": "actual_english_column_name"
}
```

### Issue: Low search accuracy

**Solution:** Adjust fuzzy matching threshold

```python
search_engine = GlossarySearchEngine(
    fuzzy_threshold=0.70  # Stricter matching
)
```

### Issue: Duplicate terms in results

**Solution:** Enable deduplication in config

```yaml
deduplication:
  enabled: true
  keep_first: true
  merge_alternatives: true
```

### Issue: Performance degradation with large glossaries

**Solution:** Enable Valkey caching

```python
search_engine = GlossarySearchEngine()
search_engine.enable_caching(
    host="localhost",
    port=6379,
    ttl=3600
)
```

---

## Performance Metrics

### Glossary Loading
- Small glossary (< 500 terms): < 100ms
- Medium glossary (500-2000 terms): 100-500ms
- Large glossary (2000+ terms): 500ms-2s

### Search Performance
- Exact match: < 1ms (with Valkey cache)
- Fuzzy match: 1-10ms (depends on glossary size)
- Pattern match: 10-50ms

### Memory Usage
- Per 1000 terms: ~1-2MB (in memory)
- Valkey cache: < 100ms latency, persistent
- Search indices: ~10% of glossary size

---

## API Reference

### GlossaryLoader

```python
class GlossaryLoader:
    def __init__(config_path: str = None)
    def load_all_glossaries() -> Tuple[List[Dict], Dict]
    def load_glossary(file_path: str, file_type: str, **options) -> List[Dict]
    def save_glossary(terms: List[Dict], output_file: str, format: str) -> None
    def deduplicate_terms(terms: List[Dict]) -> List[Dict]
    def find_duplicates(terms: List[Dict]) -> List[Tuple[Dict, Dict]]
    def merge_glossaries(glossaries: List[List[Dict]]) -> List[Dict]
```

### GlossarySearchEngine

```python
class GlossarySearchEngine:
    def __init__(min_term_length: int = 2, fuzzy_threshold: float = 0.60)
    def load_terms(terms: List[Dict]) -> None
    def search(term: str, language: str = "ko", **options) -> List[Dict]
    def enable_caching(host: str, port: int, ttl: int) -> None
    def get_statistics() -> Dict
```

---

## See Also

- [Pipeline Documentation](TRANSLATION_PIPELINE_STEPBYSTEP.md)
- [Memory System](VALKEY_INTEGRATION_SUMMARY.md)
- [Style Guides](src/STYLE_GUIDE_AB_TESTING_README.md)
