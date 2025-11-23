# TransAI Translation Pipeline - Actual Implementation

## Overview

This document describes the **actual implemented pipeline** for the TransAI translation system. This is what the code does today - no future plans, no aspirational features.

## System Architecture Reality

**What we have:**
- Excel input/output processing
- Glossary-based context building
- LLM integration (GPT-5 OWL)
- Basic term consistency tracking via Valkey
- Multiple pipeline variants for different use cases

**What we DON'T have:**
- ❌ Translation Memory (TM) matching system
- ❌ Semantic vector search (Qdrant)
- ❌ Adaptive learning (Mem0)
- ❌ Translation routing logic
- ❌ Document type detection

---

## Actual 5-Step Pipeline Process

### Step 1: Load Input Data

**Purpose**: Read source text and glossary data

**Input**:
- Excel file with source text segments
- Glossary files (JSON or Excel format)

**Process**:
```python
# Load Excel document
segments = load_excel_file(input_path)
# Returns: List of {id, source_text, reference_text, metadata}

# Load glossary
glossary_terms = GlossaryLoader().load_all_glossaries()
# Returns: List of {korean, english, source, priority} dictionaries
```

**Output**:
- List of segments to translate
- Loaded glossary (419 or 2,906 terms depending on pipeline)

**Files**:
- `src/glossary/glossary_loader.py` - Generic glossary loading
- All `production_pipeline_*.py` files use this

---

### Step 2: Build Context for Translation

**Purpose**: Gather relevant information to help LLM translate accurately

**Input**:
- Current segment(s) to translate
- Full glossary
- Previous translations from this session
- Style guide configuration

**Process**:
```python
# For each segment or batch of 5 segments:

# 2.1: Search glossary for matching terms
matching_terms = glossary_search(source_text, glossary_terms)
# Simple keyword matching - finds terms that appear in source

# 2.2: Check Valkey for locked terms (consistency)
locked_terms = valkey.get_locked_terms(document_id)
# Returns: {source_term: target_term} from previous segments

# 2.3: Get previous translations (last 3-5 segments)
previous_context = get_recent_translations(count=3)

# 2.4: Get style guide rules
style_guide = StyleGuideManager().get_current_guide()
# Returns: ~100-600 tokens of translation guidelines

# 2.5: Assemble context
context = f"""
Document Context: {style_guide}

Glossary Terms for this segment:
{matching_terms}

Previously established translations:
{locked_terms}

Recent translations for continuity:
{previous_context}
"""
```

**Output**:
- Rich context string (~200-500 tokens)
- Ready for LLM prompt

**Files**:
- `src/production_pipeline_batch_enhanced.py:213` - `build_enhanced_batch_context()`
- `src/glossary/glossary_search.py` - Term matching
- `src/memory/valkey_manager.py` - Term consistency
- `src/style_guide_config.py` - Style guide variants

---

### Step 3: Translate via LLM

**Purpose**: Call GPT-5 OWL to translate with context

**Input**:
- Context from Step 2
- Segment(s) to translate (1 or 5 depending on pipeline)

**Process**:
```python
# Build prompt
prompt = f"""
{context}

## Segments to translate:
1. {segment_1_korean}
2. {segment_2_korean}
...

## Instructions:
Translate to English following the glossary and style guide.
Maintain consistency with locked terms.

## Response Format:
1. [English translation]
2. [English translation]
...
"""

# Call GPT-5 OWL
response = openai.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": prompt}],
    text={"verbosity": "medium"},
    reasoning={"effort": "minimal"}
)

# Extract translations
translations = parse_numbered_response(response.output_text)
```

**Output**:
- List of translations
- Token usage (input + output)
- Cost ($0.00125 per 1K input, $0.01 per 1K output)

**Costs**:
- Batch mode (5 segments): ~$0.006 per segment
- Individual mode: ~$0.01 per segment

**Files**:
- All `production_pipeline_*.py` files
- Uses OpenAI SDK directly

---

### Step 4: Track Term Consistency

**Purpose**: Store term mappings for document-wide consistency

**Input**:
- Source text
- Translation
- Segment ID

**Process**:
```python
# Extract term pairs from source and translation
term_pairs = extract_term_pairs(source_text, translation, glossary_terms)

# Store in Valkey for this document session
for source_term, target_term in term_pairs:
    valkey.add_term_mapping(
        doc_id=document_id,
        source_term=source_term,
        target_term=target_term,
        segment_id=segment_id
    )

    # Lock term if it appears multiple times
    if term_frequency > 1:
        valkey.lock_term(doc_id, source_term)
```

**Output**:
- Updated term mapping database in Valkey
- Locked terms available for next segments

**Data Structure in Valkey**:
```
doc:{doc_id}:terms -> Hash of {source_term: target_term}
doc:{doc_id}:locked -> Set of locked source terms
doc:{doc_id}:metadata -> Session info (progress, status)
```

**Files**:
- `src/memory/valkey_manager.py` - Core storage
- `src/memory/session_manager.py` - Session tracking
- `src/memory/consistency_tracker.py` - Term consistency logic

---

### Step 5: Generate Output

**Purpose**: Write results to Excel with metrics

**Input**:
- All translations
- Metrics (tokens, costs, quality scores)
- Original Excel structure

**Process**:
```python
# Create results dataframe
results = []
for segment in translated_segments:
    results.append({
        'Segment ID': segment.id,
        'Source Text': segment.source,
        'Translation': segment.translation,
        'Quality Score': segment.quality_score,
        'Tokens Used': segment.tokens,
        'Cost': segment.cost,
        'Glossary Terms': segment.terms_used,
        'Method': 'LLM'  # Always LLM (no TM)
    })

df = pd.DataFrame(results)

# Write to Excel
with pd.ExcelWriter(output_path) as writer:
    df.to_excel(writer, sheet_name='Translations', index=False)

    # Add metrics sheet
    metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
```

**Output**:
- Excel file with translations
- Metrics summary
- Log file

**Metrics Tracked**:
- Total segments processed
- Average quality score (0.5-1.0 based on heuristics)
- Total tokens used
- Total cost
- Processing time
- Glossary term usage

**Files**:
- All `production_pipeline_*.py` files
- Results saved to timestamped Excel files

---

## Quality Scoring (Current Implementation)

**How quality is calculated** (from `production_pipeline_batch_enhanced.py:421`):

```python
def assess_batch_quality(korean_text, translation):
    score = 0.5  # Base score

    # Translation exists and non-empty
    if translation and len(translation.strip()) > 0:
        score += 0.2

    # Length ratio reasonable (0.5x to 3x source length)
    length_ratio = len(translation) / len(korean_text)
    if 0.5 <= length_ratio <= 3.0:
        score += 0.1

    # Glossary terms found in translation
    terms_used = count_glossary_terms_in_translation(translation)
    score += min(0.2, terms_used * 0.05)

    # Clinical terminology present
    clinical_patterns = ['study', 'clinical', 'trial', 'patient']
    pattern_matches = count_patterns(translation, clinical_patterns)
    score += min(0.1, pattern_matches * 0.02)

    return min(1.0, score)  # Cap at 1.0
```

**Note**: This is heuristic-based, not a real quality metric. It doesn't compare against reference translations.

---

## Pipeline Variants

All 6 pipelines follow the same 5-step process with minor variations:

### 1. `production_pipeline_batch_enhanced.py` ⭐ RECOMMENDED
- **Batch size**: 5 segments per API call
- **Glossary**: 2,906 terms
- **Style guide**: Configurable (10 variants)
- **Speed**: ~0.4-0.5s per segment
- **Cost**: ~$0.006 per segment

### 2. `production_pipeline_en_ko.py`
- **Direction**: EN→KO specialized
- **Glossary**: 419 clinical terms
- **Batch size**: 5 segments
- **Use case**: Clinical protocol translation

### 3. `production_pipeline_ko_en_improved.py`
- **Direction**: KO→EN
- **Special feature**: CAT tool tag preservation
- **Process**: Extract tags → Translate → Restore tags
- **Use case**: CAT tool workflows

### 4. `production_pipeline_with_style_guide.py`
- **Batch size**: 1 (individual processing)
- **Feature**: A/B tests 10 style guide variants
- **Use case**: Quality optimization experiments
- **Speed**: Slower (~2s per segment)

### 5. `production_pipeline_working.py`
- **Status**: Legacy reference
- **Glossary**: 2,906 terms
- **Use case**: Stable baseline

### 6. `production_pipeline_en_ko_improved.py`
- **Direction**: EN→KO
- **Status**: Alternative variant
- **Use case**: Similar to #2

---

## Memory Architecture (Actual)

### What Exists: Valkey Only (Tier 1)

```
┌─────────────────────────────────────┐
│         Valkey/Redis Cache          │
├─────────────────────────────────────┤
│                                     │
│  doc:{id}:metadata                 │
│  └─> Session info, progress        │
│                                     │
│  doc:{id}:terms                    │
│  └─> {source: target} mappings     │
│                                     │
│  doc:{id}:locked                   │
│  └─> Set of locked terms           │
│                                     │
└─────────────────────────────────────┘
         ▲
         │ O(1) lookups
         │ <1ms latency
         │
    Pipeline uses this
    for consistency only
```

**Usage**: Term consistency tracking only
**NOT used for**: Translation caching, TM matching, semantic search

### What Does NOT Exist

- ❌ **Qdrant** (Tier 2) - Semantic vector search
- ❌ **Mem0** (Tier 3) - Adaptive learning
- ❌ **Translation Memory** - Reusable translation database
- ❌ **Celery** - Distributed job queue

These are listed in `requirements.txt` but not used in any code.

---

## Data Flow Diagram (Actual)

```
┌─────────────┐
│ Input Excel │
│ (Korean)    │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ Load Glossary       │
│ (419-2,906 terms)   │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ For each segment or batch of 5:     │
├─────────────────────────────────────┤
│ 1. Search glossary for terms        │
│ 2. Check Valkey for locked terms    │
│ 3. Get last 3-5 translations        │
│ 4. Add style guide rules            │
│ 5. Build prompt (~200-500 tokens)   │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────┐
│ GPT-5 OWL API       │
│ (Always LLM)        │
│ No TM routing       │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Parse response      │
│ Extract translations│
└──────┬──────────────┘
       │
       ▼
┌─────────────────────────┐
│ Store term mappings     │
│ in Valkey for           │
│ next segment consistency│
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────┐
│ Calculate metrics:  │
│ - Quality score     │
│ - Tokens used       │
│ - Cost              │
└──────┬──────────────┘
       │
       ▼
┌─────────────┐
│ Excel Output│
│ (English)   │
└─────────────┘
```

---

## Performance Characteristics

### Token Usage

**Average per segment** (from actual runs):
- Source text: ~17.6 tokens
- Context (glossary + style guide + history): ~200-400 tokens
- **Total input**: ~220-420 tokens per segment
- **Output**: ~30-50 tokens per segment

### Cost Breakdown (GPT-5 OWL Pricing)

```python
# Input: $1.25 per 1M tokens
# Output: $10.00 per 1M tokens

# Batch mode (5 segments):
input_cost = 305.6 tokens * 5 * 0.00000125 = $0.00191
output_cost = 40 tokens * 5 * 0.00001 = $0.00200
total_per_batch = $0.00391
cost_per_segment = $0.00078

# Plus overhead → ~$0.006 per segment actual cost
```

### Processing Speed

- **Batch mode**: ~0.4-0.5s per segment (2.5s per 5 segments)
- **Individual mode**: ~2s per segment
- **Throughput**: ~120-150 segments/minute (batch mode)

### Quality

- **Average quality score**: 0.84 (range: 0.74-0.98)
- **Note**: Heuristic-based, not validated against references

---

## Key Files Reference

### Core Pipelines
```
src/production_pipeline_batch_enhanced.py  (RECOMMENDED)
src/production_pipeline_en_ko.py
src/production_pipeline_ko_en_improved.py
src/production_pipeline_with_style_guide.py
```

### Glossary System
```
src/glossary/glossary_loader.py         - Load glossaries
src/glossary/glossary_search.py         - Term matching
src/glossary/create_combined_glossary.py - Combine sources
```

### Memory (Valkey Only)
```
src/memory/valkey_manager.py           - Core Valkey operations
src/memory/session_manager.py          - Session tracking
src/memory/consistency_tracker.py      - Term consistency
```

### Utilities
```
src/utils/tag_handler.py               - CAT tool tags
src/utils/segment_filter.py            - Content filtering
src/style_guide_config.py              - 10 style variants
```

### Tests
```
src/tests/test_phase2_integration.py   - Integration tests
src/tests/test_valkey_integration.py   - Valkey tests
src/tests/test_be003_core.py           - Core functionality
```

---

## Configuration

### Environment Variables (.env)
```bash
OPENAI_API_KEY=sk-proj-your_key_here
VALKEY_HOST=localhost
VALKEY_PORT=6379
LOG_LEVEL=INFO
```

### Style Guide Variants

From `src/style_guide_config.py`:
1. **NONE** - No style guide (baseline)
2. **MINIMAL** - ~100 tokens
3. **COMPACT** - ~200 tokens
4. **STANDARD** - ~400 tokens (recommended)
5. **COMPREHENSIVE** - ~600 tokens
6. **CLINICAL_PROTOCOL** - EN→KO specialized
7-10. Additional regulatory variants

---

## Limitations & Trade-offs

### Current Approach

✅ **Strengths**:
- Simple, understandable architecture
- Works reliably
- Good glossary integration
- Term consistency tracking
- Multiple pipeline options

❌ **Limitations**:
- Every segment costs LLM tokens (no TM reuse)
- No semantic search
- No learning/adaptation
- Basic quality scoring
- No distributed processing

### Cost Reality

**Without Translation Memory:**
- 1,400 segments × $0.006 = ~$8.40 per document
- All costs are LLM API calls

**With TM (if implemented):**
- Could save 40-50% by reusing matches
- Would require Qdrant for semantic search
- Additional infrastructure complexity

---

## When to Use Each Pipeline

| Use Case | Pipeline | Why |
|----------|----------|-----|
| General production | batch_enhanced | Best cost/speed balance |
| EN→KO clinical docs | en_ko | Specialized 419 term glossary |
| KO→EN with CAT tools | ko_en_improved | Tag preservation |
| Quality experiments | with_style_guide | A/B testing variants |
| Baseline reference | working | Stable, proven |

---

## Maintenance Notes

### Adding New Glossary Terms
```python
# Add to existing glossary file
# Reload via GlossaryLoader
loader = GlossaryLoader()
terms = loader.load_all_glossaries()
```

### Adjusting Batch Size
```python
# In pipeline initialization
pipeline = EnhancedBatchPipeline(batch_size=10)  # Default: 5
```

### Changing Style Guide
```python
# Use different variant
pipeline = EnhancedBatchPipeline(
    style_guide_variant="comprehensive"  # vs "standard"
)
```

### Monitoring Valkey
```bash
# Check Valkey status
valkey-cli ping

# View stored keys
valkey-cli keys "doc:*"

# Check memory usage
valkey-cli INFO memory
```

---

## Summary

This is the **actual implementation** - a 5-step pipeline that:
1. Loads data and glossary
2. Builds context from glossary + history + style guide
3. Calls GPT-5 OWL for every translation
4. Tracks term consistency in Valkey
5. Outputs Excel with metrics

**It works well** for what it does. It's just simpler than what some docs describe.

**No Translation Memory. No Qdrant. No Mem0. Just glossary + smart prompting + LLM.**

---

**Document Version**: 2.0 (Simplified to match actual implementation)
**Last Updated**: November 23, 2025
**Status**: Accurate reflection of current codebase
