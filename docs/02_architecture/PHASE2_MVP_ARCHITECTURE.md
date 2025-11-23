# TransAI System Architecture (Actual Implementation)

## Overview

This document describes the actual architecture of the TransAI translation system as currently implemented.

**Architecture Philosophy**: Simple, reliable, glossary-enhanced LLM translation with term consistency tracking.

---

## System Components

```
┌──────────────────────────────────────────────────────────┐
│                TransAI Translation System                 │
└──────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
    ┌───▼────┐      ┌────▼─────┐    ┌─────▼──────┐
    │ Input  │      │ Glossary │    │  Memory    │
    │ Layer  │      │  System  │    │  (Valkey)  │
    └───┬────┘      └────┬─────┘    └─────┬──────┘
        │                │                 │
        └────────────────▼─────────────────┘
                         │
                 ┌───────▼────────┐
                 │ Pipeline Layer │
                 │  (6 variants)  │
                 └───────┬────────┘
                         │
                 ┌───────▼────────┐
                 │   LLM Layer    │
                 │  (GPT-5 OWL)   │
                 └───────┬────────┘
                         │
                  ┌──────▼───────┐
                  │ Output Layer │
                  │   (Excel)    │
                  └──────────────┘
```

---

## Layer 1: Input Processing

### Components
- **Excel Parser** - Reads source segments from Excel files
- **Data Validator** - Validates required columns and data format
- **Batch Generator** - Groups segments for batch processing (optional)

### Data Flow
```python
Input: Excel file with columns [Segment ID, Source Text, Reference, Comments]
  ↓
Parse and validate
  ↓
Output: List of segment dictionaries
```

### Implementation
- Uses `pandas` and `openpyxl` for Excel I/O
- All pipelines implement similar input handling
- Supports both individual and batch modes

---

## Layer 2: Glossary System

### Components

**Glossary Loader** (`src/glossary/glossary_loader.py`)
- Loads glossaries from JSON or Excel format
- Supports multiple glossary sources
- Handles 419-2,906 terms depending on pipeline

**Glossary Search** (`src/glossary/glossary_search.py`)
- Simple keyword matching algorithm
- Finds terms that appear in source text
- Returns matching term pairs with metadata

**Combined Glossary** (`src/glossary/create_combined_glossary.py`)
- Combines multiple glossary sources
- Priority-based deduplication
- Source tracking

### Glossary Data
- **Small glossary**: 419 clinical terms (EN-KO specialized)
- **Large glossary**: 2,906 terms (comprehensive)
- Format: `{korean, english, source, priority}`

### Usage Pattern
```python
# Load glossary
glossary_loader = GlossaryLoader()
terms = glossary_loader.load_all_glossaries()

# Search for matching terms
matches = search_glossary(source_text, terms)
# Returns: List of {korean, english} dicts found in source_text
```

---

## Layer 3: Memory Architecture (Simple)

### What Exists: Valkey Only

```
┌─────────────────────────────────┐
│     Valkey/Redis (Port 6379)    │
├─────────────────────────────────┤
│                                 │
│  Session Management:            │
│  • doc:{id}:metadata            │
│  • doc:{id}:terms               │
│  • doc:{id}:locked              │
│                                 │
│  Purpose:                       │
│  • Track term consistency       │
│  • Store session progress       │
│  • Lock terms for reuse         │
│                                 │
│  Performance:                   │
│  • O(1) lookups                 │
│  • <1ms latency                 │
│  • Connection pooling           │
│                                 │
└─────────────────────────────────┘
```

### Valkey Manager (`src/memory/valkey_manager.py`)

**Core Operations:**
```python
# Session management
create_session(doc_id, source_lang, target_lang, total_segments)
update_session_progress(doc_id, processed_count)

# Term consistency
add_term_mapping(doc_id, source_term, target_term, segment_id)
get_term_mapping(doc_id, source_term) → target_term
lock_term(doc_id, source_term)  # Prevent changes

# Session cleanup
cleanup_session(doc_id)  # Remove after completion
```

**Data Structures:**
- `doc:{id}:metadata` - Hash: session info, progress, status
- `doc:{id}:terms` - Hash: `{source_term: target_term}` mappings
- `doc:{id}:locked` - Set: locked term list

### Session Manager (`src/memory/session_manager.py`)

Wraps Valkey operations for document-level session tracking:
```python
session_mgr = SessionManager(valkey_manager)
session_id = session_mgr.create_session(doc_name, total_segments)
session_mgr.update_progress(session_id, segment_id)
session_mgr.end_session(session_id)
```

### Consistency Tracker (`src/memory/consistency_tracker.py`)

Ensures term consistency across document:
```python
tracker = ConsistencyTracker(valkey_manager, doc_id)
tracker.add_term(source, target, segment_id)
tracker.check_consistency(source)  # Get locked translation
```

### What Does NOT Exist

- ❌ **Qdrant** - No vector database for semantic search
- ❌ **Mem0** - No adaptive learning layer
- ❌ **Translation Memory** - No TM database for reuse
- ❌ **Caching of glossary results** - Not implemented despite Valkey capabilities

> **Note**: These components are listed in `requirements.txt` but are not used in any code.

---

## Layer 4: Pipeline Processing

### Available Pipelines

#### 1. Batch Enhanced Pipeline (Recommended)
**File**: `src/production_pipeline_batch_enhanced.py`

- **Batch size**: 5 segments per API call
- **Glossary**: 2,906 terms
- **Style guide**: Configurable (10 variants)
- **Speed**: ~0.4-0.5s per segment
- **Use case**: General production use

#### 2. EN-KO Specialized Pipeline
**File**: `src/production_pipeline_en_ko.py`

- **Direction**: English → Korean
- **Glossary**: 419 clinical terms (specialized)
- **Batch size**: 5 segments
- **Use case**: Clinical protocol translation

#### 3. KO-EN with Tag Preservation
**File**: `src/production_pipeline_ko_en_improved.py`

- **Direction**: Korean → English
- **Special feature**: CAT tool tag handling
- **Process**: Extract tags → Translate → Restore tags
- **Use case**: CAT tool integration

#### 4. Style Guide Testing Pipeline
**File**: `src/production_pipeline_with_style_guide.py`

- **Batch size**: 1 (individual processing)
- **Feature**: A/B test 10 style guide variants
- **Use case**: Quality optimization experiments

#### 5 & 6. Additional Variants
- `production_pipeline_working.py` - Legacy reference
- `production_pipeline_en_ko_improved.py` - Alternative EN-KO

### Common Pipeline Flow

All pipelines follow this pattern:

```
1. LOAD DATA
   └─> Read Excel + Load glossary

2. FOR EACH SEGMENT (or batch of 5):
   ├─> Search glossary for matching terms
   ├─> Check Valkey for locked terms
   ├─> Get previous 3-5 translations
   ├─> Add style guide rules
   └─> Build prompt (~200-500 tokens)

3. CALL LLM
   └─> GPT-5 OWL API (everything goes to LLM)

4. TRACK CONSISTENCY
   └─> Store term pairs in Valkey

5. OUTPUT
   └─> Write Excel with results + metrics
```

---

## Layer 5: LLM Integration

### GPT-5 OWL (Primary Model)

**API Integration**:
```python
import openai

client = openai.OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": prompt}],
    text={"verbosity": "medium"},
    reasoning={"effort": "minimal"}
)
```

**Prompt Structure**:
```
[Style Guide Rules: 100-600 tokens]

[Glossary Terms: 50-200 tokens]
- korean1 → english1
- korean2 → english2
...

[Locked Terms: 20-100 tokens]
- term1: locked_translation1
...

[Previous Context: 30-80 tokens]
Recent translations for continuity...

[Segments to Translate]
1. segment_text_1
2. segment_text_2
...

[Instructions]
Translate following glossary and style guide.
```

**Pricing** (2025 rates):
- Input: $1.25 per 1M tokens
- Output: $10.00 per 1M tokens
- **Effective cost**: ~$0.006 per segment (batch mode)

### No Fallback Models

Currently only GPT-5 OWL is used. API keys for other providers (Anthropic, Gemini, Upstage) are in `.env` but not utilized.

---

## Layer 6: Output Generation

### Excel Export

**Output Format**:
```
Sheet 1: Translations
├─ Segment ID
├─ Source Text
├─ Translation
├─ Quality Score (0.5-1.0)
├─ Tokens Used
├─ Cost
├─ Glossary Terms Used
└─ Method (always "LLM")

Sheet 2: Metrics Summary
├─ Total segments
├─ Average quality score
├─ Total tokens
├─ Total cost
├─ Processing time
└─ Glossary coverage
```

### Quality Scoring

**Current heuristic** (not validated):
```python
score = 0.5  # Base
+ 0.2 if translation exists
+ 0.1 if length ratio reasonable (0.5x-3x)
+ up to 0.2 for glossary term usage
+ up to 0.1 for clinical terminology patterns
= 0.5 to 1.0 final score
```

**Note**: This is NOT a real quality metric. It doesn't compare against reference translations or use BLEU/COMET scores.

---

## Data Flow Diagram

```
┌────────────────┐
│  User uploads  │
│  Excel file    │
└────────┬───────┘
         │
         ▼
┌────────────────────────────┐
│  Load Glossary             │
│  (419 or 2,906 terms)      │
└────────┬───────────────────┘
         │
         ▼
┌────────────────────────────┐
│  Create Valkey Session     │
│  doc:{id}:metadata         │
└────────┬───────────────────┘
         │
         ▼
┌────────────────────────────┐
│  FOR EACH SEGMENT/BATCH:   │
│  ┌──────────────────────┐  │
│  │ 1. Search glossary   │  │
│  │ 2. Check Valkey      │  │
│  │ 3. Get prev context  │  │
│  │ 4. Add style guide   │  │
│  │ 5. Build prompt      │  │
│  └──────────────────────┘  │
└────────┬───────────────────┘
         │
         ▼
┌────────────────────────────┐
│  Call GPT-5 OWL API        │
│  (No TM check, always LLM) │
└────────┬───────────────────┘
         │
         ▼
┌────────────────────────────┐
│  Parse Response            │
│  Extract Translations      │
└────────┬───────────────────┘
         │
         ▼
┌────────────────────────────┐
│  Update Valkey:            │
│  - Store term pairs        │
│  - Lock frequent terms     │
│  - Update progress         │
└────────┬───────────────────┘
         │
         ▼
┌────────────────────────────┐
│  Calculate Metrics         │
│  (tokens, cost, quality)   │
└────────┬───────────────────┘
         │
         ▼
┌────────────────────────────┐
│  Write Excel Output        │
│  (Translations + Metrics)  │
└────────────────────────────┘
```

---

## Performance Characteristics

### Throughput
- **Batch mode**: ~120-150 segments/minute
- **Individual mode**: ~30 segments/minute
- **Bottleneck**: OpenAI API latency (~2.5s per batch)

### Token Usage
- **Per segment average**: 220-420 input tokens, 30-50 output tokens
- **Batch efficiency**: 5 segments share context = lower per-segment cost

### Cost
- **Batch mode**: $0.006 per segment
- **Individual mode**: $0.01 per segment
- **1,400 segments**: ~$8.40 (batch) vs ~$14 (individual)

### Memory
- **Valkey**: ~10MB for typical session (1,400 segments)
- **Python process**: ~200-300MB
- **Minimal footprint** - no large vector databases

---

## Integration Points

### Adding New Pipelines

All pipelines should implement:
```python
class NewPipeline:
    def __init__(self, model_name, batch_size, style_guide_variant):
        # Initialize glossary, Valkey, style guide
        pass

    def translate_segment(self, segment):
        # 1. Build context
        # 2. Call LLM
        # 3. Track consistency
        # 4. Return result
        pass

    def run_pipeline(self, input_file):
        # Process all segments and generate output
        pass
```

### Adding New Glossaries

```python
# 1. Prepare glossary file (JSON or Excel)
# Format: [{korean, english, source, priority}, ...]

# 2. Load via GlossaryLoader
loader = GlossaryLoader()
new_terms = loader.load_custom_glossary("path/to/glossary.json")

# 3. Pipeline will automatically use loaded terms
```

### Extending Style Guides

```python
# In src/style_guide_config.py

class StyleGuideVariant(Enum):
    CUSTOM_NEW = "custom_new"

# Add to StyleGuideManager
def _get_custom_new_guide(self):
    return """
    [Your custom translation guidelines]
    """
```

---

## Deployment Architecture

### Local Development
```
├─ Python 3.11+ environment
├─ Valkey server (localhost:6379)
├─ OpenAI API key in .env
└─ Excel input files
```

### Production Setup (Recommended)
```
┌──────────────────────┐
│  Application Server  │
│  ├─ Python app       │
│  ├─ venv             │
│  └─ .env config      │
└──────────┬───────────┘
           │
           ├─> Valkey (Redis-compatible)
           │   └─ Managed service recommended
           │
           └─> OpenAI API
               └─ Requires valid API key
```

### Scaling Considerations

**Current limitations**:
- Single-threaded processing (no Celery)
- All state in single Valkey instance
- No load balancing

**To scale beyond ~5,000 segments/hour**:
- Add Celery for distributed processing
- Use Valkey cluster for high availability
- Implement request queuing
- Add rate limiting for API calls

---

## Error Handling

### Pipeline Error Recovery

All pipelines implement:
```python
try:
    translation = translate_batch(segments)
except OpenAIError as e:
    # Log error, return error results
    # No automatic retry (would need Celery)
    logger.error(f"API error: {e}")
    results = create_error_results(segments, e)
```

### Valkey Failover

```python
try:
    valkey_mgr = ValkeyManager()
    # Use Valkey for consistency
except ConnectionError:
    # Fall back to in-memory dictionary
    logger.warning("Valkey unavailable, using in-memory fallback")
    self.locked_terms = {}  # Local dict instead
```

**Note**: Graceful degradation to in-memory storage if Valkey unavailable.

---

## Configuration

### Environment Variables (.env)

```bash
# Required
OPENAI_API_KEY=sk-proj-your_key_here

# Optional (Valkey)
VALKEY_HOST=localhost
VALKEY_PORT=6379
VALKEY_DB=0

# Logging
LOG_LEVEL=INFO
```

### Pipeline Configuration

```python
# Batch size
pipeline = EnhancedBatchPipeline(batch_size=5)  # 1-10 supported

# Style guide
pipeline = EnhancedBatchPipeline(
    style_guide_variant="standard"  # none|minimal|standard|comprehensive
)

# Model (currently only GPT-5 OWL used)
pipeline = EnhancedBatchPipeline(model_name="Owl")
```

---

## Testing

### Test Coverage

**Integration Tests**:
- `test_phase2_integration.py` - Full pipeline workflows
- `test_valkey_integration.py` - Valkey operations
- `test_context_builder_integration.py` - Context assembly

**Unit Tests**:
- `test_be003_core.py` - Core functionality
- `test_token_optimizer_simple.py` - Token optimization

**Import Tests**:
- `test_imports.py` - Module import validation
- `production_import_test.py` - Production readiness

### Running Tests

```bash
# All tests
pytest src/tests/ -v

# Specific test
pytest src/tests/test_valkey_integration.py -v

# With coverage
pytest --cov=src src/tests/
```

---

## Architecture Decisions

### Why This Simple Architecture?

✅ **Proven to work** - Successfully processes 1,400+ segment documents
✅ **Easy to understand** - New developers can learn it quickly
✅ **Maintainable** - No complex multi-tier memory orchestration
✅ **Cost-effective** - Batch processing reduces API costs 80%
✅ **Reliable** - Fewer moving parts = fewer failure points

### What We Sacrificed

❌ **Translation Memory** - Could save 40-50% costs but adds complexity
❌ **Semantic Search** - Would need Qdrant infrastructure
❌ **Adaptive Learning** - Would need Mem0 integration
❌ **Auto Scaling** - Would need Celery/distributed architecture

### Trade-off Rationale

For medical translation use case:
- **Quality > Cost** - LLM for every segment ensures quality
- **Simplicity > Features** - Easier to debug and maintain
- **Proven > Experimental** - Stick with what works

---

## Future Extensions (If Needed)

If requirements change, these could be added:

1. **Translation Memory** (4-6 weeks)
   - Build TM database from protocol pairs
   - Implement fuzzy matching
   - Add routing logic (TM vs LLM)
   - Requires Qdrant for semantic search

2. **Distributed Processing** (2-3 weeks)
   - Implement Celery task queue
   - Add Redis/Valkey as broker
   - Create worker pool architecture

3. **Advanced Quality Scoring** (1-2 weeks)
   - Integrate BLEU/COMET scores
   - Add reference-based validation
   - Implement hallucination detection

4. **Production Monitoring** (1-2 weeks)
   - Add Prometheus metrics
   - Set up Grafana dashboards
   - Implement alerting

---

## Summary

TransAI uses a **simple, proven architecture**:

```
Excel Input
  → Load Glossary (419-2,906 terms)
  → Build Context (glossary + style + history)
  → Call GPT-5 OWL (everything)
  → Track Consistency (Valkey)
  → Excel Output
```

**Strengths**: Simple, reliable, good glossary integration, term consistency

**Limitations**: No TM (higher costs), no semantic search, no distributed processing

**Status**: Production-ready for current use cases

---

**Document Version**: 2.0 (Simplified to match implementation)
**Last Updated**: November 23, 2025
**Architecture Status**: Accurately reflects codebase
