# TransAI Source Code - Production Pipeline

## Current Structure

### Core Production Pipelines
```
src/
├── production_pipeline_batch_enhanced.py      # RECOMMENDED - General purpose with batch processing
├── production_pipeline_en_ko.py               # EN→KO clinical specialization
├── production_pipeline_en_ko_improved.py      # EN→KO with improvements
├── production_pipeline_ko_en_improved.py      # KO→EN with tag preservation
├── production_pipeline_with_style_guide.py    # Style guide variant testing
└── production_pipeline_working.py             # Legacy reference implementation
```

### Key Modules

#### Glossary Management (`glossary/`)
```
glossary/
├── glossary_loader.py                 # Load glossary files (JSON, Excel)
├── glossary_search.py                 # Fuzzy term matching engine (22.9KB)
├── create_combined_glossary.py        # Glossary creation utilities
└── __init__.py
```

#### Memory System - 3-Tier Architecture (`memory/`)
```
memory/
├── valkey_manager.py                  # Valkey/Redis integration (25KB)
│   └── Features: Connection pooling, O(1) lookups, persistence
├── session_manager.py                 # Session tracking (19KB)
│   └── Features: Document sessions, progress tracking, state management
├── consistency_tracker.py             # Term consistency (26KB)
│   └── Features: Terminology consistency across segments
├── cached_glossary_search.py          # Caching layer (19.7KB)
│   └── Features: Sub-millisecond cached lookups
├── config.py                          # Memory configuration
├── example_usage.py                   # Usage examples
└── __init__.py
```

#### Utilities (`utils/`)
```
utils/
├── tag_handler.py                     # CAT tool tag preservation (15.7KB)
│   └── Features: Extract, remove, restore tags (support for <>, </>, <1/> formats)
├── segment_filter.py                  # Content filtering (5.1KB)
└── __init__.py
```

#### Clinical Protocol System (`clinical_protocol_system/`)
```
clinical_protocol_system/
├── extract_protocol_terms.py          # Protocol term extraction (15.9KB)
├── agents/                            # AI agent configurations
├── templates/                         # Prompt templates
└── data/                              # Protocol terminology database
```

### Data Files (`data/`)
```
data/
├── production_glossary.json           # Full glossary (503KB, 2906+ terms)
├── production_glossary.xlsx           # Excel format (155KB)
├── combined_en_ko_glossary.xlsx       # Clinical terminology (20KB, 419 terms)
├── sample_glossary.json               # Example format
└── sample_test_data.json              # Test segments (15 samples)
```

### Testing (`tests/`)
```
tests/
├── test_phase2_integration.py                       # Integration tests
├── test_valkey_integration.py                       # Valkey caching tests
├── test_context_builder_integration.py              # Context building tests
├── test_enhanced_translation_integration.py         # Pipeline tests
├── test_be003_integration.py                        # BE003 workflow tests
├── test_be003_core.py                              # BE003 core tests
├── test_data_loader_performance.py                 # Performance tests
├── test_token_optimizer_simple.py                  # Token optimization tests
├── test_imports.py                                 # Import validation
├── test_package_init.py                            # Package initialization tests
├── production_import_test.py                       # Production import tests
└── valkey_integration_demo.py                      # Valkey demo
```

### Analysis & Evaluation
```
analysis/                              # Analysis tools and scripts
evaluation/                            # Evaluation metrics and reports
results/                               # Execution results and outputs
```

### Configuration & Support
```
config/                                # Configuration files
logs/                                  # Application logs
style_guide_config.py                  # 10 translation style variants
style_guide_config.json                # Style configuration data
requirements.txt                       # Python dependencies
.env                                   # Environment variables (DO NOT COMMIT)
__init__.py                            # Package initialization
```

### Documentation
```
README.md (this file)                  # Source code structure and usage
COMPLETE_PROMPT_TEMPLATE.md            # Prompt template documentation
PIPELINE_FLOW_DETAILED.md              # Detailed pipeline flow
STYLE_GUIDE_AB_TESTING_README.md       # A/B testing framework
```

## Quick Start Guide

### Setup Valkey (Recommended for Production)
```bash
# Install Valkey if not already installed
brew install valkey  # macOS
# or use Docker: docker run -d -p 6379:6379 valkey/valkey

# Start Valkey server
valkey-server

# Verify Valkey is running
valkey-cli ping
# Should respond: PONG
```

### Run Production Pipeline
```bash
# Navigate to project root
cd /Users/won.suh/Project/transai

# Activate virtual environment
source venv/bin/activate

# Run recommended pipeline (batch-enhanced)
python -m src.production_pipeline_batch_enhanced

# Or run specialized pipelines
python -m src.production_pipeline_en_ko           # EN→KO clinical
python -m src.production_pipeline_ko_en_improved  # KO→EN with tags
```

### Test Integration
```bash
# Run Valkey integration test
pytest src/tests/test_valkey_integration.py -v

# Run all integration tests
pytest src/tests/ -v

# Run specific pipeline test
pytest src/tests/test_phase2_integration.py -v
```

## Pipeline Selection Guide

| Pipeline | Use Case | Features | Best For |
|----------|----------|----------|----------|
| **batch_enhanced** | General purpose | Batch processing, flexible style guides | Most use cases |
| **en_ko** | Clinical EN→KO | Clinical specialization, 419 terms | Korean regulatory docs |
| **ko_en_improved** | General KO→EN | Tag preservation, consistency tracking | CAT tool workflows |
| **with_style_guide** | A/B testing | 10 style variants | Quality optimization |
| **working** | Reference | Legacy implementation | Learning/reference |

## Key Features

### 3-Tier Memory Architecture
1. **Tier 1 - Valkey/Redis Cache**: O(1) lookups, sub-millisecond, persistent
2. **Tier 2 - Session Memory**: Document-level term tracking, consistency
3. **Tier 3 - Style Guides**: Pre-computed variants, A/B testing

### Translation Specializations
- **Clinical Protocols**: ICH GCP-compliant prompts with 419+ medical terms
- **Regulatory Documents**: KO↔EN with tag preservation for CAT tools
- **Batch Processing**: 5+ segments per API call for 80% cost reduction

### Performance Metrics
- **Token Reduction**: 98.3% (20,473 → 413 tokens per request)
- **Processing Speed**: 720 words/minute
- **Quality Score**: 0.84 average (0.74-0.98 range)
- **Cache Lookup**: <1ms with Valkey
- **Glossary Coverage**: 89.6% of medical terms

## Module Dependencies

```
production_pipeline_*
├── glossary/
│   ├── glossary_loader.py
│   └── glossary_search.py
├── memory/
│   ├── valkey_manager.py
│   ├── session_manager.py
│   ├── consistency_tracker.py
│   └── cached_glossary_search.py
├── utils/
│   ├── tag_handler.py
│   └── segment_filter.py
└── style_guide_config.py
```

## Configuration Files

### Style Guide Configuration
```python
# 10 available variants
from src.style_guide_config import StyleGuideManager

style_mgr = StyleGuideManager()
# Available: NONE, MINIMAL, COMPACT, STANDARD, COMPREHENSIVE,
#            CLINICAL_PROTOCOL, REGULATORY_COMPLIANCE, REGULATORY_COMPLIANCE_ENHANCED
```

### Environment Configuration
```bash
# Required in src/.env
OPENAI_API_KEY=your_key_here
VALKEY_HOST=localhost
VALKEY_PORT=6379
LOG_LEVEL=INFO
```

## Testing Strategy

### Unit Tests
- Import validation tests
- Module initialization tests
- Component functionality tests

### Integration Tests
- Valkey caching integration
- Full pipeline workflows
- Context building and optimization
- Translation quality metrics

### Performance Tests
- Token usage optimization
- Data loader performance
- Memory system efficiency

## Common Tasks

### Load Custom Glossary
```python
from src.glossary.glossary_loader import GlossaryLoader

loader = GlossaryLoader()
glossary = loader.load_custom_glossary("data/your_glossary.json")
```

### Use Caching System
```python
from src.memory.valkey_manager import ValkeyManager
from src.memory.session_manager import SessionManager

valkey_mgr = ValkeyManager(host="localhost", port=6379)
session_mgr = SessionManager(valkey_mgr)
session_id = session_mgr.create_session("document.pdf", total_segments=100)
```

### Preserve CAT Tool Tags
```python
from src.utils.tag_handler import TagHandler

handler = TagHandler()
text = "Data <1>important</1> <2/>"
tags = handler.extract_tags(text)
clean = handler.remove_tags(text)
restored = handler.restore_tags(translated_text, tags)
```

## Development Guidelines

1. **Add New Pipelines**: Extend base pipeline, follow naming convention `production_pipeline_*.py`
2. **Add New Modules**: Place in appropriate subdirectory (`glossary/`, `memory/`, `utils/`, etc.)
3. **Update Tests**: Add corresponding test file in `tests/`
4. **Document Changes**: Update relevant docs in parent `docs/` directory
5. **Follow Style**: Use existing code patterns and variable naming conventions

## Directory Purpose Summary

| Directory | Purpose | Status |
|-----------|---------|--------|
| `/` | Core production pipelines | Active |
| `glossary/` | Term management and search | Active |
| `memory/` | Caching and consistency | Active |
| `utils/` | Supporting utilities | Active |
| `clinical_protocol_system/` | Medical specialization | Active |
| `tests/` | Unit and integration tests | Active |
| `data/` | Glossaries and test data | Reference |
| `analysis/` | Analysis tools | Reference |
| `evaluation/` | Evaluation metrics | Reference |
| `results/` | Execution outputs | Generated |
| `config/` | Configuration files | Reference |
| `logs/` | Application logs | Generated |