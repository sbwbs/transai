# TransAI Phase 2: Intelligent Medical Document Translation System

## 📋 Overview

Phase 2 is an advanced medical and clinical document translation system that achieves **98.3% token reduction** (20,473 → 413 tokens per request) while maintaining high translation quality. The system specializes in clinical trial, medical device, and pharmaceutical documentation translation using intelligent context building, efficient caching, and optimized LLM integration.

**Key Capabilities:**
- 98.3% reduction in LLM tokens through smart context optimization
- Clinical trial and medical device translation specialization
- 720 words/minute processing speed (10x faster than human translation)
- Sub-millisecond caching via Valkey/Redis
- 84% average translation quality score
- Support for tag preservation (CAT tool integration)

## 📂 Documentation Structure

All documentation is organized into 7 categories. Navigate by category or use the quick links below.

### Documentation Categories

```
docs/
├── 01_getting_started/              ← Start here!
│   ├── INDEX.md                     (Read first for navigation)
│   ├── SETUP_CHECKLIST.md
│   └── FILE_ORGANIZATION_GUIDE.md
│
├── 02_architecture/                 ← Understand the system
│   ├── INDEX.md
│   ├── PHASE2_MVP_ARCHITECTURE.md
│   ├── PHASE2_ARCHITECTURE_DIAGRAM.md
│   └── IMPLEMENTATION_BLUEPRINT.md
│
├── 03_core_features/                ← Learn to use
│   ├── INDEX.md
│   ├── TRANSLATION_PIPELINE_STEPBYSTEP.md
│   ├── VALKEY_INTEGRATION_SUMMARY.md
│   ├── TAG_PRESERVATION_IMPLEMENTATION.md
│   └── TECHNICAL_IMPLEMENTATION.md
│
├── 04_glossary_and_terminology/     ← Manage terms
│   ├── INDEX.md
│   ├── GLOSSARY_SYSTEM.md
│   ├── HOW_TO_ADD_GLOSSARIES.md
│   └── GLOSSARY_SEARCH_METHODS_ANALYSIS.md
│
├── 05_advanced_topics/              ← Advanced features
│   ├── INDEX.md
│   ├── TRANSLATION_PATTERNS_FOR_PROMPT.md
│   ├── PROTOCOL_PAIRING_FEASIBILITY.md
│   ├── PROTOCOL_PAIRS_USAGE_STRATEGY.md
│   └── TRANSLATION_FEEDBACK_ANALYSIS_AND_RECOMMENDATIONS.md
│
├── 06_performance_and_optimization/ ← Optimize & test
│   ├── INDEX.md
│   ├── PHASE2_MVP_TEST_PLAN.md
│   ├── PHASE2_TEST_KIT_ANALYSIS.md
│   └── TOKEN_USAGE_ANALYSIS_REPORT.md
│
└── 07_project_management/           ← Operations & security
    ├── INDEX.md
    ├── COMPLETION_REPORT.md
    └── GIT_SECURITY_CHECKLIST.md
```

---

## 🗺️ Quick Navigation

**👤 New to the project?**
→ Start with [01_getting_started/INDEX.md](01_getting_started/INDEX.md)
- Setup checklist
- Project organization
- Initial configuration

**🏗️ Want to understand the system?**
→ Go to [02_architecture/INDEX.md](02_architecture/INDEX.md)
- System architecture
- Component diagrams
- Design decisions

**⚙️ Ready to use it?**
→ Read [03_core_features/INDEX.md](03_core_features/INDEX.md)
- Translation pipeline
- Caching system
- Tag preservation

**📚 Need glossary help?**
→ See [04_glossary_and_terminology/INDEX.md](04_glossary_and_terminology/INDEX.md)
- How to add glossaries
- Glossary system overview
- Search methods

**🚀 Advanced features?**
→ Check [05_advanced_topics/INDEX.md](05_advanced_topics/INDEX.md)
- Translation optimization
- Protocol handling
- Feedback analysis

**⚡ Performance tuning?**
→ Visit [06_performance_and_optimization/INDEX.md](06_performance_and_optimization/INDEX.md)
- Testing strategy
- Token analysis
- Performance metrics

**🔐 Operations & security?**
→ Go to [07_project_management/INDEX.md](07_project_management/INDEX.md)
- Project status
- Security checklist
- Git workflow

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Valkey or Redis server** (for caching layer)
- **OpenAI API Key** (required for GPT-5 OWL and GPT-4o models)
- **macOS, Linux, or Windows with WSL**

### Installation

```bash
# 1. Navigate to project directory
cd /Users/won.suh/Project/transai

# 2. Create Python virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install Python dependencies
pip install -r src/requirements.txt

# 4. Install and start Valkey server
# Option A: macOS with Homebrew
brew install valkey
valkey-server

# Option B: Using Docker (any platform)
docker run -d -p 6379:6379 valkey/valkey

# Option C: Use system Redis (if available)
redis-server

# 5. Configure environment variables
cp src/.env.example src/.env  # Or create src/.env file manually
```

## ⚙️ Configuration

### Environment Variables (.env)

Create a `.env` file in the `src/` directory with the following variables:

```bash
# OpenAI Configuration (REQUIRED)
OPENAI_API_KEY=your_openai_api_key_here

# Alternative LLM Providers (Optional - currently not implemented in production)
ANTHROPIC_API_KEY=your_anthropic_api_key_optional
GEMINI_API_KEY=your_gemini_api_key_optional
UPSTAGE_API_KEY=your_upstage_api_key_optional

# Valkey/Redis Configuration
VALKEY_HOST=localhost
VALKEY_PORT=6379
VALKEY_DB=0

# Logging Configuration
LOG_LEVEL=INFO  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Getting API Keys

#### OpenAI API Key
1. Visit https://platform.openai.com/api-keys
2. Sign in to your OpenAI account (create one if needed)
3. Click "Create new secret key"
4. Copy the generated key to your `.env` file
5. Note: Your key will start with `sk-proj-` (DO NOT commit this to git)

## 📚 Supported Models

### Currently Implemented

| Model Name | Provider | Model ID | Status | Best For |
|-----------|----------|----------|--------|----------|
| **Owl (Primary)** | OpenAI | `gpt-5` | Active | Clinical specialization, optimal quality |
| **Falcon (Fallback)** | OpenAI | `gpt-4o` | Active | Reliable fallback, cost-efficient |

### Model Specifications

**GPT-5 OWL:**
- Input tokens: $0.015/1K tokens
- Output tokens: $0.060/1K tokens
- Context window: 128K tokens
- Max output: 8,192 tokens
- Best for: Clinical protocols, regulatory documents

**GPT-4o Falcon:**
- Input tokens: $0.005/1K tokens
- Output tokens: $0.015/1K tokens
- Context window: 128K tokens
- Max output: 4,096 tokens
- Best for: General documents, cost optimization

### Model Selection & Fallback

The system automatically:
1. Attempts translation with GPT-5 OWL (primary model)
2. Falls back to GPT-4o if OWL fails
3. Logs all fallback events with reasons

## 🏗️ Project Structure

```
transai/
├── src/                                    # Core application code
│   ├── production_pipeline_*.py            # Main translation pipelines
│   │   ├── production_pipeline_batch_enhanced.py      # RECOMMENDED - General purpose
│   │   ├── production_pipeline_en_ko.py               # EN→KO clinical specialization
│   │   ├── production_pipeline_ko_en_improved.py      # KO→EN with tag preservation
│   │   └── production_pipeline_with_style_guide.py    # Style guide variants
│   ├── glossary/                          # Glossary management
│   │   ├── glossary_loader.py             # Load glossary files
│   │   ├── glossary_search.py             # Fuzzy term matching
│   │   └── create_combined_glossary.py    # Glossary creation
│   ├── style_guide_config.py              # 10 translation style variants
│   ├── memory/                            # Caching layer (3-tier architecture)
│   │   ├── valkey_manager.py              # Valkey/Redis integration
│   │   ├── session_manager.py             # Session tracking & progress
│   │   ├── consistency_tracker.py         # Term consistency maintenance
│   │   └── cached_glossary_search.py      # Cached glossary lookups
│   ├── utils/                             # Utilities
│   │   ├── tag_handler.py                 # CAT tool tag preservation
│   │   └── segment_filter.py              # Content filtering
│   ├── clinical_protocol_system/          # Medical specialization
│   │   ├── extract_protocol_terms.py      # Protocol term extraction
│   │   ├── agents/                        # AI agent configurations
│   │   ├── templates/                     # Prompt templates
│   │   └── data/                          # Protocol terminology
│   ├── tests/                             # Unit and integration tests
│   ├── data/                              # Glossaries and test data
│   │   ├── production_glossary.json       # Full glossary (503KB)
│   │   ├── production_glossary.xlsx       # Excel format (155KB)
│   │   ├── combined_en_ko_glossary.xlsx   # Clinical terminology (20KB)
│   │   ├── sample_glossary.json           # Example format
│   │   └── sample_test_data.json          # Test segments
│   ├── analysis/                          # Analysis tools
│   ├── evaluation/                        # Evaluation metrics
│   ├── results/                           # Execution results
│   ├── config/                            # Configuration files
│   ├── logs/                              # Application logs
│   ├── requirements.txt                   # Python dependencies
│   ├── .env                               # Configuration (DO NOT COMMIT)
│   └── README.md                          # Src directory documentation
│
├── docs/                                  # Technical documentation
│   ├── README.md                          # This file (navigation & quick start)
│   └── [See Documentation Index below]
│
└── README.md                              # Root project README
```

## 🔄 Translation Pipelines

### 1. Batch Enhanced Pipeline (RECOMMENDED)

**File:** `src/production_pipeline_batch_enhanced.py`

Best for production use with optimal performance:

```python
from src.production_pipeline_batch_enhanced import EnhancedBatchPipeline

pipeline = EnhancedBatchPipeline(
    style_guide="STANDARD",
    batch_size=5,
    model_name="Owl"
)

results = pipeline.translate(input_file="input.xlsx")
```

**Performance:**
- 2.5 seconds per 5-segment batch
- Quality score: 0.84 average (0.74-0.98 range)
- Token reduction: 98.3%

### 2. Clinical Protocol Pipeline (EN→KO)

**File:** `src/production_pipeline_en_ko.py`

Specialized for English-to-Korean clinical protocols:

```python
from src.production_pipeline_en_ko import EnKoClinicialPipeline

pipeline = EnKoClinicialPipeline()
results = pipeline.translate(input_file="protocol.xlsx")
```

**Features:**
- Combined glossary (419 clinical terms)
- Regulatory compliance style guide
- Bilingual terminology formatting

### 3. KO-EN Improved Pipeline

**File:** `src/production_pipeline_ko_en_improved.py`

For Korean-to-English translation with tag preservation:

```python
from src.production_pipeline_ko_en_improved import KoEnImprovedPipeline

pipeline = KoEnImprovedPipeline()
results = pipeline.translate(input_file="document.xlsx")
```

**Features:**
- CAT tool tag preservation
- Glossary term consistency
- Hallucination detection

## 📖 Usage Examples

### Basic Translation

```python
import os
from dotenv import load_dotenv
from src.production_pipeline_batch_enhanced import EnhancedBatchPipeline

# Load environment variables
load_dotenv()

# Create pipeline
pipeline = EnhancedBatchPipeline(
    style_guide="STANDARD",
    batch_size=5
)

# Translate file
results = pipeline.translate(
    input_file="documents/sample.xlsx",
    output_file="documents/sample_translated.xlsx"
)

print(f"Processed {results['total_segments']} segments")
print(f"Average quality score: {results['avg_quality_score']:.2f}")
```

### Using Glossary Terms

```python
from src.glossary_loader import GlossaryLoader
from src.glossary_search import GlossarySearchEngine

# Load glossary
loader = GlossaryLoader()
glossary = loader.load_combined_glossary("data/sample_glossary.json")

# Search for terms
search_engine = GlossarySearchEngine(glossary)
results = search_engine.search("임상시험", top_k=5)

for match in results:
    print(f"{match['korean']} → {match['english']} ({match['score']})")
```

### Custom Style Guide

```python
from src.style_guide_config import StyleGuideManager
from src.production_pipeline_batch_enhanced import EnhancedBatchPipeline

# Configure style guide
style_manager = StyleGuideManager()
pipeline = EnhancedBatchPipeline(
    style_guide="COMPREHENSIVE",  # More tokens, higher quality
    style_guide_variant="REGULATORY_COMPLIANCE"
)

results = pipeline.translate(input_file="regulatory_docs.xlsx")
```

## 🔑 Glossary Format

The system supports glossary files in JSON format. See `data/sample_glossary.json` for a complete example.

### JSON Structure

```json
{
  "glossary_metadata": {
    "version": "1.0",
    "language_pair": "ko-en",
    "created_date": "2025-11-23",
    "description": "Sample medical terminology glossary"
  },
  "terms": [
    {
      "korean": "임상시험",
      "english": "clinical trial",
      "category": "clinical",
      "context": "Medical context or usage example",
      "frequency": "high"
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

### Adding Your Own Glossary

1. **Create a JSON file** in `data/` directory following the format above
2. **Update the pipeline** to load your glossary:

```python
loader = GlossaryLoader()
custom_glossary = loader.load_custom_glossary("data/your_glossary.json")

pipeline = EnhancedBatchPipeline(glossary=custom_glossary)
```

## 📊 Style Guides

The system provides 10 configurable style guide variants optimized for different scenarios:

| Variant | Tokens | Best For | Quality vs Speed |
|---------|--------|----------|-----------------|
| NONE | 0 | Baseline | Fastest |
| MINIMAL | 100 | Quick translations | Fast |
| COMPACT | 200 | Standard documents | Balanced |
| STANDARD | 400 | Most use cases | Balanced |
| COMPREHENSIVE | 600 | Complex documents | Quality |
| CLINICAL_PROTOCOL | 300 | EN→KO clinical | Specialized |
| REGULATORY_COMPLIANCE | 300 | KO→EN regulatory | Specialized |
| REGULATORY_COMPLIANCE_ENHANCED | 900 | Critical regulatory | Highest quality |

### Setting Style Guide

```python
pipeline = EnhancedBatchPipeline(style_guide="COMPREHENSIVE")
```

## 🧠 Memory System (Caching)

The system uses a 3-tier memory architecture for optimal performance:

### Tier 1: Valkey/Redis Cache
- Sub-millisecond O(1) lookups
- Glossary term caching
- Session management
- Connection pooling (20 connections)

### Tier 2: Session Memory
- Document-level tracking
- Term consistency across segments
- Progress tracking

### Tier 3: Style Guide Management
- Pre-computed style variants
- A/B testing framework

### Using the Memory System

```python
from src.memory.valkey_manager import ValkeyManager
from src.memory.session_manager import SessionManager

# Initialize managers
valkey_mgr = ValkeyManager(host="localhost", port=6379)
session_mgr = SessionManager(valkey_mgr)

# Create document session
session_id = session_mgr.create_session(
    document_name="clinical_protocol.pdf",
    total_segments=100
)

# Track progress
session_mgr.update_progress(session_id, segments_completed=50)

# Retrieve session status
status = session_mgr.get_session_status(session_id)
```

## 🏷️ CAT Tool Tag Preservation

The system preserves special tags used by Computer-Aided Translation tools:

### Supported Tag Types

- **Self-closing:** `<123/>`
- **Opening tags:** `<123>`
- **Closing tags:** `</123>`
- **Paired tags:** `<123>text</123>`
- **Metadata:** `[IN_ECN_301]`

### Using Tag Handler

```python
from src.utils.tag_handler import TagHandler

handler = TagHandler()

# Extract tags
text = "This is <1>important</1> clinical data <2/>."
tags = handler.extract_tags(text)
clean_text = handler.remove_tags(text)

# Restore tags after translation
translated_clean = "Ceci est <1>important</1> données cliniques <2/>."
restored = handler.restore_tags(translated_clean, tags)
```

## 📈 Performance Metrics

### Translation Performance

```
Token Reduction:      98.3% (20,473 → 413 tokens)
Processing Speed:     720 words/minute
Batch Processing:     2.5 seconds per 5 segments
Quality Score:        0.84 average (0.74-0.98 range)
Cache Lookup:         <1ms (O(1) operations)
```

### Cost per Segment

- **Average cost:** ~$0.006 per segment (using GPT-5 OWL)
- **Glossary coverage:** 89.6% of medical terms
- **Fallback rate:** <2% (automatic GPT-4o fallback)

## 🧪 Testing

### Run Unit Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_glossary_loader.py -v

# Run with coverage
pytest --cov=src tests/
```

### Sample Test Data

The project includes sample test data in `data/sample_test_data.json`:

- 15 synthetic translation segments
- Mix of KO→EN and EN→KO directions
- Multiple difficulty levels (easy, medium)
- Various medical categories (regulatory, clinical, device, operational)

## 🐛 Troubleshooting

### Common Issues

#### 1. "ModuleNotFoundError: No module named 'openai'"

**Solution:**
```bash
pip install --upgrade openai>=1.51.2
```

#### 2. "Connection refused: Cannot connect to Valkey"

**Solution:**
```bash
# Check Valkey is running
valkey-cli ping  # Should return "PONG"

# If not running, start it:
valkey-server  # Or use Docker: docker run -d -p 6379:6379 valkey/valkey
```

#### 3. "Invalid API key" error

**Solution:**
- Verify your OpenAI API key in `.env` file
- Check that key starts with `sk-proj-`
- Ensure no extra spaces or quotes around the key
- Test key at https://platform.openai.com/account/api-keys

#### 4. "GPT-5 OWL failed, using GPT-4o fallback"

This is normal behavior. The system automatically falls back to GPT-4o if GPT-5 fails. You can:
- Check logs for the specific failure reason
- Set `LOG_LEVEL=DEBUG` for detailed information
- Review API rate limits on OpenAI dashboard

## 📚 Find Specific Topics

Documentation is organized by category. Use these links to jump to what you need:

| Need Help With... | Go To... |
|---|---|
| Getting started | [01_getting_started/INDEX.md](01_getting_started/INDEX.md) |
| Understanding architecture | [02_architecture/INDEX.md](02_architecture/INDEX.md) |
| Using translation features | [03_core_features/INDEX.md](03_core_features/INDEX.md) |
| Managing glossaries | [04_glossary_and_terminology/INDEX.md](04_glossary_and_terminology/INDEX.md) |
| Advanced optimization | [05_advanced_topics/INDEX.md](05_advanced_topics/INDEX.md) |
| Performance & testing | [06_performance_and_optimization/INDEX.md](06_performance_and_optimization/INDEX.md) |
| Project & security | [07_project_management/INDEX.md](07_project_management/INDEX.md) |

**Each category has an INDEX.md file that navigates the documents in that section.**

## 🔧 Development

### Project Dependencies

Core dependencies are listed in `requirements.txt`. Key packages:

- **LLM Integration:** `openai>=1.51.2`
- **Caching:** `valkey>=6.1.1`
- **Data Processing:** `pandas>=2.0.0`, `openpyxl>=3.1.0`
- **Async:** `asyncio`, `aiohttp`
- **Config:** `python-dotenv>=1.0.0`
- **Testing:** `pytest`, `pytest-asyncio`, `pytest-mock`

### Adding New Features

1. Create feature branch: `git checkout -b feature/description`
2. Implement changes following existing code style
3. Add unit tests in `tests/`
4. Update documentation in `docs/`
5. Test with sample data in `data/sample_test_data.json`

## 📝 License

[Add your license information here]

## 📧 Support

For questions or issues:
1. Check existing documentation in `docs/`
2. Review sample code in `src/` with inline comments
3. Check test files for usage examples

## ✨ Key Features Summary

✅ **98.3% token reduction** through intelligent context optimization
✅ **Clinical specialization** with medical device/trial terminology
✅ **Fast processing** - 720 words/minute (10x human speed)
✅ **High quality** - 84% average quality score
✅ **Smart caching** - Sub-millisecond Valkey integration
✅ **CAT integration** - Tag preservation for workflow tools
✅ **Flexible styling** - 10 style guide variants
✅ **Auto fallback** - Graceful degradation to GPT-4o
✅ **Production ready** - Comprehensive error handling and logging

---

**Version:** 1.0.0
**Last Updated:** November 23, 2025
**Status:** Production Ready
