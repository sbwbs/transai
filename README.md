# TransAI: Medical Document Translation System

> Glossary-enhanced LLM translation for clinical trials and medical device documentation with 80% API call reduction through batch processing.

## 📋 Quick Links

- **📖 Getting Started:** [01_getting_started/](docs/01_getting_started/INDEX.md)
- **📚 Full Documentation:** [docs/README.md](docs/README.md)
- **📊 System Status:** [Completion Report](docs/07_project_management/COMPLETION_REPORT.md)
- **🔐 Security Info:** [Git Security Checklist](docs/07_project_management/GIT_SECURITY_CHECKLIST.md)

## 🚀 Overview

TransAI is a production-ready medical document translation system specializing in clinical trials, pharmaceutical, and medical device documentation translation. Built with glossary-enhanced prompting, term consistency tracking (Valkey), and GPT-5 OWL.

## 📂 Project Structure

```
transai/
├── src/                                  # Application source code
│   ├── production_pipeline_*.py           # 6 translation pipelines (entry points)
│   ├── glossary/                         # Glossary management system
│   │   ├── glossary_loader.py            # Generic glossary loader
│   │   ├── glossary_search.py            # Keyword term matching
│   │   ├── create_combined_glossary.py   # Glossary utilities
│   │   └── glossary_config.example.yaml  # Configuration template
│   ├── memory/                           # Term consistency (Valkey cache only)
│   │   ├── valkey_manager.py             # Valkey integration
│   │   ├── session_manager.py            # Session tracking
│   │   ├── consistency_tracker.py        # Term consistency
│   │   └── cached_glossary_search.py     # Cached searches
│   ├── utils/                            # Utilities
│   │   ├── tag_handler.py                # CAT tool tag preservation
│   │   └── segment_filter.py             # Content filtering
│   ├── clinical_protocol_system/         # Medical specialization
│   ├── tests/                            # Test suite (11+ test files)
│   ├── data/                             # Glossaries & test data
│   ├── style_guide_config.py             # 10 translation style variants
│   ├── requirements.txt                  # Dependencies
│   └── .env.example                      # Configuration template
│
├── docs/                                 # Organized documentation (29 files)
│   ├── 01_getting_started/               # Onboarding (3 docs + INDEX)
│   ├── 02_architecture/                  # System design (4 docs + INDEX)
│   ├── 03_core_features/                 # Feature usage (5 docs + INDEX)
│   ├── 04_glossary_and_terminology/      # Glossary guide (4 docs + INDEX)
│   ├── 05_advanced_topics/               # Advanced features (5 docs + INDEX)
│   ├── 06_performance_and_optimization/  # Performance (4 docs + INDEX)
│   ├── 07_project_management/            # Operations (3 docs + INDEX)
│   └── README.md                         # Documentation hub
│
├── README.md                             # This file (project overview)
└── .git/                                 # Version control
```

## ⚡ Quick Start (5 minutes)

### 1. Set Up Python Environment
```bash
cd transai
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r src/requirements.txt
```

### 2. Start Valkey Server
```bash
# Option A: Homebrew (macOS)
brew install valkey && valkey-server

# Option B: Docker
docker run -d -p 6379:6379 valkey/valkey
```

### 3. Configure API Keys
```bash
# Create src/.env (copy from src/.env.example)
cp src/.env.example src/.env

# Edit and add your OpenAI API key
OPENAI_API_KEY=sk-proj-your_actual_key_here
```

### 4. Run First Translation
```python
from src.production_pipeline_batch_enhanced import EnhancedBatchPipeline

pipeline = EnhancedBatchPipeline(style_guide="STANDARD")
results = pipeline.run_enhanced_batch_pipeline(
    input_file="path/to/your/document.xlsx"
)
print(f"Quality Score: {results['avg_quality_score']:.2f}")
```

For more examples, see [docs/README.md](docs/README.md)

## 📚 Documentation Structure

Documentation is organized into 7 categories with 29 files total. Each category has an INDEX.md for easy navigation.

| Category | Purpose | Files |
|----------|---------|-------|
| **[01_getting_started](docs/01_getting_started/INDEX.md)** | Setup & onboarding | 3 docs |
| **[02_architecture](docs/02_architecture/INDEX.md)** | System design & diagrams | 4 docs |
| **[03_core_features](docs/03_core_features/INDEX.md)** | How to use the system | 5 docs |
| **[04_glossary_and_terminology](docs/04_glossary_and_terminology/INDEX.md)** | Manage glossaries | 4 docs |
| **[05_advanced_topics](docs/05_advanced_topics/INDEX.md)** | Advanced features | 5 docs |
| **[06_performance_and_optimization](docs/06_performance_and_optimization/INDEX.md)** | Testing & optimization | 4 docs |
| **[07_project_management](docs/07_project_management/INDEX.md)** | Operations & security | 3 docs |

**Start here:** [docs/README.md](docs/README.md) - Main documentation hub with quick navigation

## 🔑 Supported Models

| Model | Provider | Status | Use Case |
|-------|----------|--------|----------|
| **GPT-5** | OpenAI | Primary | Clinical specialization & quality |
| **GPT-4o** | OpenAI | Fallback | Cost-efficient alternative |

## 🔐 Security

✅ **No API Keys** - All replaced with placeholders
✅ **No Customer Data** - Replaced with synthetic samples
✅ **No Proprietary Docs** - Only open-source friendly files
✅ **Git Ready** - Safe to commit immediately

See [GIT_SECURITY_CHECKLIST.md](docs/07_project_management/GIT_SECURITY_CHECKLIST.md) for security best practices.

## 📊 System Specifications

| Metric | Value |
|--------|-------|
| API Call Reduction | 80% (batch processing) |
| Processing Speed | 120-150 segments/minute |
| Quality Score | 0.84 average (heuristic) |
| Valkey Latency | <1ms (term lookups) |
| Batch Size | 5 segments per API call |
| Batch Time | ~2.5 seconds |
| LLM Model | GPT-5 OWL (primary only) |

## 🎯 Usage Examples

### Basic Translation
```python
from src.production_pipeline_batch_enhanced import EnhancedBatchPipeline

pipeline = EnhancedBatchPipeline(style_guide="STANDARD")
results = pipeline.translate(input_file="documents/sample.xlsx")
print(f"Quality Score: {results['avg_quality_score']:.2f}")
```

### Using Glossary
```python
from src.glossary_search import GlossarySearchEngine

engine = GlossarySearchEngine(glossary)
matches = engine.search("임상시험", top_k=5)
```

See **[docs/README.md](docs/README.md)** for more examples.

## 📁 Key Source Files

**Main Pipelines** (`src/`):
- `production_pipeline_batch_enhanced.py` - Recommended for production
- `production_pipeline_en_ko.py` - English→Korean specialization
- `production_pipeline_ko_en_improved.py` - Korean→English with tag preservation

**Core Modules:**
- `glossary_loader.py` - Load and process glossaries
- `glossary_search.py` - Fuzzy term matching
- `style_guide_config.py` - 10 configurable style guide variants
- `analyze_token_usage.py` - Token optimization analysis

**Memory & Caching:**
- `memory/valkey_manager.py` - Valkey/Redis integration
- `memory/session_manager.py` - Session tracking
- `memory/consistency_tracker.py` - Term consistency

**Utilities:**
- `utils/tag_handler.py` - CAT tool tag preservation
- `utils/segment_filter.py` - Content filtering

## 🧪 Testing

```bash
# Run all tests
pytest src/tests/ -v

# Run specific test
pytest src/tests/test_valkey_integration.py -v

# With coverage
pytest --cov=src src/tests/
```

**Sample Data Available:**
- `src/data/sample_glossary.json` - 15 medical terms + abbreviations
- `src/data/sample_test_data.json` - 15 synthetic translation segments (KO↔EN)
- `src/data/combined_en_ko_glossary.xlsx` - 419 clinical terms

## 🐛 Troubleshooting

### "Cannot connect to Valkey"
```bash
# Verify Valkey is running
valkey-cli ping  # Should return: PONG
```

### "Invalid API key"
- Check `.env` file in `src/` directory
- Verify key starts with `sk-proj-`
- Get key from: https://platform.openai.com/api-keys

### Import errors
```bash
# Reinstall dependencies
pip install --upgrade -r src/requirements.txt
```

See [01_getting_started/SETUP_CHECKLIST.md](docs/01_getting_started/SETUP_CHECKLIST.md) for more troubleshooting.

## 🔄 Pipeline Variants

10 style guide variants available:

| Variant | Tokens | Best For |
|---------|--------|----------|
| NONE | 0 | Baseline |
| MINIMAL | 100 | Quick drafts |
| STANDARD | 400 | Most cases |
| COMPREHENSIVE | 600 | Quality-focused |
| CLINICAL_PROTOCOL | 300 | Medical docs |
| REGULATORY_COMPLIANCE | 300 | Legal docs |

See [docs/README.md](docs/README.md) for complete list.

## 🚀 Deployment

Ready for:
- ✅ Local development
- ✅ Team collaboration (git)
- ✅ Cloud deployment (AWS, GCP, Azure)
- ✅ Docker containerization
- ✅ CI/CD integration

## 📞 Getting Help

1. **Start:** [Getting Started Guide](docs/01_getting_started/INDEX.md) - Setup & onboarding
2. **Learn:** [Complete README](docs/README.md) - All topics with examples
3. **Explore:** [Documentation Hub](docs/) - 29 documents organized by category

## 🎉 Status

**Current Status:** ✅ Production Ready

- **Architecture:** Proven and tested
- **Code Quality:** High with comprehensive error handling
- **Documentation:** Organized & comprehensive (29 files in 7 categories)
- **Testing:** Full test suite with 11+ test files
- **Security:** Zero secrets, no sensitive data
- **Glossary System:** Refactored to be generic and flexible
- **Ready for:** Immediate deployment

## 📝 License

[Add your license here]

## 📧 Support & Learning

For questions, issues, or to learn more:
1. Check [Troubleshooting](docs/01_getting_started/SETUP_CHECKLIST.md#troubleshooting) in Getting Started guide
2. Review [Complete README](docs/README.md) - Comprehensive usage guide
3. Browse [Documentation Categories](docs/README.md#documentation-structure) - Find what you need

---

**Version:** 1.0.0
**Last Updated:** November 23, 2025
**Status:** Production Ready ✅

**Quick Links:**
- 🚀 [Getting Started](docs/01_getting_started/INDEX.md)
- 📖 [Documentation Hub](docs/README.md)
- 🏗️ [Architecture Overview](docs/02_architecture/INDEX.md)
- 💻 [How to Use](docs/03_core_features/INDEX.md)
