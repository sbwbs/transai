# Glossary & Terminology Management

Everything you need to know about managing glossaries and terminology in TransAI. Learn how to create, load, search, and maintain glossary databases.

## Documents in this Category

### 1. [GLOSSARY_SYSTEM.md](GLOSSARY_SYSTEM.md)
**Purpose:** Complete glossary system documentation
- Architecture and design
- Configuration guide
- Usage examples
- Input/output formats
- Best practices
- API reference

**Read this if:** You want comprehensive glossary documentation

### 2. [HOW_TO_ADD_GLOSSARIES.md](HOW_TO_ADD_GLOSSARIES.md)
**Purpose:** Practical guide for adding your own glossaries
- Step-by-step setup instructions
- Excel file format examples
- JSON and CSV formats
- Configuration examples
- Validation and testing
- Troubleshooting

**Read this if:** You want to add your own glossaries to the system

### 3. [GLOSSARY_SEARCH_METHODS_ANALYSIS.md](GLOSSARY_SEARCH_METHODS_ANALYSIS.md)
**Purpose:** Analysis of glossary search algorithms
- Search method comparison
- Fuzzy matching algorithms
- Performance analysis
- Optimization strategies
- Algorithm selection guide

**Read this if:** You want to understand how glossary searching works

---

## Quick Start: Adding Your Glossary

### 3 Steps to Add a Glossary

```
Step 1: Create Excel/JSON file
Step 2: Place in data/glossaries/
Step 3: Load and use
```

**Detailed guide:** [HOW_TO_ADD_GLOSSARIES.md](HOW_TO_ADD_GLOSSARIES.md)

---

## Glossary Formats Supported

| Format | Best For | Example |
|--------|----------|---------|
| Excel (.xlsx) | Large glossaries, easy editing | Medical device terms |
| JSON | Metadata, alternatives, structure | Complex terminology |
| CSV | Simple, portable | Basic term pairs |

---

## Reading Paths

### Path 1: Quick Setup (30 minutes)
1. [HOW_TO_ADD_GLOSSARIES.md](HOW_TO_ADD_GLOSSARIES.md) - "Option 1: Quick Start"
2. Create your Excel file
3. Use in pipeline

### Path 2: Comprehensive Understanding (2 hours)
1. [GLOSSARY_SYSTEM.md](GLOSSARY_SYSTEM.md) - Overview
2. [HOW_TO_ADD_GLOSSARIES.md](HOW_TO_ADD_GLOSSARIES.md) - All options
3. [GLOSSARY_SEARCH_METHODS_ANALYSIS.md](GLOSSARY_SEARCH_METHODS_ANALYSIS.md) - How searching works

### Path 3: Advanced Optimization (3+ hours)
1. [GLOSSARY_SYSTEM.md](GLOSSARY_SYSTEM.md) - Full docs
2. [GLOSSARY_SEARCH_METHODS_ANALYSIS.md](GLOSSARY_SEARCH_METHODS_ANALYSIS.md) - Algorithms
3. Implement custom glossary sources

---

## Common Glossary Tasks

### Add a Simple Glossary
**Time:** 5 minutes
```
1. Create Excel with 2 columns: Korean, English
2. Copy to src/data/glossaries/
3. Load with: loader.load_glossary("file.xlsx")
```
→ See [HOW_TO_ADD_GLOSSARIES.md](HOW_TO_ADD_GLOSSARIES.md) - "Option 1: Quick Start"

### Set Up Configuration-Based Loading
**Time:** 10 minutes
```
1. Create glossary_config.yaml
2. Define all glossary sources
3. Load with: GlossaryLoader(config_path="...")
```
→ See [HOW_TO_ADD_GLOSSARIES.md](HOW_TO_ADD_GLOSSARIES.md) - "Option 2: Configuration-Based"

### Add Multiple Glossaries from Different Domains
**Time:** 30 minutes
```
1. Create separate files for each domain
2. Configure in YAML
3. Load all at once with config
```
→ See [GLOSSARY_SYSTEM.md](GLOSSARY_SYSTEM.md) - "Use Case 1"

### Migrate Existing Glossaries
**Time:** 20 minutes
```
1. Export existing glossaries
2. Format to Excel/JSON
3. Add to configuration
4. Validate with test searches
```
→ See [HOW_TO_ADD_GLOSSARIES.md](HOW_TO_ADD_GLOSSARIES.md) - "Validation & Testing"

### Optimize Glossary Search
**Time:** 1 hour
```
1. Analyze search performance
2. Adjust fuzzy threshold
3. Enable caching
4. Monitor results
```
→ See [GLOSSARY_SEARCH_METHODS_ANALYSIS.md](GLOSSARY_SEARCH_METHODS_ANALYSIS.md)

---

## Glossary File Structures

### Minimal Excel (2 columns)
```
Korean          | English
임상시험        | clinical trial
피험자          | study subject
```

### Standard Excel (4+ columns)
```
Korean     | English        | Category   | Frequency
임상시험   | clinical trial | clinical   | high
피험자     | study subject  | clinical   | high
```

### JSON with Metadata
```json
{
  "glossary_metadata": { ... },
  "terms": [
    {
      "korean": "임상시험",
      "english": "clinical trial",
      "alternatives": { ... }
    }
  ]
}
```

---

## Performance Tips

✅ **DO:**
- Use configuration files for multiple glossaries
- Enable Valkey caching for production
- Index glossaries by domain
- Keep glossaries under 5000 terms per file

❌ **DON'T:**
- Hardcode glossary paths
- Mix multiple languages per file
- Use inconsistent formatting
- Skip validation before production

---

## Glossary Statistics

### Current Production Glossary
- **Total Terms:** 2906
- **Coding Form:** 1400 medical device terms
- **Clinical Trials:** 419 protocol terms
- **Coverage:** 89.6% of medical terminology

### Performance Metrics
- **Load Time:** < 1 second for 3000 terms
- **Search Time:** < 10ms for fuzzy match
- **Cache Hit:** Sub-millisecond with Valkey
- **Memory Usage:** ~3MB for 3000 terms

---

## Troubleshooting

**Q: Glossary terms not found during translation**
A: See [HOW_TO_ADD_GLOSSARIES.md](HOW_TO_ADD_GLOSSARIES.md) - "Troubleshooting" section

**Q: How do I validate my glossary?**
A: See [HOW_TO_ADD_GLOSSARIES.md](HOW_TO_ADD_GLOSSARIES.md) - "Validation & Testing" section

**Q: Why is search slow?**
A: See [GLOSSARY_SEARCH_METHODS_ANALYSIS.md](GLOSSARY_SEARCH_METHODS_ANALYSIS.md) - Performance section

**Q: How do I merge glossaries?**
A: See [GLOSSARY_SYSTEM.md](GLOSSARY_SYSTEM.md) - "Use Case 4: Merge Multiple External Glossaries"

---

## API Quick Reference

```python
# Load glossary
from src.glossary.glossary_loader import GlossaryLoader
loader = GlossaryLoader()
terms = loader.load_glossary("glossary.xlsx", "excel")

# Search glossary
from src.glossary.glossary_search import GlossarySearchEngine
engine = GlossarySearchEngine()
results = engine.search("임상시험")

# Use in pipeline
from src.production_pipeline_batch_enhanced import EnhancedBatchPipeline
pipeline = EnhancedBatchPipeline(glossary=terms)
```

---

## Related Documentation

- **Need to run translations?** → [03_core_features](../03_core_features/)
- **Want to understand architecture?** → [02_architecture](../02_architecture/)
- **Optimizing performance?** → [06_performance_and_optimization](../06_performance_and_optimization/)
