# File Organization Guide - What to Commit

## 📊 Analysis of src/ Root Files

### Test & Demo Files in src/ Root (15 files)

**These are LOOSE files that should be moved to tests/ or deleted:**

| File | Type | Keep? | Reason |
|------|------|-------|--------|
| `test_20_samples.py` | Test | ❌ Move | Duplicate - in tests/ as properly organized test |
| `test_en_ko_enhanced_glossary.py` | Test | ❌ Move | Specific test - belongs in tests/ |
| `test_enhanced_prompts_ab.py` | Test | ❌ Move | A/B testing - belongs in tests/ |
| `test_improved_pipelines.py` | Test | ❌ Move | Pipeline test - belongs in tests/ |
| `test_ko_en_batch.py` | Test | ❌ Move | Batch test - belongs in tests/ |
| `test_new_priority.py` | Test | ❌ Move | Priority test - belongs in tests/ |
| `test_priority.py` | Test | ❌ Move | Priority test - duplicate |
| `test_reference_evaluation.py` | Test | ❌ Move | Evaluation test - belongs in tests/ |
| `test_tag_preservation.py` | Test | ❌ Move | Tag test - belongs in tests/ |
| `test_valkey_integration.py` | Test | ⚠️ Check | Also in tests/ - **KEEP proper version in tests/** |
| `run_full_dataset_parallel.py` | Script | ❌ Delete | One-off testing script |
| `run_full_production_test.py` | Script | ❌ Delete | One-off testing script |
| `run_improved_pipelines_full.py` | Script | ❌ Delete | One-off testing script |
| `run_parallel_production_test.py` | Script | ❌ Delete | One-off testing script |
| `show_prompt_template.py` | Script | ❌ Delete | One-off debugging script |

### Recommendation

**Move or delete 15 loose test files from src/ root:**
- Tests should be in `src/tests/` only
- One-off scripts should be in `src/archive/scripts/` or deleted

---

## 📂 Current Structure Issues

### ❌ Problem: Test Files Scattered Across Root

```
src/
├── test_20_samples.py              ← Should not be here
├── test_en_ko_enhanced_glossary.py ← Should not be here
├── test_enhanced_prompts_ab.py     ← Should not be here
├── test_improved_pipelines.py      ← Should not be here
├── test_ko_en_batch.py             ← Should not be here
├── test_new_priority.py            ← Should not be here
├── test_priority.py                ← Should not be here
├── test_reference_evaluation.py    ← Should not be here
├── test_tag_preservation.py        ← Should not be here
├── test_valkey_integration.py      ← Duplicate
├── run_*.py (4 files)              ← Should not be here
├── show_prompt_template.py         ← Should not be here
├── tests/                          ← Proper location
│   ├── test_*.py (proper tests)
│   └── *.py (proper test files)
```

### ✅ Solution: Clean Organization

```
src/
├── production_pipeline_*.py        ✓ Keep
├── glossary_loader.py              ✓ Keep
├── glossary_search.py              ✓ Keep
├── style_guide_config.py           ✓ Keep
├── analyze_token_usage.py          ✓ Keep
├── memory/                         ✓ Keep
├── utils/                          ✓ Keep
├── clinical_protocol_system/       ✓ Keep
├── tests/                          ✓ Keep (proper location)
│   ├── test_*.py
│   ├── *_test.py
│   └── test_valkey_integration.py (proper version)
├── archive/scripts/                ✓ Can move loose scripts here
│   ├── run_full_production_test.py
│   ├── run_parallel_production_test.py
│   ├── show_prompt_template.py
│   └── etc.
└── data/                           ✓ Keep
```

---

## 🎯 Action Plan

### 1. Delete One-Off Test Scripts from src/ Root

These are temporary testing scripts that shouldn't be in the main codebase:

```bash
cd /Users/won.suh/Project/transai/src

# Delete one-off scripts
rm -f run_full_dataset_parallel.py
rm -f run_full_production_test.py
rm -f run_improved_pipelines_full.py
rm -f run_parallel_production_test.py
rm -f show_prompt_template.py
```

**Rationale:** These are development/debugging scripts, not part of the main codebase.

### 2. Move Individual Test Files to tests/ or Delete

Check each test file:

```bash
# These are probably redundant with tests/ versions
rm -f test_20_samples.py
rm -f test_en_ko_enhanced_glossary.py
rm -f test_enhanced_prompts_ab.py
rm -f test_improved_pipelines.py
rm -f test_ko_en_batch.py
rm -f test_new_priority.py
rm -f test_priority.py
rm -f test_reference_evaluation.py
rm -f test_tag_preservation.py
rm -f test_valkey_integration.py  # Keep proper version in tests/
```

**Rationale:** Tests belong in `tests/` directory. Root-level test files are development artifacts.

### 3. Verify tests/ Has All Necessary Tests

```bash
ls -la tests/
# Should have:
# - test_phase2_integration.py (main integration test)
# - test_valkey_integration.py (proper version)
# - test_context_builder_integration.py
# - test_enhanced_translation_integration.py
# - test_be003_integration.py
# - etc.
```

---

## 📋 Files That SHOULD Be Committed

### Production Code (Keep)
```
✓ production_pipeline_batch_enhanced.py
✓ production_pipeline_en_ko.py
✓ production_pipeline_en_ko_improved.py
✓ production_pipeline_ko_en_improved.py
✓ production_pipeline_with_style_guide.py
✓ production_pipeline_working.py
✓ glossary_loader.py
✓ glossary_search.py
✓ create_combined_glossary.py
✓ style_guide_config.py
✓ analyze_token_usage.py
✓ translation_qa.py
✓ reference_evaluation_system.py
```

### Supporting Modules (Keep)
```
✓ memory/valkey_manager.py
✓ memory/session_manager.py
✓ memory/consistency_tracker.py
✓ memory/cached_glossary_search.py
✓ utils/tag_handler.py
✓ utils/segment_filter.py
✓ clinical_protocol_system/*.py
```

### Tests (Keep - but organized in tests/)
```
✓ tests/test_phase2_integration.py
✓ tests/test_valkey_integration.py
✓ tests/test_context_builder_integration.py
✓ tests/test_enhanced_translation_integration.py
✓ tests/test_be003_integration.py
✓ tests/test_be003_core.py
✓ tests/test_data_loader_performance.py
✓ tests/test_imports.py
✓ tests/test_package_init.py
✓ tests/test_token_optimizer_simple.py
✓ tests/production_import_test.py
✓ tests/valkey_integration_demo.py
```

### Configuration (Keep)
```
✓ requirements.txt
✓ .env (sanitized - in .gitignore)
✓ config/
```

### Data (Keep - sample only)
```
✓ data/sample_glossary.json
✓ data/sample_test_data.json
✓ data/production_glossary.json
✓ data/combined_en_ko_glossary.xlsx
```

---

## 📦 Files That Should NOT Be Committed

### Loose Test Files in src/ Root
```
✗ test_20_samples.py
✗ test_en_ko_enhanced_glossary.py
✗ test_enhanced_prompts_ab.py
✗ test_improved_pipelines.py
✗ test_ko_en_batch.py
✗ test_new_priority.py
✗ test_priority.py
✗ test_reference_evaluation.py
✗ test_tag_preservation.py
✗ test_valkey_integration.py (root version - use tests/ version)
```

### One-Off Scripts
```
✗ run_full_dataset_parallel.py
✗ run_full_production_test.py
✗ run_improved_pipelines_full.py
✗ run_parallel_production_test.py
✗ show_prompt_template.py
```

### Generated Files (Already in .gitignore)
```
✗ en_ko_results_*.xlsx (test outputs)
✗ logs/ (application logs)
✗ results/ (test results)
✗ __pycache__/ (Python cache)
✗ .pytest_cache/ (pytest cache)
```

---

## ✅ Recommended Clean-Up

### Option 1: Delete (Recommended for One-Off Scripts)
Most appropriate if these are temporary development/testing files.

```bash
cd /Users/won.suh/Project/transai/src
# Delete one-off test scripts
rm -f run_*.py show_*.py
# Delete redundant test files in root
rm -f test_*.py
```

### Option 2: Archive (If You Want to Keep for Reference)
If you want to keep these for historical reference:

```bash
# Move to archive
mkdir -p archive/old_tests
mv src/test_*.py archive/old_tests/
mv src/run_*.py archive/old_tests/
mv src/show_*.py archive/old_tests/
```

### Option 3: Hybrid (Recommended)
Keep proper tests in `tests/`, delete one-off scripts:

```bash
# Delete one-off scripts
rm -f src/run_*.py
rm -f src/show_*.py

# Delete loose test files (keep tests/ versions)
rm -f src/test_*.py
```

---

## 🎯 Final File Count After Cleanup

**Before:** 58 files in src/ root (messy)
**After:** ~13 files in src/ root (clean)
- 6-7 production pipeline files
- 3-4 core modules
- requirements.txt, .env
- 1 README-like file

**Tests:** 12 organized files in tests/

---

## 📝 Commit What's Left

After cleanup, commit:

```bash
cd /Users/won.suh/Project/transai

# Add all cleaned code
git add src/production_pipeline_*.py
git add src/glossary_*.py
git add src/memory/
git add src/utils/
git add src/tests/
git add src/data/sample*.json
git add src/requirements.txt

# Add documentation
git add docs/
git add README.md
git add .gitignore

# Commit
git commit -m "Clean production codebase - remove test scripts from root

- Moved tests to proper tests/ directory
- Removed one-off development scripts
- Kept production pipelines and core modules
- Organized structure following Python conventions"
```

---

## 🔍 How to Identify What to Delete

Ask yourself about each file:

1. **Is this part of the main codebase?**
   - YES → Keep
   - NO → Delete or archive

2. **Does this run in production?**
   - YES → Keep
   - NO → Delete or archive

3. **Is this a development/testing artifact?**
   - YES → Delete (unless historical value)
   - NO → Keep

4. **Are there better organized versions elsewhere?**
   - YES → Delete the loose version
   - NO → Keep

---

## 📚 References

- [.gitignore Documentation](../.gitignore) - What's automatically excluded
- [Git Security Checklist](GIT_SECURITY_CHECKLIST.md) - What not to commit
- [Setup Checklist](SETUP_CHECKLIST.md) - Running tests properly

---

**Summary:** Delete 15 loose test/script files from src/ root. Keep production code and tests/ directory. Commit clean, organized codebase.

**Status:** Ready for cleanup ✅
