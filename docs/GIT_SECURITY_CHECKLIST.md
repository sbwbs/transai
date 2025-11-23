# Git Security Checklist

**Before committing or pushing to git, verify no sensitive data is included.**

## 🔐 What Should NEVER Be Committed

### API Keys & Credentials
- ❌ `.env` file with real API keys
- ❌ `*.key`, `*.pem` files
- ❌ `credentials.json`, `secrets.json`
- ❌ AWS/Azure configuration files

### Customer Data
- ❌ Excel files (`.xlsx`, `.xls`) - EXCEPT `src/data/sample_*.json`
- ❌ CSV files (`.csv`, `.tsv`)
- ❌ Database files (`.db`, `.sqlite`)
- ❌ Customer-specific test results
- ❌ Translation output files with real data

### Generated/Temporary Files
- ❌ `logs/` directory
- ❌ `results/` directory
- ❌ `src/results/` directory
- ❌ `__pycache__/` directories
- ❌ `.pytest_cache/`
- ❌ Virtual environment (`venv/`)

## ✅ What SHOULD Be Committed

### Code
- ✅ `src/*.py` - All Python modules
- ✅ `src/memory/` - Memory layer code
- ✅ `src/utils/` - Utility modules
- ✅ `src/tests/` - Test suite
- ✅ `src/clinical_protocol_system/` - Domain code

### Documentation
- ✅ `docs/` - All markdown documentation
- ✅ `README.md` - Project overview

### Configuration
- ✅ `src/requirements.txt` - Dependencies
- ✅ `.gitignore` - This security file
- ✅ `src/.env.example` (if created with dummy values)

### Sample Data
- ✅ `src/data/sample_glossary.json` - Example glossary
- ✅ `src/data/sample_test_data.json` - Example test data

## 🛡️ Pre-Commit Checklist

Before running `git commit`:

```bash
# 1. Check git status to see what will be committed
git status

# 2. Verify no data files are staged
git diff --cached --name-only | grep -E "\.(xlsx|csv|db|log)$"
# Should return NOTHING - if it does, remove with: git reset <file>

# 3. Verify .env is NOT staged
git diff --cached src/.env
# Should return NOTHING

# 4. Check for any unexpected files
git diff --cached --stat
```

## 🚀 Safe Commit Process

### Step 1: Review Changes
```bash
# See what's staged for commit
git status

# See detailed changes
git diff --cached
```

### Step 2: Remove Sensitive Files
```bash
# If you accidentally staged a data file, unstage it:
git reset src/results/*.xlsx
git reset src/en_ko_results_*.xlsx

# Or unstage everything and re-add carefully:
git reset
git add .  # This will skip ignored files automatically
```

### Step 3: Double-Check
```bash
# Verify only safe files are staged
git diff --cached --name-only

# Check for .env file
git diff --cached | grep -i "api\|key\|secret"
# Should return NOTHING
```

### Step 4: Commit
```bash
git commit -m "Your commit message"
```

## 📋 Files in .gitignore

The `.gitignore` file is configured to automatically ignore:

### Data Files
```
*.xlsx           # Excel files
*.xls            # Excel files
*.csv            # CSV files
*.tsv            # Tab-separated values
*.db             # Databases
*.sqlite         # SQLite databases
```

### Results & Logs
```
src/results/     # Translation results
src/logs/        # Application logs
logs/            # System logs
*.log            # Log files
```

### Environment
```
.env             # Environment variables with secrets
.env.local       # Local overrides
```

### Python
```
venv/            # Virtual environment
__pycache__/     # Python cache
*.pyc            # Compiled Python
.pytest_cache/   # Pytest cache
```

## ⚠️ Common Mistakes

### ❌ Mistake: Committing .env with API key
```bash
git add .env           # DON'T DO THIS
git commit -m "Add config"
```
**Prevention:** `.env` is in `.gitignore` - won't be added by default

### ❌ Mistake: Committing customer data Excel file
```bash
git add src/results/customer_translation.xlsx
git commit -m "Add results"
```
**Prevention:** All `.xlsx` files are in `.gitignore`

### ❌ Mistake: Force-adding ignored file
```bash
git add -f src/results/data.xlsx  # The -f bypasses .gitignore!
git commit -m "Add data"
```
**Prevention:** Never use `git add -f` for data files

### ✅ Solution: Check before committing
```bash
# Always use this before commit
git status
git diff --cached
```

## 🔍 If You Accidentally Committed Secrets

### Immediate Action (Not Yet Pushed)
```bash
# If just committed but not pushed:
git reset --soft HEAD~1   # Undo commit but keep changes
git reset src/.env        # Unstage the file
git commit -m "Remove .env from commit"
```

### After Push (Critical!)
```bash
# 1. Regenerate API keys immediately
# 2. Remove from git history with BFG or git-filter-repo:
git filter-branch --tree-filter 'rm -f src/.env' -- --all
git push origin --force-with-lease
# 3. Notify team of key rotation
```

## 📝 Creating .env Template

To help users, create `src/.env.example` with placeholder values:

```bash
# Example .env file for configuration
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_optional
VALKEY_HOST=localhost
VALKEY_PORT=6379
VALKEY_DB=0
LOG_LEVEL=INFO
```

**Then commit only the .example file:**
```bash
git add src/.env.example   # Safe - no real keys
git commit -m "Add .env.example template"
```

## 🔔 CI/CD Integration

If using GitHub Actions or similar CI/CD:

### Pre-commit Hook (Optional but Recommended)
```bash
# In .git/hooks/pre-commit
#!/bin/bash
# Prevent committing .env files
if git diff --cached --name-only | grep -E "\.env$|\.xlsx|\.csv"; then
    echo "ERROR: Attempting to commit sensitive files!"
    echo "Check .gitignore for patterns"
    exit 1
fi
```

### GitHub Secrets for Actions
Store all API keys in GitHub Secrets, not in code:
- Settings → Secrets and variables → Actions
- Add OPENAI_API_KEY, etc.
- Reference as `${{ secrets.OPENAI_API_KEY }}` in workflows

## ✅ Security Verification

Run this before major pushes:

```bash
# 1. Check for API key patterns in staged files
git diff --cached | grep -iE "sk-proj|api[_-]?key|secret|password"

# 2. Check for .env files
git ls-files | grep -E "\.env$|\.env\..*"

# 3. Check for data files
git ls-files | grep -E "\.(xlsx|csv|db|sqlite)$" | grep -v "sample"

# 4. All should be empty - if not, remove files
```

## 📚 Related Documentation

- [Completion Report](COMPLETION_REPORT.md) - Security cleanup details
- [Migration Summary](MIGRATION_SUMMARY.md) - What was removed
- [Setup Checklist](SETUP_CHECKLIST.md) - Configuration guide

## 🎯 TL;DR

1. **Before every commit:** Run `git status` and `git diff --cached`
2. **Never commit:** `.env`, `*.xlsx`, `*.csv`, `logs/`, `results/`
3. **Safe to commit:** `src/*.py`, `docs/`, `sample_*.json`
4. **Accidentally committed?** Use `git reset` to undo before pushing
5. **Already pushed?** Rotate API keys immediately and contact security

---

**Last Updated:** November 23, 2025
**Status:** Security Critical ⚠️
