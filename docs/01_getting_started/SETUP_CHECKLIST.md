# Setup & Configuration Checklist

Complete this checklist to get the Phase 2 translation system running in your environment.

---

## 🔧 Pre-Installation (5 minutes)

### System Requirements
- [ ] Python 3.11 or higher installed
- [ ] macOS, Linux, or Windows with WSL
- [ ] 2GB minimum free disk space
- [ ] Administrator/sudo access (for Valkey installation)

**Verify Python version:**
```bash
python3 --version  # Should be 3.11+
pip --version      # Should work
```

### Get API Keys Ready
- [ ] Have OpenAI account (https://openai.com)
- [ ] Generated OpenAI API key (https://platform.openai.com/api-keys)
- [ ] API key starts with `sk-proj-`

---

## 📦 Step 1: Clone/Copy Repository (2 minutes)

```bash
# Navigate to project
cd /Users/won.suh/Project/transai/phase2
```

- [ ] Project directory accessible
- [ ] README.md visible
- [ ] src/ folder exists
- [ ] data/ folder exists

---

## 🐍 Step 2: Python Virtual Environment (3 minutes)

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Verify activation (should show (venv) in terminal)
which python
```

- [ ] venv directory created
- [ ] Terminal shows (venv) prefix
- [ ] `which python` points to venv/bin/python
- [ ] pip works inside venv

---

## 📥 Step 3: Install Dependencies (5 minutes)

```bash
# Upgrade pip
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

**Installation time:** 3-5 minutes depending on internet speed

- [ ] pip installed successfully
- [ ] All dependencies installed without errors
- [ ] No version conflicts

**Key packages installed:**
- openai>=1.51.2
- valkey>=6.1.1
- pandas>=2.0.0
- openpyxl>=3.1.0
- pytest (for testing)

---

## 💾 Step 4: Install & Start Valkey/Redis (3-5 minutes)

### Option A: Homebrew (macOS) - Recommended
```bash
# Install Valkey
brew install valkey

# Start Valkey server
valkey-server

# In another terminal, verify connection
valkey-cli ping  # Should respond: PONG
```

- [ ] Valkey installed
- [ ] `valkey-server` running in one terminal
- [ ] `valkey-cli ping` returns PONG

### Option B: Docker (Any Platform)
```bash
# Run Valkey in Docker
docker run -d -p 6379:6379 valkey/valkey

# Verify
docker ps | grep valkey
```

- [ ] Docker installed
- [ ] Docker container running
- [ ] Port 6379 exposed

### Option C: System Redis (If Available)
```bash
# Start Redis (if installed)
redis-server

# Verify
redis-cli ping  # Should respond: PONG
```

- [ ] Redis server running
- [ ] `redis-cli ping` returns PONG

**IMPORTANT: Keep Valkey/Redis running for all operations. Use separate terminal.**

---

## 🔑 Step 5: Configure Environment Variables (2 minutes)

```bash
# Edit .env file in project root
nano .env  # or your favorite editor
```

**Update the following:**

```bash
# REQUIRED: Your actual OpenAI API key
OPENAI_API_KEY=sk-proj-your_actual_key_here

# Optional (currently not used in production)
ANTHROPIC_API_KEY=your_key_optional
GEMINI_API_KEY=your_key_optional
UPSTAGE_API_KEY=your_key_optional

# Valkey Configuration (use defaults unless changed)
VALKEY_HOST=localhost
VALKEY_PORT=6379
VALKEY_DB=0

# Logging Level
LOG_LEVEL=INFO  # Options: DEBUG, INFO, WARNING, ERROR
```

**To get OpenAI API key:**
1. Visit https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Copy the generated key
4. Paste into `.env` file as shown above

- [ ] `.env` file edited
- [ ] OPENAI_API_KEY has actual key (not placeholder)
- [ ] VALKEY_HOST is localhost
- [ ] VALKEY_PORT is 6379
- [ ] File saved

**⚠️ SECURITY: Never commit .env file to git. Add to .gitignore:**
```bash
echo ".env" >> .gitignore
```

---

## 🧪 Step 6: Test Installation (3 minutes)

### Test Python Environment
```bash
# Should show paths inside venv/
python -c "import sys; print(sys.prefix)"

# Test imports
python -c "import openai; print(f'OpenAI version: {openai.__version__}')"
python -c "import valkey; print(f'Valkey installed: OK')"
python -c "import pandas; print(f'Pandas installed: OK')"
```

- [ ] Python points to venv
- [ ] OpenAI imports successfully
- [ ] Valkey imports successfully
- [ ] Pandas imports successfully

### Test Valkey Connection
```bash
# From project directory
python -c "
import valkey
r = valkey.Valkey(host='localhost', port=6379)
print('Valkey connection:', r.ping())
"
```

- [ ] Valkey connection test passes
- [ ] Output shows: Valkey connection: True

### Test API Key
```bash
# Quick OpenAI test
python -c "
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
print('OpenAI client created successfully')
print(f'API key loaded: {len(os.getenv(\"OPENAI_API_KEY\", \"\")) > 0}')
"
```

- [ ] OpenAI client creates without errors
- [ ] API key loaded from .env

---

## ✅ Step 7: Run Sample Translation (5 minutes)

### Using Sample Data
```bash
# Copy and modify example code
python << 'EOF'
from src.production_pipeline_batch_enhanced import EnhancedBatchPipeline
from src.glossary_loader import GlossaryLoader

# Load sample glossary
loader = GlossaryLoader()
glossary = loader.load_custom_glossary("data/sample_glossary.json")
print(f"Loaded {len(glossary.get('terms', []))} glossary terms")

# Create pipeline
pipeline = EnhancedBatchPipeline(style_guide="STANDARD")
print("Pipeline initialized successfully")
EOF
```

- [ ] Sample glossary loads without errors
- [ ] Pipeline initializes successfully
- [ ] No import errors

### View Sample Test Data
```bash
# Show sample data structure
python -c "
import json
with open('data/sample_test_data.json') as f:
    data = json.load(f)
    print(f\"Sample test segments: {len(data['test_segments'])}\")
    print(f\"First segment: {data['test_segments'][0]}\")
"
```

- [ ] Sample test data loads
- [ ] Contains 15 segments

---

## 🧬 Step 8: Run Unit Tests (5 minutes)

```bash
# Run pytest on sample data
pytest tests/ -v --tb=short

# Or run specific test
pytest tests/test_glossary_loader.py -v
```

**Expected outcome:**
- Some tests may pass
- Some may require specific data files
- No import errors

- [ ] pytest runs without errors
- [ ] Tests execute (pass or skip gracefully)
- [ ] No module import failures

---

## 📖 Step 9: Review Documentation (10 minutes)

Read these in order:

1. [ ] **README.md** (547 lines)
   - Overview of system
   - Quick start guide
   - Supported models
   - Usage examples

2. [ ] **MIGRATION_SUMMARY.md**
   - What was cleaned
   - Security checklist
   - Features ready to use

3. [ ] **TECHNICAL_IMPLEMENTATION.md**
   - Architecture deep dive
   - Component details
   - Data flow

4. [ ] **docs/PHASE2_MVP_ARCHITECTURE.md**
   - System architecture
   - Component relationships

---

## 🎯 Step 10: Ready for Development (Variable)

### Now You Can:

- [ ] Read documents in `data/` with custom pipeline
- [ ] Create your own glossary files
- [ ] Add custom style guides
- [ ] Run translations with your data
- [ ] Extend the system with new features
- [ ] Customize pipeline settings

### Common First Tasks:

1. **Create your glossary:**
   ```bash
   # Copy sample format
   cp data/sample_glossary.json data/my_glossary.json
   # Edit my_glossary.json with your terms
   ```

2. **Create test documents:**
   ```bash
   # Create Excel file with:
   # Column A: Korean text
   # Column B: (translation will go here)
   ```

3. **Run first translation:**
   See **README.md** → "Usage Examples" section

---

## ⚙️ Configuration Reference

### .env Variables

| Variable | Default | Required | Purpose |
|----------|---------|----------|---------|
| OPENAI_API_KEY | (none) | YES | OpenAI authentication |
| VALKEY_HOST | localhost | NO | Cache server address |
| VALKEY_PORT | 6379 | NO | Cache server port |
| VALKEY_DB | 0 | NO | Cache database index |
| LOG_LEVEL | INFO | NO | Logging verbosity |

### Style Guides

| Style | Tokens | Best For |
|-------|--------|----------|
| NONE | 0 | Baseline speed |
| MINIMAL | 100 | Quick translations |
| STANDARD | 400 | Most use cases |
| COMPREHENSIVE | 600 | Quality-focused |
| CLINICAL_PROTOCOL | 300 | Medical protocols |
| REGULATORY_COMPLIANCE | 300 | Legal documents |

See README.md for complete list.

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'openai'"
```bash
# Ensure venv is activated
source venv/bin/activate

# Reinstall requirements
pip install --upgrade -r requirements.txt
```

### "Connection refused: Cannot connect to Valkey"
```bash
# Check if Valkey is running
valkey-cli ping

# If not, start it
valkey-server &
```

### "Invalid API key" error
- Verify key starts with `sk-proj-`
- Check for extra spaces in `.env`
- Verify key at https://platform.openai.com/account/api-keys
- Ensure .env is in project root (not subdirectory)

### "Permission denied" errors
```bash
# Fix permissions
chmod u+w .env
chmod -R u+w data/
```

### Import errors after installation
```bash
# Deactivate and reactivate venv
deactivate
source venv/bin/activate

# Reinstall packages
pip install -r requirements.txt
```

---

## ✨ Success Criteria

You've successfully set up when:

- [x] Python 3.11+ with venv activated
- [x] All dependencies installed (`pip list` shows openai, valkey, pandas)
- [x] Valkey/Redis running (responds to `valkey-cli ping` with PONG)
- [x] `.env` file configured with real OpenAI API key
- [x] OpenAI client initializes without errors
- [x] Sample glossary loads (15+ terms)
- [x] README and MIGRATION_SUMMARY reviewed
- [x] Ready to use/develop

---

## 📞 Quick Reference Commands

```bash
# Activate environment
source venv/bin/activate

# Install/update dependencies
pip install -r requirements.txt

# Start Valkey (separate terminal)
valkey-server

# Check Valkey connection
valkey-cli ping

# Run tests
pytest tests/ -v

# Python interactive mode
python

# Deactivate venv when done
deactivate
```

---

## 📚 Next Steps

1. **Read README.md** - Full system documentation
2. **Try sample translation** - See README.md "Usage Examples"
3. **Create custom glossary** - Follow "Glossary Format" section
4. **Build your workflow** - Customize pipelines for your needs
5. **Extend system** - Add features as needed (see "Development" section)

---

**Status:** Production Ready ✅
**Last Updated:** November 23, 2025

For detailed information, see **README.md** and other documentation files.
