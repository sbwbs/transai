# Core Features

Learn about the main features of TransAI and how to use them. These documents explain how the translation pipeline, caching system, and other core components work.

## Documents in this Category

### 1. [TRANSLATION_PIPELINE_STEPBYSTEP.md](TRANSLATION_PIPELINE_STEPBYSTEP.md)
**Purpose:** Detailed walkthrough of the translation pipeline
- Step-by-step pipeline workflow
- How documents are processed
- Stage-by-stage explanations
- Input and output handling

**Read this if:** You want to understand how documents flow through the system

### 2. [VALKEY_INTEGRATION_SUMMARY.md](VALKEY_INTEGRATION_SUMMARY.md)
**Purpose:** Caching system and memory architecture
- Valkey/Redis integration details
- 3-tier memory architecture
- Performance optimization
- Caching strategies

**Read this if:** You want to understand or optimize the caching system

### 3. [TAG_PRESERVATION_IMPLEMENTATION.md](TAG_PRESERVATION_IMPLEMENTATION.md)
**Purpose:** CAT tool tag handling and preservation
- Tag extraction and restoration
- CAT tool integration
- Supported tag formats
- Usage examples

**Read this if:** You need to preserve formatting tags from CAT tools

### 4. [TECHNICAL_IMPLEMENTATION.md](TECHNICAL_IMPLEMENTATION.md)
**Purpose:** Deep dive into technical implementation details
- Implementation approaches for each component
- Code patterns and conventions
- Error handling strategies
- Integration details

**Read this if:** You're diving deep into the codebase

---

## Feature Quick Reference

| Feature | Document | Purpose |
|---------|----------|---------|
| Document Translation | [TRANSLATION_PIPELINE_STEPBYSTEP.md](TRANSLATION_PIPELINE_STEPBYSTEP.md) | How translations are processed |
| Performance Caching | [VALKEY_INTEGRATION_SUMMARY.md](VALKEY_INTEGRATION_SUMMARY.md) | Speed up lookups with Valkey |
| Tag Preservation | [TAG_PRESERVATION_IMPLEMENTATION.md](TAG_PRESERVATION_IMPLEMENTATION.md) | Keep formatting tags intact |
| Technical Details | [TECHNICAL_IMPLEMENTATION.md](TECHNICAL_IMPLEMENTATION.md) | Implementation specifics |

---

## Reading Path

**For Translation Users:**
1. [TRANSLATION_PIPELINE_STEPBYSTEP.md](TRANSLATION_PIPELINE_STEPBYSTEP.md) - Understand the flow
2. [VALKEY_INTEGRATION_SUMMARY.md](VALKEY_INTEGRATION_SUMMARY.md) - Learn about caching
3. Start translating!

**For CAT Tool Users:**
1. [TAG_PRESERVATION_IMPLEMENTATION.md](TAG_PRESERVATION_IMPLEMENTATION.md) - How tags are preserved
2. [TRANSLATION_PIPELINE_STEPBYSTEP.md](TRANSLATION_PIPELINE_STEPBYSTEP.md) - Full pipeline context

**For Developers:**
1. [TRANSLATION_PIPELINE_STEPBYSTEP.md](TRANSLATION_PIPELINE_STEPBYSTEP.md) - Overview
2. [TECHNICAL_IMPLEMENTATION.md](TECHNICAL_IMPLEMENTATION.md) - Implementation details
3. [VALKEY_INTEGRATION_SUMMARY.md](VALKEY_INTEGRATION_SUMMARY.md) - Performance optimization

---

## Core Features at a Glance

✨ **Translation Pipeline**
- Batch processing for efficiency
- Glossary integration
- Quality scoring
- Error handling

⚡ **Caching System**
- Sub-millisecond lookups
- Persistent term storage
- Session-based tracking
- Automatic consistency

🏷️ **Tag Preservation**
- CAT tool format support
- Automatic extraction/restoration
- Multiple tag types
- Metadata preservation

🔧 **Technical Excellence**
- Robust error handling
- Comprehensive logging
- Performance optimization
- Clean code architecture

---

## Common Tasks

**How do I translate a document?**
→ See [TRANSLATION_PIPELINE_STEPBYSTEP.md](TRANSLATION_PIPELINE_STEPBYSTEP.md)

**How do I improve performance?**
→ See [VALKEY_INTEGRATION_SUMMARY.md](VALKEY_INTEGRATION_SUMMARY.md)

**How do I keep my tags intact?**
→ See [TAG_PRESERVATION_IMPLEMENTATION.md](TAG_PRESERVATION_IMPLEMENTATION.md)

**Where's the technical implementation?**
→ See [TECHNICAL_IMPLEMENTATION.md](TECHNICAL_IMPLEMENTATION.md)

---

## Related Documentation

- **Need glossary help?** → [04_glossary_and_terminology](../04_glossary_and_terminology/)
- **Want to optimize?** → [06_performance_and_optimization](../06_performance_and_optimization/)
- **Understanding architecture?** → [02_architecture](../02_architecture/)
