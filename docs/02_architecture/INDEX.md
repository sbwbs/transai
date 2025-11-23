# Architecture & Design

Understand how TransAI is structured and designed. These documents explain the system architecture, components, and design decisions.

## Documents in this Category

### 1. [PHASE2_MVP_ARCHITECTURE.md](PHASE2_MVP_ARCHITECTURE.md)
**Purpose:** High-level system architecture overview
- System components and their relationships
- Data flow diagrams
- Component interactions
- Architecture decisions

**Read this if:** You want to understand how TransAI's main components work together

### 2. [PHASE2_ARCHITECTURE_DIAGRAM.md](PHASE2_ARCHITECTURE_DIAGRAM.md)
**Purpose:** Visual architecture diagrams and explanations
- System diagrams with visual representations
- Component hierarchies
- Integration points
- Visual data flow

**Read this if:** You prefer visual explanations of the architecture

### 3. [IMPLEMENTATION_BLUEPRINT.md](IMPLEMENTATION_BLUEPRINT.md)
**Purpose:** Detailed technical implementation specifications
- Implementation approaches
- Technical decisions
- Code organization strategies
- Best practices for implementation

**Read this if:** You're implementing new features or modifying existing code

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│         TransAI System Architecture                  │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────┐         ┌──────────────┐        │
│  │   Input      │         │   Pipeline   │        │
│  │   (Excel)    │ ────→   │   Processing │        │
│  └──────────────┘         └──────────────┘        │
│         ▲                         │                 │
│         │                         ▼                 │
│         │                  ┌──────────────┐        │
│         └──────────────────│  Glossary &  │        │
│                            │  Memory      │        │
│                            │  (Valkey)    │        │
│                            └──────────────┘        │
│                                   │                 │
│                                   ▼                 │
│                            ┌──────────────┐        │
│                            │   LLM API    │        │
│                            │  (GPT-5 OWL) │        │
│                            └──────────────┘        │
│                                   │                 │
│                                   ▼                 │
│                            ┌──────────────┐        │
│                            │   Output     │        │
│                            │   (Excel)    │        │
│                            └──────────────┘        │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## Reading Path

1. **Start here:** [PHASE2_MVP_ARCHITECTURE.md](PHASE2_MVP_ARCHITECTURE.md) - Get overview
2. **Visual learner?** → [PHASE2_ARCHITECTURE_DIAGRAM.md](PHASE2_ARCHITECTURE_DIAGRAM.md)
3. **Implementing changes?** → [IMPLEMENTATION_BLUEPRINT.md](IMPLEMENTATION_BLUEPRINT.md)

---

## Key Concepts

**You should understand:**
- ✅ How pipelines process documents
- ✅ Role of glossary system
- ✅ Memory/caching architecture
- ✅ LLM integration points
- ✅ Data flow through the system

---

## Next Steps

- **Ready to use?** → Go to [03_core_features](../03_core_features/)
- **Understanding glossaries?** → Go to [04_glossary_and_terminology](../04_glossary_and_terminology/)
- **Optimizing performance?** → Go to [06_performance_and_optimization](../06_performance_and_optimization/)
