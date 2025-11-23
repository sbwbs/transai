#!/usr/bin/env python3
"""
Style Guide Configuration for A/B Testing
Configurable variants to test quality vs. token efficiency
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import json
import os
from datetime import datetime


class StyleGuideVariant(Enum):
    """Available style guide variants for A/B testing"""
    NONE = "none"                    # No style guide (baseline)
    MINIMAL = "minimal"              # Essential only (~100 tokens)
    COMPACT = "compact"              # Condensed version (~200 tokens)
    STANDARD = "standard"            # Full style guide (~400 tokens)
    COMPREHENSIVE = "comprehensive"  # Extended with examples (~600 tokens)
    CLINICAL_PROTOCOL = "clinical_protocol"  # EN-KO Clinical Protocol specialized (~300 tokens)
    CLINICAL_PROTOCOL_STRICT = "clinical_protocol_strict"  # EN-KO Strict literal translation (~250 tokens)
    REGULATORY_COMPLIANCE = "regulatory_compliance"  # KO-EN Regulatory compliance (~300 tokens)
    REGULATORY_COMPLIANCE_ENHANCED = "regulatory_compliance_enhanced"  # KO-EN with examples (~900 tokens)
    CLINICAL_PROTOCOL_STRICT_ENHANCED = "clinical_protocol_strict_enhanced"  # EN-KO with examples (~900 tokens)
    CUSTOM = "custom"                # User-defined configuration


@dataclass
class StyleGuideConfig:
    """Configuration for style guide variants"""
    variant: StyleGuideVariant
    name: str
    description: str
    estimated_tokens: int
    quality_score: float  # Expected quality improvement (0.0-1.0)
    token_efficiency: float  # Token reduction maintained (0.0-1.0)
    enabled: bool = True
    custom_rules: Optional[Dict] = None


class StyleGuideManager:
    """Manages different style guide variants for A/B testing"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "style_guide_config.json"
        self.variants = self._load_variants()
        self.current_variant = StyleGuideVariant.STANDARD
        self.experiment_mode = False
        self.experiment_results = {}
        
    def _load_variants(self) -> Dict[StyleGuideVariant, StyleGuideConfig]:
        """Load style guide variants from configuration"""
        variants = {
            StyleGuideVariant.NONE: StyleGuideConfig(
                variant=StyleGuideVariant.NONE,
                name="No Style Guide",
                description="Baseline translation without style instructions",
                estimated_tokens=0,
                quality_score=0.0,
                token_efficiency=1.0,
                enabled=True
            ),
            
            StyleGuideVariant.MINIMAL: StyleGuideConfig(
                variant=StyleGuideVariant.MINIMAL,
                name="Minimal Style Guide",
                description="Essential ICH-GCP requirements only",
                estimated_tokens=100,
                quality_score=0.3,
                token_efficiency=0.95,
                enabled=True
            ),
            
            StyleGuideVariant.COMPACT: StyleGuideConfig(
                variant=StyleGuideVariant.COMPACT,
                name="Compact Style Guide",
                description="Condensed version with key rules",
                estimated_tokens=200,
                quality_score=0.6,
                token_efficiency=0.90,
                enabled=True
            ),
            
            StyleGuideVariant.STANDARD: StyleGuideConfig(
                variant=StyleGuideVariant.STANDARD,
                name="Standard Style Guide",
                description="Full clinical protocol style guide",
                estimated_tokens=400,
                quality_score=0.8,
                token_efficiency=0.85,
                enabled=True
            ),
            
            StyleGuideVariant.COMPREHENSIVE: StyleGuideConfig(
                variant=StyleGuideVariant.COMPREHENSIVE,
                name="Comprehensive Style Guide",
                description="Extended with examples and detailed rules",
                estimated_tokens=600,
                quality_score=0.9,
                token_efficiency=0.80,
                enabled=True
            ),
            
            StyleGuideVariant.CLINICAL_PROTOCOL: StyleGuideConfig(
                variant=StyleGuideVariant.CLINICAL_PROTOCOL,
                name="EN-KO Clinical Protocol",
                description="Specialized for EN→KO clinical protocol translation",
                estimated_tokens=300,
                quality_score=0.85,
                token_efficiency=0.88,
                enabled=True
            ),
            
            StyleGuideVariant.CLINICAL_PROTOCOL_STRICT: StyleGuideConfig(
                variant=StyleGuideVariant.CLINICAL_PROTOCOL_STRICT,
                name="EN-KO Strict Literal",
                description="Strict literal translation for regulatory review",
                estimated_tokens=250,
                quality_score=0.92,
                token_efficiency=0.90,
                enabled=True
            ),
            
            StyleGuideVariant.REGULATORY_COMPLIANCE: StyleGuideConfig(
                variant=StyleGuideVariant.REGULATORY_COMPLIANCE,
                name="KO-EN Regulatory",
                description="KO→EN with hallucination prevention and conciseness",
                estimated_tokens=300,
                quality_score=0.88,
                token_efficiency=0.85,
                enabled=True
            ),

            StyleGuideVariant.REGULATORY_COMPLIANCE_ENHANCED: StyleGuideConfig(
                variant=StyleGuideVariant.REGULATORY_COMPLIANCE_ENHANCED,
                name="KO-EN Enhanced with Examples",
                description="KO→EN with style guide + few-shot examples",
                estimated_tokens=900,
                quality_score=0.93,
                token_efficiency=0.75,
                enabled=True
            ),

            StyleGuideVariant.CLINICAL_PROTOCOL_STRICT_ENHANCED: StyleGuideConfig(
                variant=StyleGuideVariant.CLINICAL_PROTOCOL_STRICT_ENHANCED,
                name="EN-KO Enhanced with Examples",
                description="EN→KO with style guide + few-shot examples",
                estimated_tokens=900,
                quality_score=0.95,
                token_efficiency=0.75,
                enabled=True
            )
        }
        
        # Load custom variants if config file exists
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    custom_config = json.load(f)
                    for variant_name, config_data in custom_config.get('custom_variants', {}).items():
                        if variant_name not in [v.value for v in StyleGuideVariant]:
                            custom_variant = StyleGuideVariant(variant_name)
                            variants[custom_variant] = StyleGuideConfig(
                                variant=custom_variant,
                                **config_data
                            )
            except Exception as e:
                print(f"Warning: Could not load custom style guide config: {e}")
        
        return variants
    
    def get_style_guide(self, variant: StyleGuideVariant) -> str:
        """Get the style guide content for the specified variant"""
        if variant == StyleGuideVariant.NONE:
            return ""
        
        elif variant == StyleGuideVariant.MINIMAL:
            return self._get_minimal_style_guide()
        
        elif variant == StyleGuideVariant.COMPACT:
            return self._get_compact_style_guide()
        
        elif variant == StyleGuideVariant.STANDARD:
            return self._get_standard_style_guide()
        
        elif variant == StyleGuideVariant.COMPREHENSIVE:
            return self._get_comprehensive_style_guide()
        
        elif variant == StyleGuideVariant.CLINICAL_PROTOCOL:
            return self._get_en_ko_clinical_protocol_style_guide()
        
        elif variant == StyleGuideVariant.CLINICAL_PROTOCOL_STRICT:
            return self._get_en_ko_clinical_protocol_strict_style_guide()
        
        elif variant == StyleGuideVariant.REGULATORY_COMPLIANCE:
            return self._get_ko_en_regulatory_compliance_style_guide()

        elif variant == StyleGuideVariant.REGULATORY_COMPLIANCE_ENHANCED:
            return self._get_ko_en_enhanced_with_examples()

        elif variant == StyleGuideVariant.CLINICAL_PROTOCOL_STRICT_ENHANCED:
            return self._get_en_ko_enhanced_with_examples()

        elif variant == StyleGuideVariant.CUSTOM:
            return self._get_custom_style_guide()

        else:
            return self._get_standard_style_guide()
    
    def _get_minimal_style_guide(self) -> str:
        """Minimal essential style guide (~100 tokens)"""
        return """\n## Style: ICH-GCP Clinical Protocol
- Use formal professional register
- 임상시험→Clinical Study, 시험대상자→Study Subject
- Follow ICH-GCP terminology standards
- Maintain regulatory compliance"""
    
    def _get_compact_style_guide(self) -> str:
        """Compact style guide (~200 tokens)"""
        return """\n## Style: ICH-GCP Clinical Protocol

**REGISTER & FORMALITY:**
- Formal professional register (합니다→will/shall)
- Neutral tone, declarative statements

**TERMINOLOGY:**
- 임상시험→Clinical Study (protocol context)
- 시험대상자→Study Subject (trial context)
- 이상반응→Adverse Event (not side effect)
- 임상시험용 의약품→Investigational Product

**COMPLIANCE:**
- Follow ICH-GCP E6(R2) standards
- Subject safety priority
- Written informed consent required"""
    
    def _get_standard_style_guide(self) -> str:
        """Standard style guide (~400 tokens)"""
        return """\n## Clinical Protocol Style Guide (ICH-GCP E6(R2))

**REGISTER & FORMALITY:**
- Use formal professional register throughout
- Transform Korean honorifics (합니다/습니다) → neutral professional (will/shall)
- Use declarative statements for procedures, conditional for contingencies
- Maintain neutral, objective tone without cultural hierarchical markers

**SENTENCE STRUCTURE:**
- Break long Korean sentences (>20 words) into 2-3 shorter English sentences
- Maximum 25 words per English sentence for regulatory clarity
- Use active voice for procedures ("The investigator will assess...")
- Use passive voice for results ("Efficacy will be evaluated...")

**TERMINOLOGY CONSISTENCY:**
- 임상시험 → Clinical Study (NOT Clinical Trial in protocol context)
- 임상시험용 의약품 → Investigational Product (NOT test drug)
- 시험대상자 → Study Subject (NOT patient in trial context)
- 이상반응 → Adverse Event (NOT side effect)
- 중대한 이상반응 → Serious Adverse Event

**REGULATORY COMPLIANCE:**
- Include: "This study will be conducted in accordance with Declaration of Helsinki, ICH-GCP"
- Priority: "The safety and well-being of study subjects is the highest priority"
- Consent: "All study subjects must provide written informed consent"
- Risk: "The risk-benefit ratio has been assessed and documented" """
    
    def _get_comprehensive_style_guide(self) -> str:
        """Comprehensive style guide with examples (~600 tokens)"""
        return """\n## Comprehensive Clinical Protocol Style Guide (ICH-GCP E6(R2))

**REGISTER & FORMALITY:**
- Use formal professional register throughout all sections
- Transform Korean honorifics (합니다/습니다) → neutral professional (will/shall)
- Use declarative statements for procedures, conditional for contingencies
- Maintain neutral, objective tone without cultural hierarchical markers
- Authority: Use declarative statements for procedures, conditional for contingencies

**SENTENCE STRUCTURE TRANSFORMATION:**
- Break long Korean sentences (>20 words) into 2-3 shorter English sentences
- Maximum 25 words per English sentence for regulatory clarity
- Maintain logical flow and causal relationships
- Use active voice for procedures ("The investigator will assess...")
- Use passive voice for results ("Efficacy will be evaluated...")
- Requirements: Use modal verbs ("Subjects must provide...")

**TERMINOLOGY CONSISTENCY STANDARDS:**
- 임상시험 → Clinical Study (NOT Clinical Trial in protocol context)
- 임상시험용 의약품 → Investigational Product (NOT test drug)
- 시험대상자 → Study Subject (NOT patient in trial context)
- 이상반응 → Adverse Event (NOT side effect)
- 중대한 이상반응 → Serious Adverse Event
- 동의서 → Informed Consent (NOT consent form)

**CONTEXT-DEPENDENT TERMINOLOGY:**
- 환자 → Study Subject (trial context), Patient (medical context)
- 치료 → Intervention (trial context), Treatment (medical context)
- 효과 → Efficacy/Effectiveness (trial context), Effect (general context)

**REGULATORY COMPLIANCE LANGUAGE:**
- Framework: "This study will be conducted in accordance with the Declaration of Helsinki, ICH-GCP, and all applicable national regulations"
- Safety Priority: "The safety and well-being of study subjects is the highest priority"
- Informed Consent: "All study subjects must provide written informed consent before participation"
- Risk Assessment: "The risk-benefit ratio has been assessed and documented in the protocol"
- Monitoring: "Continuous safety monitoring will ensure subject protection throughout the study"

**CULTURAL ADAPTATION PATTERNS:**
- Neutralize Korean hierarchical language patterns (습니다 → will/shall)
- Maintain professional authority without cultural markers
- Use direct requirement statements (must/shall) for obligations
- Transform indirect obligation expressions to direct requirements

**ABBREVIATION STANDARDS:**
- First Use: Always spell out with abbreviation in parentheses
- Subsequent Use: Abbreviation only within the same section
- Cross-References: Spell out when referring across major sections
- Example: "The Investigational Product (IP) will be administered... Later, the IP dosing schedule..." """
    
    def _get_custom_style_guide(self) -> str:
        """Custom style guide based on user configuration"""
        if self.variants.get(StyleGuideVariant.CUSTOM) and self.variants[StyleGuideVariant.CUSTOM].custom_rules:
            return self._build_custom_style_guide(self.variants[StyleGuideVariant.CUSTOM].custom_rules)
        return self._get_standard_style_guide()
    
    def _get_en_ko_clinical_protocol_style_guide(self) -> str:
        """EN-KO Clinical Protocol style guide (~250 tokens)"""
        return """\n## EN→KO Clinical Protocol Style Guide

**TERMINOLOGY CONSISTENCY:**
- Clinical Study Protocol → 임상시험계획서
- Phase 1/2/3 → 제1상/제2상/제3상
- Open-label → 공개 라벨  
- Dose Escalation → 용량 증량
- Multicenter → 다기관
- Safety → 안전성
- Pharmacokinetics → 약동학
- Acute Myeloid Leukemia → 급성 골수성 백혈병

**BILINGUAL FORMAT:**
- Medical conditions: Korean(English, ABBREV) → 급성 골수성 백혈병(Acute Myeloid Leukemia, AML)
- Technical terms: Korean(English) → 최대 내약 용량(maximum tolerated dose)
- Drug/protocol codes: Keep unchanged

**FORMAL REGISTER (Natural Korean Flow):**
- Statements: ~다/~된다 endings
- Procedures: ~실시된다/~수행된다  
- Requirements: ~해야 한다
- Definitions: ~으로 정의된다

**SENTENCE STRUCTURE:**
- Adapt English SVO to Korean SOV naturally
- Break long compound sentences for Korean flow
- Use passive voice for procedural language
- Move time expressions to sentence beginning"""
    
    def _build_custom_style_guide(self, custom_rules: Dict) -> str:
        """Build custom style guide from user rules"""
        style_guide = "\n## Custom Clinical Protocol Style Guide\n"
        
        for section, rules in custom_rules.items():
            style_guide += f"\n**{section.upper()}:**\n"
            if isinstance(rules, list):
                for rule in rules:
                    style_guide += f"- {rule}\n"
            elif isinstance(rules, dict):
                for key, value in rules.items():
                    style_guide += f"- {key}: {value}\n"
            else:
                style_guide += f"- {rules}\n"
        
        return style_guide
    
    def _get_en_ko_clinical_protocol_strict_style_guide(self) -> str:
        """EN-KO Strict Literal Translation style guide (~250 tokens)"""
        return """\n## 🔒 EN→KO Strict Literal Translation Guide

**CORE PRINCIPLE: DIRECT TRANSLATION ONLY**
- 직역 최우선: 원문의 의미만 정확히 전달
- 정보 추가 절대 금지: 원문에 없는 내용 추가 불허
- 주관적 해석 금지: 평가나 판단 표현 사용 불가

**MANDATORY REGULATORY TERMS:**
- Title Page → 제목페이지 (NOT 표지)
- Sponsor Representative → 의뢰자 대표자 (NOT 의뢰자)
- Clinical Study Protocol → 임상시험계획서
- Informed Consent → 동의서
- Adverse Event → 이상반응
- Investigational Product → 임상시험용 의약품

**TRANSLATION APPROACH:**
- 보수적 번역: 영문본과 한글본 대조 심사 고려
- 표준화된 용어: 식약처 임상시험 용어집 기준
- 격식있는 문체: 합니다체 사용
- 객관적 표현: "적정함", "우수함" 등 주관적 표현 금지

**STRUCTURE:**
- 어순 조정: 영어 SVO → 한국어 SOV
- 문법적 조정만 허용: 조사, 어미 등
- 문장 분할: 긴 문장은 자연스럽게 분리"""
    
    def _get_ko_en_regulatory_compliance_style_guide(self) -> str:
        """KO-EN Regulatory Compliance with Anti-Hallucination style guide (~300 tokens)"""
        return """\n## 🚨 KO→EN Regulatory Compliance Guide

**CRITICAL ANTI-HALLUCINATION RULES:**
- TRANSLATE ONLY SOURCE CONTENT: Never add degrees, titles, or info not in Korean
- NO ASSUMPTIONS: "교수" = "Professor" ONLY (not "MD, PhD, Professor")
- EXACT INFORMATION PARITY: English must match Korean exactly
- NO ELABORATION: Direct, literal translation required

**MANDATORY TERMS:**
- 임상시험 → clinical study (NOT clinical trial in protocol)
- 교수 → Professor (NEVER add MD/PhD unless stated)
- 이상반응 → adverse event (NOT side effect)
- 시험대상자 → study subject (NOT patient in trial context)
- 의뢰자 대표자 → sponsor representative

**CONCISENESS REQUIREMENTS:**
- Minimal word count: Use only necessary words
- No redundant phrases: Avoid "as mentioned above", "it should be noted"
- Direct statements: Professional, regulatory tone
- Abbreviation consistency: Introduce once, use consistently

**REGULATORY WRITING STYLE:**
- ICH-GCP compliant terminology
- Professional, not technical writing tone
- Objective, declarative statements
- No mid-sentence unnecessary capitalization

**QUALITY CONTROL:**
- Every added word must exist in Korean source
- Every English sentence must have Korean equivalent
- Prevent over-explanation or interpretation"""

    def _get_ko_en_enhanced_with_examples(self) -> str:
        """KO-EN Enhanced with Generalizable Style Guide + Few-Shot Examples (~900 tokens)"""
        return """\n## 🔒 KO→EN Clinical Protocol Translation Guide (Enhanced)

### PART 1: GENERALIZABLE STYLE & TERMINOLOGY

**TONE & REGISTER (extracted from professional translations):**
- Formal professional register WITHOUT Korean honorifics
  ❌ "합니다/습니다" → ✅ "will/shall/must"
- Objective, declarative statements (regulatory tone)
- No subjective evaluation (avoid "appropriate", "satisfactory")
- ICH-GCP compliant professional language

**SENTENCE TRANSFORMATION PATTERNS:**
- Korean long sentences (20+ words) → 2-3 English sentences (max 25 words each)
- Maintain causal/logical relationships when splitting
- Active voice: procedures ("The investigator will assess...")
- Passive voice: results ("Safety will be evaluated...")
- Modal verbs: requirements ("Subjects must provide...")

**TERMINOLOGY CONSISTENCY (mandatory terms):**
- 임상시험계획서 → Clinical Study Protocol (NOT Clinical Trial Protocol)
- 의뢰자 대리인 → Sponsor Representative (NOT Sponsor Agent/Delegate)
- 이상반응 → Adverse Event (NOT Side Effect)
- 시험대상자 → Study Subject (NOT Patient in trial context)
- 동의서 → Informed Consent (NOT Consent Form)
- 임상시험용 의약품 → Investigational Product (NOT Test Drug)

**REGULATORY COMPLIANCE LANGUAGE:**
- Framework phrase: "in accordance with [Declaration of Helsinki/ICH-GCP]"
- Safety priority: "safety and well-being of study subjects"
- Consent requirement: "written informed consent"
- NO title/degree additions: 교수 = "Professor" ONLY (never add MD/PhD)

**ABBREVIATION HANDLING:**
- First mention: Spell out with abbreviation in parentheses
- Subsequent: Use abbreviation only

---

### PART 2: FEW-SHOT LEARNING EXAMPLES

**Example 1 - Tag Preservation:**
KO: [임상시험 계획서 개요]
EN: [Protocol Synopsis]
✓ Tags unchanged, direct terminology

**Example 2 - Regulatory Compliance Statement:**
KO: 본 임상시험 계획서에 포함된 모든 정보는 임상시험책임자 및 임상시험 담당자, 임상시험심사위원회, 규제기관을 위해 제공된 것으로서, 의뢰자의 사전 서면 동의 없이 제3자에게 공개될 수 없습니다.
EN: All information contained in this protocol is intended to be provided to the principal investigators and sub-investigators, Institutional Review Board, and regulatory authorities and shall not be disclosed to any third party without prior written consent of the sponsor.
✓ Formal tone, regulatory terminology, structured enumeration, no honorifics

**Example 3 - Complex Protocol Title:**
KO: 비미란성 위식도역류질환 환자에서 DWP14012의 유효성 및 안전성을 평가하기 위한 다기관, 이중눈가림, 무작위배정, 위약대조, 평행군, 3상, 치료적 확증 임상시험
EN: A multi-center, double-blind, randomized, placebo-controlled, parallel-group, phase 3, therapeutic confirmatory clinical trial to evaluate the efficacy and safety of DWP14012 in patients with non-erosive gastroesophageal reflux disease
✓ Technical accuracy, hyphenated compound adjectives, disease name precision

**Example 4 - ICH-GCP Commitment:**
KO: 본인은 본 임상시험을 헬싱키 선언, International council for harmonisation of technical requirements for pharmaceuticals for human use-good clinical practice (ICH-GCP) 및 적용되는 모든 해당 국가의 관련규정에 따라 진행할 것입니다.
EN: I will conduct this study in accordance with the Declaration of Helsinki, International Council for Harmonization of Technical Requirements for Pharmaceuticals for Human Use-Good Clinical Practice (ICH-GCP), and all applicable national regulations.
✓ Compliance framework, abbreviation introduction, professional commitment tone

**Example 5 - Professional Attestation:**
KO: 본인은 본 임상시험 계획서를 읽고 검토하였고, 본 임상시험 계획서가 임상시험을 진행하는 데에 있어 필요한 모든 정보를 포함하고 있음을 이해하였으며 이에 동의합니다.
EN: I have read and reviewed this protocol, and I understand and agree that it contains all necessary information to conduct this study.
✓ Concise, direct, professional (no over-explanation)

---

**CRITICAL ANTI-HALLUCINATION RULE:**
교수 = "Professor" ONLY. NEVER add "MD", "PhD", "Dr." unless explicitly stated in Korean."""

    def _get_en_ko_enhanced_with_examples(self) -> str:
        """EN-KO Enhanced with Generalizable Style Guide + Few-Shot Examples (~900 tokens)"""
        return """\n## 🔒 EN→KO Strict Literal Clinical Protocol Translation Guide (Enhanced)

### PART 1: GENERALIZABLE STYLE & TERMINOLOGY

**TONE & REGISTER (extracted from professional translations):**
- 격식있는 합니다체 (formal -합니다 style)
- 객관적 서술: 주관적 평가 표현 금지 (적정함, 우수함 등)
- 자연스러운 한국어 어순 (SOV)
- 의학/규제 전문 용어 사용

**SENTENCE TRANSFORMATION PATTERNS:**
- English SVO → Korean SOV 자연스럽게 전환
- 시간 표현: 문장 앞으로 이동
- 긴 영어 복합문: 한국어 2-3문장으로 자연스럽게 분할
- 수동태 선호: 절차적 언어에서 (~실시된다, ~수행된다)
- 문장 종결: 진술문(~다/~된다), 절차(~실시된다), 요구사항(~해야 한다)

**MANDATORY REGULATORY TERMS (필수 규제 용어):**
- Title Page → 제목페이지 (NOT 표지)
- Sponsor Representative → 의뢰자 대리인 (NOT 의뢰자 대표자)
- Clinical Study Protocol → 임상시험계획서
- Informed Consent → 동의서
- Adverse Event → 이상반응
- Investigational Product → 임상시험용 의약품
- Phase 1/2/3 → 제1상/제2상/제3상

**BILINGUAL TERMINOLOGY FORMAT:**
- Medical conditions: Korean(English, ABBREV)
  Example: 급성 골수성 백혈병(Acute Myeloid Leukemia, AML)
- Technical terms: Korean(English)
  Example: 최대 내약 용량(maximum tolerated dose)
- Drug/protocol codes: KEEP UNCHANGED
  Example: ZE46-0134 → ZE46-0134

**TRANSLATION APPROACH:**
- 직역 최우선: 원문 의미만 정확히 전달
- 정보 추가 절대 금지: 원문에 없는 내용 불허
- 보수적 번역: 영문본-한글본 대조 심사 고려
- 식약처 임상시험 용어집 기준

---

### PART 2: FEW-SHOT LEARNING EXAMPLES

**Example 1 - Phase 1 Protocol Title with Bilingual Format:**
EN: A Phase 1, Open-label, Dose Escalation and Dose Expansion, Multicenter Clinical Trial to Evaluate the Safety, Pharmacokinetics, Pharmacodynamics, and Preliminary Efficacy of ZE46-0134 in Adults with FLT3 mutated Relapsed or Refractory Acute Myeloid Leukemia (AML)
KO: FLT3 돌연변이 재발성 또는 불응성 급성 골수성 백혈병(Acute Myeloid Leukemia, AML) 성인 환자를 대상으로 ZE46-0134의 안전성, 약동학, 약력학 및 예비 유효성을 평가하기 위한 제1상, 공개 라벨, 용량 증량 및 용량 확장, 다기관 임상시험
✓ Bilingual medical term format, natural SOV order, technical accuracy, drug code unchanged

**Example 2 - Sponsor Information:**
EN: Lomond Therapeutics AU Pty Ltd (A subsidiary of Lomond Therapeutics, LLC)
KO: Lomond Therapeutics AU Pty Ltd (Lomond Therapeutics, LLC의 자회사)
✓ Company names unchanged, natural Korean possessive structure

**Example 3 - Signature Block (Mandatory Term):**
EN: Signature of Sponsor Representative
KO: 의뢰자 대리인의 서명
✓ MUST use 대리인 (NOT 대표자), natural possessive form

**Example 4 - Formal Attestation:**
EN: By my signature, I confirm that I have reviewed this protocol and find its content to be acceptable.
KO: 본인은 서명을 통해 본 임상시험계획서를 검토했으며 그 내용이 수용 가능함을 확인합니다.
✓ Formal 합니다체, time expression to beginning, natural flow

**Example 5 - Printed Name Format:**
EN: Printed Name of Sponsor Representative
KO: 의뢰자 대리인 이름(정자체)
✓ Mandatory term + bilingual clarification format

---

**CRITICAL RULES:**
- 원문에 없는 정보 추가 절대 금지
- 태그 보존: 모든 <태그>와 [메타데이터] 정확히 유지
- 의뢰자 대리인 (NOT 대표자) - 필수 용어"""

    def set_variant(self, variant: StyleGuideVariant) -> None:
        """Set the current style guide variant"""
        if variant in self.variants and self.variants[variant].enabled:
            self.current_variant = variant
            print(f"✅ Style guide variant set to: {self.variants[variant].name}")
        else:
            print(f"❌ Style guide variant '{variant.value}' not available or disabled")
    
    def enable_experiment_mode(self, variants: List[StyleGuideVariant]) -> None:
        """Enable A/B testing mode with specified variants"""
        self.experiment_mode = True
        self.experiment_variants = [v for v in variants if v in self.variants and self.variants[v].enabled]
        print(f"🧪 Experiment mode enabled with variants: {[v.value for v in self.experiment_variants]}")
    
    def get_experiment_variant(self, segment_id: int) -> StyleGuideVariant:
        """Get style guide variant for A/B testing (round-robin)"""
        if not self.experiment_mode:
            return self.current_variant
        
        variant_index = segment_id % len(self.experiment_variants)
        return self.experiment_variants[variant_index]
    
    def record_experiment_result(self, variant: StyleGuideVariant, segment_id: int, 
                               quality_score: float, token_count: int, processing_time: float) -> None:
        """Record experiment results for analysis"""
        if variant not in self.experiment_results:
            self.experiment_results[variant] = []
        
        self.experiment_results[variant].append({
            'segment_id': segment_id,
            'quality_score': quality_score,
            'token_count': token_count,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_experiment_summary(self) -> Dict:
        """Get summary of experiment results"""
        if not self.experiment_mode:
            return {}
        
        summary = {}
        for variant, results in self.experiment_results.items():
            if results:
                avg_quality = sum(r['quality_score'] for r in results) / len(results)
                avg_tokens = sum(r['token_count'] for r in results) / len(results)
                avg_time = sum(r['processing_time'] for r in results) / len(results)
                
                summary[variant.value] = {
                    'name': self.variants[variant].name,
                    'segments_translated': len(results),
                    'average_quality_score': avg_quality,
                    'average_token_count': avg_tokens,
                    'average_processing_time': avg_time,
                    'token_efficiency': self.variants[variant].token_efficiency
                }
        
        return summary
    
    def save_experiment_results(self, filename: str = None) -> None:
        """Save experiment results to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"style_guide_experiment_{timestamp}.json"
        
        results = {
            'experiment_config': {
                'variants_tested': [v.value for v in self.experiment_variants],
                'total_segments': sum(len(r) for r in self.experiment_results.values())
            },
            'variant_results': self.get_experiment_summary(),
            'detailed_results': self.experiment_results
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Experiment results saved to: {filename}")
    
    def get_available_variants(self) -> List[StyleGuideVariant]:
        """Get list of available and enabled variants"""
        return [v for v, config in self.variants.items() if config.enabled]
    
    def print_variant_info(self) -> None:
        """Print information about available variants"""
        print("\n🎨 Available Style Guide Variants:")
        print("-" * 80)
        for variant, config in self.variants.items():
            if config.enabled:
                status = "✓" if variant == self.current_variant else " "
                print(f"{status} {variant.value:15} | {config.name:25} | "
                      f"Tokens: {config.estimated_tokens:4} | "
                      f"Quality: {config.quality_score:.1f} | "
                      f"Efficiency: {config.token_efficiency:.2f}")
        print("-" * 80)


# Example usage and testing
if __name__ == "__main__":
    # Initialize style guide manager
    manager = StyleGuideManager()
    
    # Show available variants
    manager.print_variant_info()
    
    # Test different variants
    print("\n🧪 Testing Style Guide Variants:")
    for variant in [StyleGuideVariant.NONE, StyleGuideVariant.MINIMAL, StyleGuideVariant.STANDARD, StyleGuideVariant.CLINICAL_PROTOCOL]:
        style_guide = manager.get_style_guide(variant)
        token_count = len(style_guide) // 4
        print(f"\n{variant.value.upper()} ({token_count} tokens):")
        print(style_guide[:200] + "..." if len(style_guide) > 200 else style_guide)
    
    # Enable experiment mode
    manager.enable_experiment_mode([StyleGuideVariant.NONE, StyleGuideVariant.STANDARD])
    
    # Simulate experiment
    for i in range(5):
        variant = manager.get_experiment_variant(i)
        style_guide = manager.get_style_guide(variant)
        token_count = len(style_guide) // 4
        quality_score = 0.5 + (0.5 if variant != StyleGuideVariant.NONE else 0.0)
        
        manager.record_experiment_result(variant, i, quality_score, token_count, 1.0)
        print(f"Segment {i}: {variant.value} → Quality: {quality_score:.2f}, Tokens: {token_count}")
    
    # Show results
    print("\n📊 Experiment Summary:")
    summary = manager.get_experiment_summary()
    for variant, data in summary.items():
        print(f"{variant}: {data['segments_translated']} segments, "
              f"Avg Quality: {data['average_quality_score']:.2f}, "
              f"Avg Tokens: {data['average_token_count']}")
    
    # Save results
    manager.save_experiment_results()
