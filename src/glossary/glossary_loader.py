#!/usr/bin/env python3
"""
Generic Glossary Loader for TransAI
Load and process glossary files from multiple sources with flexible column mapping
Supports Excel, JSON, and CSV formats with configurable source definitions
"""

import pandas as pd
import json
import os
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import logging
import yaml

class GlossaryLoader:
    """Generic glossary loader with flexible source configuration"""

    def __init__(self, config_path: Optional[str] = None, log_level: str = "INFO"):
        """
        Initialize glossary loader

        Args:
            config_path: Path to YAML/JSON configuration file
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.setup_logging(log_level)
        self.config = self._load_config(config_path) if config_path else {}
        self.sources = self.config.get('sources', [])

    def setup_logging(self, level: str = "INFO"):
        """Setup logging configuration"""
        logging.basicConfig(level=getattr(logging, level))
        self.logger = logging.getLogger(__name__)

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML or JSON file"""
        self.logger.info(f"📋 Loading configuration from: {config_path}")

        try:
            path = Path(config_path)

            if path.suffix.lower() in ['.yaml', '.yml']:
                with open(path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                self.logger.error(f"❌ Unsupported config format: {path.suffix}")
                return {}

            self.logger.info(f"✅ Configuration loaded with {len(config.get('sources', []))} sources")
            return config

        except Exception as e:
            self.logger.error(f"❌ Error loading config: {e}")
            return {}

    def load_glossary(self,
                     file_path: str,
                     file_type: str,
                     sheet_name: Optional[str] = None,
                     mapping: Optional[Dict[str, str]] = None,
                     alternatives_separator: str = "|",
                     confidence_score: float = 0.85,
                     source_name: str = None) -> List[Dict]:
        """
        Load glossary from a single file with flexible column mapping

        Args:
            file_path: Path to glossary file
            file_type: File type ('excel', 'json', 'csv')
            sheet_name: Sheet name for Excel files
            mapping: Column name mapping {'korean': 'col_name', 'english': 'col_name', ...}
            alternatives_separator: Character separating alternative terms
            confidence_score: Confidence score for terms (0.0-1.0)
            source_name: Name of glossary source

        Returns:
            List of glossary term dictionaries
        """
        file_path = str(file_path)
        source_name = source_name or Path(file_path).stem

        self.logger.info(f"📚 Loading glossary: {file_path}")
        self.logger.info(f"   Type: {file_type}, Source: {source_name}")

        try:
            if file_type.lower() == 'excel':
                return self._load_from_excel(
                    file_path, sheet_name, mapping, alternatives_separator,
                    confidence_score, source_name
                )
            elif file_type.lower() == 'json':
                return self._load_from_json(file_path, confidence_score, source_name)
            elif file_type.lower() == 'csv':
                return self._load_from_csv(
                    file_path, mapping, alternatives_separator,
                    confidence_score, source_name
                )
            else:
                self.logger.error(f"❌ Unsupported file type: {file_type}")
                return []

        except Exception as e:
            self.logger.error(f"❌ Error loading glossary: {e}")
            return []

    def _load_from_excel(self,
                        file_path: str,
                        sheet_name: Optional[str],
                        mapping: Optional[Dict],
                        alternatives_separator: str,
                        confidence_score: float,
                        source_name: str) -> List[Dict]:
        """Load glossary from Excel file"""
        try:
            xl_file = pd.ExcelFile(file_path)
            self.logger.info(f"   Sheets available: {xl_file.sheet_names}")

            # Use provided sheet name or first sheet
            target_sheet = sheet_name or xl_file.sheet_names[0]
            df = pd.read_excel(file_path, sheet_name=target_sheet)

            self.logger.info(f"   Sheet '{target_sheet}': {len(df)} rows")
            self.logger.info(f"   Columns: {list(df.columns)}")

            # Auto-detect columns if mapping not provided
            if not mapping:
                mapping = self._auto_detect_columns(df)
                self.logger.info(f"   Auto-detected mapping: {mapping}")

            return self._process_dataframe(
                df, mapping, alternatives_separator, confidence_score, source_name
            )

        except Exception as e:
            self.logger.error(f"❌ Error loading Excel file: {e}")
            return []

    def _load_from_json(self,
                       file_path: str,
                       confidence_score: float,
                       source_name: str) -> List[Dict]:
        """Load glossary from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            terms = []

            # Handle both direct array and metadata-wrapped formats
            if isinstance(data, list):
                terms_data = data
                metadata = {}
            else:
                terms_data = data.get('terms', [])
                metadata = data.get('glossary_metadata', {})

            self.logger.info(f"   Loaded {len(terms_data)} terms from JSON")

            for term in terms_data:
                if isinstance(term, dict):
                    # Normalize term structure
                    normalized_term = {
                        'korean': term.get('korean', ''),
                        'english': term.get('english', ''),
                        'source': source_name,
                        'confidence_score': confidence_score,
                        'category': term.get('category', ''),
                        'frequency': term.get('frequency', ''),
                        'context': term.get('context', '')
                    }

                    # Include alternatives if present
                    if 'alternatives' in term:
                        normalized_term['alternatives'] = term['alternatives']

                    if normalized_term['korean'] and normalized_term['english']:
                        terms.append(normalized_term)

            return terms

        except Exception as e:
            self.logger.error(f"❌ Error loading JSON file: {e}")
            return []

    def _load_from_csv(self,
                      file_path: str,
                      mapping: Optional[Dict],
                      alternatives_separator: str,
                      confidence_score: float,
                      source_name: str) -> List[Dict]:
        """Load glossary from CSV file"""
        try:
            df = pd.read_csv(file_path)

            self.logger.info(f"   CSV file: {len(df)} rows")
            self.logger.info(f"   Columns: {list(df.columns)}")

            # Auto-detect columns if mapping not provided
            if not mapping:
                mapping = self._auto_detect_columns(df)
                self.logger.info(f"   Auto-detected mapping: {mapping}")

            return self._process_dataframe(
                df, mapping, alternatives_separator, confidence_score, source_name
            )

        except Exception as e:
            self.logger.error(f"❌ Error loading CSV file: {e}")
            return []

    def _auto_detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Auto-detect Korean and English columns"""
        mapping = {}

        for col in df.columns:
            # Skip if already found
            if 'korean' in mapping and 'english' in mapping:
                break

            sample_text = str(df[col].iloc[0]) if len(df) > 0 else ""
            has_korean = any('\u3131' <= char <= '\uD79D' for char in sample_text)

            if has_korean and 'korean' not in mapping:
                mapping['korean'] = col
            elif not has_korean and 'english' not in mapping:
                mapping['english'] = col

        return mapping

    def _process_dataframe(self,
                          df: pd.DataFrame,
                          mapping: Dict[str, str],
                          alternatives_separator: str,
                          confidence_score: float,
                          source_name: str) -> List[Dict]:
        """Process DataFrame rows into glossary term objects"""
        terms = []
        korean_col = mapping.get('korean')
        english_col = mapping.get('english')
        category_col = mapping.get('category')
        frequency_col = mapping.get('frequency')
        context_col = mapping.get('context')

        if not korean_col or not english_col:
            self.logger.error(f"❌ Missing required columns in mapping")
            return []

        for idx, row in df.iterrows():
            korean = str(row.get(korean_col, '')).strip() if pd.notna(row.get(korean_col)) else ''
            english = str(row.get(english_col, '')).strip() if pd.notna(row.get(english_col)) else ''

            if not korean or not english:
                continue

            # Handle alternative terms separated by alternatives_separator
            ko_parts = [t.strip() for t in korean.split(alternatives_separator)]
            en_parts = [t.strip() for t in english.split(alternatives_separator)]

            term = {
                'korean': ko_parts[0],
                'english': en_parts[0],
                'source': source_name,
                'confidence_score': confidence_score
            }

            # Add optional fields
            if category_col and pd.notna(row.get(category_col)):
                term['category'] = str(row.get(category_col))

            if frequency_col and pd.notna(row.get(frequency_col)):
                term['frequency'] = str(row.get(frequency_col))

            if context_col and pd.notna(row.get(context_col)):
                term['context'] = str(row.get(context_col))

            # Include alternatives if present
            if len(ko_parts) > 1 or len(en_parts) > 1:
                term['alternatives'] = {
                    'korean': ko_parts[1:] if len(ko_parts) > 1 else [],
                    'english': en_parts[1:] if len(en_parts) > 1 else []
                }

            terms.append(term)

        self.logger.info(f"✅ Extracted {len(terms)} terms")
        return terms

    def load_all_glossaries(self) -> Tuple[List[Dict], Dict]:
        """Load all glossaries from configuration"""
        if not self.sources:
            self.logger.warning("⚠️ No glossary sources configured")
            return [], {}

        all_terms = []
        stats = {'sources': {}, 'total_terms': 0}

        for source_config in self.sources:
            if not source_config.get('enabled', True):
                self.logger.info(f"⏭️ Skipping disabled source: {source_config.get('name')}")
                continue

            try:
                source_name = source_config.get('name', 'unknown')
                file_path = source_config.get('path')
                file_type = source_config.get('type', 'excel')
                sheet_name = source_config.get('sheet')
                mapping = source_config.get('mapping')
                confidence_score = source_config.get('confidence_score', 0.85)
                alternatives_separator = source_config.get('alternatives_separator', '|')

                terms = self.load_glossary(
                    file_path=file_path,
                    file_type=file_type,
                    sheet_name=sheet_name,
                    mapping=mapping,
                    alternatives_separator=alternatives_separator,
                    confidence_score=confidence_score,
                    source_name=source_name
                )

                all_terms.extend(terms)
                stats['sources'][source_name] = len(terms)

            except Exception as e:
                self.logger.error(f"❌ Error loading source {source_config.get('name')}: {e}")

        stats['total_terms'] = len(all_terms)

        # Log summary
        self.logger.info(f"📊 Glossary loading summary:")
        for source_name, count in stats['sources'].items():
            self.logger.info(f"   {source_name}: {count} terms")
        self.logger.info(f"   Total: {stats['total_terms']} terms")

        return all_terms, stats

    def save_glossary(self,
                     terms: List[Dict],
                     output_file: str,
                     format: str = "json",
                     include_metadata: bool = True) -> None:
        """
        Save glossary to file

        Args:
            terms: List of glossary term dictionaries
            output_file: Path to output file
            format: Output format ('json', 'excel', 'csv')
            include_metadata: Include metadata in output
        """
        self.logger.info(f"💾 Saving {len(terms)} terms to: {output_file}")

        try:
            os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

            if format.lower() == 'json':
                self._save_json(terms, output_file, include_metadata)
            elif format.lower() == 'excel':
                self._save_excel(terms, output_file)
            elif format.lower() == 'csv':
                self._save_csv(terms, output_file)
            else:
                self.logger.error(f"❌ Unsupported format: {format}")

            self.logger.info(f"✅ Glossary saved successfully")

        except Exception as e:
            self.logger.error(f"❌ Error saving glossary: {e}")
            raise

    def _save_json(self, terms: List[Dict], output_file: str, include_metadata: bool) -> None:
        """Save glossary as JSON"""
        if include_metadata:
            data = {
                "glossary_metadata": {
                    "version": "1.0",
                    "language_pair": "ko-en",
                    "created_date": pd.Timestamp.now().isoformat(),
                    "total_terms": len(terms),
                    "sources": list(set(t.get('source') for t in terms))
                },
                "terms": terms
            }
        else:
            data = terms

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save_excel(self, terms: List[Dict], output_file: str) -> None:
        """Save glossary as Excel"""
        df = pd.DataFrame(terms)

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Main glossary sheet
            df.to_excel(writer, sheet_name='Combined_Glossary', index=False)

            # Sheet by source if multiple sources
            sources = df['source'].unique()
            for source in sources:
                source_df = df[df['source'] == source]
                sheet_name = str(source)[:31]  # Excel sheet name limit
                source_df.to_excel(writer, sheet_name=sheet_name, index=False)

    def _save_csv(self, terms: List[Dict], output_file: str) -> None:
        """Save glossary as CSV"""
        df = pd.DataFrame(terms)
        df.to_csv(output_file, index=False, encoding='utf-8')

def main():
    """Main function to load glossaries based on configuration"""
    import sys

    print("📚 Generic Glossary Loader")
    print("=" * 70)

    # Check for config file argument
    config_file = sys.argv[1] if len(sys.argv) > 1 else None

    if config_file:
        print(f"\n📋 Loading configuration from: {config_file}")
        loader = GlossaryLoader(config_path=config_file)
        all_terms, stats = loader.load_all_glossaries()
    else:
        print("\n⚠️ No configuration file provided")
        print("Usage: python glossary_loader.py <config.yaml|config.json>")
        print("\nExample usage:")
        print("  python glossary_loader.py glossary_config.yaml")
        return

    # Save production-ready glossary
    output_dir = "./data"
    output_file = f"{output_dir}/production_glossary.json"
    os.makedirs(output_dir, exist_ok=True)

    loader.save_glossary(
        terms=all_terms,
        output_file=output_file,
        format="json",
        include_metadata=True
    )

    # Also save as Excel
    excel_file = f"{output_dir}/production_glossary.xlsx"
    loader.save_glossary(
        terms=all_terms,
        output_file=excel_file,
        format="excel"
    )

    # Show sample terms
    print(f"\n📝 Sample Terms:")
    for i, term in enumerate(all_terms[:5]):
        alternatives = ""
        if term.get('alternatives'):
            alternatives = f" (alternatives: {term['alternatives']['english'][:1]})"
        print(f"   {i+1}. {term['korean']} → {term['english']}{alternatives}")

    # Show statistics
    print(f"\n📊 Glossary Summary:")
    for source, count in stats['sources'].items():
        print(f"   {source}: {count} terms")
    print(f"   Total: {stats['total_terms']} terms")

    print(f"\n✅ Glossary loaded and saved successfully!")
    print(f"   JSON: {output_file}")
    print(f"   Excel: {excel_file}")


if __name__ == "__main__":
    main()