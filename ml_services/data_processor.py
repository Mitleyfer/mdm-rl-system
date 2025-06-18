import re
import json
import hashlib
import logging
import phonetics

import numpy as np
import polars as pl

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple


logger = logging.getLogger(__name__)

class DataProcessor:
    """
    High-performance data processing pipeline using Polars
    Handles data loading, profiling, standardization, and feature extraction
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.profiling_cache = {}

    def _default_config(self) -> Dict:
        return {
            'chunk_size': 10000,
            'max_sample_size': 1000,
            'encoding': 'utf-8',
            'date_formats': ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y'],
            'phone_regex': r'[\d\s\-\(\)\+]+',
            'email_regex': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        }

    async def load_dataset(self, file_path: str, dataset_type: str) -> Dict:
        """Load dataset from file with automatic format detection"""
        logger.info(f"Loading dataset from {file_path}")

        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()

        try:
            if file_extension == '.csv':
                df = await self._load_csv(file_path)
            elif file_extension == '.json':
                df = await self._load_json(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                df = await self._load_excel(file_path)
            elif file_extension == '.parquet':
                df = pl.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

            profile = await self._profile_data(df)

            df = self._standardize_columns(df, dataset_type)

            records = df.to_dicts()

            return {
                'records': records,
                'profile': profile,
                'dataset_type': dataset_type,
                'columns': df.columns,
                'shape': df.shape
            }

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    async def _load_csv(self, file_path: Path) -> pl.DataFrame:
        """Load CSV with automatic delimiter detection"""
        with open(file_path, 'r', encoding=self.config['encoding']) as f:
            sample = f.read(1024)

        delimiters = [',', ';', '\t', '|']
        delimiter_counts = {d: sample.count(d) for d in delimiters}
        delimiter = max(delimiter_counts, key=delimiter_counts.get)

        df = pl.read_csv(
            file_path,
            separator=delimiter,
            encoding=self.config['encoding'],
            null_values=['', 'NULL', 'null', 'None', 'NaN', 'nan', 'N/A', 'n/a'],
            try_parse_dates=True
        )

        return df

    async def _load_json(self, file_path: Path) -> pl.DataFrame:
        """Load JSON file"""
        with open(file_path, 'r', encoding=self.config['encoding']) as f:
            data = json.load(f)

        if isinstance(data, list):
            df = pl.DataFrame(data)
        elif isinstance(data, dict):
            if all(isinstance(v, list) for v in data.values()):
                df = pl.DataFrame(data)
            else:
                df = pl.DataFrame([data])
        else:
            raise ValueError("Unsupported JSON structure")

        return df

    async def _load_excel(self, file_path: Path) -> pl.DataFrame:
        """Load Excel file"""
        import pandas as pd

        pd_df = pd.read_excel(file_path, engine='openpyxl')

        df = pl.from_pandas(pd_df)

        return df

    async def _profile_data(self, df: pl.DataFrame) -> Dict:
        """Profile dataset to understand characteristics"""
        profile = {
            'num_records': len(df),
            'num_columns': len(df.columns),
            'columns': {},
            'quality_metrics': {}
        }

        for col in df.columns:
            col_profile = {
                'dtype': str(df[col].dtype),
                'null_count': df[col].null_count(),
                'null_ratio': df[col].null_count() / len(df) if len(df) > 0 else 0,
                'unique_count': df[col].n_unique(),
                'unique_ratio': df[col].n_unique() / len(df) if len(df) > 0 else 0
            }

            if df[col].dtype in [pl.Int32, pl.Int64, pl.Float32, pl.Float64]:
                stats = df[col].describe()
                col_profile.update({
                    'mean': float(stats['mean'][0]) if stats['mean'][0] is not None else None,
                    'std': float(stats['std'][0]) if stats['std'][0] is not None else None,
                    'min': float(stats['min'][0]) if stats['min'][0] is not None else None,
                    'max': float(stats['max'][0]) if stats['max'][0] is not None else None
                })

            elif df[col].dtype == pl.Utf8:
                non_null = df[col].drop_nulls()
                if len(non_null) > 0:
                    col_profile['sample_values'] = non_null.head(5).to_list()

                    patterns = self._detect_patterns(non_null)
                    if patterns:
                        col_profile['detected_patterns'] = patterns

            profile['columns'][col] = col_profile

        total_cells = len(df) * len(df.columns)
        total_nulls = sum(col_prof['null_count'] for col_prof in profile['columns'].values())

        profile['quality_metrics'] = {
            'completeness': 1 - (total_nulls / total_cells) if total_cells > 0 else 0,
            'avg_null_ratio': total_nulls / total_cells if total_cells > 0 else 0,
            'columns_with_nulls': sum(1 for col_prof in profile['columns'].values() if col_prof['null_count'] > 0)
        }

        return profile

    def _detect_patterns(self, series: pl.Series) -> List[str]:
        """Detect common patterns in string data"""
        patterns = []
        sample = series.head(100).to_list()

        phone_matches = sum(1 for val in sample if val and re.match(self.config['phone_regex'], str(val)))
        if phone_matches > len(sample) * 0.7:
            patterns.append('phone')

        email_matches = sum(1 for val in sample if val and re.match(self.config['email_regex'], str(val)))
        if email_matches > len(sample) * 0.7:
            patterns.append('email')

        for date_format in self.config['date_formats']:
            date_matches = 0
            for val in sample:
                if val:
                    try:
                        datetime.strptime(str(val), date_format)
                        date_matches += 1
                    except:
                        pass
            if date_matches > len(sample) * 0.7:
                patterns.append(f'date:{date_format}')
                break

        return patterns

    def _standardize_columns(self, df: pl.DataFrame, dataset_type: str) -> pl.DataFrame:
        """Standardize column names based on dataset type"""
        column_mappings = {
            'customer': {
                'fname': 'first_name',
                'lname': 'last_name',
                'firstname': 'first_name',
                'lastname': 'last_name',
                'phone_number': 'phone',
                'telephone': 'phone',
                'email_address': 'email',
                'street': 'street_name',
                'streetname': 'street_name',
                'zip_code': 'zip',
                'postal_code': 'zip'
            },
            'product': {
                'product_name': 'name',
                'product_title': 'name',
                'description': 'description',
                'desc': 'description',
                'manufacturer': 'brand',
                'make': 'brand',
                'model_number': 'model',
                'sku': 'sku',
                'upc': 'upc'
            },
            'healthcare': {
                'provider_name': 'name',
                'npi': 'npi',
                'specialty': 'specialty',
                'practice_name': 'organization',
                'org_name': 'organization'
            }
        }

        mappings = column_mappings.get(dataset_type, {})

        rename_dict = {}
        for col in df.columns:
            col_lower = col.lower().replace(' ', '_')
            if col_lower in mappings:
                rename_dict[col] = mappings[col_lower]
            else:
                rename_dict[col] = col_lower

        return df.rename(rename_dict)

    async def extract_features(self, data: Dict) -> Dict:
        """Extract features for ML models"""
        logger.info("Extracting features from dataset")

        records = data['records']
        profile = data.get('profile', {})

        features = {
            'statistics': self._calculate_statistics(records, profile),
            'quality_issues': self._identify_quality_issues(records, profile),
            'key_attributes': self._identify_key_attributes(records, profile),
            'complexity_score': self._calculate_complexity_score(records, profile),
            'blocking_keys': self._suggest_blocking_keys(records, profile)
        }

        if 'ground_truth' in data:
            features['sample_pairs'] = self._create_sample_pairs(records, data['ground_truth'])
            features['labels'] = self._create_labels(features['sample_pairs'], data['ground_truth'])

        return features

    def _calculate_statistics(self, records: List[Dict], profile: Dict) -> Dict:
        """Calculate dataset statistics"""
        stats = {
            'num_records': len(records),
            'avg_completeness': profile.get('quality_metrics', {}).get('completeness', 0),
            'avg_fields_per_record': 0,
            'field_coverage': {}
        }

        if records:
            field_counts = [len([v for v in record.values() if v is not None]) for record in records]
            stats['avg_fields_per_record'] = np.mean(field_counts)

            all_fields = set()
            for record in records:
                all_fields.update(record.keys())

            for field in all_fields:
                coverage = sum(1 for r in records if r.get(field) is not None) / len(records)
                stats['field_coverage'][field] = coverage

            string_fields = ['first_name', 'last_name', 'street_name', 'name']
            for field in string_fields:
                if field in all_fields:
                    lengths = [len(str(r.get(field, ''))) for r in records if r.get(field)]
                    if lengths:
                        stats[f'avg_{field}_length'] = np.mean(lengths)

        sample_size = min(1000, len(records))
        if sample_size > 10:
            sample_indices = np.random.choice(len(records), sample_size, replace=False)
            duplicates = 0

            for i in range(sample_size):
                for j in range(i + 1, sample_size):
                    if self._records_similar(
                            records[sample_indices[i]],
                            records[sample_indices[j]]
                    ):
                        duplicates += 1

            stats['duplicate_ratio'] = duplicates / (sample_size * (sample_size - 1) / 2)
        else:
            stats['duplicate_ratio'] = 0

        stats['missing_ratio'] = profile.get('quality_metrics', {}).get('avg_null_ratio', 0)

        return stats

    def _records_similar(self, record1: Dict, record2: Dict) -> bool:
        """Quick similarity check for duplicate estimation"""
        key_fields = ['first_name', 'last_name', 'name', 'street_name']

        matches = 0
        compared = 0

        for field in key_fields:
            if field in record1 and field in record2:
                val1 = str(record1.get(field, '')).lower().strip()
                val2 = str(record2.get(field, '')).lower().strip()

                if val1 and val2:
                    compared += 1
                    if val1 == val2:
                        matches += 1

        return matches / compared > 0.8 if compared > 0 else False

    def _identify_quality_issues(self, records: List[Dict], profile: Dict) -> List[str]:
        """Identify data quality issues"""
        issues = []

        if profile.get('quality_metrics', {}).get('avg_null_ratio', 0) > 0.3:
            issues.append('high_missing_values')

        for col_name, col_profile in profile.get('columns', {}).items():
            if col_profile.get('unique_ratio', 0) > 0.9 and col_profile.get('dtype') == 'Utf8':
                issues.append(f'high_cardinality_{col_name}')

            patterns = col_profile.get('detected_patterns', [])
            if 'phone' in patterns and col_profile.get('null_ratio', 0) < 0.5:
                sample_values = col_profile.get('sample_values', [])
                formats = set()
                for val in sample_values:
                    if val:
                        if '-' in str(val):
                            formats.add('dash')
                        if '(' in str(val):
                            formats.add('parentheses')
                        if ' ' in str(val):
                            formats.add('space')

                if len(formats) > 1:
                    issues.append('inconsistent_phone_format')

        stats = self._calculate_statistics(records, profile)
        if stats.get('duplicate_ratio', 0) > 0.1:
            issues.append('high_duplicate_ratio')

        return issues

    def _identify_key_attributes(self, records: List[Dict], profile: Dict) -> List[str]:
        """Identify key attributes for matching"""
        key_attributes = []

        for col_name, col_profile in profile.get('columns', {}).items():
            if (col_profile.get('null_ratio', 1) < 0.2 and
                    col_profile.get('unique_ratio', 0) > 0.7):
                key_attributes.append(col_name)

        priority_fields = ['name', 'first_name', 'last_name', 'email', 'phone',
                           'npi', 'sku', 'upc']

        for field in priority_fields:
            if field in profile.get('columns', {}) and field not in key_attributes:
                if profile['columns'][field].get('null_ratio', 1) < 0.5:
                    key_attributes.append(field)

        return key_attributes[:10]

    def _calculate_complexity_score(self, records: List[Dict], profile: Dict) -> float:
        """Calculate dataset complexity score (0-1)"""
        complexity_factors = []

        missing_ratio = profile.get('quality_metrics', {}).get('avg_null_ratio', 0)
        complexity_factors.append(missing_ratio)

        high_cardinality_fields = sum(
            1 for col_profile in profile.get('columns', {}).values()
            if col_profile.get('unique_ratio', 0) > 0.9
        )
        cardinality_factor = high_cardinality_fields / len(profile.get('columns', {}))
        complexity_factors.append(cardinality_factor)

        issues = self._identify_quality_issues(records, profile)
        issue_factor = len(issues) / 10
        complexity_factors.append(min(1.0, issue_factor))

        if records:
            field_counts = [len([v for v in r.values() if v is not None]) for r in records[:100]]
            if field_counts:
                variability = np.std(field_counts) / np.mean(field_counts)
                complexity_factors.append(min(1.0, variability))

        return np.mean(complexity_factors) if complexity_factors else 0.5

    def _suggest_blocking_keys(self, records: List[Dict], profile: Dict) -> List[Dict]:
        """Suggest blocking keys for efficient matching"""
        suggestions = []

        for col_name, col_profile in profile.get('columns', {}).items():
            if (col_profile.get('null_ratio', 1) < 0.1 and
                    0.01 < col_profile.get('unique_ratio', 0) < 0.5):

                suggestions.append({
                    'field': col_name,
                    'method': 'exact',
                    'expected_blocks': int(col_profile.get('unique_count', 100))
                })

        name_fields = ['last_name', 'first_name', 'name']
        for field in name_fields:
            if field in profile.get('columns', {}):
                if profile['columns'][field].get('null_ratio', 1) < 0.2:
                    suggestions.append({
                        'field': field,
                        'method': 'soundex',
                        'expected_blocks': 500
                    })

        numeric_fields = ['zip', 'address_number']
        for field in numeric_fields:
            if field in profile.get('columns', {}):
                col_prof = profile['columns'][field]
                if (col_prof.get('dtype') in ['Int32', 'Int64', 'Float32', 'Float64'] and
                        col_prof.get('null_ratio', 1) < 0.3):
                    suggestions.append({
                        'field': field,
                        'method': 'sorted_neighborhood',
                        'window_size': 10
                    })

        return suggestions[:3]

    def _create_sample_pairs(self, records: List[Dict], ground_truth: List[Tuple[int, int]],
                             n_samples: int = 1000) -> List[Tuple[Dict, Dict]]:
        """Create sample pairs for training"""
        pairs = []

        for i, j in ground_truth[:n_samples // 2]:
            if i < len(records) and j < len(records):
                pairs.append((records[i], records[j]))

        n_negative = n_samples - len(pairs)
        positive_set = set(ground_truth)

        attempts = 0
        while len(pairs) < n_samples and attempts < n_samples * 10:
            i = np.random.randint(0, len(records))
            j = np.random.randint(0, len(records))

            if i != j and (i, j) not in positive_set and (j, i) not in positive_set:
                pairs.append((records[i], records[j]))

            attempts += 1

        return pairs

    def _create_labels(self, pairs: List[Tuple[Dict, Dict]],
                       ground_truth: List[Tuple[int, int]]) -> List[int]:
        """Create labels for sample pairs"""
        truth_hashes = set()

        for i, j in ground_truth:
            # Note: This is simplified - in production, record IDs needed
            truth_hashes.add((self._hash_record(i), self._hash_record(j)))
            truth_hashes.add((self._hash_record(j), self._hash_record(i)))

        labels = []
        for record1, record2 in pairs:
            hash_pair = (self._hash_record(record1), self._hash_record(record2))
            labels.append(1 if hash_pair in truth_hashes else 0)

        return labels

    def _hash_record(self, record: Any) -> str:
        """Create hash for record comparison"""
        if isinstance(record, dict):
            sorted_items = sorted(record.items())
            record_str = str(sorted_items)
        else:
            record_str = str(record)

        return hashlib.md5(record_str.encode()).hexdigest()

    def standardize_names(self, names: List[str]) -> List[str]:
        """Standardize name formats"""
        standardized = []

        for name in names:
            if name:
                name = name.upper()

                name = ' '.join(name.split())

                prefixes = ['MR', 'MRS', 'MS', 'DR', 'PROF']
                suffixes = ['JR', 'SR', 'II', 'III', 'IV', 'PHD', 'MD']

                parts = name.split()
                if parts and parts[0] in prefixes:
                    parts = parts[1:]
                if parts and parts[-1] in suffixes:
                    parts = parts[:-1]

                standardized.append(' '.join(parts))
            else:
                standardized.append('')

        return standardized

    def standardize_addresses(self, addresses: List[str]) -> List[str]:
        """Standardize address formats"""
        standardized = []

        abbreviations = {
            'STREET': 'ST',
            'AVENUE': 'AVE',
            'ROAD': 'RD',
            'BOULEVARD': 'BLVD',
            'DRIVE': 'DR',
            'LANE': 'LN',
            'COURT': 'CT',
            'PLACE': 'PL',
            'NORTH': 'N',
            'SOUTH': 'S',
            'EAST': 'E',
            'WEST': 'W'
        }

        for address in addresses:
            if address:
                address = address.upper()

                for full, abbr in abbreviations.items():
                    address = address.replace(f' {full} ', f' {abbr} ')
                    address = address.replace(f' {full}', f' {abbr}')

                address = ' '.join(address.split())

                standardized.append(address)
            else:
                standardized.append('')

        return standardized

    def create_blocking_key(self, record: Dict, method: str = 'soundex') -> Optional[str]:
        """Create blocking key for a record"""
        if method == 'soundex':
            last_name = record.get('last_name', '')
            if last_name:
                return phonetics.soundex(last_name)

        elif method == 'first_letter':
            last_name = record.get('last_name', '')
            zip_code = record.get('zip', '')

            if last_name:
                key = last_name[0].upper()
                if zip_code:
                    key += str(zip_code)[:3]
                return key

        elif method == 'sorted_neighborhood':
            last_name = record.get('last_name', '')
            street = record.get('street_name', '')

            key_parts = []
            if last_name:
                key_parts.append(last_name.upper())
            if street:
                key_parts.append(street.upper())

            return ' '.join(key_parts) if key_parts else None

        return None