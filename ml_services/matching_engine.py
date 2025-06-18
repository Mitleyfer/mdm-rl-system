import logging
import asyncio
import jellyfish
import phonetics

import numpy as np
import polars as pl

from collections import defaultdict

from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class MatchingEngine:
    """
    High-performance matching engine for MDM
    Implements various matching algorithms with rule-based configuration
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.executor = ThreadPoolExecutor(max_workers=self.config['max_workers'])

    def _default_config(self) -> Dict:
        return {
            'max_workers': 4,
            'batch_size': 1000,
            'similarity_metrics': ['exact', 'fuzzy', 'phonetic', 'semantic'],
            'default_thresholds': {
                'name': 0.85,
                'address': 0.80,
                'phone': 0.95,
                'email': 0.98
            }
        }

    async def match_dataset(self, data: Dict, rules: Dict) -> Dict:
        """
        Perform matching on entire dataset using provided rules
        """
        logger.info(f"Starting matching on {len(data['records'])} records")
        start_time = datetime.now()

        records = data['records']

        blocks = self._create_blocks(records, rules)

        all_matches = []
        total_comparisons = 0

        matching_tasks = []
        for block_key, block_records in blocks.items():
            if len(block_records) > 1:
                task = self._match_block(block_records, rules)
                matching_tasks.append(task)

        block_results = await asyncio.gather(*matching_tasks)

        for matches, comparisons in block_results:
            all_matches.extend(matches)
            total_comparisons += comparisons

        unique_matches = self._deduplicate_matches(all_matches)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        metrics = {
            'total_records': len(records),
            'total_matches': len(unique_matches),
            'total_comparisons': total_comparisons,
            'processing_time': processing_time,
            'comparisons_per_second': total_comparisons / processing_time if processing_time > 0 else 0
        }

        if 'ground_truth' in data:
            accuracy_metrics = self._calculate_accuracy_metrics(
                unique_matches,
                data['ground_truth']
            )
            metrics.update(accuracy_metrics)

        return {
            'matches': unique_matches,
            'metrics': metrics,
            'rules_used': rules
        }

    def _create_blocks(self, records: List[Dict], rules: Dict) -> Dict[str, List[Tuple[int, Dict]]]:
        """Create blocks based on blocking strategy"""
        blocks = defaultdict(list)
        blocking_key = rules.get('blocking_key', 'sorted_neighborhood')

        for idx, record in enumerate(records):
            if blocking_key == 'exact':
                block_field = rules.get('block_field', 'zip')
                key = record.get(block_field, 'MISSING')
                blocks[key].append((idx, record))

            elif blocking_key == 'soundex':
                last_name = record.get('last_name', '')
                if last_name:
                    key = phonetics.soundex(last_name)
                    blocks[key].append((idx, record))
                else:
                    blocks['MISSING'].append((idx, record))

            elif blocking_key == 'first_letter':
                last_name = record.get('last_name', '')
                if last_name:
                    key = last_name[0].upper()
                    blocks[key].append((idx, record))
                else:
                    blocks['MISSING'].append((idx, record))

            elif blocking_key == 'sorted_neighborhood':
                last_name = record.get('last_name', '')
                if last_name:
                    blocks[f'LN_{last_name[:3].upper()}'].append((idx, record))

                street = record.get('street_name', '')
                if street:
                    blocks[f'ST_{street[:3].upper()}'].append((idx, record))

                zip_code = record.get('zip', '')
                if zip_code:
                    blocks[f'ZIP_{str(zip_code)[:3]}'].append((idx, record))

            else:
                blocks['ALL'].append((idx, record))

        logger.info(f"Created {len(blocks)} blocks")
        return blocks

    async def _match_block(self, block_records: List[Tuple[int, Dict]], rules: Dict) -> Tuple[List[Tuple[int, int]], int]:
        """Match records within a block"""
        matches = []
        comparisons = 0

        for i in range(len(block_records)):
            for j in range(i + 1, len(block_records)):
                idx1, record1 = block_records[i]
                idx2, record2 = block_records[j]

                similarity = self._calculate_similarity(record1, record2, rules)

                if self._is_match(similarity, rules):
                    matches.append((idx1, idx2))

                comparisons += 1

        return matches, comparisons

    def _calculate_similarity(self, record1: Dict, record2: Dict, rules: Dict) -> Dict[str, float]:
        """Calculate similarity scores between two records"""
        similarity = {}

        name_sim = self._name_similarity(record1, record2, rules)
        similarity['name'] = name_sim

        address_sim = self._address_similarity(record1, record2, rules)
        similarity['address'] = address_sim

        phone_sim = self._phone_similarity(record1, record2)
        similarity['phone'] = phone_sim

        email_sim = self._email_similarity(record1, record2)
        similarity['email'] = email_sim

        weights = {
            'name': rules.get('name_weight', 0.4),
            'address': rules.get('address_weight', 0.3),
            'phone': rules.get('phone_weight', 0.2),
            'email': rules.get('email_weight', 0.1)
        }

        overall = 0
        total_weight = 0

        for field, score in similarity.items():
            if score is not None:
                overall += score * weights.get(field, 0)
                total_weight += weights.get(field, 0)

        similarity['overall'] = overall / total_weight if total_weight > 0 else 0

        return similarity

    def _name_similarity(self, record1: Dict, record2: Dict, rules: Dict) -> Optional[float]:
        """Calculate name similarity"""
        first1 = str(record1.get('first_name', '')).upper().strip()
        first2 = str(record2.get('first_name', '')).upper().strip()
        last1 = str(record1.get('last_name', '')).upper().strip()
        last2 = str(record2.get('last_name', '')).upper().strip()

        if not (first1 or last1) or not (first2 or last2):
            return None

        scores = []

        if rules.get('exact_weight', 0.3) > 0:
            exact_score = 1.0 if (first1 == first2 and last1 == last2) else 0.0
            scores.append(('exact', exact_score, rules.get('exact_weight', 0.3)))

        if rules.get('fuzzy_weight', 0.7) > 0:
            if first1 and first2:
                first_jw = jellyfish.jaro_winkler_similarity(first1, first2)
            else:
                first_jw = 0.5

            if last1 and last2:
                last_jw = jellyfish.jaro_winkler_similarity(last1, last2)
            else:
                last_jw = 0.5

            fuzzy_score = (first_jw + last_jw) / 2
            scores.append(('fuzzy', fuzzy_score, rules.get('fuzzy_weight', 0.7)))

        if rules.get('enable_phonetic', True):
            if last1 and last2:
                sound1 = phonetics.soundex(last1)
                sound2 = phonetics.soundex(last2)
                phonetic_score = 1.0 if sound1 == sound2 else 0.0
            else:
                phonetic_score = 0.0

            scores.append(('phonetic', phonetic_score, 0.1))

        if first1 and first2 and last1 and last2:
            swap_score = jellyfish.jaro_winkler_similarity(first1, last2) * \
                         jellyfish.jaro_winkler_similarity(last1, first2)
            if swap_score > 0.8:
                scores.append(('swap', swap_score, 0.1))

        total_score = 0
        total_weight = 0

        for _, score, weight in scores:
            total_score += score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0

    def _address_similarity(self, record1: Dict, record2: Dict, rules: Dict) -> Optional[float]:
        """Calculate address similarity"""
        num1 = str(record1.get('address_number', '')).strip()
        num2 = str(record2.get('address_number', '')).strip()
        street1 = str(record1.get('street_name', '')).upper().strip()
        street2 = str(record2.get('street_name', '')).upper().strip()
        city1 = str(record1.get('city', '')).upper().strip()
        city2 = str(record2.get('city', '')).upper().strip()
        zip1 = str(record1.get('zip', '')).strip()
        zip2 = str(record2.get('zip', '')).strip()

        if not any([num1, street1, city1, zip1]) or not any([num2, street2, city2, zip2]):
            return None

        scores = []

        if num1 and num2:
            num_score = 1.0 if num1 == num2 else 0.0
            scores.append(('number', num_score, 0.3))

        if street1 and street2:
            street_score = jellyfish.jaro_winkler_similarity(street1, street2)

            if rules.get('enable_abbreviation', True):
                street1_expanded = self._expand_street_abbreviations(street1)
                street2_expanded = self._expand_street_abbreviations(street2)

                if street1_expanded != street1 or street2_expanded != street2:
                    expanded_score = jellyfish.jaro_winkler_similarity(
                        street1_expanded,
                        street2_expanded
                    )
                    street_score = max(street_score, expanded_score)

            scores.append(('street', street_score, 0.4))

        if city1 and city2:
            city_score = jellyfish.jaro_winkler_similarity(city1, city2)
            scores.append(('city', city_score, 0.2))

        if zip1 and zip2:
            zip1_5 = zip1[:5]
            zip2_5 = zip2[:5]
            zip_score = 1.0 if zip1_5 == zip2_5 else 0.0
            scores.append(('zip', zip_score, 0.1))

        total_score = 0
        total_weight = 0

        for _, score, weight in scores:
            total_score += score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0

    def _phone_similarity(self, record1: Dict, record2: Dict) -> Optional[float]:
        """Calculate phone similarity"""
        phone1 = str(record1.get('phone', '')).strip()
        phone2 = str(record2.get('phone', '')).strip()

        if not phone1 or not phone2:
            return None

        phone1_normalized = ''.join(c for c in phone1 if c.isdigit())
        phone2_normalized = ''.join(c for c in phone2 if c.isdigit())

        if len(phone1_normalized) >= 10 and len(phone2_normalized) >= 10:
            return 1.0 if phone1_normalized[-10:] == phone2_normalized[-10:] else 0.0
        else:
            return 1.0 if phone1_normalized == phone2_normalized else 0.0

    def _email_similarity(self, record1: Dict, record2: Dict) -> Optional[float]:
        """Calculate email similarity"""
        email1 = str(record1.get('email', '')).lower().strip()
        email2 = str(record2.get('email', '')).lower().strip()

        if not email1 or not email2:
            return None

        if email1 == email2:
            return 1.0

        if '@' in email1 and '@' in email2:
            local1, domain1 = email1.split('@', 1)
            local2, domain2 = email2.split('@', 1)

            if local1 == local2:
                return 0.8

            local_sim = jellyfish.jaro_winkler_similarity(local1, local2)
            if local_sim > 0.9:
                return local_sim * 0.8

        return 0.0

    def _expand_street_abbreviations(self, street: str) -> str:
        """Expand common street abbreviations"""
        abbreviations = {
            'ST': 'STREET',
            'AVE': 'AVENUE',
            'RD': 'ROAD',
            'BLVD': 'BOULEVARD',
            'DR': 'DRIVE',
            'LN': 'LANE',
            'CT': 'COURT',
            'PL': 'PLACE',
            'N': 'NORTH',
            'S': 'SOUTH',
            'E': 'EAST',
            'W': 'WEST'
        }

        words = street.split()
        expanded = []

        for word in words:
            if word in abbreviations:
                expanded.append(abbreviations[word])
            else:
                expanded.append(word)

        return ' '.join(expanded)

    def _is_match(self, similarity: Dict[str, float], rules: Dict) -> bool:
        """Determine if similarity scores indicate a match"""
        thresholds = {
            'name': rules.get('name_threshold', 0.85),
            'address': rules.get('address_threshold', 0.80),
            'phone': rules.get('phone_threshold', 0.95),
            'email': rules.get('email_threshold', 0.98)
        }

        fields_above_threshold = 0
        fields_compared = 0

        for field, threshold in thresholds.items():
            if similarity.get(field) is not None:
                fields_compared += 1
                if similarity[field] >= threshold:
                    fields_above_threshold += 1

        overall_threshold = rules.get('overall_threshold', 0.8)
        if similarity.get('overall', 0) >= overall_threshold:
            return True

        min_fields = rules.get('min_matching_fields', 2)
        if fields_above_threshold >= min_fields:
            return True

        if similarity.get('name', 0) >= 0.95 or similarity.get('email', 0) >= 0.99:
            return True

        return False

    def _deduplicate_matches(self, matches: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Remove duplicate matches and ensure consistency"""
        unique_matches = set()

        for i, j in matches:
            if i < j:
                unique_matches.add((i, j))
            else:
                unique_matches.add((j, i))

        return list(unique_matches)

    def _calculate_accuracy_metrics(self,
                                    predicted_matches: List[Tuple[int, int]],
                                    ground_truth: List[Tuple[int, int]]) -> Dict[str, float]:
        """Calculate precision, recall, and F1 score"""
        predicted_set = set(predicted_matches)
        truth_set = set(ground_truth)

        true_positives = len(predicted_set & truth_set)

        false_positives = len(predicted_set - truth_set)

        false_negatives = len(truth_set - predicted_set)

        precision = true_positives / (true_positives + false_positives) if predicted_set else 0
        recall = true_positives / (true_positives + false_negatives) if truth_set else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }

    def match_pair(self, record1: Dict, record2: Dict, rules: Dict) -> Tuple[bool, Dict[str, float]]:
        """Match a single pair of records"""
        similarity = self._calculate_similarity(record1, record2, rules)
        is_match = self._is_match(similarity, rules)
        return is_match, similarity

    def batch_match(self, record: Dict, candidates: List[Dict], rules: Dict) -> List[Tuple[int, Dict[str, float]]]:
        """Find all matches for a record from a list of candidates"""
        matches = []

        for idx, candidate in enumerate(candidates):
            is_match, similarity = self.match_pair(record, candidate, rules)
            if is_match:
                matches.append((idx, similarity))

        matches.sort(key=lambda x: x[1].get('overall', 0), reverse=True)

        return matches