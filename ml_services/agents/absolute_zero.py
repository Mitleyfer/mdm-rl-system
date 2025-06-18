import torch
import random
import logging
import asyncio

import numpy as np
import torch.nn as nn
import torch.optim as optim

from datetime import datetime
from collections import deque
from typing import Dict, List, Tuple, Optional, Any


logger = logging.getLogger(__name__)

class TaskGenerator(nn.Module):
    """Neural network for generating synthetic matching tasks"""

    def __init__(self, latent_dim: int = 64, output_dim: int = 128):
        super(TaskGenerator, self).__init__()
        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, output_dim)

        self.activation = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(256)

    def forward(self, z):
        x = self.activation(self.batch_norm1(self.fc1(z)))
        x = self.activation(self.batch_norm2(self.fc2(x)))
        x = self.activation(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

class SelfPlayEnvironment:
    """
    Environment for self-play learning in MDM context
    Generates synthetic data matching scenarios and verifies outcomes
    """

    def __init__(self, config: Dict):
        self.config = config
        self.task_complexity = config.get('initial_complexity', 0.1)
        self.curriculum_rate = config.get('curriculum_rate', 0.01)

        self.name_patterns = [
            "{first} {last}",
            "{first} {middle} {last}",
            "{last}, {first}",
            "{first} {last} {suffix}"
        ]

        self.address_patterns = [
            "{number} {street} {type}",
            "{number} {direction} {street} {type}",
            "{street} {number}",
            "{number} {street} {type}, {unit}"
        ]

        self.corruption_types = [
            'typo', 'abbreviation', 'missing', 'swap', 'duplicate'
        ]

    def generate_task(self) -> Dict:
        """Generate a synthetic matching task with known ground truth"""
        n_records = int(100 * (1 + self.task_complexity * 9))  # 100-1000 records
        n_duplicates = int(n_records * 0.1 * (1 + self.task_complexity))

        records = []
        ground_truth = []

        unique_entities = []
        for i in range(n_records - n_duplicates):
            entity = self._generate_entity(i)
            unique_entities.append(entity)
            records.append(entity)

        for i in range(n_duplicates):
            original_idx = random.randint(0, len(unique_entities) - 1)
            original = unique_entities[original_idx]

            duplicate = self._create_variation(original, self.task_complexity)
            records.append(duplicate)

            ground_truth.append((original_idx, len(records) - 1))

        shuffle_map = list(range(len(records)))
        random.shuffle(shuffle_map)
        shuffled_records = [records[i] for i in shuffle_map]

        inverse_map = {v: k for k, v in enumerate(shuffle_map)}
        adjusted_truth = [
            (inverse_map[pair[0]], inverse_map[pair[1]])
            for pair in ground_truth
        ]

        return {
            'records': shuffled_records,
            'ground_truth': adjusted_truth,
            'complexity': self.task_complexity,
            'metadata': {
                'n_records': n_records,
                'n_duplicates': n_duplicates,
                'corruption_rate': self.task_complexity
            }
        }

    def _generate_entity(self, entity_id: int) -> Dict:
        """Generate a synthetic entity record"""
        first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Emma', 'Robert', 'Lisa']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis']
        streets = ['Main', 'Oak', 'Elm', 'Park', 'Washington', 'Lake', 'Hill', 'Forest']

        record = {
            'id': f"ENT_{entity_id:06d}",
            'first_name': random.choice(first_names),
            'last_name': random.choice(last_names),
            'middle_name': random.choice(['A', 'B', 'C', 'D', '']) if random.random() > 0.5 else '',
            'address_number': random.randint(1, 9999),
            'street_name': random.choice(streets),
            'street_type': random.choice(['St', 'Ave', 'Rd', 'Blvd', 'Dr']),
            'city': random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston']),
            'state': random.choice(['NY', 'CA', 'IL', 'TX']),
            'zip': f"{random.randint(10000, 99999)}",
            'phone': f"{random.randint(200, 999)}-{random.randint(200, 999)}-{random.randint(1000, 9999)}",
            'email': None
        }

        if random.random() > 0.3:
            record['email'] = f"{record['first_name'].lower()}.{record['last_name'].lower()}@example.com"

        return record

    def _create_variation(self, original: Dict, complexity: float) -> Dict:
        """Create a variation of a record based on complexity"""
        variation = original.copy()

        n_corruptions = random.randint(1, max(1, int(5 * complexity)))

        for _ in range(n_corruptions):
            corruption_type = random.choice(self.corruption_types)
            field = random.choice(['first_name', 'last_name', 'address_number',
                                   'street_name', 'phone', 'email'])

            if corruption_type == 'typo' and field in ['first_name', 'last_name', 'street_name']:
                value = str(variation.get(field, ''))
                if len(value) > 2:
                    pos = random.randint(0, len(value) - 1)
                    value = value[:pos] + random.choice('abcdefghijklmnopqrstuvwxyz') + value[pos+1:]
                    variation[field] = value

            elif corruption_type == 'abbreviation' and field == 'street_type':
                abbrevs = {'Street': ['St', 'St.', 'Str'],
                           'Avenue': ['Ave', 'Ave.', 'Av'],
                           'Road': ['Rd', 'Rd.'],
                           'Boulevard': ['Blvd', 'Blvd.', 'Bl']}
                current = variation.get('street_type', 'St')
                for full, abbr_list in abbrevs.items():
                    if current in abbr_list:
                        variation['street_type'] = random.choice(abbr_list)
                        break

            elif corruption_type == 'missing':
                if field in variation and field not in ['first_name', 'last_name']:
                    variation[field] = None

            elif corruption_type == 'swap' and field == 'first_name':
                variation['first_name'], variation['last_name'] = \
                    variation.get('last_name', ''), variation.get('first_name', '')

        return variation

    def verify_matches(self, proposed_matches: List[Tuple[int, int]],
                       ground_truth: List[Tuple[int, int]]) -> Dict:
        """Verify matching results against ground truth"""
        proposed_set = set(proposed_matches)
        truth_set = set(ground_truth)

        true_positives = len(proposed_set & truth_set)
        false_positives = len(proposed_set - truth_set)
        false_negatives = len(truth_set - proposed_set)

        precision = true_positives / (true_positives + false_positives) if proposed_set else 0
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

    def update_complexity(self, performance: float):
        """Update task complexity based on performance (curriculum learning)"""
        if performance > 0.9:
            self.task_complexity = min(1.0, self.task_complexity + self.curriculum_rate)
        elif performance < 0.7:
            self.task_complexity = max(0.1, self.task_complexity - self.curriculum_rate * 0.5)

class AbsoluteZeroAgent:
    """
    Absolute Zero agent adapted for MDM self-play learning
    Learns without external training data through self-generated tasks
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.task_generator = TaskGenerator(
            latent_dim=self.config['latent_dim'],
            output_dim=128
        ).to(self.device)

        self.policy_network = self._build_policy_network().to(self.device)

        self.generator_optimizer = optim.Adam(
            self.task_generator.parameters(),
            lr=self.config['learning_rate']
        )
        self.policy_optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=self.config['learning_rate']
        )

        self.environment = SelfPlayEnvironment(self.config)

        self.experience_buffer = deque(maxlen=self.config['buffer_size'])

        self.performance_history = []
        self.generated_tasks = 0

    def _default_config(self) -> Dict:
        return {
            'latent_dim': 64,
            'learning_rate': 1e-4,
            'batch_size': 32,
            'buffer_size': 5000,
            'initial_complexity': 0.1,
            'curriculum_rate': 0.01,
            'episodes_per_task': 10,
            'self_play_iterations': 1000
        }

    def _build_policy_network(self) -> nn.Module:
        """Build policy network for matching decisions"""
        return nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    async def initialize(self):
        """Initialize the Absolute Zero agent"""
        logger.info("Initializing Absolute Zero Agent...")

        try:
            self.load_model("models/absolute_zero_checkpoint.pth")
            logger.info("Loaded pre-trained Absolute Zero model")
        except:
            logger.info("Starting Absolute Zero from scratch")

        await self._initial_self_play()

    async def shutdown(self):
        """Save model and experiences"""
        self.save_model("models/absolute_zero_checkpoint.pth")

    async def _initial_self_play(self):
        """Run initial self-play episodes"""
        logger.info("Running initial self-play...")

        for i in range(min(10, self.config['self_play_iterations'] // 100)):
            task = self.environment.generate_task()

            results = await self._solve_task(task)

            self.experience_buffer.append({
                'task': task,
                'results': results,
                'complexity': task['complexity']
            })

            self.environment.update_complexity(results['metrics']['f1_score'])

        logger.info(f"Initial self-play completed with {len(self.experience_buffer)} experiences")

    def _encode_record_pair(self, record1: Dict, record2: Dict) -> torch.Tensor:
        """Encode a pair of records for the policy network"""
        features = []

        for field in ['first_name', 'last_name', 'street_name']:
            val1 = str(record1.get(field, '')).lower()
            val2 = str(record2.get(field, '')).lower()

            features.append(1.0 if val1 == val2 else 0.0)

            if val1 and val2:
                max_len = max(len(val1), len(val2))
                lev_dist = self._levenshtein_distance(val1, val2)
                features.append(1.0 - lev_dist / max_len)
            else:
                features.append(0.0)

            min_len = min(len(val1), len(val2))
            if min_len > 0:
                prefix_match = sum(c1 == c2 for c1, c2 in zip(val1, val2)) / min_len
                features.append(prefix_match)
            else:
                features.append(0.0)

        for field in ['address_number', 'zip']:
            val1 = record1.get(field)
            val2 = record2.get(field)

            if val1 is not None and val2 is not None:
                try:
                    num1 = float(str(val1).replace('-', ''))
                    num2 = float(str(val2).replace('-', ''))
                    features.append(1.0 if num1 == num2 else 0.0)
                    features.append(1.0 / (1.0 + abs(num1 - num2)))
                except:
                    features.extend([0.0, 0.0])
            else:
                features.extend([0.0, 0.0])

        for field in ['first_name', 'last_name', 'phone', 'email']:
            features.append(
                1.0 if (record1.get(field) is None) == (record2.get(field) is None) else 0.0
            )

        while len(features) < 256:
            features.append(0.0)

        return torch.FloatTensor(features[:256])

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    async def _solve_task(self, task: Dict) -> Dict:
        """Attempt to solve a self-generated matching task"""
        records = task['records']
        ground_truth = task['ground_truth']

        proposed_matches = []

        with torch.no_grad():
            for i in range(len(records)):
                for j in range(i + 1, len(records)):
                    pair_features = self._encode_record_pair(records[i], records[j]).to(self.device)

                    match_prob = self.policy_network(pair_features.unsqueeze(0)).item()

                    if match_prob > 0.5:
                        proposed_matches.append((i, j))

        metrics = self.environment.verify_matches(proposed_matches, ground_truth)

        return {
            'proposed_matches': proposed_matches,
            'metrics': metrics,
            'n_comparisons': len(records) * (len(records) - 1) // 2
        }

    def _train_on_experience(self, experience: Dict) -> float:
        """Train policy network on a self-play experience"""
        task = experience['task']
        records = task['records']
        ground_truth = task['ground_truth']

        positive_pairs = ground_truth
        negative_pairs = []

        all_pairs = set()
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                if (i, j) not in ground_truth and (j, i) not in ground_truth:
                    all_pairs.add((i, j))

        if all_pairs:
            negative_pairs = random.sample(
                list(all_pairs),
                min(len(positive_pairs) * 2, len(all_pairs))
            )

        features = []
        labels = []

        for i, j in positive_pairs:
            features.append(self._encode_record_pair(records[i], records[j]))
            labels.append(1.0)

        for i, j in negative_pairs:
            features.append(self._encode_record_pair(records[i], records[j]))
            labels.append(0.0)

        if not features:
            return 0.0

        features = torch.stack(features).to(self.device)
        labels = torch.FloatTensor(labels).to(self.device)

        predictions = self.policy_network(features).squeeze()

        loss = nn.BCELoss()(predictions, labels)

        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.policy_optimizer.step()

        return loss.item()

    async def learn(self, data: Dict, features: Dict, matches: Dict) -> Dict:
        """
        Learn through self-play, optionally incorporating real data
        """
        logger.info("Absolute Zero Agent learning through self-play...")

        total_iterations = self.config['self_play_iterations']
        losses = []
        performances = []

        if 'records' in data and len(data['records']) > 0:
            logger.info("Incorporating real data into self-play...")

            real_task = {
                'records': data['records'],
                'ground_truth': data.get('ground_truth', []),
                'complexity': 0.5,
                'metadata': {'source': 'real_data'}
            }

            results = await self._solve_task(real_task)

            self.experience_buffer.append({
                'task': real_task,
                'results': results,
                'complexity': real_task['complexity']
            })

            self.environment.task_complexity = max(0.3, results['metrics']['f1_score'])

        for iteration in range(total_iterations):
            task = self.environment.generate_task()
            self.generated_tasks += 1

            results = await self._solve_task(task)

            self.experience_buffer.append({
                'task': task,
                'results': results,
                'complexity': task['complexity']
            })

            if len(self.experience_buffer) >= self.config['batch_size']:
                batch_losses = []

                batch = random.sample(
                    list(self.experience_buffer),
                    self.config['batch_size']
                )

                for exp in batch:
                    loss = self._train_on_experience(exp)
                    batch_losses.append(loss)

                avg_loss = np.mean(batch_losses)
                losses.append(avg_loss)

            performances.append(results['metrics']['f1_score'])

            self.environment.update_complexity(results['metrics']['f1_score'])

            if iteration % 100 == 0:
                recent_perf = np.mean(performances[-100:]) if performances else 0
                recent_loss = np.mean(losses[-100:]) if losses else 0
                logger.info(f"Iteration {iteration}: "
                            f"Performance: {recent_perf:.3f}, "
                            f"Loss: {recent_loss:.4f}, "
                            f"Complexity: {self.environment.task_complexity:.3f}")

        self.performance_history.extend(performances)

        return {
            'iterations': total_iterations,
            'avg_performance': np.mean(performances),
            'final_complexity': self.environment.task_complexity,
            'avg_loss': np.mean(losses) if losses else 0,
            'total_tasks_generated': self.generated_tasks
        }

    async def generate_rules(self, features: Dict) -> Dict:
        """
        Generate matching rules based on learned policy
        """
        logger.info("Generating rules from Absolute Zero learning...")

        test_scenarios = self._generate_test_scenarios()

        thresholds = {}
        weights = {}

        name_scores = []
        for scenario in test_scenarios['name_variations']:
            pair_features = self._encode_record_pair(scenario[0], scenario[1])
            with torch.no_grad():
                score = self.policy_network(pair_features.unsqueeze(0).to(self.device)).item()
            name_scores.append((scenario[2], score))

        name_scores.sort(key=lambda x: x[0])
        name_threshold = self._find_threshold(name_scores)
        thresholds['name_threshold'] = name_threshold

        address_scores = []
        for scenario in test_scenarios['address_variations']:
            pair_features = self._encode_record_pair(scenario[0], scenario[1])
            with torch.no_grad():
                score = self.policy_network(pair_features.unsqueeze(0).to(self.device)).item()
            address_scores.append((scenario[2], score))

        address_scores.sort(key=lambda x: x[0])
        thresholds['address_threshold'] = self._find_threshold(address_scores)

        feature_importance = await self._estimate_feature_importance()

        total_importance = sum(feature_importance.values())
        for feature, importance in feature_importance.items():
            weights[f"{feature}_weight"] = importance / total_importance

        rules = {
            **thresholds,
            **weights,
            'enable_phonetic': True,
            'enable_abbreviation': True,
            'blocking_key': 'sorted_neighborhood',
            'confidence_threshold': 0.5
        }

        if self.environment.task_complexity > 0.7:
            for key in thresholds:
                rules[key] *= 0.95

        logger.info(f"Generated rules from {self.generated_tasks} self-play tasks")
        return rules

    def _generate_test_scenarios(self) -> Dict[str, List]:
        """Generate test scenarios for rule extraction"""
        scenarios = {
            'name_variations': [],
            'address_variations': [],
            'phone_variations': []
        }

        base_record = {
            'first_name': 'John',
            'last_name': 'Smith',
            'address_number': '123',
            'street_name': 'Main',
            'street_type': 'St',
            'phone': '555-123-4567'
        }

        name_variations = [
            ('John Smith', 'John Smith', 1.0),
            ('John Smith', 'Jon Smith', 0.9),
            ('John Smith', 'John Smyth', 0.85),
            ('John Smith', 'J Smith', 0.8),
            ('John Smith', 'Jonathan Smith', 0.75),
            ('John Smith', 'Jack Smith', 0.6),
            ('John Smith', 'John Jones', 0.3),
            ('John Smith', 'Jane Smith', 0.4),
            ('John Smith', 'Bob Johnson', 0.1)
        ]

        for name1, name2, similarity in name_variations:
            record1 = base_record.copy()
            record2 = base_record.copy()

            parts1 = name1.split()
            parts2 = name2.split()

            record1['first_name'] = parts1[0]
            record1['last_name'] = parts1[1] if len(parts1) > 1 else ''
            record2['first_name'] = parts2[0]
            record2['last_name'] = parts2[1] if len(parts2) > 1 else ''

            scenarios['name_variations'].append((record1, record2, similarity))

        address_variations = [
            ('123 Main St', '123 Main St', 1.0),
            ('123 Main St', '123 Main Street', 0.95),
            ('123 Main St', '123 Main', 0.85),
            ('123 Main St', '123 Main Ave', 0.7),
            ('123 Main St', '124 Main St', 0.8),
            ('123 Main St', '321 Main St', 0.6),
            ('123 Main St', '456 Oak St', 0.2)
        ]

        for addr1, addr2, similarity in address_variations:
            record1 = base_record.copy()
            record2 = base_record.copy()

            parts1 = addr1.split()
            parts2 = addr2.split()

            if len(parts1) >= 2:
                record1['address_number'] = parts1[0]
                record1['street_name'] = parts1[1]
                record1['street_type'] = parts1[2] if len(parts1) > 2 else 'St'

            if len(parts2) >= 2:
                record2['address_number'] = parts2[0]
                record2['street_name'] = parts2[1]
                record2['street_type'] = parts2[2] if len(parts2) > 2 else 'St'

            scenarios['address_variations'].append((record1, record2, similarity))

        return scenarios

    def _find_threshold(self, scores: List[Tuple[float, float]]) -> float:
        """Find optimal threshold from similarity-score pairs"""
        for similarity, match_prob in scores:
            if match_prob > 0.5:
                return max(0.5, similarity * 0.95)

        return 0.85

    async def _estimate_feature_importance(self) -> Dict[str, float]:
        """Estimate feature importance through ablation study"""
        if len(self.experience_buffer) == 0:
            return {
                'name': 0.4,
                'address': 0.3,
                'phone': 0.2,
                'email': 0.1
            }

        experience = random.choice(list(self.experience_buffer))
        task = experience['task']
        records = task['records']
        ground_truth = task['ground_truth']

        if not ground_truth:
            return {
                'name': 0.4,
                'address': 0.3,
                'phone': 0.2,
                'email': 0.1
            }

        baseline_results = await self._solve_task(task)
        baseline_f1 = baseline_results['metrics']['f1_score']

        importance = {}

        feature_groups = {
            'name': ['first_name', 'last_name'],
            'address': ['address_number', 'street_name', 'street_type'],
            'phone': ['phone'],
            'email': ['email']
        }

        for group_name, fields in feature_groups.items():
            ablated_records = []
            for record in records:
                ablated = record.copy()
                for field in fields:
                    ablated[field] = None
                ablated_records.append(ablated)

            ablated_task = {
                'records': ablated_records,
                'ground_truth': ground_truth,
                'complexity': task['complexity']
            }

            ablated_results = await self._solve_task(ablated_task)
            ablated_f1 = ablated_results['metrics']['f1_score']

            importance[group_name] = max(0, baseline_f1 - ablated_f1)

        total = sum(importance.values())
        if total > 0:
            for key in importance:
                importance[key] /= total

        return importance

    def save_model(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'task_generator_state_dict': self.task_generator.state_dict(),
            'policy_network_state_dict': self.policy_network.state_dict(),
            'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'task_complexity': self.environment.task_complexity,
            'generated_tasks': self.generated_tasks,
            'config': self.config
        }, path)
        logger.info(f"Absolute Zero model saved to {path}")

    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.task_generator.load_state_dict(checkpoint['task_generator_state_dict'])
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.environment.task_complexity = checkpoint.get('task_complexity', 0.1)
        self.generated_tasks = checkpoint.get('generated_tasks', 0)
        logger.info(f"Absolute Zero model loaded from {path}")

    async def health_check(self) -> str:
        """Check agent health"""
        try:
            z = torch.randn(1, self.config['latent_dim']).to(self.device)
            _ = self.task_generator(z)

            test_features = torch.randn(1, 256).to(self.device)
            _ = self.policy_network(test_features)

            task = self.environment.generate_task()
            if len(task['records']) == 0:
                return "unhealthy: environment not generating tasks"

            return "healthy"
        except Exception as e:
            return f"unhealthy: {str(e)}"