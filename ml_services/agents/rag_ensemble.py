import faiss
import torch
import pickle
import logging
import asyncio

import numpy as np

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    pipeline
)
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class KnowledgeBase:
    """Vector database for storing and retrieving matching scenarios"""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []
        self.encoder = None

    def add_scenario(self, scenario: Dict, embedding: np.ndarray):
        """Add a matching scenario to the knowledge base"""
        self.index.add(embedding.reshape(1, -1))
        self.metadata.append(scenario)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Retrieve k most similar scenarios"""
        if self.index.ntotal == 0:
            return []

        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.metadata):
                results.append(self.metadata[idx])

        return results

    def save(self, path: str):
        """Save knowledge base to disk"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.metadata", 'wb') as f:
            pickle.dump(self.metadata, f)

    def load(self, path: str):
        """Load knowledge base from disk"""
        if Path(f"{path}.index").exists():
            self.index = faiss.read_index(f"{path}.index")
            with open(f"{path}.metadata", 'rb') as f:
                self.metadata = pickle.load(f)

class RAGEnsembleAgent:
    """
    RAG-enhanced ensemble learning agent that uses multiple LLMs
    from Hugging Face for rule generation and optimization
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}

        self.knowledge_base = KnowledgeBase()

        self.executor = ThreadPoolExecutor(max_workers=4)

        self.model_weights = {}

    def _default_config(self) -> Dict:
        return {
            'models': [
                'sentence-transformers/all-MiniLM-L6-v2',
                'microsoft/deberta-v3-base',
                'bert-base-uncased',
                'roberta-base'
            ],
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'max_length': 512,
            'num_retrievals': 5,
            'temperature': 0.7,
            'ensemble_strategy': 'weighted_vote'
        }

    async def initialize(self):
        """Initialize all models and components"""
        logger.info("Initializing RAG Ensemble Agent...")

        embedding_model_name = self.config['embedding_model']
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name).to(self.device)

        for model_name in self.config['models'][1:]:
            try:
                logger.info(f"Loading model: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=2
                ).to(self.device)

                self.models[model_name] = model
                self.tokenizers[model_name] = tokenizer
                self.model_weights[model_name] = 1.0

            except Exception as e:
                logger.warning(f"Failed to load model {model_name}: {e}")

        kb_path = "models/rag_knowledge_base"
        try:
            self.knowledge_base.load(kb_path)
            logger.info(f"Loaded knowledge base with {self.knowledge_base.index.ntotal} scenarios")
        except:
            logger.info("Starting with empty knowledge base")

    async def shutdown(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        self.knowledge_base.save("models/rag_knowledge_base")

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding vector"""
        inputs = self.embedding_tokenizer(
            text,
            max_length=self.config['max_length'],
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            embeddings = self.embedding_model(**inputs).last_hidden_state
            attention_mask = inputs['attention_mask']
            embeddings = (embeddings * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)

        return embeddings.cpu().numpy()[0]

    def _create_scenario_description(self, data: Dict, features: Dict) -> str:
        """Create textual description of matching scenario"""
        stats = features.get('statistics', {})

        description = f"""
        Data Matching Scenario:
        - Domain: {data.get('dataset_type', 'unknown')}
        - Records: {stats.get('num_records', 0)}
        - Completeness: {stats.get('avg_completeness', 0):.2%}
        - Missing Values: {stats.get('missing_ratio', 0):.2%}
        - Estimated Duplicates: {stats.get('duplicate_ratio', 0):.2%}
        - Key Attributes: {', '.join(features.get('key_attributes', []))}
        - Data Quality Issues: {', '.join(features.get('quality_issues', []))}
        """

        return description.strip()

    def _create_rule_prompt(self, scenario: str, retrieved_scenarios: List[Dict]) -> str:
        """Create prompt for rule generation based on scenario and retrieved examples"""
        prompt = f"Given the following data matching scenario:\n{scenario}\n\n"

        if retrieved_scenarios:
            prompt += "Similar successful scenarios:\n"
            for i, prev_scenario in enumerate(retrieved_scenarios[:3]):
                prompt += f"\nExample {i+1}:\n"
                prompt += f"Context: {prev_scenario.get('description', 'N/A')}\n"
                prompt += f"Rules: {prev_scenario.get('rules', {})}\n"
                prompt += f"Performance: F1={prev_scenario.get('f1_score', 0):.3f}\n"

        prompt += "\nRecommended matching rules configuration:"

        return prompt

    async def _ensemble_inference(self, record_pair: Tuple[Dict, Dict]) -> Dict[str, float]:
        """Run ensemble inference on a record pair"""
        results = {}

        text = self._format_record_pair(record_pair)

        tasks = []
        for model_name, model in self.models.items():
            task = asyncio.create_task(
                self._single_model_inference(model_name, model, text)
            )
            tasks.append(task)

        model_outputs = await asyncio.gather(*tasks)

        weighted_scores = {'match': 0.0, 'no_match': 0.0}
        total_weight = sum(self.model_weights.values())

        for model_name, output in zip(self.models.keys(), model_outputs):
            if output is not None:
                weight = self.model_weights[model_name] / total_weight
                weighted_scores['match'] += output['match'] * weight
                weighted_scores['no_match'] += output['no_match'] * weight
                results[model_name] = output

        results['ensemble'] = weighted_scores
        return results

    async def _single_model_inference(self, model_name: str, model: Any, text: str) -> Optional[Dict]:
        """Run inference with a single model"""
        try:
            tokenizer = self.tokenizers[model_name]
            inputs = tokenizer(
                text,
                max_length=self.config['max_length'],
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            return {
                'match': probs[0][1].item(),
                'no_match': probs[0][0].item()
            }
        except Exception as e:
            logger.warning(f"Inference failed for {model_name}: {e}")
            return None

    def _format_record_pair(self, record_pair: Tuple[Dict, Dict]) -> str:
        """Format record pair for model input"""
        r1, r2 = record_pair

        text = "Compare these records:\n"
        text += f"Record 1: {self._record_to_string(r1)}\n"
        text += f"Record 2: {self._record_to_string(r2)}\n"
        text += "Are these the same entity?"

        return text

    def _record_to_string(self, record: Dict) -> str:
        """Convert record to string representation"""
        parts = []
        for key, value in record.items():
            if value:
                parts.append(f"{key}: {value}")
        return ", ".join(parts)

    async def learn(self, data: Dict, features: Dict, matches: Dict) -> Dict:
        """
        Learn from dataset by updating knowledge base and model weights
        """
        logger.info("RAG Ensemble Agent learning from dataset...")

        scenario_desc = self._create_scenario_description(data, features)
        scenario_embedding = self._encode_text(scenario_desc)

        similar_scenarios = self.knowledge_base.search(scenario_embedding, k=self.config['num_retrievals'])

        if 'sample_pairs' in data and 'labels' in data:
            ensemble_performance = await self._evaluate_ensemble(
                data['sample_pairs'],
                data['labels']
            )

            self._update_model_weights(ensemble_performance)

        current_scenario = {
            'description': scenario_desc,
            'features': features,
            'rules': matches.get('rules_used', {}),
            'f1_score': matches.get('metrics', {}).get('f1_score', 0),
            'timestamp': np.datetime64('now')
        }
        self.knowledge_base.add_scenario(current_scenario, scenario_embedding)

        return {
            'scenarios_retrieved': len(similar_scenarios),
            'knowledge_base_size': self.knowledge_base.index.ntotal,
            'ensemble_performance': ensemble_performance if 'sample_pairs' in data else None
        }

    async def _evaluate_ensemble(self, sample_pairs: List[Tuple], labels: List[int]) -> Dict:
        """Evaluate ensemble performance on labeled pairs"""
        correct_by_model = {name: 0 for name in self.models.keys()}
        correct_by_model['ensemble'] = 0

        for pair, label in zip(sample_pairs[:100], labels[:100]):
            results = await self._ensemble_inference(pair)

            for model_name in self.models.keys():
                if model_name in results:
                    pred = 1 if results[model_name]['match'] > 0.5 else 0
                    if pred == label:
                        correct_by_model[model_name] += 1

            ensemble_pred = 1 if results['ensemble']['match'] > 0.5 else 0
            if ensemble_pred == label:
                correct_by_model['ensemble'] += 1

        n_samples = min(len(sample_pairs), 100)
        performance = {
            name: correct / n_samples
            for name, correct in correct_by_model.items()
        }

        return performance

    def _update_model_weights(self, performance: Dict):
        """Update model weights based on performance"""
        alpha = 0.1

        for model_name in self.models.keys():
            if model_name in performance:
                old_weight = self.model_weights[model_name]
                new_weight = old_weight * (1 - alpha) + performance[model_name] * alpha
                self.model_weights[model_name] = max(0.1, new_weight)

    async def generate_rules(self, features: Dict) -> Dict:
        """
        Generate optimized rules using RAG and ensemble voting
        """
        logger.info("Generating rules with RAG Ensemble...")

        scenario_desc = self._create_scenario_description({'dataset_type': 'current'}, features)
        scenario_embedding = self._encode_text(scenario_desc)

        similar_scenarios = self.knowledge_base.search(
            scenario_embedding,
            k=self.config['num_retrievals']
        )

        if similar_scenarios:
            rule_aggregates = {}
            total_weight = 0

            for scenario in similar_scenarios:
                weight = scenario.get('f1_score', 0.5)
                rules = scenario.get('rules', {})

                for key, value in rules.items():
                    if isinstance(value, (int, float)):
                        if key not in rule_aggregates:
                            rule_aggregates[key] = 0
                        rule_aggregates[key] += value * weight

                total_weight += weight

            if total_weight > 0:
                base_rules = {
                    key: value / total_weight
                    for key, value in rule_aggregates.items()
                }
            else:
                base_rules = self._default_rules()
        else:
            base_rules = self._default_rules()

        adjustments = await self._ensemble_rule_adjustments(features)

        final_rules = base_rules.copy()
        for key, adjustment in adjustments.items():
            if key in final_rules and isinstance(adjustment, (int, float)):
                final_rules[key] = np.clip(final_rules[key] + adjustment, 0.0, 1.0)

        logger.info(f"Generated rules based on {len(similar_scenarios)} similar scenarios")
        return final_rules

    async def _ensemble_rule_adjustments(self, features: Dict) -> Dict:
        """Get rule adjustments from ensemble"""
        adjustments = {}

        quality_score = features.get('statistics', {}).get('avg_completeness', 0.5)

        if quality_score < 0.7:
            adjustments['name_threshold'] = -0.05
            adjustments['address_threshold'] = -0.08
        elif quality_score > 0.9:
            adjustments['name_threshold'] = 0.03
            adjustments['address_threshold'] = 0.05

        return adjustments

    def _default_rules(self) -> Dict:
        """Default rule configuration"""
        return {
            'name_threshold': 0.85,
            'address_threshold': 0.80,
            'phone_threshold': 0.95,
            'email_threshold': 0.98,
            'fuzzy_weight': 0.7,
            'exact_weight': 0.3,
            'enable_phonetic': True,
            'enable_abbreviation': True,
            'enable_semantic': True,
            'semantic_weight': 0.2
        }

    async def health_check(self) -> str:
        """Check agent health"""
        try:
            test_text = "Test health check"
            _ = self._encode_text(test_text)

            for model_name, model in self.models.items():
                test_input = torch.randn(1, 10).to(self.device)
                if next(model.parameters()).device != self.device:
                    return f"unhealthy: {model_name} on wrong device"

            return "healthy"
        except Exception as e:
            return f"unhealthy: {str(e)}"