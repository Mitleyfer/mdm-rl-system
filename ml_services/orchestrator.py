import asyncio
import logging

import numpy as np

from enum import Enum
from datetime import datetime
from core.database import get_db
from .rule_manager import RuleManage
from .agents.rlhf_agent import RLHFAgent
from .data_processor import DataProcessor
from .matching_engine import MatchingEngine
from utils.metrics import calculate_metrics
from .agents.classical_rl import ClassicalRLAgent
from .agents.rag_ensemble import RAGEnsembleAgent
from typing import Dict, List, Any, Optional, Tuple
from .agents.absolute_zero import AbsoluteZeroAgent

logger = logging.getLogger(__name__)

class AgentType(Enum):
    CLASSICAL_RL = "classical_rl"
    RAG_ENSEMBLE = "rag_ensemble"
    RLHF = "rlhf"
    ABSOLUTE_ZERO = "absolute_zero"
    HYBRID = "hybrid"

class MLOrchestrator:
    """
    Central orchestrator for multi-paradigm reinforcement learning system.
    Manages agent selection, coordination, and learning cycles.
    """

    def __init__(self):
        self.agents = {}
        self.data_processor = DataProcessor()
        self.matching_engine = MatchingEngine()
        self.rule_manager = RuleManager()
        self.current_agent = None
        self.performance_history = []
        self.is_initialized = False

    async def initialize(self):
        """Initialize all ML agents and services"""
        try:
            logger.info("Initializing ML Orchestrator...")

            self.agents[AgentType.CLASSICAL_RL] = ClassicalRLAgent()
            self.agents[AgentType.RAG_ENSEMBLE] = RAGEnsembleAgent()
            self.agents[AgentType.RLHF] = RLHFAgent()
            self.agents[AgentType.ABSOLUTE_ZERO] = AbsoluteZeroAgent()

            for agent_type, agent in self.agents.items():
                await agent.initialize()
                logger.info(f"Initialized {agent_type.value} agent")

            await self.rule_manager.load_rules()

            self.is_initialized = True
            logger.info("ML Orchestrator initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ML Orchestrator: {e}")
            raise

    async def shutdown(self):
        """Shutdown all agents and services"""
        for agent in self.agents.values():
            await agent.shutdown()

    async def process_dataset(
            self,
            dataset_id: str,
            file_path: str,
            dataset_name: str,
            dataset_type: str
    ):
        """Process uploaded dataset through the ML pipeline"""
        try:
            await self._update_dataset_status(dataset_id, "loading", 0)

            logger.info(f"Processing dataset {dataset_id}: {dataset_name}")
            data = await self.data_processor.load_dataset(file_path, dataset_type)

            await self._update_dataset_status(dataset_id, "preprocessing", 20)

            features = await self.data_processor.extract_features(data)

            await self._update_dataset_status(dataset_id, "matching", 40)

            initial_matches = await self.matching_engine.match_dataset(
                data,
                self.rule_manager.get_active_rules()
            )

            await self._update_dataset_status(dataset_id, "learning", 60)

            selected_agent = await self._select_agent(data, features, initial_matches)

            learning_results = await selected_agent.learn(
                data=data,
                features=features,
                matches=initial_matches
            )

            await self._update_dataset_status(dataset_id, "optimizing", 80)

            optimized_rules = await selected_agent.generate_rules(features)

            validation_results = await self._validate_rules(
                optimized_rules,
                data,
                initial_matches
            )

            if validation_results['improvement'] > 0:
                await self.rule_manager.update_rules(optimized_rules)
                logger.info(f"Rules updated with {validation_results['improvement']:.2%} improvement")

            await self._update_dataset_status(dataset_id, "completed", 100)

            await self._store_results(dataset_id, {
                "dataset_name": dataset_name,
                "dataset_type": dataset_type,
                "records_processed": len(data),
                "initial_performance": initial_matches['metrics'],
                "final_performance": validation_results['metrics'],
                "improvement": validation_results['improvement'],
                "agent_used": selected_agent.__class__.__name__
            })

        except Exception as e:
            logger.error(f"Failed to process dataset {dataset_id}: {e}")
            await self._update_dataset_status(dataset_id, "failed", -1)
            raise

    async def _select_agent(
            self,
            data: Dict,
            features: Dict,
            initial_matches: Dict
    ) -> Any:
        """
        Intelligently select the most appropriate agent based on context
        """
        data_profile = {
            "size": len(data.get("records", [])),
            "quality_score": features.get("quality_score", 0.5),
            "domain": data.get("dataset_type", "unknown"),
            "has_labels": "labels" in data,
            "complexity": features.get("complexity_score", 0.5)
        }

        if not data_profile["has_labels"] and data_profile["size"] < 1000:
            logger.info("Selected Absolute Zero agent (no labels, small dataset)")
            return self.agents[AgentType.ABSOLUTE_ZERO]

        elif data_profile["has_labels"] and data_profile["quality_score"] > 0.8:
            logger.info("Selected Classical RL agent (high quality labeled data)")
            return self.agents[AgentType.CLASSICAL_RL]

        elif data_profile["complexity"] > 0.7:
            logger.info("Selected RAG Ensemble agent (complex matching)")
            return self.agents[AgentType.RAG_ENSEMBLE]

        elif self.performance_history and len(self.performance_history) > 10:
            logger.info("Selected RLHF agent (sufficient feedback history)")
            return self.agents[AgentType.RLHF]

        else:
            logger.info("Using hybrid multi-agent approach")
            return await self._create_hybrid_agent()

    async def _create_hybrid_agent(self):
        """Create a hybrid agent that combines multiple paradigms"""
        class HybridAgent:
            def __init__(self, agents):
                self.agents = agents

            async def learn(self, data, features, matches):
                """Ensemble learning from multiple agents"""
                results = []
                for agent in self.agents.values():
                    try:
                        result = await agent.learn(data, features, matches)
                        results.append(result)
                    except Exception as e:
                        logger.warning(f"Agent {agent} failed: {e}")

                return self._aggregate_results(results)

            async def generate_rules(self, features):
                """Generate rules by combining agent proposals"""
                all_rules = []
                weights = []

                for agent_type, agent in self.agents.items():
                    try:
                        rules = await agent.generate_rules(features)
                        all_rules.append(rules)
                        weights.append(self._get_agent_weight(agent_type))
                    except Exception as e:
                        logger.warning(f"Rule generation failed for {agent_type}: {e}")

                return self._combine_rules(all_rules, weights)

            def _aggregate_results(self, results):
                """Aggregate learning results from multiple agents"""
                if not results:
                    return {}

                aggregated = {
                    "loss": np.mean([r.get("loss", 0) for r in results]),
                    "metrics": {},
                    "insights": []
                }

                metric_keys = set()
                for r in results:
                    metric_keys.update(r.get("metrics", {}).keys())

                for key in metric_keys:
                    values = [r["metrics"].get(key, 0) for r in results if "metrics" in r]
                    aggregated["metrics"][key] = np.mean(values) if values else 0

                return aggregated

            def _combine_rules(self, all_rules, weights):
                """Combine rules from multiple agents with weights"""
                if not all_rules:
                    return {}

                combined = {}
                weights = np.array(weights)
                weights = weights / weights.sum()

                rule_keys = set()
                for rules in all_rules:
                    rule_keys.update(rules.keys())

                for key in rule_keys:
                    values = []
                    rule_weights = []
                    for i, rules in enumerate(all_rules):
                        if key in rules:
                            values.append(rules[key])
                            rule_weights.append(weights[i])

                    if values:
                        if isinstance(values[0], (int, float)):
                            combined[key] = np.average(values, weights=rule_weights)
                        else:
                            idx = np.argmax(rule_weights)
                            combined[key] = values[idx]

                return combined

            def _get_agent_weight(self, agent_type):
                """Get weight for agent based on historical performance"""
                # TODO: Implement performance-based weighting
                return 1.0

        return HybridAgent(self.agents)

    async def _validate_rules(
            self,
            rules: Dict,
            data: Dict,
            baseline_matches: Dict
    ) -> Dict:
        """Validate new rules against baseline performance"""
        try:
            new_matches = await self.matching_engine.match_dataset(data, rules)

            baseline_metrics = baseline_matches.get("metrics", {})
            new_metrics = calculate_metrics(
                new_matches["matches"],
                data.get("ground_truth", [])
            )

            improvement = (
                    new_metrics.get("f1_score", 0) -
                    baseline_metrics.get("f1_score", 0)
            )

            return {
                "metrics": new_metrics,
                "improvement": improvement,
                "is_valid": improvement >= -0.02
            }

        except Exception as e:
            logger.error(f"Rule validation failed: {e}")
            return {
                "metrics": {},
                "improvement": 0,
                "is_valid": False
            }

    async def get_dataset_status(self, dataset_id: str) -> Optional[Dict]:
        """Get current processing status for a dataset"""
        async with get_db() as db:
            result = await db.fetch_one(
                "SELECT * FROM dataset_processing WHERE id = :id",
                {"id": dataset_id}
            )
            return dict(result) if result else None

    async def _update_dataset_status(
            self,
            dataset_id: str,
            status: str,
            progress: int
    ):
        """Update dataset processing status"""
        async with get_db() as db:
            await db.execute(
                """
                INSERT INTO dataset_processing (id, status, progress, updated_at)
                VALUES (:id, :status, :progress, :updated_at)
                ON CONFLICT (id) DO UPDATE SET
                    status = :status,
                    progress = :progress,
                    updated_at = :updated_at
                """,
                {
                    "id": dataset_id,
                    "status": status,
                    "progress": progress,
                    "updated_at": datetime.utcnow()
                }
            )

    async def _store_results(self, dataset_id: str, results: Dict):
        """Store processing results"""
        async with get_db() as db:
            await db.execute(
                """
                UPDATE dataset_processing 
                SET results = :results, completed_at = :completed_at
                WHERE id = :id
                """,
                {
                    "id": dataset_id,
                    "results": results,
                    "completed_at": datetime.utcnow()
                }
            )

        self.performance_history.append({
            "dataset_id": dataset_id,
            "timestamp": datetime.utcnow(),
            "metrics": results.get("final_performance", {}),
            "agent": results.get("agent_used", "unknown")
        })

    async def health_check(self) -> Dict[str, str]:
        """Check health of all ML services"""
        status = {}
        for agent_type, agent in self.agents.items():
            try:
                if hasattr(agent, 'health_check'):
                    agent_status = await agent.health_check()
                else:
                    agent_status = "healthy"
                status[agent_type.value] = agent_status
            except Exception as e:
                status[agent_type.value] = f"unhealthy: {str(e)}"

        return status

    async def get_active_model_info(self) -> Dict[str, Any]:
        """Get information about currently active models"""
        info = {
            "orchestrator_version": "1.0.0",
            "active_agents": list(self.agents.keys()),
            "current_rules": await self.rule_manager.get_rule_summary(),
            "performance_summary": self._get_performance_summary()
        }
        return info

    def _get_performance_summary(self) -> Dict[str, float]:
        """Get summary of recent performance"""
        if not self.performance_history:
            return {}

        recent = self.performance_history[-10:]

        return {
            "avg_f1_score": np.mean([h["metrics"].get("f1_score", 0) for h in recent]),
            "avg_precision": np.mean([h["metrics"].get("precision", 0) for h in recent]),
            "avg_recall": np.mean([h["metrics"].get("recall", 0) for h in recent]),
            "total_datasets_processed": len(self.performance_history)
        }