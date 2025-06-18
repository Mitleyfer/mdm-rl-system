import torch
import logging
import asyncio

import numpy as np
import torch.nn as nn
import torch.optim as optim

from datetime import datetime
from collections import deque
from typing import Dict, List, Tuple, Optional


logger = logging.getLogger(__name__)

class PreferenceModel(nn.Module):
    """Neural network for learning human preferences"""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super(PreferenceModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class FeedbackBuffer:
    """Buffer for storing human feedback"""

    def __init__(self, capacity: int = 5000):
        self.buffer = deque(maxlen=capacity)
        self.preference_pairs = deque(maxlen=capacity // 2)

    def add_feedback(self, state: Dict, action: Dict, feedback: float, metadata: Dict = None):
        """Add single feedback instance"""
        self.buffer.append({
            'state': state,
            'action': action,
            'feedback': feedback,
            'metadata': metadata or {},
            'timestamp': datetime.utcnow()
        })

    def add_preference(self, state: Dict, action_a: Dict, action_b: Dict, preference: int):
        """Add preference between two actions (0: A better, 1: B better, 0.5: equal)"""
        self.preference_pairs.append({
            'state': state,
            'action_a': action_a,
            'action_b': action_b,
            'preference': preference,
            'timestamp': datetime.utcnow()
        })

    def get_recent_feedback(self, n: int = 100) -> List[Dict]:
        """Get n most recent feedback instances"""
        return list(self.buffer)[-n:]

    def get_preference_pairs(self, n: int = 50) -> List[Dict]:
        """Get n most recent preference pairs"""
        return list(self.preference_pairs)[-n:]

class RLHFAgent:
    """
    Reinforcement Learning from Human Feedback agent
    Incorporates domain expertise through interactive learning
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = 64
        self.action_dim = 32

        self.preference_model = PreferenceModel(
            self.state_dim + self.action_dim
        ).to(self.device)

        self.preference_optimizer = optim.Adam(
            self.preference_model.parameters(),
            lr=self.config['learning_rate']
        )

        self.policy_network = self._build_policy_network().to(self.device)
        self.policy_optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=self.config['learning_rate']
        )

        self.feedback_buffer = FeedbackBuffer()

        self.uncertainty_threshold = self.config['uncertainty_threshold']
        self.query_budget = self.config['query_budget']
        self.queries_made = 0

        self.training_history = []

    def _default_config(self) -> Dict:
        return {
            'learning_rate': 3e-4,
            'batch_size': 32,
            'uncertainty_threshold': 0.3,
            'query_budget': 100,
            'min_feedback_for_training': 50,
            'preference_weight': 0.7,
            'direct_feedback_weight': 0.3
        }

    def _build_policy_network(self) -> nn.Module:
        """Build policy network for generating actions"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.action_dim),
            nn.Tanh()
        )

    async def initialize(self):
        """Initialize the RLHF agent"""
        logger.info("Initializing RLHF Agent...")

        try:
            self._load_feedback_history("models/rlhf_feedback.pkl")
            logger.info(f"Loaded {len(self.feedback_buffer.buffer)} feedback instances")
        except:
            logger.info("Starting with empty feedback buffer")

        try:
            self.load_model("models/rlhf_checkpoint.pth")
            logger.info("Loaded pre-trained RLHF models")
        except:
            logger.info("Starting with fresh RLHF models")

    async def shutdown(self):
        """Save models and feedback"""
        self.save_model("models/rlhf_checkpoint.pth")
        self._save_feedback_history("models/rlhf_feedback.pkl")

    def _encode_state(self, data: Dict, features: Dict) -> torch.Tensor:
        """Encode state for RLHF agent"""
        state_vector = []

        stats = features.get('statistics', {})
        state_vector.extend([
            stats.get('num_records', 0) / 100000,
            stats.get('avg_completeness', 0.5),
            stats.get('missing_ratio', 0.1),
            stats.get('duplicate_ratio', 0.05),
            features.get('complexity_score', 0.5)
        ])

        metrics = features.get('current_metrics', {})
        state_vector.extend([
            metrics.get('precision', 0.5),
            metrics.get('recall', 0.5),
            metrics.get('f1_score', 0.5)
        ])

        state_vector.extend([0] * (self.state_dim - len(state_vector)))

        return torch.FloatTensor(state_vector[:self.state_dim])

    def _encode_action(self, rules: Dict) -> torch.Tensor:
        """Encode rule configuration as action"""
        action_vector = []

        for key in ['name_threshold', 'address_threshold', 'phone_threshold',
                    'email_threshold', 'fuzzy_weight', 'exact_weight']:
            action_vector.append(rules.get(key, 0.5))

        action_vector.append(1.0 if rules.get('enable_phonetic', True) else 0.0)
        action_vector.append(1.0 if rules.get('enable_abbreviation', True) else 0.0)

        action_vector.extend([0] * (self.action_dim - len(action_vector)))

        return torch.FloatTensor(action_vector[:self.action_dim])

    def _decode_action(self, action_tensor: torch.Tensor) -> Dict:
        """Decode action tensor to rule configuration"""
        action = action_tensor.cpu().numpy()

        rules = {
            'name_threshold': np.clip(action[0], 0.5, 1.0),
            'address_threshold': np.clip(action[1], 0.5, 1.0),
            'phone_threshold': np.clip(action[2], 0.7, 1.0),
            'email_threshold': np.clip(action[3], 0.8, 1.0),
            'fuzzy_weight': np.clip(action[4], 0.0, 1.0),
            'exact_weight': np.clip(action[5], 0.0, 1.0),
            'enable_phonetic': action[6] > 0.5,
            'enable_abbreviation': action[7] > 0.5
        }

        total_weight = rules['fuzzy_weight'] + rules['exact_weight']
        if total_weight > 0:
            rules['fuzzy_weight'] /= total_weight
            rules['exact_weight'] /= total_weight

        return rules

    async def should_query_human(self, state: torch.Tensor, action: torch.Tensor) -> bool:
        """Determine if human feedback should be requested"""
        if self.queries_made >= self.query_budget:
            return False

        self.preference_model.train()
        uncertainties = []

        with torch.no_grad():
            combined = torch.cat([state, action]).unsqueeze(0).to(self.device)

            for _ in range(10):
                pred = self.preference_model(combined)
                uncertainties.append(pred.item())

        uncertainty = np.std(uncertainties)

        return uncertainty > self.uncertainty_threshold

    async def request_feedback(self, context: Dict) -> Optional[Dict]:
        """
        Request human feedback (in production, this would interface with UI)
        For now, simulate feedback based on rule quality
        """
        # In production, this would:
        # 1. Send request to UI/API
        # 2. Wait for human response
        # 3. Return structured feedback

        await asyncio.sleep(0.1)

        rules = context.get('proposed_rules', {})
        current_metrics = context.get('current_metrics', {})

        avg_threshold = np.mean([
            rules.get('name_threshold', 0.85),
            rules.get('address_threshold', 0.80)
        ])

        if current_metrics.get('f1_score', 0) < 0.7:
            if avg_threshold > 0.85:
                feedback_score = 0.3
                suggestion = "Thresholds too high for this data quality"
            else:
                feedback_score = 0.7
                suggestion = "Consider enabling more features"
        else:
            feedback_score = 0.9
            suggestion = "Good configuration for this dataset"

        self.queries_made += 1

        return {
            'score': feedback_score,
            'suggestion': suggestion,
            'expert_id': 'simulated',
            'response_time': 0.1
        }

    def train_preference_model(self):
        """Train preference model on collected feedback"""
        feedback_data = self.feedback_buffer.get_recent_feedback()
        preference_pairs = self.feedback_buffer.get_preference_pairs()

        if len(feedback_data) < self.config['min_feedback_for_training']:
            return None

        total_loss = 0
        n_updates = 0

        if feedback_data:
            batch_size = min(self.config['batch_size'], len(feedback_data))
            indices = np.random.choice(len(feedback_data), batch_size, replace=False)

            states = []
            actions = []
            targets = []

            for idx in indices:
                feedback = feedback_data[idx]
                state = self._encode_state({}, feedback['state'])
                action = self._encode_action(feedback['action'])
                states.append(state)
                actions.append(action)
                targets.append(feedback['feedback'])

            states = torch.stack(states).to(self.device)
            actions = torch.stack(actions).to(self.device)
            targets = torch.FloatTensor(targets).to(self.device)

            inputs = torch.cat([states, actions], dim=1)
            predictions = self.preference_model(inputs).squeeze()

            loss = nn.MSELoss()(predictions, targets)

            self.preference_optimizer.zero_grad()
            loss.backward()
            self.preference_optimizer.step()

            total_loss += loss.item()
            n_updates += 1

        if preference_pairs and len(preference_pairs) >= 10:
            batch_size = min(self.config['batch_size'] // 2, len(preference_pairs))
            indices = np.random.choice(len(preference_pairs), batch_size, replace=False)

            for idx in indices:
                pref = preference_pairs[idx]
                state = self._encode_state({}, pref['state']).to(self.device)
                action_a = self._encode_action(pref['action_a']).to(self.device)
                action_b = self._encode_action(pref['action_b']).to(self.device)

                input_a = torch.cat([state, action_a]).unsqueeze(0)
                input_b = torch.cat([state, action_b]).unsqueeze(0)

                pred_a = self.preference_model(input_a)
                pred_b = self.preference_model(input_b)

                if pref['preference'] == 0:
                    loss = -torch.log(torch.sigmoid(pred_a - pred_b))
                elif pref['preference'] == 1:
                    loss = -torch.log(torch.sigmoid(pred_b - pred_a))
                else:
                    loss = (pred_a - pred_b) ** 2

                self.preference_optimizer.zero_grad()
                loss.backward()
                self.preference_optimizer.step()

                total_loss += loss.item()
                n_updates += 1

        return total_loss / n_updates if n_updates > 0 else None

    def train_policy(self, states: torch.Tensor, rewards: torch.Tensor):
        """Train policy network using learned reward model"""
        actions = self.policy_network(states)

        inputs = torch.cat([states, actions], dim=1)
        predicted_rewards = self.preference_model(inputs).squeeze()

        loss = -predicted_rewards.mean()

        action_std = actions.std(dim=0).mean()
        entropy_bonus = -0.01 * torch.log(action_std + 1e-8)
        loss += entropy_bonus

        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.policy_optimizer.step()

        return loss.item()

    async def learn(self, data: Dict, features: Dict, matches: Dict) -> Dict:
        """
        Learn from dataset with human feedback integration
        """
        logger.info("RLHF Agent learning with human feedback...")

        state = self._encode_state(data, features).to(self.device)

        with torch.no_grad():
            action = self.policy_network(state.unsqueeze(0)).squeeze()

        proposed_rules = self._decode_action(action)

        should_query = await self.should_query_human(state, action)

        if should_query:
            context = {
                'proposed_rules': proposed_rules,
                'current_metrics': matches.get('metrics', {}),
                'data_stats': features.get('statistics', {})
            }

            feedback = await self.request_feedback(context)

            if feedback:
                self.feedback_buffer.add_feedback(
                    state=features,
                    action=proposed_rules,
                    feedback=feedback['score'],
                    metadata={'suggestion': feedback['suggestion']}
                )

                logger.info(f"Received feedback: {feedback['score']:.2f} - {feedback['suggestion']}")

        pref_loss = self.train_preference_model()

        if len(self.feedback_buffer.buffer) >= self.config['batch_size']:
            feedback_samples = np.random.choice(
                self.feedback_buffer.get_recent_feedback(),
                size=self.config['batch_size'],
                replace=True
            )

            states = torch.stack([
                self._encode_state({}, f['state'])
                for f in feedback_samples
            ]).to(self.device)

            rewards = torch.FloatTensor([
                f['feedback'] for f in feedback_samples
            ]).to(self.device)

            policy_loss = self.train_policy(states, rewards)
        else:
            policy_loss = None

        return {
            'feedback_collected': len(self.feedback_buffer.buffer),
            'queries_made': self.queries_made,
            'preference_loss': pref_loss,
            'policy_loss': policy_loss,
            'should_query': should_query
        }

    async def generate_rules(self, features: Dict) -> Dict:
        """
        Generate rules using policy trained with human feedback
        """
        logger.info("Generating rules with RLHF policy...")

        state = self._encode_state({}, features).to(self.device)

        with torch.no_grad():
            action = self.policy_network(state.unsqueeze(0)).squeeze()

        rules = self._decode_action(action)

        if len(self.feedback_buffer.buffer) > 50:
            candidates = []
            scores = []

            for _ in range(10):
                noise = torch.randn_like(action) * 0.1
                candidate_action = action + noise
                candidate_rules = self._decode_action(candidate_action)

                with torch.no_grad():
                    input_tensor = torch.cat([state, candidate_action]).unsqueeze(0).to(self.device)
                    score = self.preference_model(input_tensor).item()

                candidates.append(candidate_rules)
                scores.append(score)

            best_idx = np.argmax(scores)
            rules = candidates[best_idx]
            logger.info(f"Selected rules with preference score: {scores[best_idx]:.3f}")

        return rules

    def save_model(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'preference_model_state_dict': self.preference_model.state_dict(),
            'policy_network_state_dict': self.policy_network.state_dict(),
            'preference_optimizer_state_dict': self.preference_optimizer.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'queries_made': self.queries_made,
            'config': self.config
        }, path)
        logger.info(f"RLHF models saved to {path}")

    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.preference_model.load_state_dict(checkpoint['preference_model_state_dict'])
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.preference_optimizer.load_state_dict(checkpoint['preference_optimizer_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.queries_made = checkpoint.get('queries_made', 0)
        logger.info(f"RLHF models loaded from {path}")

    def _save_feedback_history(self, path: str):
        """Save feedback buffer to disk"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'buffer': list(self.feedback_buffer.buffer),
                'preference_pairs': list(self.feedback_buffer.preference_pairs)
            }, f)

    def _load_feedback_history(self, path: str):
        """Load feedback buffer from disk"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            for feedback in data['buffer']:
                self.feedback_buffer.buffer.append(feedback)
            for pref in data['preference_pairs']:
                self.feedback_buffer.preference_pairs.append(pref)

    async def health_check(self) -> str:
        """Check agent health"""
        try:
            test_state = torch.randn(1, self.state_dim).to(self.device)
            test_action = self.policy_network(test_state)

            test_input = torch.cat([test_state, test_action], dim=1)
            _ = self.preference_model(test_input)

            return "healthy"
        except Exception as e:
            return f"unhealthy: {str(e)}"