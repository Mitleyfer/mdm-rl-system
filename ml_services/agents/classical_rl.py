import torch
import random
import logging

import numpy as np
import torch.nn as nn
import torch.optim as optim

from collections import deque
from typing import Dict, List, Tuple, Optional


logger = logging.getLogger(__name__)

class DQN(nn.Module):
    """Deep Q-Network for MDM rule optimization"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)

        self.dropout = nn.Dropout(0.2)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

class ReplayBuffer:
    """Experience replay buffer for DQN training"""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class ClassicalRLAgent:
    """
    Classical Reinforcement Learning agent using Deep Q-Learning
    for MDM rule optimization
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = 128
        self.action_dim = 50

        self.q_network = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.config['learning_rate']
        )
        self.loss_fn = nn.MSELoss()

        self.memory = ReplayBuffer(self.config['buffer_size'])

        self.epsilon = self.config['epsilon_start']
        self.epsilon_min = self.config['epsilon_end']
        self.epsilon_decay = self.config['epsilon_decay']
        self.gamma = self.config['gamma']
        self.batch_size = self.config['batch_size']
        self.update_target_every = self.config['update_target_every']

        self.steps = 0
        self.episodes = 0
        self.training_history = []

    def _default_config(self) -> Dict:
        return {
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'batch_size': 32,
            'buffer_size': 10000,
            'update_target_every': 1000
        }

    async def initialize(self):
        """Initialize the agent"""
        logger.info("Initializing Classical RL Agent...")
        try:
            self.load_model("models/classical_rl_checkpoint.pth")
            logger.info("Loaded pre-trained model")
        except:
            logger.info("Starting with fresh model")

    async def shutdown(self):
        """Cleanup resources"""
        self.save_model("models/classical_rl_checkpoint.pth")

    def _encode_state(self, data: Dict, features: Dict, rules: Dict) -> np.ndarray:
        """
        Encode current state for the RL agent
        State includes: rule configuration, data statistics, performance metrics
        """
        state_vector = []

        rule_features = []
        for key in ['name_threshold', 'address_threshold', 'phone_threshold',
                    'email_threshold', 'fuzzy_weight', 'exact_weight']:
            rule_features.append(rules.get(key, 0.5))
        state_vector.extend(rule_features)

        data_stats = features.get('statistics', {})
        stat_features = [
            data_stats.get('num_records', 0) / 1000000,
            data_stats.get('avg_completeness', 0.5),
            data_stats.get('avg_name_length', 20) / 100,
            data_stats.get('missing_ratio', 0.1),
            data_stats.get('duplicate_ratio', 0.05)
        ]
        state_vector.extend(stat_features)

        perf_metrics = features.get('current_metrics', {})
        perf_features = [
            perf_metrics.get('precision', 0.5),
            perf_metrics.get('recall', 0.5),
            perf_metrics.get('f1_score', 0.5),
            perf_metrics.get('processing_time', 1.0) / 10
        ]
        state_vector.extend(perf_features)

        state_vector.extend([0] * (self.state_dim - len(state_vector)))

        return np.array(state_vector[:self.state_dim], dtype=np.float32)

    def _decode_action(self, action_idx: int) -> Dict:
        """
        Decode action index to rule modification
        Actions include: adjust thresholds, change weights, enable/disable rules
        """
        action_mappings = {
            0: {'name_threshold': 0.05},
            1: {'name_threshold': -0.05},
            2: {'address_threshold': 0.05},
            3: {'address_threshold': -0.05},
            4: {'phone_threshold': 0.05},
            5: {'phone_threshold': -0.05},
            6: {'email_threshold': 0.05},
            7: {'email_threshold': -0.05},

            8: {'fuzzy_weight': 0.1},
            9: {'fuzzy_weight': -0.1},
            10: {'exact_weight': 0.1},
            11: {'exact_weight': -0.1},

            12: {'enable_phonetic': True},
            13: {'enable_phonetic': False},
            14: {'enable_abbreviation': True},
            15: {'enable_abbreviation': False},

            16: {'blocking_key': 'first_letter'},
            17: {'blocking_key': 'soundex'},
            18: {'blocking_key': 'sorted_neighborhood'},

            19: {}
        }

        if action_idx >= 20:
            base_actions = list(action_mappings.values())[:10]
            idx1 = (action_idx - 20) // 10
            idx2 = (action_idx - 20) % 10
            if idx1 < len(base_actions) and idx2 < len(base_actions):
                combined = {}
                combined.update(base_actions[idx1])
                combined.update(base_actions[idx2])
                return combined

        return action_mappings.get(action_idx, {})

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return None

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        current_q_values = self.q_network(state).gather(1, action.unsqueeze(1))

        with torch.no_grad():
            next_q_values = self.target_network(next_state).max(1)[0]
            target_q_values = reward + (self.gamma * next_q_values * (1 - done))

        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        if self.steps % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.steps += 1

        return loss.item()

    async def learn(self, data: Dict, features: Dict, matches: Dict) -> Dict:
        """
        Learn from a dataset through episodic training
        """
        logger.info("Classical RL Agent learning from dataset...")

        current_rules = matches.get('rules_used', self._default_rules())
        state = self._encode_state(data, features, current_rules)

        episode_rewards = []
        losses = []

        for episode in range(self.config.get('episodes_per_dataset', 100)):
            episode_reward = 0
            rules = current_rules.copy()

            for step in range(self.config.get('max_steps_per_episode', 50)):
                action = self.select_action(state)

                rule_modification = self._decode_action(action)
                rules = self._apply_rule_modification(rules, rule_modification)

                reward = await self._calculate_reward(data, rules, features)

                next_features = features.copy()
                next_features['current_metrics'] = self._estimate_metrics(rules)
                next_state = self._encode_state(data, next_features, rules)

                done = step == self.config.get('max_steps_per_episode', 50) - 1
                self.memory.push(state, action, reward, next_state, done)

                loss = self.train_step()
                if loss is not None:
                    losses.append(loss)

                state = next_state
                episode_reward += reward

                if done:
                    break

            episode_rewards.append(episode_reward)

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_loss = np.mean(losses[-100:]) if losses else 0
                logger.info(f"Episode {episode}: Avg Reward: {avg_reward:.3f}, "
                            f"Avg Loss: {avg_loss:.4f}, Epsilon: {self.epsilon:.3f}")

        self.episodes += len(episode_rewards)

        return {
            "episodes_trained": len(episode_rewards),
            "avg_reward": np.mean(episode_rewards),
            "final_epsilon": self.epsilon,
            "avg_loss": np.mean(losses) if losses else 0,
            "total_episodes": self.episodes
        }

    async def generate_rules(self, features: Dict) -> Dict:
        """
        Generate optimized rules based on learned policy
        """
        logger.info("Generating optimized rules...")

        best_rules = self._default_rules()
        best_score = -float('inf')

        state = self._encode_state({}, features, best_rules)

        for _ in range(50):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor).squeeze().cpu().numpy()

            action = np.argmax(q_values)

            modification = self._decode_action(action)
            candidate_rules = self._apply_rule_modification(best_rules.copy(), modification)

            score = self._estimate_rule_score(candidate_rules, features)

            if score > best_score:
                best_rules = candidate_rules
                best_score = score

                next_features = features.copy()
                next_features['current_metrics'] = self._estimate_metrics(best_rules)
                state = self._encode_state({}, next_features, best_rules)
            else:
                break

        logger.info(f"Generated rules with estimated score: {best_score:.3f}")
        return best_rules

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
            'blocking_key': 'sorted_neighborhood'
        }

    def _apply_rule_modification(self, rules: Dict, modification: Dict) -> Dict:
        """Apply modification to rules with bounds checking"""
        for key, value in modification.items():
            if key in rules:
                if isinstance(value, (int, float)):
                    rules[key] = np.clip(rules[key] + value, 0.0, 1.0)
                else:
                    rules[key] = value
            else:
                rules[key] = value

        return rules

    async def _calculate_reward(self, data: Dict, rules: Dict, features: Dict) -> float:
        """
        Calculate reward based on matching performance
        In production, this would run actual matching and compute metrics
        """
        estimated_metrics = self._estimate_metrics(rules)

        reward = (
                estimated_metrics['f1_score'] * 0.5 +
                estimated_metrics['precision'] * 0.3 +
                estimated_metrics['recall'] * 0.2 -
                estimated_metrics['processing_time'] * 0.1
        )

        return reward

    def _estimate_metrics(self, rules: Dict) -> Dict:
        """
        Estimate performance metrics based on rules
        In production, this would be learned from historical data
        """
        avg_threshold = np.mean([
            rules.get('name_threshold', 0.85),
            rules.get('address_threshold', 0.80),
            rules.get('phone_threshold', 0.95),
            rules.get('email_threshold', 0.98)
        ])

        precision = min(0.95, avg_threshold + 0.1)
        recall = max(0.6, 1.0 - avg_threshold * 0.3)
        f1_score = 2 * (precision * recall) / (precision + recall)

        processing_time = 1.0
        if rules.get('enable_phonetic', True):
            processing_time += 0.2
        if rules.get('enable_abbreviation', True):
            processing_time += 0.1

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'processing_time': processing_time
        }

    def _estimate_rule_score(self, rules: Dict, features: Dict) -> float:
        """Estimate overall score for a rule configuration"""
        metrics = self._estimate_metrics(rules)

        data_quality = features.get('statistics', {}).get('avg_completeness', 0.5)

        if data_quality < 0.7:
            score = (
                    metrics['f1_score'] * 0.4 +
                    metrics['recall'] * 0.4 +
                    metrics['precision'] * 0.2
            )
        else:
            score = (
                    metrics['f1_score'] * 0.6 +
                    metrics['precision'] * 0.2 +
                    metrics['recall'] * 0.2
            )

        return score

    def save_model(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes,
            'config': self.config
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        logger.info(f"Model loaded from {path}")

    async def health_check(self) -> str:
        """Check agent health"""
        try:
            test_state = torch.randn(1, self.state_dim).to(self.device)
            with torch.no_grad():
                _ = self.q_network(test_state)
            return "healthy"
        except Exception as e:
            return f"unhealthy: {str(e)}"