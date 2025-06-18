import json
import yaml
import logging

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class RuleManager:
    """
    Manages matching rules configuration and versioning
    """

    def __init__(self, config_path: str = "config/rules"):
        self.config_path = Path(config_path)
        self.config_path.mkdir(parents=True, exist_ok=True)

        self.current_rules = None
        self.rules_history = []
        self.default_rules = self._load_default_rules()

    def _load_default_rules(self) -> Dict[str, Any]:
        """Load default rule configuration"""
        return {
            'name_threshold': 0.85,
            'address_threshold': 0.80,
            'phone_threshold': 0.95,
            'email_threshold': 0.98,

            'fuzzy_weight': 0.7,
            'exact_weight': 0.3,
            'name_weight': 0.4,
            'address_weight': 0.3,
            'phone_weight': 0.2,
            'email_weight': 0.1,

            'enable_phonetic': True,
            'enable_abbreviation': True,
            'enable_semantic': False,
            'semantic_weight': 0.2,

            'blocking_key': 'sorted_neighborhood',
            'block_field': 'zip',
            'min_block_size': 2,
            'max_block_size': 1000,

            'min_matching_fields': 2,
            'overall_threshold': 0.8,
            'confidence_threshold': 0.5,

            'version': 1,
            'created_at': datetime.utcnow().isoformat(),
            'created_by': 'system',
            'description': 'Default matching rules'
        }

    async def load_rules(self) -> Dict[str, Any]:
        """Load current active rules"""
        try:
            from core.database import get_active_rules
            db_rules = await get_active_rules()

            if db_rules:
                self.current_rules = db_rules
                logger.info(f"Loaded rules version {db_rules.get('version', 'unknown')} from database")
                return self.current_rules

            rules_file = self.config_path / "active_rules.json"
            if rules_file.exists():
                with open(rules_file, 'r') as f:
                    self.current_rules = json.load(f)
                logger.info("Loaded rules from file")
                return self.current_rules

            self.current_rules = self.default_rules.copy()
            logger.info("Using default rules")
            return self.current_rules

        except Exception as e:
            logger.error(f"Failed to load rules: {e}")
            self.current_rules = self.default_rules.copy()
            return self.current_rules

    async def update_rules(self, new_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Update rules with validation"""
        try:
            validated_rules = self._validate_rules(new_rules)

            validated_rules['updated_at'] = datetime.utcnow().isoformat()
            validated_rules['version'] = self.current_rules.get('version', 0) + 1

            # Save to database
            from core.database import save_rules
            await save_rules(validated_rules)

            rules_file = self.config_path / "active_rules.json"
            with open(rules_file, 'w') as f:
                json.dump(validated_rules, f, indent=2)

            if self.current_rules:
                self._archive_rules(self.current_rules)

            self.current_rules = validated_rules

            logger.info(f"Updated rules to version {validated_rules['version']}")
            return validated_rules

        except Exception as e:
            logger.error(f"Failed to update rules: {e}")
            raise

    def _validate_rules(self, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate rule configuration"""
        validated = {}

        threshold_fields = [
            'name_threshold', 'address_threshold',
            'phone_threshold', 'email_threshold',
            'overall_threshold', 'confidence_threshold'
        ]

        for field in threshold_fields:
            if field in rules:
                value = float(rules[field])
                if 0 <= value <= 1:
                    validated[field] = value
                else:
                    raise ValueError(f"{field} must be between 0 and 1")
            else:
                validated[field] = self.default_rules.get(field, 0.5)

        weight_groups = [
            ['fuzzy_weight', 'exact_weight'],
            ['name_weight', 'address_weight', 'phone_weight', 'email_weight']
        ]

        for group in weight_groups:
            group_weights = {}
            total = 0

            for field in group:
                if field in rules:
                    value = float(rules[field])
                    if 0 <= value <= 1:
                        group_weights[field] = value
                        total += value
                    else:
                        raise ValueError(f"{field} must be between 0 and 1")

            if total > 0:
                for field, value in group_weights.items():
                    validated[field] = value / total
            else:
                for field in group:
                    validated[field] = self.default_rules.get(field, 1.0 / len(group))

        boolean_fields = ['enable_phonetic', 'enable_abbreviation', 'enable_semantic']
        for field in boolean_fields:
            if field in rules:
                validated[field] = bool(rules[field])
            else:
                validated[field] = self.default_rules.get(field, False)

        if 'blocking_key' in rules:
            valid_keys = ['exact', 'soundex', 'first_letter', 'sorted_neighborhood']
            if rules['blocking_key'] in valid_keys:
                validated['blocking_key'] = rules['blocking_key']
            else:
                raise ValueError(f"blocking_key must be one of {valid_keys}")
        else:
            validated['blocking_key'] = self.default_rules['blocking_key']

        integer_fields = ['min_matching_fields', 'min_block_size', 'max_block_size']
        for field in integer_fields:
            if field in rules:
                value = int(rules[field])
                if value > 0:
                    validated[field] = value
                else:
                    raise ValueError(f"{field} must be positive")
            else:
                validated[field] = self.default_rules.get(field, 1)

        for key, value in rules.items():
            if key not in validated and not key.startswith('_'):
                validated[key] = value

        return validated

    def _archive_rules(self, rules: Dict[str, Any]):
        """Archive rules to history"""
        archive_file = self.config_path / f"rules_v{rules.get('version', 0)}.json"
        with open(archive_file, 'w') as f:
            json.dump(rules, f, indent=2)

        self.rules_history.append({
            'version': rules.get('version', 0),
            'archived_at': datetime.utcnow().isoformat(),
            'file': str(archive_file)
        })

    def get_active_rules(self) -> Optional[Dict[str, Any]]:
        """Get current active rules"""
        return self.current_rules

    async def get_rule_summary(self) -> Dict[str, Any]:
        """Get summary of current rules"""
        if not self.current_rules:
            await self.load_rules()

        return {
            'version': self.current_rules.get('version', 0),
            'thresholds': {
                k: v for k, v in self.current_rules.items()
                if k.endswith('_threshold')
            },
            'weights': {
                k: v for k, v in self.current_rules.items()
                if k.endswith('_weight')
            },
            'features': {
                k: v for k, v in self.current_rules.items()
                if k.startswith('enable_')
            },
            'blocking_strategy': self.current_rules.get('blocking_key', 'unknown'),
            'last_updated': self.current_rules.get('updated_at', 'unknown')
        }

    def export_rules(self, format: str = 'json') -> str:
        """Export rules in specified format"""
        if not self.current_rules:
            return ""

        if format == 'json':
            return json.dumps(self.current_rules, indent=2)
        elif format == 'yaml':
            return yaml.dump(self.current_rules, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def import_rules(self, content: str, format: str = 'json') -> Dict[str, Any]:
        """Import rules from string content"""
        if format == 'json':
            rules = json.loads(content)
        elif format == 'yaml':
            rules = yaml.safe_load(content)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return self._validate_rules(rules)

    def get_rules_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get rules version history"""
        history = []

        archive_files = sorted(
            self.config_path.glob("rules_v*.json"),
            key=lambda x: int(x.stem.split('_v')[1]),
            reverse=True
        )

        for file in archive_files[:limit]:
            with open(file, 'r') as f:
                rules = json.load(f)
                history.append({
                    'version': rules.get('version', 0),
                    'created_at': rules.get('created_at', 'unknown'),
                    'created_by': rules.get('created_by', 'unknown'),
                    'description': rules.get('description', ''),
                    'file': str(file)
                })

        return history