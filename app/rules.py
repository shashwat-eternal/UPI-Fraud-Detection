
from typing import Any, Dict, List
from pydantic import BaseModel


class RuleConfig(BaseModel):
    max_24h_velocity: int = 8
    max_single_amount: float = 50000.0
    blacklisted_banks: list[str] = ["SUSPICIOUS_BANK_TEST", "FRAUD_BANK_X"]
    enable_multi_flag_rule: bool = True


# Global active rule configuration state
_RULE_CONFIG = RuleConfig()


def get_rule_config() -> RuleConfig:
    return _RULE_CONFIG


def update_rule_config(new_config: RuleConfig) -> RuleConfig:
    global _RULE_CONFIG
    _RULE_CONFIG = new_config
    return _RULE_CONFIG


def evaluate_rules(txn: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates a transaction against all active business rules.
    Returns list of triggered rules, total rule score, and highest severity.
    """
    config = get_rule_config()
    triggered_rules: List[Dict[str, Any]] = []

    sender_bank = str(txn.get("sender_bank", "")).upper()
    receiver_bank = str(txn.get("receiver_bank", "")).upper()
    amount = float(txn.get("transaction_amount", 0.0))
    txns_24h = int(txn.get("transactions_last_24h", 0))

    is_new = bool(txn.get("is_new_beneficiary", False))
    loc_mismatch = bool(txn.get("location_mismatch_flag", False))
    device_change = bool(txn.get("device_change_flag", False))

    # 1. Blacklist Check (Severity: CRITICAL, Score: 0.99)
    blacklisted = [b.upper() for b in config.blacklisted_banks]
    if sender_bank in blacklisted or receiver_bank in blacklisted:
        triggered_rules.append({
            "rule_id": "RULE_001_BLACKLIST",
            "rule_name": "Blacklisted Bank / Entity",
            "severity": "CRITICAL",
            "score": 0.99,
            "description": f"Transaction involves blacklisted entity ({sender_bank} -> {receiver_bank}).",
        })

    # 2. Extreme Velocity Check (Severity: HIGH, Score: 0.85)
    if txns_24h >= config.max_24h_velocity:
        triggered_rules.append({
            "rule_id": "RULE_002_VELOCITY_BREACH",
            "rule_name": "24h Velocity Limit Exceeded",
            "severity": "HIGH",
            "score": 0.85,
            "description": f"Transaction frequency ({txns_24h} in 24h) exceeds policy limit of {config.max_24h_velocity}.",
        })

    # 3. High Value Cap Check (Severity: MEDIUM, Score: 0.65)
    if amount >= config.max_single_amount:
        triggered_rules.append({
            "rule_id": "RULE_003_AMOUNT_CAP",
            "rule_name": "High Value Transaction Cap",
            "severity": "MEDIUM",
            "score": 0.65,
            "description": f"Transaction amount (₹{amount:,.2f}) exceeds single transfer cap of ₹{config.max_single_amount:,.2f}.",
        })

    # 4. Multi-Flag Security Alert (Severity: HIGH, Score: 0.90)
    if config.enable_multi_flag_rule and (is_new and loc_mismatch and device_change):
        triggered_rules.append({
            "rule_id": "RULE_004_TRIPLE_FLAG",
            "rule_name": "Triple Security Discrepancy",
            "severity": "HIGH",
            "score": 0.90,
            "description": "Simultaneous new beneficiary, location mismatch, and device change detected.",
        })

    if not triggered_rules:
        rule_risk_score = 0.0
        max_severity = "NONE"
    else:
        rule_risk_score = max(r["score"] for r in triggered_rules)
        severities = [r["severity"] for r in triggered_rules]
        if "CRITICAL" in severities:
            max_severity = "CRITICAL"
        elif "HIGH" in severities:
            max_severity = "HIGH"
        else:
            max_severity = "MEDIUM"

    return {
        "triggered_rules": triggered_rules,
        "rule_risk_score": round(rule_risk_score, 4),
        "max_severity": max_severity,
        "total_rules_checked": 4,
    }
