
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    sender_bank: str = Field(..., examples=["HDFC"])
    receiver_bank: str = Field(..., examples=["SBI"])
    transaction_amount: float = Field(..., gt=0, examples=[1500.0])
    timestamp: datetime = Field(..., examples=["2026-07-18T23:45:00"])
    transaction_location: str = Field(..., examples=["Lucknow"])
    device_type: Literal["Android", "iOS", "Web"] = Field(..., examples=["Android"])
    is_new_beneficiary: bool = Field(..., examples=[True])
    location_mismatch_flag: bool = Field(..., examples=[False])
    device_change_flag: bool = Field(..., examples=[False])
    transactions_last_24h: int = Field(..., ge=0, examples=[2])


class RiskFactor(BaseModel):
    feature_name: str
    impact_score: float
    reason_code: str
    category: Literal["risk", "trust"]


class ExplanationResponse(BaseModel):
    risk_drivers: list[RiskFactor] = []
    trust_factors: list[RiskFactor] = []
    summary: str


class RuleResult(BaseModel):
    rule_id: str
    rule_name: str
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "NONE"]
    score: float
    description: str


class RuleSummary(BaseModel):
    triggered_rules: list[RuleResult] = []
    rule_risk_score: float
    max_severity: str
    total_rules_checked: int = 4


class RuleConfigSchema(BaseModel):
    max_24h_velocity: int = Field(8, ge=1)
    max_single_amount: float = Field(50000.0, gt=0)
    blacklisted_banks: list[str] = Field(default_factory=lambda: ["SUSPICIOUS_BANK_TEST", "FRAUD_BANK_X"])
    enable_multi_flag_rule: bool = True


class PredictionResponse(BaseModel):
    prediction: Literal["Fraud", "No Fraud"]
    fraud_probability: float
    message: str
    model_used: str = "RandomForest"
    composite_risk_score: float | None = None
    rule_summary: RuleSummary | None = None
    explanation: ExplanationResponse | None = None


