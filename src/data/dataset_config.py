

from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    name: str
    path: str
    target_column: str
    fraud_labels: set          # values in target_column meaning "fraud"
    id_columns: list = field(default_factory=list)      # dropped entirely
    timestamp_column: str | None = None                  # full datetime column
    seconds_column: str | None = None                    # OR: seconds-since-start column
    amount_column: str | None = None
    categorical_columns: list = field(default_factory=list)
    flag_columns: list = field(default_factory=list)     # pre-existing binary 0/1 columns
    numeric_columns: list = field(default_factory=list)  # any other raw numeric columns
    velocity_column: str | None = None                   # pre-existing count/frequency column


UPI_CONFIG = DatasetConfig(
    name="UPI Transactions (synthetic)",
    path="data/raw/upi_transactions.csv",
    target_column="transaction_status",
    fraud_labels={"Fraud"},
    id_columns=["transaction_id", "sender_account_name", "receiver_account_name"],
    timestamp_column="timestamp",
    amount_column="transaction_amount",
    categorical_columns=["sender_bank", "receiver_bank", "transaction_location", "device_type"],
    flag_columns=["is_new_beneficiary", "location_mismatch_flag", "device_change_flag"],
    numeric_columns=[],
    velocity_column="transactions_last_24h",
)

CREDITCARD_CONFIG = DatasetConfig(
    name="ULB Credit Card Fraud Detection (real data)",
    path="data/external/creditcard.csv",
    target_column="Class",
    fraud_labels={1},
    id_columns=[],
    timestamp_column=None,
    seconds_column="Time",   # seconds since first transaction; used to derive hour-of-day
    amount_column="Amount",
    categorical_columns=[],
    flag_columns=[],
    numeric_columns=[f"V{i}" for i in range(1, 29)],  # PCA-anonymized features V1-V28
    velocity_column=None,
)

DATASET_REGISTRY = {
    "upi": UPI_CONFIG,
    "creditcard": CREDITCARD_CONFIG,
}
