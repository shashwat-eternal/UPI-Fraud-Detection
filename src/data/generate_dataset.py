

import numpy as np
import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta

# ---- Config ----
N_RECORDS = 150_000
FRAUD_RATE = 0.08          # ~8% fraud, realistic class imbalance (paper balances this later via SMOTE)
RANDOM_SEED = 42
OUTPUT_PATH = "data/raw/upi_transactions.csv"

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
fake = Faker("en_IN")
Faker.seed(RANDOM_SEED)

BANKS = ["SBI", "HDFC", "ICICI", "Axis", "PNB", "Kotak", "BOB", "Canara", "Union Bank", "IDFC First"]
DEVICE_TYPES = ["Android", "iOS", "Web"]
LOCATIONS = [
    "Lucknow", "Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Chennai",
    "Kolkata", "Pune", "Jaipur", "Patna", "Bhopal", "Kanpur", "Rural-UP",
    "Rural-Bihar", "Rural-MP", "Rural-Rajasthan"
]

START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 12, 31)


def random_timestamp(is_fraud: bool) -> datetime:
    """Fraudulent transactions skew toward late-night hours."""
    delta_days = (END_DATE - START_DATE).days
    day_offset = random.randint(0, delta_days)
    base_date = START_DATE + timedelta(days=day_offset)

    if is_fraud and random.random() < 0.55:
        hour = random.choice([0, 1, 2, 3, 4, 23])
    else:
        hour = int(np.clip(np.random.normal(loc=14, scale=5), 0, 23))

    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return base_date.replace(hour=hour, minute=minute, second=second)


def random_amount(is_fraud: bool) -> float:
    """Fraud amounts are bimodal: very small test transactions or unusually large ones."""
    if is_fraud:
        if random.random() < 0.4:
            amt = np.random.uniform(1, 50)          # small "testing" transactions
        else:
            amt = np.random.lognormal(mean=9.5, sigma=0.9)  # large unusual transfers
    else:
        amt = np.random.lognormal(mean=6.5, sigma=1.0)      # typical everyday spend

    return round(float(np.clip(amt, 1, 200000)), 2)


def account_name_initials(name: str) -> str:
    parts = name.split()
    return ".".join([p[0].upper() for p in parts]) + "."


def generate_row(i: int) -> dict:
    is_fraud = random.random() < FRAUD_RATE

    sender_name = fake.name()
    receiver_name = fake.name()

    is_new_beneficiary = random.random() < (0.65 if is_fraud else 0.12)
    location_mismatch = random.random() < (0.5 if is_fraud else 0.05)
    device_change = random.random() < (0.45 if is_fraud else 0.06)
    txns_last_24h = np.random.poisson(lam=6 if is_fraud else 2)

    row = {
        "transaction_id": f"TXN{100000 + i}",
        "sender_account_name": account_name_initials(sender_name),
        "receiver_account_name": account_name_initials(receiver_name),
        "sender_bank": random.choice(BANKS),
        "receiver_bank": random.choice(BANKS),
        "transaction_amount": random_amount(is_fraud),
        "timestamp": random_timestamp(is_fraud).isoformat(),
        "transaction_location": random.choice(LOCATIONS),
        "device_type": random.choice(DEVICE_TYPES),
        "is_new_beneficiary": int(is_new_beneficiary),
        "location_mismatch_flag": int(location_mismatch),
        "device_change_flag": int(device_change),
        "transactions_last_24h": int(txns_last_24h),
        "transaction_status": "Fraud" if is_fraud else "No Fraud",
    }
    return row


def main():
    print(f"Generating {N_RECORDS:,} synthetic UPI transactions "
          f"(~{FRAUD_RATE*100:.0f}% fraud rate)...")

    rows = [generate_row(i) for i in range(N_RECORDS)]
    df = pd.DataFrame(rows)

    # Shuffle so fraud/no-fraud rows aren't in generation order
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved to {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print(df["transaction_status"].value_counts(normalize=True).round(4))


if __name__ == "__main__":
    main()
