import asyncio
import random
import traceback
from datetime import datetime, timezone

from app.predict import predict_transaction

# Optional logging
try:
    from app.db import log_prediction
    DB_AVAILABLE = True
except Exception:
    DB_AVAILABLE = False


BANKS = [
    "SBI",
    "HDFC",
    "ICICI",
    "Axis",
    "PNB",
    "Kotak",
    "BOB",
    "Canara",
    "Union Bank",
    "IDFC First",
]

DEVICE_TYPES = [
    "Android",
    "iOS",
    "Web",
]

LOCATIONS = [
    "Lucknow",
    "Delhi",
    "Mumbai",
    "Bengaluru",
    "Hyderabad",
    "Chennai",
    "Kolkata",
    "Pune",
    "Jaipur",
    "Patna",
    "Bhopal",
    "Kanpur",
    "Rural-UP",
    "Rural-Bihar",
    "Rural-MP",
    "Rural-Rajasthan",
]


def _generate_transaction(scenario: str | None = None, fraud_bias: float = 0.12):
    """Generate a transaction based on regular random distribution or an attack scenario."""

    if scenario == "sim_swap":
        sender = random.choice(BANKS)
        receiver = random.choice(BANKS)
        amount = round(random.uniform(45000, 95000), 2)
        hour = random.choice([1, 2, 3])
        new_beneficiary = True
        location_mismatch = True
        device_change = True
        txns = random.randint(6, 12)
    elif scenario == "micro_probe":
        sender = random.choice(BANKS)
        receiver = random.choice(BANKS)
        amount = round(random.uniform(1.0, 15.0), 2)
        hour = random.randint(0, 23)
        new_beneficiary = False
        location_mismatch = False
        device_change = False
        txns = random.randint(9, 16)
    elif scenario == "blacklist":
        sender = "SUSPICIOUS_BANK_TEST"
        receiver = random.choice(BANKS)
        amount = round(random.uniform(1000, 25000), 2)
        hour = random.randint(8, 22)
        new_beneficiary = True
        location_mismatch = False
        device_change = False
        txns = random.randint(1, 4)
    elif scenario == "device_hijack":
        sender = random.choice(BANKS)
        receiver = random.choice(BANKS)
        amount = round(random.uniform(8000, 35000), 2)
        hour = random.randint(0, 23)
        new_beneficiary = True
        location_mismatch = True
        device_change = True
        txns = random.randint(4, 8)
    else:
        suspicious = random.random() < fraud_bias
        if suspicious:
            amount = random.choice([
                round(random.uniform(1, 40), 2),
                round(random.uniform(20000, 90000), 2)
            ])
            hour = random.choice([0, 1, 2, 3, 4, 23])
            new_beneficiary = random.random() < 0.7
            location_mismatch = random.random() < 0.6
            device_change = random.random() < 0.5
            txns = random.randint(5, 12)
        else:
            amount = round(random.uniform(50, 5000), 2)
            hour = random.randint(7, 22)
            new_beneficiary = random.random() < 0.1
            location_mismatch = random.random() < 0.05
            device_change = random.random() < 0.05
            txns = random.randint(0, 4)
        sender = random.choice(BANKS)
        receiver = random.choice(BANKS)

    now = datetime.now(timezone.utc).replace(
        hour=hour,
        minute=random.randint(0, 59),
        second=random.randint(0, 59),
        microsecond=0,
    )

    return {
        "sender_bank": sender,
        "receiver_bank": receiver,
        "transaction_amount": amount,
        "timestamp": now.isoformat(),
        "transaction_location": random.choice(LOCATIONS),
        "device_type": random.choice(DEVICE_TYPES),
        "is_new_beneficiary": new_beneficiary,
        "location_mismatch_flag": location_mismatch,
        "device_change_flag": device_change,
        "transactions_last_24h": txns,
    }


async def transaction_stream(interval_seconds: float = 1.0, scenario: str | None = None):
    """
    Infinite async generator for the Live Dashboard.
    """

    transaction_id = 1

    while True:

        try:

            txn = _generate_transaction(scenario=scenario)


            result = predict_transaction(txn)

            # Save to DB if available
            if DB_AVAILABLE:
                try:
                    log_prediction(
                        txn,
                        result,
                        source="stream"
                    )
                except Exception as db_error:
                    print("\nDatabase logging failed:")
                    print(db_error)

            record = {
                "id": transaction_id,
                "transaction": txn,
                "prediction": result.get("prediction", "Unknown"),
                "fraud_probability": float(
                    result.get("fraud_probability", 0)
                ),
                "message": result.get(
                    "message",
                    ""
                ),
            }

            transaction_id += 1

            yield record

        except Exception as e:

            print("\n==============================")
            print("STREAM ERROR")
            print("==============================")
            traceback.print_exc()
            print("==============================\n")

        await asyncio.sleep(interval_seconds)