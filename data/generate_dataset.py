"""
Synthetic dataset generator for AI Payment Recovery Agent.

Generates:
  - payments.csv         : failed payment events (with realistic noisy failure reasons)
  - customer_history.csv : per-customer features used by the recovery scorer

Design notes:
  - failure_reason_raw simulates messy real-world bank/gateway error strings.
    The AI classifier's job is to map these to a clean category. Some are
    ambiguous/ noisy on purpose so accuracy isn't trivially 100%.
  - "ground_truth_recovered" simulates what actually happened after the
    failure (for offline evaluation only -- your AI does NOT see this column
    when making decisions, it's used purely to score your model afterward).
"""

import csv
import random
from datetime import datetime, timedelta

random.seed(42)

N_CUSTOMERS = 250
N_PAYMENTS = 600

FAILURE_TEMPLATES = {
    "insufficient_funds": [
        "Insufficient balance in account",
        "INSUFFICIENT_FUNDS",
        "Txn declined - low balance",
        "Your account has insufficient funds for this transaction",
    ],
    "card_expired": [
        "Card has expired",
        "EXPIRED_CARD",
        "This card is no longer valid",
    ],
    "bank_declined": [
        "Declined by issuing bank",
        "DO_NOT_HONOR",
        "Bank declined the transaction. Contact your bank.",
        "Issuer declined transaction (generic decline)",
    ],
    "network_error": [
        "Gateway timeout",
        "Network error, please retry",
        "TIMEOUT_ERROR - no response from processor",
    ],
    "invalid_cvv": [
        "Incorrect CVV entered",
        "CVV verification failed",
    ],
    "other": [
        "Transaction failed",
        "Payment could not be processed",
        "Unknown error occurred",
    ],
}

# Recovery likelihood varies meaningfully by failure type -- this is the
# signal your model should learn to pick up on.
RECOVERY_BASE_RATE = {
    "insufficient_funds": 0.55,   # often fixed by payday, retry works sometimes
    "card_expired": 0.15,         # needs new card, rarely self-resolves
    "bank_declined": 0.35,
    "network_error": 0.85,        # usually transient, retry works well
    "invalid_cvv": 0.60,          # user re-enters correctly
    "other": 0.30,
}

CHANNELS = ["email", "sms", "whatsapp"]


def gen_customers(n):
    customers = []
    for i in range(n):
        cid = f"CUST{i+1:04d}"
        total_past = random.randint(1, 60)
        past_failures = random.randint(0, min(10, total_past))
        successful = total_past - past_failures
        customers.append({
            "customer_id": cid,
            "total_past_payments": total_past,
            "successful_payments": successful,
            "past_failure_count": past_failures,
            "avg_days_to_recovery": round(random.uniform(0.2, 6.0), 1) if past_failures > 0 else 0.0,
            "preferred_channel": random.choice(CHANNELS),
            "account_age_days": random.randint(10, 1200),
        })
    return customers


def gen_payments(customers, n):
    payments = []
    base_date = datetime(2026, 8, 1)
    for i in range(n):
        cust = random.choice(customers)
        true_category = random.choices(
            list(FAILURE_TEMPLATES.keys()),
            weights=[0.30, 0.12, 0.25, 0.18, 0.08, 0.07],
        )[0]
        raw_message = random.choice(FAILURE_TEMPLATES[true_category])

        # add label noise: 8% chance the raw message is a vague "other"-style
        # string even for a real category, to simulate messy gateway logs
        if random.random() < 0.08:
            raw_message = random.choice(FAILURE_TEMPLATES["other"])

        amount_paise = random.choice([49900, 99900, 149900, 299900, 499900, 999900, 1999900])
        failed_at = base_date + timedelta(
            days=random.randint(0, 20), hours=random.randint(0, 23), minutes=random.randint(0, 59)
        )

        # ground truth recovery outcome, influenced by failure type + customer history
        base_rate = RECOVERY_BASE_RATE[true_category]
        history_bonus = 0.15 if cust["successful_payments"] > cust["past_failure_count"] * 3 else 0.0
        recovered = random.random() < min(0.95, base_rate + history_bonus)

        payments.append({
            "payment_id": f"PAY{i+1:05d}",
            "customer_id": cust["customer_id"],
            "amount_paise": amount_paise,
            "currency": "INR",
            "failure_reason_raw": raw_message,
            "true_failure_category": true_category,  # for eval only
            "failed_at": failed_at.isoformat(),
            "subscription_id": f"SUB{random.randint(1,150):04d}" if random.random() < 0.6 else "",
            "attempt_number": 1,
            "ground_truth_recovered": recovered,  # for eval only, not model input
        })
    return payments


def main():
    customers = gen_customers(N_CUSTOMERS)
    payments = gen_payments(customers, N_PAYMENTS)

    with open("customer_history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(customers[0].keys()))
        writer.writeheader()
        writer.writerows(customers)

    with open("payments.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(payments[0].keys()))
        writer.writeheader()
        writer.writerows(payments)

    print(f"Generated {len(customers)} customers -> customer_history.csv")
    print(f"Generated {len(payments)} payments -> payments.csv")
    print("\nNOTE: 'true_failure_category' and 'ground_truth_recovered' are")
    print("evaluation-only columns. Your classifier/scorer should NOT read")
    print("these as input -- use them only to compute accuracy/precision/recall.")


if __name__ == "__main__":
    main()
