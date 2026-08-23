"""
Audit Logger — AI Payment Recovery Agent

Append-only structured audit trail. Every AI decision, policy check, and
action gets logged here so that for any payment, you can reconstruct:
  - what happened
  - why the AI made the decision it did
  - what action was taken
  - who/what approved it
  - what the result was

This directly implements section 12 (Auditability) from the buildathon brief.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path(__file__).parent.parent / "docs" / "audit_log.jsonl"


def log_event(payment_id: str, event: str, actor: str, details: dict | None = None):
    """
    Append one audit event. Kept deliberately simple (JSONL, one event per
    line) so it's trivial to read, grep, or load into the dashboard later --
    no database dependency needed to demo auditability.
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payment_id": payment_id,
        "event": event,
        "actor": actor,  # "ai_agent" | "policy_engine" | "human" | "system"
        "details": details or {},
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def read_audit_trail(payment_id: str | None = None):
    """Read back the audit log, optionally filtered to one payment."""
    if not LOG_PATH.exists():
        return []
    entries = []
    with open(LOG_PATH) as f:
        for line in f:
            entry = json.loads(line)
            if payment_id is None or entry["payment_id"] == payment_id:
                entries.append(entry)
    return entries


if __name__ == "__main__":
    # quick smoke test
    log_event("PAY00001", "Payment failed", "system", {"amount_paise": 299900})
    log_event("PAY00001", "AI classified failure", "ai_agent",
               {"category": "bank_declined", "confidence": 0.94})
    log_event("PAY00001", "Recovery strategy selected", "ai_agent",
               {"strategy": "retry_delayed", "reasoning": "example reasoning"})
    trail = read_audit_trail("PAY00001")
    print(f"Logged {len(trail)} events for PAY00001:")
    for e in trail:
        print(f"  {e['timestamp']}  [{e['actor']}]  {e['event']}")
