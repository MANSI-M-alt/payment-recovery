"""
Strategy Selector — AI Payment Recovery Agent

This is the genuine "AI reasoning -> decision" step in the pipeline (as
opposed to the classifier/scorer, which are prediction models). Given:
  - the failure classification + confidence
  - the recovery probability score
  - customer history
  - the policy engine's allowed_actions for this payment

...it uses an LLM to reason over these signals and pick ONE strategy,
with a short explanation. Critically: the LLM can only choose from
policy.allowed_actions -- it cannot invent an action outside what the
policy engine has already bounded. This keeps the "AI decides" step
genuinely bounded rather than trusting the LLM blindly.

Reliability: if the API call fails (no key, network issue, malformed
response), this falls back to a deterministic rule-based decision rather
than crashing the pipeline -- an unhandled LLM failure should never block
a financial workflow.

Requires ANTHROPIC_API_KEY in your .env file. Without it, runs in
fallback-only mode (still fully functional, just rule-based instead of
LLM-reasoned) -- useful for testing without burning API credits.
"""

import os
import json
import requests
from dotenv import load_dotenv

from policy_engine import evaluate_policy, PolicyDecision
from audit_logger import log_event

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-5"


STRATEGY_DESCRIPTIONS = {
    "retry_now": "Immediately retry the payment. Best for transient failures on customers with strong payment history.",
    "retry_delayed": "Wait and retry later (e.g. next day). Best when the failure may resolve itself with time (e.g. insufficient funds, likely payday-related).",
    "notify_customer": "Send the customer a message asking them to update payment info or retry manually. Best for issues the customer needs to fix themselves (e.g. expired card).",
    "escalate_to_human": "Hand off to a human agent. Best for low-confidence classifications, large amounts, or repeated failures.",
}


def build_prompt(payment_context: dict, policy: PolicyDecision) -> str:
    allowed = policy.allowed_actions
    options_text = "\n".join(
        f"- {a}: {STRATEGY_DESCRIPTIONS[a]}" for a in allowed
    )
    return f"""You are a payment recovery decision assistant for an Indian payments company. A customer's payment failed. Based on the signals below, choose exactly ONE recovery strategy from the allowed options, and explain your reasoning in 1-2 sentences.

Payment context:
- Amount: ₹{payment_context['amount_paise'] / 100:.0f}
- Attempt number: {payment_context['attempt_number']}
- Predicted failure category: {payment_context['predicted_failure_category']} (classifier confidence: {payment_context['classifier_confidence']:.2f})
- Recovery probability (model score): {payment_context['recovery_probability']:.2f}
- Customer's past success rate: {payment_context['past_success_rate']:.2f}
- Customer's past failure count: {payment_context['past_failure_count']}
- Customer's preferred contact channel: {payment_context['preferred_channel']}

Allowed strategies for this payment (already bounded by policy -- do not suggest anything outside this list):
{options_text}

Respond ONLY in this exact JSON format, nothing else:
{{"strategy": "<one of the allowed strategy names exactly as written>", "reasoning": "<1-2 sentence explanation>"}}"""


def fallback_decision(payment_context: dict, policy: PolicyDecision) -> dict:
    """Deterministic rule-based fallback if the LLM call fails or no API key is set."""
    allowed = policy.allowed_actions
    prob = payment_context["recovery_probability"]

    if "retry_now" in allowed and prob >= 0.7:
        strategy = "retry_now"
        reasoning = f"Fallback rule: recovery probability {prob:.2f} is high, retrying immediately."
    elif "retry_delayed" in allowed and prob >= 0.4:
        strategy = "retry_delayed"
        reasoning = f"Fallback rule: recovery probability {prob:.2f} is moderate, delaying retry."
    elif "notify_customer" in allowed:
        strategy = "notify_customer"
        reasoning = f"Fallback rule: recovery probability {prob:.2f} is low, notifying customer instead of auto-retrying."
    else:
        strategy = "escalate_to_human"
        reasoning = "Fallback rule: no safe automated option applies, escalating."

    return {"strategy": strategy, "reasoning": reasoning, "source": "fallback_rules"}


def select_strategy(payment_context: dict) -> dict:
    """
    Main entry point. Returns:
        {
          "strategy": str,
          "reasoning": str,
          "source": "llm" | "fallback_rules",
          "policy": PolicyDecision,
        }
    Also writes audit log entries for the policy check and the decision.
    """
    payment_id = payment_context["payment_id"]

    policy = evaluate_policy(
        amount_paise=payment_context["amount_paise"],
        attempt_number=payment_context["attempt_number"],
        classifier_confidence=payment_context["classifier_confidence"],
        recovery_probability=payment_context["recovery_probability"],
    )
    log_event(payment_id, "Policy evaluated", "policy_engine", {
        "allowed_actions": policy.allowed_actions,
        "requires_human_approval": policy.requires_human_approval,
        "hard_stop": policy.hard_stop,
        "reason": policy.reason,
    })

    if policy.hard_stop:
        result = {
            "strategy": policy.allowed_actions[0],
            "reasoning": policy.reason,
            "source": "policy_hard_stop",
        }
        log_event(payment_id, "Recovery stopped by policy", "policy_engine", result)
        return {**result, "policy": policy}

    if not ANTHROPIC_API_KEY:
        result = fallback_decision(payment_context, policy)
    else:
        try:
            prompt = build_prompt(payment_context, policy)
            response = requests.post(
                ANTHROPIC_API_URL,
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": MODEL,
                    "max_tokens": 200,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=15,
            )
            response.raise_for_status()
            text = response.json()["content"][0]["text"].strip()
            parsed = json.loads(text)

            if parsed["strategy"] not in policy.allowed_actions:
                raise ValueError(
                    f"LLM chose '{parsed['strategy']}' which is outside "
                    f"allowed actions {policy.allowed_actions}"
                )
            result = {**parsed, "source": "llm"}
        except Exception as e:
            log_event(payment_id, "LLM call failed, using fallback", "system", {"error": str(e)})
            result = fallback_decision(payment_context, policy)

    log_event(payment_id, "Recovery strategy selected", "ai_agent", {
        "strategy": result["strategy"],
        "reasoning": result["reasoning"],
        "source": result["source"],
        "requires_human_approval": policy.requires_human_approval,
    })

    return {**result, "policy": policy}


if __name__ == "__main__":
    example = {
        "payment_id": "PAY00001",
        "amount_paise": 299900,
        "attempt_number": 1,
        "predicted_failure_category": "bank_declined",
        "classifier_confidence": 0.94,
        "recovery_probability": 0.62,
        "past_success_rate": 0.85,
        "past_failure_count": 1,
        "preferred_channel": "sms",
    }

    if not ANTHROPIC_API_KEY:
        print("NOTE: ANTHROPIC_API_KEY not set in .env -- running in fallback-rules mode.")
        print("Add your key to .env to see real LLM reasoning.\n")

    result = select_strategy(example)
    print(f"Strategy: {result['strategy']}")
    print(f"Reasoning: {result['reasoning']}")
    print(f"Source: {result['source']}")
    print(f"Requires human approval: {result['policy'].requires_human_approval}")
