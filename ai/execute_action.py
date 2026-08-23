"""
Execute Action — AI Payment Recovery Agent

Takes a strategy decision (from strategy_selector.py) and actually
executes it:
  - retry_now / retry_delayed / notify_customer -> create a Razorpay
    Payment Link in test mode and log it
  - escalate_to_human -> no Razorpay action, just logged for human pickup
  - if requires_human_approval is True -> the action is NOT auto-executed;
    it's logged as "pending approval" instead (this is the gate actually
    holding, not just being computed and ignored)

Reliability: if the Razorpay API call fails for any reason (bad keys,
network, rate limit), this catches it, logs a failed action to the audit
trail, and returns a clear failure result -- it does not crash the batch
run or silently pretend the action succeeded.
"""

from razorpay_client import create_recovery_payment_link
from audit_logger import log_event

ACTIONS_REQUIRING_PAYMENT_LINK = {"retry_now", "retry_delayed", "notify_customer"}


def execute_action(payment_context: dict, decision: dict) -> dict:
    """
    payment_context: dict with payment_id, amount_paise, customer_name,
                      customer_contact
    decision: the dict returned by strategy_selector.select_strategy()
              (has 'strategy', 'reasoning', 'source', 'policy')

    Returns a result dict describing what actually happened.
    """
    payment_id = payment_context["payment_id"]
    strategy = decision["strategy"]
    policy = decision["policy"]

    # Gate check: if human approval is required, do NOT auto-execute.
    if policy.requires_human_approval and not policy.hard_stop:
        result = {
            "payment_id": payment_id,
            "action_type": "pending_human_approval",
            "result": "pending",
            "detail": f"Strategy '{strategy}' selected but requires human "
                      f"approval before execution: {policy.reason}",
        }
        log_event(payment_id, "Action pending human approval", "system", result)
        return result

    # Hard stop / escalate -> no automated Razorpay action
    if strategy == "escalate_to_human" or policy.hard_stop:
        result = {
            "payment_id": payment_id,
            "action_type": "escalated",
            "result": "escalated",
            "detail": decision["reasoning"],
        }
        log_event(payment_id, "Escalated to human, no automated action taken", "system", result)
        return result

    # Automated action: create a recovery payment link
    if strategy in ACTIONS_REQUIRING_PAYMENT_LINK:
        try:
            link = create_recovery_payment_link(
                amount_paise=payment_context["amount_paise"],
                customer_name=payment_context.get("customer_name", "Customer"),
                customer_contact=payment_context.get("customer_contact", "+910000000000"),
                description=f"Payment recovery ({strategy}) for {payment_id}",
                reference_id=payment_id,
            )
            result = {
                "payment_id": payment_id,
                "action_type": strategy,
                "result": "link_created",
                "razorpay_link_id": link["link_id"],
                "razorpay_short_url": link["short_url"],
                "razorpay_status": link["status"],
            }
            log_event(payment_id, "Razorpay recovery action executed", "ai_agent", result)
            return result
        except Exception as e:
            # Graceful failure: log it, don't crash the batch
            result = {
                "payment_id": payment_id,
                "action_type": strategy,
                "result": "failed",
                "error": str(e),
            }
            log_event(payment_id, "Razorpay action failed", "system", result)
            return result

    # Fallback for any unexpected strategy value
    result = {
        "payment_id": payment_id,
        "action_type": strategy,
        "result": "no_action_defined",
        "detail": f"No execution handler defined for strategy '{strategy}'",
    }
    log_event(payment_id, "No action handler for strategy", "system", result)
    return result


if __name__ == "__main__":
    from strategy_selector import select_strategy

    # Case 1: small amount, should attempt to execute (will fail gracefully
    # if no real Razorpay keys are configured -- that's expected and fine,
    # it proves the error handling works)
    example = {
        "payment_id": "PAY00001",
        "amount_paise": 49900,
        "attempt_number": 1,
        "predicted_failure_category": "network_error",
        "classifier_confidence": 0.9,
        "recovery_probability": 0.85,
        "past_success_rate": 0.9,
        "past_failure_count": 0,
        "preferred_channel": "sms",
        "customer_name": "Test Customer",
        "customer_contact": "+919876543210",
    }
    decision = select_strategy(example)
    print(f"Decision: {decision['strategy']} (approval required: {decision['policy'].requires_human_approval})")
    result = execute_action(example, decision)
    print(f"Execution result: {result}")

    # Case 2: large amount, should be gated (pending approval, no auto-action)
    print()
    example2 = dict(example, payment_id="PAY00002", amount_paise=1999900)
    decision2 = select_strategy(example2)
    print(f"Decision: {decision2['strategy']} (approval required: {decision2['policy'].requires_human_approval})")
    result2 = execute_action(example2, decision2)
    print(f"Execution result: {result2}")
