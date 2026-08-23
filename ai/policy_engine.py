"""
Policy Engine — AI Payment Recovery Agent

Defines hard, non-negotiable limits on what the AI strategy selector is
allowed to decide autonomously. This is deliberately separate from the AI
reasoning step: the AI can recommend anything, but the policy engine has
final say on whether that action can execute automatically or needs a
human approval gate.

This directly implements the "Explainable / Bounded / Gated" requirement
from the buildathon brief (section 4, "Critical Requirement").
"""

from dataclasses import dataclass

# --- Hard limits (tune these, but they exist for a reason: no single
#     automated action should be able to cause runaway cost or annoy a
#     customer into churning) ---
AUTO_ACTION_MAX_AMOUNT_PAISE = 500_000          # ₹5,000 -- above this, human approval required
MAX_RETRY_ATTEMPTS = 3                          # stop retrying after this many attempts
MIN_CONFIDENCE_FOR_AUTO_ACTION = 0.55           # below this classifier confidence, escalate
LOW_RECOVERY_PROBABILITY_THRESHOLD = 0.25       # below this, don't bother auto-retrying


@dataclass
class PolicyDecision:
    allowed_actions: list          # which strategies the AI is permitted to choose from
    requires_human_approval: bool
    hard_stop: bool                # true if we should NOT attempt any further recovery
    reason: str                    # human-readable explanation, goes in the audit log


def evaluate_policy(
    amount_paise: int,
    attempt_number: int,
    classifier_confidence: float,
    recovery_probability: float,
) -> PolicyDecision:
    """
    Determines which recovery strategies are permitted for this payment,
    BEFORE the AI strategy selector is even asked to reason about it.
    The AI chooses among allowed_actions only -- it cannot override a
    hard_stop or bypass the approval gate.
    """

    # Hard stop: too many attempts already, don't keep hammering the customer
    if attempt_number > MAX_RETRY_ATTEMPTS:
        return PolicyDecision(
            allowed_actions=["escalate_to_human"],
            requires_human_approval=True,
            hard_stop=True,
            reason=f"Attempt {attempt_number} exceeds max retry limit "
                   f"({MAX_RETRY_ATTEMPTS}). Stopping automated recovery, "
                   f"escalating to human review.",
        )

    # Hard stop: recovery probability too low to justify bothering the customer again
    if recovery_probability < LOW_RECOVERY_PROBABILITY_THRESHOLD and attempt_number > 1:
        return PolicyDecision(
            allowed_actions=["notify_customer", "escalate_to_human"],
            requires_human_approval=False,
            hard_stop=False,
            reason=f"Recovery probability ({recovery_probability:.2f}) is below "
                   f"threshold ({LOW_RECOVERY_PROBABILITY_THRESHOLD}) on a repeat "
                   f"attempt. Automated retry not permitted; notify or escalate only.",
        )

    # Approval gate: amount too large for fully automatic action
    requires_approval = amount_paise > AUTO_ACTION_MAX_AMOUNT_PAISE

    # Approval gate: classifier isn't confident enough to trust blindly
    low_confidence = classifier_confidence < MIN_CONFIDENCE_FOR_AUTO_ACTION
    requires_approval = requires_approval or low_confidence

    allowed = ["retry_now", "retry_delayed", "notify_customer", "escalate_to_human"]

    reasons = []
    if amount_paise > AUTO_ACTION_MAX_AMOUNT_PAISE:
        reasons.append(
            f"amount ₹{amount_paise/100:.0f} exceeds auto-action limit "
            f"₹{AUTO_ACTION_MAX_AMOUNT_PAISE/100:.0f}"
        )
    if low_confidence:
        reasons.append(
            f"classifier confidence ({classifier_confidence:.2f}) below "
            f"threshold ({MIN_CONFIDENCE_FOR_AUTO_ACTION})"
        )

    reason_text = (
        "Requires human approval: " + "; ".join(reasons)
        if requires_approval
        else "Within policy limits -- eligible for fully automated action."
    )

    return PolicyDecision(
        allowed_actions=allowed,
        requires_human_approval=requires_approval,
        hard_stop=False,
        reason=reason_text,
    )
