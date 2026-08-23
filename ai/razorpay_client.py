"""
Razorpay Client — AI Payment Recovery Agent

Thin wrapper around the Razorpay Python SDK, scoped to exactly what the
recovery agent needs: creating a Payment Link (the recovery action) and
checking its status (verification).

Why Payment Links specifically: creating one is a genuine server-side API
call needing no customer interaction, so it's fully automatable -- but
*completing* one requires the customer to actually pay (OTP/3-D Secure
etc.), which cannot and should not be scripted server-side. This mirrors
real-world payment security constraints rather than faking a full auto-pay
loop. It's also literally the right tool for recovery: send a link via
SMS/email/WhatsApp, customer taps it, pays, done.

Requires RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET in .env (test mode keys).
"""

import os
import razorpay
from dotenv import load_dotenv

load_dotenv()

KEY_ID = os.getenv("RAZORPAY_KEY_ID")
KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")


def get_client():
    if not KEY_ID or not KEY_SECRET:
        raise RuntimeError(
            "RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET not set in .env. "
            "Add your test-mode keys to run real Razorpay calls."
        )
    return razorpay.Client(auth=(KEY_ID, KEY_SECRET))


def create_recovery_payment_link(
    amount_paise: int,
    customer_name: str,
    customer_contact: str,
    description: str,
    reference_id: str,
) -> dict:
    """
    Creates a Razorpay Payment Link in test mode -- the bounded action
    executed for retry_now / retry_delayed / notify_customer strategies.

    Returns a dict with the link id, short_url, and status. Raises on
    hard API failure -- caller (execute_action.py) is responsible for
    catching this and logging a failed action to the audit trail rather
    than crashing the pipeline.
    """
    client = get_client()
    payload = {
        "amount": amount_paise,
        "currency": "INR",
        "description": description,
        "customer": {
            "name": customer_name,
            "contact": customer_contact,
        },
        "notify": {"sms": True, "email": False},
        "reminder_enable": True,
        "reference_id": reference_id,
    }
    link = client.payment_link.create(payload)
    return {
        "link_id": link["id"],
        "short_url": link["short_url"],
        "status": link["status"],  # created / paid / expired / cancelled
    }


def fetch_payment_link_status(link_id: str) -> dict:
    """Verification step: check whether a previously-created link has been paid."""
    client = get_client()
    link = client.payment_link.fetch(link_id)
    return {
        "link_id": link["id"],
        "status": link["status"],
        "amount_paid": link.get("amount_paid", 0),
    }


if __name__ == "__main__":
    if not KEY_ID or not KEY_SECRET:
        print("No Razorpay test keys found in .env -- add RAZORPAY_KEY_ID and")
        print("RAZORPAY_KEY_SECRET (from your Razorpay Dashboard test mode) to test this.")
    else:
        print("Creating a test recovery payment link...")
        result = create_recovery_payment_link(
            amount_paise=49900,
            customer_name="Test Customer",
            customer_contact="+919999999999",
            description="Payment recovery test -- Buildathon demo",
            reference_id="TEST-PAY-001",
        )
        print(result)
        print("\nFetching status back...")
        status = fetch_payment_link_status(result["link_id"])
        print(status)
