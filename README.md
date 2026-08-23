# AI Payment Recovery Agent

> Razorpay AI Buildathon — Track 03: AI Revenue Recovery

## 1. Problem
Merchants lose recurring revenue when subscription/checkout payments fail.
Most failures are never retried intelligently — retries are either blind
(retry everything, annoying customers and wasting attempts) or absent
(no retry at all, revenue is just written off).

## 2. Solution
An AI agent that, for every failed payment:
1. Classifies the likely failure reason from noisy gateway error text
2. Scores the likelihood that this customer/payment can be recovered
3. Selects a bounded recovery strategy (immediate retry / delayed retry /
   notify customer / escalate to human)
4. Checks the action against policy (amount limits, retry limits)
5. Executes the action in Razorpay test mode
6. Verifies the result and logs everything to an audit trail

## 3. Status
🚧 In progress — buildathon deadline Sept 4, 2026.

- [x] Synthetic dataset generated (`data/`)
- [x] Failure classifier (96% test accuracy — see `docs/classifier_metrics.json`)
- [x] Recovery scorer (0.74 ROC-AUC — see `docs/scorer_metrics.json`)
- [x] Strategy selector + policy gate (LLM reasoning, bounded by policy engine)
- [x] Razorpay test-mode integration (Payment Links, with graceful failure handling)
- [ ] Razorpay test-mode integration
- [ ] Backend API + audit log
- [ ] Frontend dashboard
- [ ] Batch evaluation + metrics
- [ ] Pitch video

## 4. Repo structure
```
frontend/    React dashboard
backend/     FastAPI app (routers, models, services)
ai/          Classifier, scorer, strategy-selection logic
data/        Synthetic dataset + generator script
tests/       Tests
docs/        Architecture diagrams, workflow docs
```

## 5. Dataset
See `data/generate_dataset.py`. Generates:
- `payments.csv` — 600 synthetic failed payments with noisy real-world-style
  failure messages
- `customer_history.csv` — 250 customers with payment history features

`true_failure_category` and `ground_truth_recovered` columns exist for
**evaluation only** — the AI does not see these as input.

## 6. Setup
```bash
# regenerate dataset
cd data && python3 generate_dataset.py

# backend (once built)
cd backend && pip install -r requirements.txt && uvicorn app.main:app --reload

# frontend (once built)
cd frontend && npm install && npm run dev
```

## 7. Evaluation

### Failure classifier
TF-IDF (char n-grams) + Logistic Regression, trained on `payments.csv`,
evaluated on a held-out 25% test split (150 payments).

- **Overall accuracy: 96.0%**
- Full per-class precision/recall/F1 and confusion matrix in
  `docs/classifier_metrics.json`
- Weakest class: `other` (catch-all/ambiguous messages), precision 0.67 —
  expected, since this category is intentionally noisy in the synthetic
  data. Low-confidence predictions here are a useful signal for routing
  to human review rather than acting automatically.

Recovery scorer and end-to-end batch recovery rate: TBD.

### Recovery scorer
Random Forest, trained on classifier output + customer history features
(past success rate, account age, past failure count, etc.), evaluated on
a held-out 25% test split (150 payments).

- **ROC-AUC: 0.739**
- **Precision: 0.72, Recall: 0.63, F1: 0.67** (recovered class)
- Top predictive features: customer's past success rate, account age,
  and predicted failure category — full ranking in `docs/scorer_metrics.json`
- Deliberately not near-perfect: recovery outcomes are genuinely
  probabilistic, and an AUC this high on realistic noisy data is more
  credible than a suspiciously perfect score would be

End-to-end batch recovery rate (full pipeline on 600 payments): TBD.

### Strategy selector + policy gate
The strategy selector is the LLM-reasoning step: given the classifier
output, recovery score, and customer history, it picks one recovery
action (`retry_now`, `retry_delayed`, `notify_customer`,
`escalate_to_human`) with a short explanation.

Critically, it does **not** decide freely — a rule-based `policy_engine.py`
runs first and computes which actions are even permitted:

- **Bounded**: auto-retry attempts are capped (stops after 3 failed
  attempts rather than retrying forever)
- **Gated**: payments above ₹5,000, or classifications the model isn't
  confident about, require human approval before any automated action
- **Explainable**: every decision includes a 1–2 sentence reasoning string
- **Reliable**: if the LLM call fails or is unavailable, a deterministic
  rule-based fallback keeps the pipeline running rather than crashing —
  tested and verified in `ai/strategy_selector.py`

Every policy check and decision is written to an append-only audit log
(`docs/audit_log.jsonl` via `ai/audit_logger.py`), so any payment's full
decision trail — what happened, why, what was decided, and whether it
required human approval — can be reconstructed after the fact.

Example verified behaviors:
- A ₹19,999 payment was correctly flagged as requiring human approval
  (exceeds the ₹5,000 auto-action limit) even though a strategy was still
  recommended
- A payment on its 5th failed attempt was correctly hard-stopped rather
  than retried again

### Razorpay test-mode integration
Recovery actions (`retry_now`, `retry_delayed`, `notify_customer`) create
a real Razorpay **Payment Link** in test mode via `ai/razorpay_client.py`
and `ai/execute_action.py`.

Payment Links were chosen deliberately: creating one is a genuine
server-side API call needing no customer interaction, so the *creation*
step is fully automatable end-to-end. *Completing* one requires the
customer to actually pay (OTP/card entry), which cannot and should not be
scripted server-side — this mirrors real payment security constraints
rather than faking a full auto-pay loop.

Verified behaviors:
- If `requires_human_approval` is true, the action is **not** auto-executed
  — it's logged as pending approval instead, proving the gate actually
  blocks execution rather than just being computed and ignored
- If the Razorpay API call fails (bad keys, network issue), the failure is
  caught, logged to the audit trail, and returned clearly — the pipeline
  does not crash or silently pretend success

## 8. Limitations
- Dataset is synthetic; failure-text patterns are simulated, not pulled
  from real Razorpay gateway logs
- `other` failure category has lower classification precision by design
  (ambiguous/noisy messages) — flagged for human review at low confidence
  rather than silently misclassified

## 9. Future improvements
TBD
