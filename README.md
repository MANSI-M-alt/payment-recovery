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
- [ ] Recovery scorer
- [ ] Strategy selector + policy gate
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

## 8. Limitations
- Dataset is synthetic; failure-text patterns are simulated, not pulled
  from real Razorpay gateway logs
- `other` failure category has lower classification precision by design
  (ambiguous/noisy messages) — flagged for human review at low confidence
  rather than silently misclassified

## 9. Future improvements
TBD
