# 5-minute pitch — shot list and script

Every number below is the real measured value as of the last full pipeline run.
If you re-run `python run.py all`, re-read them off the README and
`reports/model_selection.md` before filming — and never say a number you have
not just seen on screen.

**Before you hit record**

```bash
python run.py api                      # terminal 1, leave it up
python run.py app                      # terminal 2 → localhost:8501
curl -X POST http://127.0.0.1:8000/gate/reset   # clean gate window
```

Have open: the Streamlit app (3 tabs), one terminal for `scripts/demo_gate.py`,
and the repo in an editor. Close everything else — a notification popping up
mid-take costs you a re-shoot.

---

## Beat 1 — the problem (0:00–0:30)

> "Fraud and chargebacks eat merchant margin. But the obvious fix eats more of
> it. In this dataset legitimate transactions outnumber fraud 579 to one — so a
> model that's slightly too aggressive costs a merchant more in declined real
> customers than the fraud ever did.
>
> So the interesting question isn't 'can you detect fraud.' It's **where do you
> put the threshold, and what stops the system when it goes wrong.** That's
> what I built."

*On screen: the README architecture diagram.*

---

## Beat 2 — live demo (0:30–1:30)

*Streamlit → Live Score tab. Draw a known-fraud row, hit Score.*

> "This is a real transaction from a held-out split the model has never seen.
> Ground truth: fraud. The service scores it 1.0000, and because that's above
> the block threshold, it blocks — in about 20 milliseconds.
>
> And it tells me *why*: the top three contributing features by SHAP value.
> These are anonymised PCA components — the dataset authors never published
> what they mean, and I'm not going to invent a story about what V17 is. What I
> can say is exactly how much each one moved this decision."

**If you draw a fraud the model misses** — it will happen, recall is 76%, not
100% — do not re-roll on camera. The app labels it plainly as a missed fraud.
Say: *"There's one it missed. Recall is 76%, and the cost curve is exactly how
I decided that 76% was the right number to stop at rather than pushing higher
and declining more real customers."* That answer is worth more than a clean take.

*Draw a legitimate row. Show it allowed.*

> "Same service, a legitimate transaction, allowed. Note there's a third
> outcome — review. Borderline scores go to a human queue instead of being
> auto-decided. Uncertainty routes to a person."

---

## Beat 3 — the cost curve (1:30–2:30) ← **the beat that proves you read the brief**

*Model Report tab. Point at `cost_curve.png`.*

> "Here's the part I care most about. My threshold is not 0.5. Nobody's
> threshold should be 0.5.
>
> I wrote down two assumptions, and they're in `config.py`, not buried in a
> function. A missed fraud costs the merchant the **actual amount** of that
> transaction — the chargeback claws back the full value. I use the real amount
> per transaction, not an average, because fraud amounts here are heavily
> skewed: median 9, max 2,125. Averaging would let a model that misses the
> expensive frauds look identical to one that misses the cheap ones.
>
> A false positive — a real customer declined at checkout — costs a flat 50, in
> support time and lost goodwill. That's a number *I chose*. I can't measure it
> from this dataset. Change it in the config and this whole curve moves, and
> the service picks up the new threshold automatically.
>
> Sweeping every threshold from 0.01 to 0.99 gives this curve — and I pick the
> minimum **on validation**, then apply it to test. That matters: my first
> version took the argmin on the test split and quoted the cost there, which
> fits the threshold to the same 71 frauds it's scored against. That was worth
> an 11% optimism, and I fixed it.
>
> The honest number: threshold **0.09**, we catch **80%** of fraud, wrongly
> decline **16** customers out of 42,488, and lose **4,038** instead of
> **9,241** doing nothing. A **56% reduction** in fraud losses.
>
> And notice the shape: the curve is almost flat across the top half of the
> threshold range. That flatness *is* the finding — it says the exact threshold
> barely matters here, and what matters is having a cost model at all."

---

## Beat 4 — the gate (2:30–3:30) ← **the clip worth more than any slide**

*Terminal: `python scripts/demo_gate.py --n 80`.*

> "Now the part that makes this a risk *manager* and not a classifier behind
> HTTP.
>
> I'm sending 80 known-fraud transactions in a burst. The model flags 63 of
> them as blocks — and it's *right*, every one of those is real fraud. Watch
> what the service actually does."

*Let the downgrade lines scroll. Stop talking and let it happen.*

> "It blocked the first one, then it stopped — 4 auto-blocks out of 63 the
> model wanted. There's a hard cap: this service
> will never auto-block more than 5% of the last 200 decisions. Past that,
> blocks get force-downgraded to review, and the reason is written to the audit
> log.
>
> That's deliberate. If this model ever starts wanting to block everything —
> drift, an attack, a bad deploy — the merchant gets a review queue, **not an
> outage**. The blast radius is bounded by construction, not by hoping the
> model stays correct."

*Streamlit → Audit Trail tab.*

> "And every one of those decisions is here. Timestamp, hash of the input —
> never the raw transaction — the score, the decision, **the decision the model
> wanted**, whether the gate intervened and why, the rule id, and the model
> version. The blue line is the rolling block rate pressing against the cap.
> That's the bound holding."

---

## Beat 5 — honest metrics (3:30–4:30)

*Model Report tab, before/after table.*

> "PR-AUC 0.806 on a held-out split that no training or tuning step ever
> touched. That's roughly 480 times better than chance, since the base rate is
> 0.0017.
>
> I'm reporting PR-AUC, not accuracy. Predicting 'legitimate' for everything
> scores 99.83% accuracy on this data and catches zero fraud. ROC-AUC isn't
> much better as a headline — its denominator has 284,000 true negatives, so
> thousands of false positives barely move it.
>
> Optuna ran 50 trials optimising PR-AUC on the **validation** split. The test
> split is loaded in exactly one file, once, after the final model is fitted.
> `tune_optuna.py` never imports it — that's checkable in about ten seconds."

**Then say the uncomfortable thing, because it's the strongest thing you have:**

> "And here's the result I think actually matters. Tuning **didn't work.**
>
> Optuna improved validation PR-AUC, 0.845 to 0.853, so I selected the tuned
> model — before looking at test. On test it scored 0.806, against the
> baseline's 0.815. Worse. On cost it's worse too: 4,038 against 3,535.
>
> So I checked whether that gap is even real. Paired bootstrap, 2,000
> resamples: the difference is minus 0.009, with a 95% interval from minus 0.04
> to plus 0.02. It straddles zero. The two models are **statistically
> indistinguishable**, because this test split contains 71 frauds and the
> confidence interval on a single PR-AUC is about plus-or-minus 0.08 wide.
>
> Which means: on a dataset with this few positives, 50 trials of
> hyperparameter search buys essentially nothing that survives held-out data.
> More labelled fraud would help far more than more search. I checked the same
> way whether LightGBM, CatBoost or an ensemble of all three would help — every
> paired interval straddles zero too. The model family isn't the bottleneck;
> 492 labelled frauds is.
>
> And I'm still shipping the tuned model, even though the baseline looks cheaper
> on test — because the selection rule was fixed before I looked, and re-picking
> the winner afterwards is the exact thing I'm criticising.
>
> I could have re-run that search with different seeds until the table looked
> like an improvement — on noise this size it eventually would. I'd rather show
> you the confidence interval. If I showed you 0.99 here, you should assume
> leakage, not brilliance: my split drops the 1,081 exact duplicate rows before
> splitting and asserts disjointness by content hash, and there's a test that
> deliberately poisons a split to prove that guard actually fires."

---

## Beat 6 — what's next (4:30–5:00)

> "Limitations, before you ask: one dataset, one geography, two days in 2013.
> No drift monitoring — the block-rate gate catches the symptom, not the cause.
> The false-positive cost is my assumption, not a measurement. And the audit
> log is append-only by convention, not cryptographically tamper-evident.
> They're all in the README.
>
> Next: drift monitoring with automatic threshold recalibration, a real
> Razorpay webhook integration, hash-chained audit records, and extending past
> card fraud to return and refund abuse — the other half of the margin problem.
>
> Thanks."

---

## Rehearsal checklist

- [ ] Fresh clone, fresh venv, full pipeline runs with no manual steps
- [ ] API cold-start done **before** recording (first request loads the model)
- [ ] Gate window reset so the first burst block actually blocks
- [ ] A known-fraud row and a known-legit row both drawn successfully once
- [ ] Every number you plan to say is visible on screen when you say it
- [ ] Run the whole sequence twice, on the machine you'll present from
