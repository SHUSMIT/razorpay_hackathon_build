# 5-minute pitch — shot list and script

The numbers below are filled in from the current run
(`reports/assessment.json`, shipped model `ensemble`, test PR-AUC
75.93%). If you re-run the pipeline, run
`python scripts/update_readme.py` and re-read them off the README before
filming. **Never say a number you have not just seen on screen.**

**Before you hit record**

```bash
python run.py api                                  # terminal 1, leave it up
python run.py app                                  # terminal 2 -> localhost:8501
```

Have open: the dashboard (4 tabs), one terminal, and `docs/FINDINGS.md` in an
editor. Close everything else — a notification mid-take costs a re-shoot.

---

## Beat 1 — the problem (0:00–0:30)

> "Fraud and chargebacks eat merchant margin. But the obvious fix eats more of
> it. In this data legitimate transactions outnumber fraud about 570 to one, so
> a model that's slightly too aggressive costs a merchant more in declined real
> customers than the fraud ever did.
>
> So the question isn't 'can you detect fraud.' It's **where do you put the
> threshold, what stops the system when it goes wrong, and can a human
> understand why it decided what it decided.**"

*On screen: the README architecture diagram.*

---

## Beat 2 — a decision a human can act on (0:30–1:20)

*Dashboard → Live Score → "Known fraud" → Draw → Score it.*

> "This is a real transaction from a held-out month the model has never seen.
> Ground truth: fraud. It scores **94%**, and because that's above the
> block threshold, it blocks — in about **50** milliseconds.
>
> And look at *why*. Not a SHAP value on an anonymous component — an actual
> reason: this merchant category is fraudulent **0.6**% of the time against
> **0.09**% overall; this amount is higher than **89**% of legitimate
> transactions. A reviewer can act on that."

*Then draw a "Known legitimate" row and score it → allow.*

**Then deliberately draw frauds until one is missed**, and say so:

> "Recall is **66**%, so it misses some. There's one. I'd rather show you
> that than pretend otherwise — and the cost curve in a minute is exactly why
> we don't chase 100%."

---

## Beat 3 — the finding that changes the model (1:20–2:30)

**This is the beat that separates this from a notebook.** Have
`docs/FINDINGS.md` on screen.

> "Three things nearly gave me a great-looking number that was worthless.
>
> **One.** A random split scores about 61% here. Time-based, it collapsed to
> 0.09%. There are only 2,000 cardholders in this data, so income, debt and
> credit score together fingerprint a *person* — a random split just lets the
> model recognise who got defrauded before. Dropping those took it from 0.09%
> to 4.7%: a fifty-two times improvement, from *removing* features.
>
> **Two.** Then it hit 87%, which was too good, so I checked. One column,
> `merchant_state`, scored 89% *by itself* — better than the whole model —
> because 95 to 98% of later fraud sits in a single value. That's a label
> artefact, not a signal. Dropping it cost me two thirds of my headline
> number, and I dropped it.
>
> **Three.** And the useful one: the fraud mechanism **inverts**. It's 83%
> card-not-present in the training years, and 86% chip-present in the recent
> ones. Eight years of history was teaching the wrong pattern."

*If you have 15 seconds spare, or if a judge asks how much to trust the number:*

> "And I'll get ahead of one thing. Most of my remaining score leans on a
> single flag — whether the merchant ZIP is missing. In this data that marks
> almost all the fraud, which is a quirk of how it was generated, not how
> payments work. Take that feature away and I score 11%, not 76%. Eleven is
> the number I'd expect on real traffic. Everything around the model — the
> threshold, the gate, the review queue, the audit trail — doesn't care which
> of those two numbers is true."

---

## Beat 4 — what we did about the drift (2:30–3:10)

> "So training weights decay with age — every row stays in, but its influence
> halves every so many days. And I didn't pick that half-life; Optuna tunes it
> alongside every other hyper-parameter. It chose **60** days.
>
> That took validation from 15% to 46% on identical features. Not a better
> model — a model that stopped being taught by a world that no longer exists."

*On screen: the Model Performance tab.*

---

## Beat 5 — confidence-based decisioning, live (3:10–4:10)

*Live Score tab → score a borderline transaction → Review Queue.*

> "A human sees transactions only when the calibrated model score sits in the
> review band. High-confidence fraud is blocked automatically; low-risk traffic
> proceeds. The queue is for uncertainty, not for a volume throttle."

*Then Review Queue tab — resolve one case.*

> "And those aren't dropped on the floor. Here's the queue, with the same
> plain-language evidence, and a human approves or declines. That resolution
> goes into the audit trail too."

---

## Beat 6 — the money, and the honesty (4:10–4:45)

*Model Performance tab, cost curve.*

> "The threshold isn't 0.5, and it isn't tuned to the test set. It's the
> cost minimum, chosen on validation data the model never trained on.
>
> The assumption you should push on: a missed fraud costs the full transaction
> amount, and wrongly blocking a real customer costs a flat 50 in support and
> goodwill. That second number is an assumption, not a measurement — change it
> and the threshold moves. On held-out data this avoids **52**% of the
> avoidable loss.
>
> Picking the threshold on the test split itself would have looked
> **530** better. That's selection bias, and it's in the report as a number
> rather than quietly taken."

---

## Beat 7 — what's next (4:45–5:00)

> "Next: drift monitoring in production — this dataset proves why that matters
> more than another point of PR-AUC. Then a durable review queue, and a real
> gateway integration.
>
> The limitations are written down in the README before anyone asks."

---

## If something breaks on camera

- **API offline** — the sidebar says so; restart terminal 1.
- **Empty review queue** — score a borderline transaction from Live Score.
- **A drawn fraud scores low** — use it. That is Beat 2's honest moment.
