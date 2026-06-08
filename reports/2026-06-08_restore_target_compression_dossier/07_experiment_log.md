# 07 — Experiment Log (append-only)

**Date opened:** 2026-06-08. Append-only Popperian ledger. Every entry links its pre-registration (`05` / a `preregister` artifact) and states its verdict against the **pre-committed falsifiers**. Negatives are recorded with the same prominence as wins (use the postmortem template). Newest at the bottom; never edit past entries except to add a cross-link.

Entry template (from the skill):
```
### EXP-NN · <title> · <date> · <SUCCESS|FALSIFIED|INCONCLUSIVE>
- Plan (pre-reg): <link>
- Variable: <the one thing changed>
- Driver / artifact / results: <script · artifact ts · location>
- Readout: <fast probe> → <full metrics vs locked baseline>
- Verdict vs falsifiers (plan §5): <which fired / none> ⇒ <verdict>
- Decision: <next step per plan §7>
```

---

*(No experiments logged yet. EXP-01 is gated behind Phase-1 pre-flight — see `04_roadmap.md` / `03 §D`.)*

> Note: the synthetic reproduction in `reports/investigation_metrics_08062026/FINDINGS.md` (raw 4.79 vs log1p 1.66 vs zeros 2.16) is **diagnostic evidence**, not a pre-registered experiment of this dossier. It motivates EXP-01; it does not substitute for it (synthetic data, not real models vs the gold standard).
