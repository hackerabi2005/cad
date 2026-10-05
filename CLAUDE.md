# Guidelines and Invariants

## Guidelines
1. **Think Before Coding**: Don't assume. Don't hide confusion. Surface tradeoffs. State assumptions explicitly.
2. **Simplicity First**: Minimum code that solves the problem. Nothing speculative.
3. **Surgical Changes**: Touch only what you must. Clean up only your own mess.
4. **Goal-Driven Execution**: Define success criteria. Loop until verified.

## Invariants
- `LAD`, `LCX`, `RCA`, `Cath` never enter X; preprocessing only inside Pipelines.
- Raw data and the original mesh are never edited; derive into new files.
- Vessel and feature names come only from `vessels.json` and `schema.json`.
- The disclaimer is always visible in the UI ("Decision support / educational use only — not a substitute for formal diagnostic imaging.").
- Pin and log library versions; fixed seeds.
- Any new dependency needs a one-line justification.
- Cath == OR(vessels) holds after one documented alignment.
- Fast, exact SHAP explanations for served model.
