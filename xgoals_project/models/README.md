# Models

Save trained models from the notebook here using joblib, e.g.

- baseline: `models/xg_baseline.joblib`
- 360 model: `models/xg_360.joblib`

Example (in notebook):

```python
import joblib
joblib.dump(baseline_model, "models/xg_baseline.joblib")
joblib.dump(model_360, "models/xg_360.joblib")
```
