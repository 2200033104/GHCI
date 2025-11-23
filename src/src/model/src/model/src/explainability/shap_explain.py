# src/explainability/shap_explain.py
import shap
import numpy as np
from typing import List

def explain_instance(model, vectorizer, text: str, top_k=5):
    """
    Returns a list of (feature, contribution) pairs for the instance.
    This uses kernel explainer on the vectorized input for model-agnostic explanations.
    For tree models with a .get_booster(), TreeExplainer would be faster.
    """
    # vectorize
    x = vectorizer.transform([text])
    # Try specialized explainer for tree models
    try:
        explainer = shap.Explainer(model.named_steps["clf"])
        shap_values = explainer(x)
    except Exception:
        background = vectorizer.transform([""])
        explainer = shap.KernelExplainer(lambda z: model.predict_proba(z), background)
        shap_values = explainer.shap_values(x, nsamples=100)

    # For simplicity return top_k tokens by absolute contribution
    try:
        # shap_values may be list (for multiclass)
        if isinstance(shap_values, list):
            # pick class with highest predicted prob
            class_idx = int(np.argmax(model.predict_proba([text])[0]))
            sv = shap_values[class_idx][0]
        else:
            sv = shap_values.values[0]
        feature_names = vectorizer.get_feature_names_out()
        contrib = list(zip(feature_names, sv))
        contrib_sorted = sorted(contrib, key=lambda x: abs(x[1]), reverse=True)[:top_k]
        return contrib_sorted
    except Exception as e:
        return [("shap_error", str(e))]
