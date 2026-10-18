def decision_for_probability(probability: float) -> str:
    if not 0 <= probability <= 1:
        raise ValueError("Fraud probability must be between 0 and 1.")
    if probability >= 0.70:
        return "BLOCK"
    if probability >= 0.30:
        return "REVIEW"
    return "APPROVE"
