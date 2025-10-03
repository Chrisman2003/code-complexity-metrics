# Data cleaning, filtering, feature engineering
def normalize_by_loc(records):
    for rec in records:
        loc = rec["lines"] or 1
        rec["cognitive_per_line"] = rec["cognitive"] / loc
        rec["cyclomatic_per_line"] = rec["cyclomatic"] / loc
        rec["nesting_per_line"] = rec["nesting"] / loc
        rec["sloc_per_line"] = rec["sloc"] / loc
        # Halstead metrics often return a dict; normalize if numeric value exists
        if isinstance(rec["halstead"], dict) and "effort" in rec["halstead"]:
            rec["halstead_effort_per_line"] = rec["halstead"]["effort"] / loc
    return records
