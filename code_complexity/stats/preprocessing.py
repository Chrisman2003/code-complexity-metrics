# Data cleaning, filtering, feature engineering
def normalize_by_loc(records):
    for rec in records:
        loc = rec["lines"] or 1
        rec["cognitive_per_line"] = rec["cognitive"] / loc
        rec["cyclomatic_per_line"] = rec["cyclomatic"] / loc
        rec["nesting_per_line"] = rec["nesting"] / loc
        rec["sloc_per_line"] = rec["sloc"] / loc
        rec["halstead_per_line"] = rec["halstead"] / loc
    return records
