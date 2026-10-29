def dedupe(items):
    """Remove duplicates, keeping the first occurrence of each item in order."""
    return list(dict.fromkeys(items))
