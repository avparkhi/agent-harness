def parse_line(line):
    """Split one CSV line into fields."""
    return [field.strip() for field in line.split(",")]


def load_contacts(text):
    return [parse_line(line) for line in text.splitlines() if line.strip()]
