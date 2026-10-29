def add_tag(tag, tags=None):
    """Add a tag to a list of tags and return the list."""
    if tags is None:
        tags = []
    tags.append(tag)
    return tags
