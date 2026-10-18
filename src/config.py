def normalize_service_url(value: str) -> str:
    url = value.strip().rstrip("/")
    if url.startswith(("http://", "https://")):
        return url
    if "." not in url:
        return f"https://{url}.onrender.com"
    return f"https://{url}"
