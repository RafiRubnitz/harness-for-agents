"""In-memory ``SecretStore``."""


class FakeSecretStore:
    """Secrets kept in a dict instead of the OS keyring."""

    def __init__(self, values: dict[str, str] | None = None) -> None:
        self.values: dict[str, str] = dict(values or {})

    def get(self, name: str) -> str | None:
        """Return the secret, or ``None`` when it is not set."""
        return self.values.get(name)

    def set(self, name: str, value: str) -> None:
        """Store the secret."""
        self.values[name] = value
