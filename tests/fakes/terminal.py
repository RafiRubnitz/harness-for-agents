"""FakeTerminal: scripted input lines, recorded output."""

from dataclasses import dataclass, field


@dataclass
class FakeTerminal:
    """Answers ``read`` from a list and keeps everything that was written."""

    inputs: list[str] = field(default_factory=list[str])
    written: list[str] = field(default_factory=list[str])
    prompts: list[str] = field(default_factory=list[str])
    styles: list[str | None] = field(default_factory=list[str | None])

    def write(self, text: str) -> None:
        """Record raw text."""
        self.written.append(text)

    def line(self, text: str = "", style: str | None = None) -> None:
        """Record one line and the style it was written in."""
        self.written.append(f"{text}\n")
        self.styles.append(style)

    async def read(self, prompt: str) -> str | None:
        """Return the next scripted line, or ``None`` when the script is over."""
        self.prompts.append(prompt)
        return self.inputs.pop(0) if self.inputs else None

    @property
    def output(self) -> str:
        """Everything written so far."""
        return "".join(self.written)
