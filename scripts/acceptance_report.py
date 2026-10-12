"""Report which spec acceptance criteria are covered by tests.

Reads the criteria from ``tests/data/acceptance_criteria.json`` and the
``@pytest.mark.acceptance("<file>-<section>.<n>")`` marks from the test tree,
then writes a table per spec section and a summary.

Usage:
    uv run python scripts/acceptance_report.py [--missing] [--strict]
"""

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = ROOT / "tests"
CRITERIA_FILE = TESTS_DIR / "data" / "acceptance_criteria.json"
MARK_NAME = "acceptance"
ENCODING = "utf-8"
EXIT_OK = 0
EXIT_UNCOVERED = 1
COVERED = "x"
UNCOVERED = " "
TEXT_WIDTH = 70


def mark_ref(decorator: ast.expr) -> str | None:
    """The reference of one ``pytest.mark.acceptance("...")`` decorator, if it is one."""
    if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
        return None
    if decorator.func.attr != MARK_NAME or not decorator.args:
        return None
    first = decorator.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def collect_marks(tests_dir: Path) -> dict[str, list[str]]:
    """Map every marked reference to the tests that carry it."""
    marks: dict[str, list[str]] = defaultdict(list)
    for path in sorted(tests_dir.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding=ENCODING), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for decorator in node.decorator_list:
                ref = mark_ref(decorator)
                if ref is not None:
                    marks[ref].append(f"{path.relative_to(ROOT).as_posix()}::{node.name}")
    return marks


def load_criteria(path: Path) -> dict[str, list[str]]:
    """The criteria texts per ``<file>-<section>``."""
    raw: dict[str, list[str]] = json.loads(path.read_text(encoding=ENCODING))
    return raw


def shorten(text: str) -> str:
    """One line of bounded width."""
    flat = " ".join(text.split())
    return flat if len(flat) <= TEXT_WIDTH else f"{flat[: TEXT_WIDTH - 1]}…"


def render(
    criteria: dict[str, list[str]], marks: dict[str, list[str]], *, missing_only: bool
) -> tuple[list[str], int, int]:
    """The report lines plus the covered and total counts."""
    lines: list[str] = []
    covered = total = 0
    known: set[str] = set()
    for section, texts in criteria.items():
        lines.append(f"\n## {section}")
        for number, text in enumerate(texts, start=1):
            ref = f"{section}.{number}"
            known.add(ref)
            tests = marks.get(ref, [])
            total += 1
            covered += bool(tests)
            if tests and missing_only:
                continue
            lines.append(f"[{COVERED if tests else UNCOVERED}] {ref:<9} {shorten(text)}")
            lines.extend(f"      - {test}" for test in tests)
    unknown = sorted(set(marks) - known)
    if unknown:
        lines.append("\n## marks that match no criterion")
        lines.extend(f"    {ref}: {', '.join(marks[ref])}" for ref in unknown)
    return lines, covered, total


def main(argv: list[str] | None = None) -> int:
    """Write the report; with ``--strict`` fail when a criterion has no test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--missing", action="store_true", help="list only uncovered criteria")
    parser.add_argument("--strict", action="store_true", help="exit 1 when anything is uncovered")
    args = parser.parse_args(argv)
    lines, covered, total = render(
        load_criteria(CRITERIA_FILE), collect_marks(TESTS_DIR), missing_only=args.missing
    )
    lines.append(f"\ncovered {covered} of {total} acceptance criteria")
    sys.stdout.reconfigure(encoding=ENCODING)  # type: ignore[union-attr]
    sys.stdout.write("\n".join(lines).lstrip("\n") + "\n")
    return EXIT_UNCOVERED if args.strict and covered < total else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
