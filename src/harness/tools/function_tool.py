"""The ``@tool`` decorator: an async function becomes a tool (08 §4.2).

The schema comes from the type hints and the description from the docstring.
"""

import inspect
import json
from collections.abc import Awaitable, Callable
from typing import Any, get_type_hints

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError, create_model

from harness.core.consts import (
    DEFAULT_TOOL_TIMEOUT_S,
    META_ERROR_CODE,
    Decision,
    ErrorCode,
    RiskLevel,
    ToolSource,
)
from harness.core.records import ApprovalPreview, ToolDef, ToolOutput
from harness.core.types import TextBlock
from harness.tools.consts import CONTEXT_PARAM, MSG_BAD_INPUT, MSG_NOT_ASYNC
from harness.tools.types import ToolContext, ToolTraits

ToolFunction = Callable[..., Awaitable[object]]
PreviewFunction = Callable[..., Awaitable[ApprovalPreview | None]]

_TITLE_KEY = "title"


def text_output(text: str, *, metadata: dict[str, JsonValue] | None = None) -> ToolOutput:
    """A successful result with one text block."""
    return ToolOutput(content=[TextBlock(text=text)], metadata=metadata or {})


def error_output(text: str, code: ErrorCode = ErrorCode.TOOL_FAILED) -> ToolOutput:
    """A failed result the model can read; errors are values, not exceptions."""
    return ToolOutput(
        content=[TextBlock(text=text)], is_error=True, metadata={META_ERROR_CODE: code.value}
    )


def _without_titles(node: JsonValue) -> JsonValue:
    if isinstance(node, dict):
        return {key: _without_titles(value) for key, value in node.items() if key != _TITLE_KEY}
    if isinstance(node, list):
        return [_without_titles(item) for item in node]
    return node


def _input_model(function: ToolFunction, name: str) -> type[BaseModel]:
    hints = get_type_hints(function)
    fields: dict[str, Any] = {}
    for parameter in inspect.signature(function).parameters.values():
        if parameter.name == CONTEXT_PARAM:
            continue
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        fields[parameter.name] = (hints.get(parameter.name, str), default)
    return create_model(f"{name}_input", __config__=ConfigDict(extra="forbid"), **fields)


def _schema(model: type[BaseModel]) -> dict[str, JsonValue]:
    cleaned = _without_titles(model.model_json_schema())
    return cleaned if isinstance(cleaned, dict) else {}


def normalise(result: object) -> ToolOutput:
    """Turn whatever a tool function returned into a ``ToolOutput``."""
    if isinstance(result, ToolOutput):
        return result
    if isinstance(result, str):
        return text_output(result)
    return text_output(json.dumps(result, ensure_ascii=False, default=str))


class FunctionTool:
    """A ``Tool`` backed by one async function."""

    def __init__(
        self,
        function: ToolFunction,
        definition: ToolDef,
        traits: ToolTraits,
        model: type[BaseModel],
        preview: PreviewFunction | None = None,
    ) -> None:
        self._function = function
        self._definition = definition
        self._traits = traits
        self._model = model
        self._preview = preview
        self._wants_context = CONTEXT_PARAM in inspect.signature(function).parameters

    @property
    def definition(self) -> ToolDef:
        """The tool definition."""
        return self._definition

    @property
    def traits(self) -> ToolTraits:
        """How rules read this tool's input."""
        return self._traits

    def _arguments(self, args: dict[str, JsonValue], ctx: ToolContext) -> dict[str, Any]:
        parsed = self._model.model_validate(args)
        arguments: dict[str, Any] = {
            name: getattr(parsed, name) for name in parsed.model_fields_set
        }
        if self._wants_context:
            arguments[CONTEXT_PARAM] = ctx
        return arguments

    async def run(self, args: dict[str, JsonValue], ctx: ToolContext) -> ToolOutput:
        """Validate the input, call the function and normalise what it returns."""
        try:
            arguments = self._arguments(args, ctx)
        except ValidationError as error:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in issue['loc'])}: {issue['msg']}"
                for issue in error.errors()
            )
            return error_output(MSG_BAD_INPUT.format(tool=self._definition.name, error=problems))
        return normalise(await self._function(**arguments))

    async def preview(self, args: dict[str, JsonValue], ctx: ToolContext) -> ApprovalPreview | None:
        """Call the preview function, if the tool has one."""
        if self._preview is None:
            return None
        try:
            arguments = self._arguments(args, ctx)
        except ValidationError:
            return None
        return await self._preview(**arguments)


def tool(
    *,
    name: str | None = None,
    group: str | None = None,
    read_only: bool = False,
    reason_required: bool | None = None,
    exclusive: bool = False,
    risk: RiskLevel = RiskLevel.MEDIUM,
    default_permission: Decision = Decision.ASK,
    timeout_s: int = DEFAULT_TOOL_TIMEOUT_S,
    source: ToolSource = ToolSource.CUSTOM,
    traits: ToolTraits | None = None,
    preview: PreviewFunction | None = None,
) -> Callable[[ToolFunction], FunctionTool]:
    """Turn an async function into a tool.

    A parameter named ``ctx`` receives the ``ToolContext`` and is left out of the schema.

    Raises:
        TypeError: The decorated function is not async.
    """

    def wrap(function: ToolFunction) -> FunctionTool:
        tool_name = name or function.__name__
        if not inspect.iscoroutinefunction(function):
            raise TypeError(MSG_NOT_ASYNC.format(name=tool_name))
        model = _input_model(function, tool_name)
        definition = ToolDef(
            name=tool_name,
            description=inspect.getdoc(function) or tool_name,
            input_schema=_schema(model),
            source=source,
            group=group,
            read_only=read_only,
            reason_required=reason_required,
            exclusive=exclusive,
            risk=risk,
            default_permission=default_permission,
            timeout_s=timeout_s,
        )
        return FunctionTool(function, definition, traits or ToolTraits(), model, preview)

    return wrap
