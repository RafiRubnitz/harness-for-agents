"""The two kinds of hook: a Python function in this process and an external program (05 §5)."""

import asyncio
import importlib.util
import inspect
import os
import shutil
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

from pydantic import ValidationError

from harness.config import HooksConfig
from harness.config import consts as config_consts
from harness.core.consts import ErrorCode
from harness.core.errors import HookError
from harness.core.hook_types import HookInput, HookOutput, HookSpec
from harness.hooks import consts
from harness.hooks.utils import fill, known_placeholder, placeholders, split_command

HookFunction = Callable[[HookInput], Awaitable[object]]


class Runner(Protocol):
    """Runs one hook once."""

    def check(self) -> None:
        """Load whatever the hook needs now.

        Raises:
            HookError: The hook cannot be loaded.
        """
        ...

    async def run(self, inp: HookInput) -> HookOutput:
        """Return what the hook answered.

        Raises:
            HookError: The hook could not be loaded, failed, or answered nonsense.
        """
        ...


def _load_error(message: str, spec: HookSpec) -> HookError:
    return HookError(message, hook=spec.name, event=spec.event, code=ErrorCode.HOOK_LOAD_FAILED)


def _invalid(spec: HookSpec, error: object) -> HookError:
    return HookError(
        consts.MSG_INVALID_OUTPUT.format(hook=spec.name, error=error),
        hook=spec.name,
        event=spec.event,
        code=ErrorCode.HOOK_INVALID_OUTPUT,
    )


class ModuleCache:
    """Imports hook modules from the hooks directory and re-imports a file that changed."""

    def __init__(self, hooks_dir: Path) -> None:
        self._dir = hooks_dir
        self._loaded: dict[Path, tuple[tuple[int, int], ModuleType]] = {}

    def clear(self) -> None:
        """Forget every imported module."""
        self._loaded.clear()

    def path_of(self, module: str) -> Path | None:
        """The file of a dotted module name, or ``None`` when the name is not a module name."""
        parts = module.split(consts.MODULE_SEPARATOR)
        if not all(part.isidentifier() for part in parts):
            return None
        return self._dir.joinpath(*parts).with_suffix(consts.PYTHON_SUFFIX)

    def function(self, path: Path, name: str, spec: HookSpec) -> HookFunction:
        """Return the async hook function ``name`` of the module at ``path``.

        Raises:
            HookError: The file is missing, fails to import, or has no such async function.
        """
        found = getattr(self._module(path, spec), name, None)
        if not inspect.iscoroutinefunction(found):
            raise _load_error(consts.MSG_FUNCTION_MISSING.format(path=path, function=name), spec)
        return cast("HookFunction", found)

    def _module(self, path: Path, spec: HookSpec) -> ModuleType:
        try:
            stat = path.stat()
        except OSError:
            raise _load_error(consts.MSG_MODULE_MISSING.format(path=path), spec) from None
        stamp = (stat.st_mtime_ns, stat.st_size)
        cached = self._loaded.get(path)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        if str(self._dir) not in sys.path:  # hook modules may import their neighbours
            sys.path.append(str(self._dir))
        name = consts.USER_MODULE_PREFIX + "_".join(
            path.relative_to(self._dir).with_suffix("").parts
        )
        module_spec = importlib.util.spec_from_file_location(name, path)
        if module_spec is None or module_spec.loader is None:
            raise _load_error(consts.MSG_MODULE_MISSING.format(path=path), spec)
        module = importlib.util.module_from_spec(module_spec)
        try:
            module_spec.loader.exec_module(module)
        except Exception as error:  # user code may fail in any way
            message = consts.MSG_MODULE_BROKEN.format(path=path, error=repr(error))
            raise _load_error(message, spec) from error
        self._loaded[path] = (stamp, module)
        return module


class PythonRunner:
    """``python:<module>:<function>``: an async function called inside this process."""

    def __init__(self, spec: HookSpec, path: Path, function: str, modules: ModuleCache) -> None:
        self._spec = spec
        self._path = path
        self._function = function
        self._modules = modules

    def check(self) -> None:
        """Import the module now so that a broken hook is reported before it is needed."""
        self._modules.function(self._path, self._function, self._spec)

    async def run(self, inp: HookInput) -> HookOutput:
        """Call the function; a ``dict`` or ``None`` answer is accepted like a ``HookOutput``."""
        answer = await self._modules.function(self._path, self._function, self._spec)(inp)
        if answer is None:
            return HookOutput()
        try:
            return HookOutput.model_validate(answer)
        except ValidationError as error:
            raise _invalid(self._spec, error) from error


class CommandRunner:
    """An external program: ``HookInput`` as JSON on stdin, ``HookOutput`` as JSON on stdout."""

    def __init__(self, spec: HookSpec, argv: list[str], base_dir: Path, *, fills: bool) -> None:
        self._spec = spec
        self._argv = argv
        self._base_dir = base_dir
        self._fills = fills

    def check(self) -> None:
        """Nothing more to check: the program was found when the runner was built."""

    def _cwd(self, inp: HookInput) -> Path:
        if inp.working_dir and Path(inp.working_dir).is_dir():
            return Path(inp.working_dir)
        return self._base_dir

    async def run(self, inp: HookInput) -> HookOutput:
        """Run the program without a shell and read its answer."""
        name = self._spec.name
        argv = [fill(arg, inp, name) for arg in self._argv] if self._fills else self._argv
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._cwd(inp),
            env={**os.environ, consts.ENV_IO_ENCODING: consts.OUTPUT_ENCODING},
        )
        try:
            out, err = await process.communicate(inp.model_dump_json().encode())
        finally:
            if process.returncode is None:
                await _kill(process)
        if process.returncode != 0:
            stderr = err.decode(consts.OUTPUT_ENCODING, "replace").strip()
            message = consts.MSG_EXIT_CODE.format(
                hook=name, code=process.returncode, stderr=stderr[-consts.STDERR_SHOWN_CHARS :]
            )
            raise HookError(message, hook=name, event=self._spec.event)
        text = out.decode(consts.OUTPUT_ENCODING, "replace").strip()
        if not text:
            return HookOutput()
        try:
            return HookOutput.model_validate_json(text)
        except ValidationError as error:
            raise _invalid(self._spec, error) from error


async def _kill(process: asyncio.subprocess.Process) -> None:
    """End a hook program that was cut off (timeout or cancellation)."""
    try:
        process.kill()
    except ProcessLookupError:
        return
    try:
        async with asyncio.timeout(consts.KILL_WAIT_S):
            await asyncio.shield(process.wait())
    except TimeoutError:
        pass


def _python_runner(spec: HookSpec, modules: ModuleCache) -> PythonRunner:
    module, _, function = spec.run[len(consts.RUN_PYTHON_PREFIX) :].rpartition(":")
    path = modules.path_of(module) if module and function.isidentifier() else None
    if path is None:
        raise _load_error(consts.MSG_BAD_PYTHON_REF.format(run=spec.run), spec)
    return PythonRunner(spec, path, function, modules)


def _file_runner(spec: HookSpec, path: Path, runner: str, base_dir: Path) -> CommandRunner:
    if not path.is_absolute():
        path = base_dir / path
    if not path.is_file():
        raise _load_error(consts.MSG_FILE_MISSING.format(path=path), spec)
    if runner == config_consts.HOOK_RUNNER_PYTHON:
        return CommandRunner(spec, [sys.executable, str(path)], base_dir, fills=False)
    if runner == config_consts.HOOK_RUNNER_DIRECT:
        return CommandRunner(spec, [str(path)], base_dir, fills=False)
    message = consts.MSG_NO_RUNNER.format(suffix=path.suffix, runner=runner)
    raise _load_error(message, spec)


def _line_runner(spec: HookSpec, parts: list[str], base_dir: Path) -> CommandRunner:
    for name in (name for part in parts for name in placeholders(part)):
        if not known_placeholder(name):
            raise _load_error(consts.MSG_UNKNOWN_PLACEHOLDER.format(name=name), spec)
    program = shutil.which(parts[0])
    if program is None:
        raise _load_error(consts.MSG_PROGRAM_MISSING.format(program=parts[0]), spec)
    return CommandRunner(spec, [program, *parts[1:]], base_dir, fills=True)


def build(
    spec: HookSpec, settings: HooksConfig, modules: ModuleCache, default_dir: Path
) -> PythonRunner | CommandRunner:
    """Turn a hook's ``run`` into its runner (05 §6).

    Args:
        spec: The hook.
        settings: The ``hooks`` config section (the suffix-to-runner table).
        modules: Where Python hook modules are imported from.
        default_dir: Base of relative paths when the hook does not come from a file.

    Raises:
        HookError: The hook cannot be loaded (``hook.load_failed``).
    """
    unknown = sorted(set(spec.match) - consts.MATCH_KEYS)
    if unknown:
        raise _load_error(consts.MSG_UNKNOWN_MATCH.format(keys=", ".join(unknown)), spec)
    if spec.run.startswith(consts.RUN_PYTHON_PREFIX):
        return _python_runner(spec, modules)
    try:
        parts = split_command(spec.run)
    except ValueError as error:
        raise _load_error(consts.MSG_BAD_COMMAND.format(error=error), spec) from error
    base_dir = Path(spec.source_file).parent if spec.source_file else default_dir
    runner = settings.runners.get(Path(parts[0]).suffix.lower())
    if len(parts) == 1 and runner is not None:
        return _file_runner(spec, Path(parts[0]), runner, base_dir)
    return _line_runner(spec, parts, base_dir)
