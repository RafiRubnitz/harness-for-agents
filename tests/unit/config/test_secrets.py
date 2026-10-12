"""Secret references and masking."""

import pytest

from harness.config.consts import SECRET_MASK_FILL, SECRET_MASK_VISIBLE
from harness.config.secrets import bad_refs, expand, mask, secret_names
from harness.core.consts import ErrorCode
from harness.core.errors import ConfigError, SecretMissingError
from tests.consts import TEST_API_KEY
from tests.fakes.secrets import FakeSecretStore

SECRET_NAME = "anthropic"
ENV_NAME = "MY_KEY"
ENV_VALUE = "from-env"


def test_expand_reads_secret_and_env() -> None:
    store = FakeSecretStore({SECRET_NAME: TEST_API_KEY})

    assert expand(f"${{secret:{SECRET_NAME}}}", store, {}) == TEST_API_KEY
    assert expand(f"${{env:{ENV_NAME}}}", store, {ENV_NAME: ENV_VALUE}) == ENV_VALUE


@pytest.mark.acceptance("06-6.2")
def test_expand_missing_secret_raises_secret_missing() -> None:
    with pytest.raises(SecretMissingError) as raised:
        expand(f"${{secret:{SECRET_NAME}}}", FakeSecretStore(), {})

    assert raised.value.code is ErrorCode.CONFIG_SECRET_MISSING


def test_expand_missing_env_raises_config_error() -> None:
    with pytest.raises(ConfigError):
        expand(f"${{env:{ENV_NAME}}}", FakeSecretStore(), {})


@pytest.mark.acceptance("06-6.2")
@pytest.mark.parametrize("ref", ["${ANTHROPIC_API_KEY}", "${secret:}", "${other:x}"])
def test_reference_without_valid_prefix_is_reported(ref: str) -> None:
    found = bad_refs({"providers": {"anthropic": {"api_key": ref}}})

    assert [key for key, _ in found] == ["providers.anthropic.api_key"]


def test_hook_commands_are_not_scanned_for_references() -> None:
    assert bad_refs({"hooks": {"specs": [{"run": "echo ${HOME}"}]}}) == []


def test_secret_names_lists_referenced_secrets() -> None:
    data = {"providers": {"anthropic": {"api_key": f"${{secret:{SECRET_NAME}}}"}}}

    assert secret_names(data) == [("providers.anthropic.api_key", SECRET_NAME)]


def test_mask_shows_only_the_edges() -> None:
    masked = mask(TEST_API_KEY)

    assert masked.startswith(TEST_API_KEY[:SECRET_MASK_VISIBLE])
    assert masked.endswith(TEST_API_KEY[-SECRET_MASK_VISIBLE:])
    assert SECRET_MASK_FILL in masked
    assert TEST_API_KEY not in masked


def test_mask_hides_short_values_entirely() -> None:
    assert mask("abc") == SECRET_MASK_FILL
