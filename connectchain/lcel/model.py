# Copyright 2023 American Express Travel Related Services Company, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
"""LCEL model module"""
import logging
import os
from typing import Any

from langchain.schema.language_model import BaseLanguageModel
from langchain_openai import AzureOpenAI, ChatOpenAI
from pydantic import SecretStr

from connectchain.utils import Config, SessionMap, get_token_from_env
from connectchain.utils.llm_proxy_wrapper import wrap_llm_with_proxy

logger = logging.getLogger(__name__)


class LCELModelException(BaseException):
    """Base exception for the LCEL model"""


def model(index: Any = "1") -> BaseLanguageModel:
    """
    Though name of this method may be confusing, the purpose is to keep
    the LCEL notation nearly intact
    """
    llm = _get_model_(index)
    return llm


def _get_model_(index: Any) -> BaseLanguageModel:
    """Get the model config based on the models defined in the config"""
    config = Config.from_env()
    try:
        models = config.models
    except KeyError as ex:
        raise LCELModelException("No models defined in config") from ex
    model_config = models[index]
    if model_config is None:
        raise LCELModelException(f'Model config at index "{index}" is not defined')

    needs_eas = False
    try:
        if (
            hasattr(config, "eas")
            and config.eas
            and hasattr(config.eas, "id_key")
            and config.eas.id_key
            and not getattr(model_config, "bypass_eas", False)
        ):
            needs_eas = True
    except (AttributeError, KeyError):
        needs_eas = False

    if model_config.provider != "openai":
        needs_eas = False

    if needs_eas:
        model_instance = _get_openai_model_(index, config, model_config)
    else:
        model_instance = _get_direct_model_(model_config)

    if model_instance is None:
        raise LCELModelException("Not implemented")
    try:
        proxy_config = model_config.proxy
    except KeyError:
        pass
    if proxy_config is None:
        try:
            proxy_config = config.proxy
        except KeyError:
            pass
    if proxy_config is not None:
        wrap_llm_with_proxy(model_instance, proxy_config)  # type: ignore[arg-type]
    return model_instance


def _get_openai_model_(index: Any, config: Any, model_config: Any) -> BaseLanguageModel:
    """Get the OpenAI LLM instance"""
    model_session_key = SessionMap.uuid_from_config(config, model_config)
    auth_token = os.getenv(model_session_key)
    session_map = SessionMap(config.eas.token_refresh_interval)
    if auth_token is None or session_map.is_expired(model_session_key):
        auth_token = get_token_from_env(index)
        os.environ[model_session_key] = auth_token
        if model_config.type == "chat":
            llm: BaseLanguageModel = _get_chat_model_(auth_token, model_config)
        else:
            llm = _get_azure_model_(auth_token, model_config)
        session_map.new_session(model_session_key, llm)  # type: ignore[arg-type]
        return llm
    return session_map.get_llm(model_session_key)  # type: ignore[return-value]


def _get_chat_model_(auth_token: str, model_config: Any) -> ChatOpenAI:
    """Get a ChatOpenAI instance"""
    return ChatOpenAI(
        model=model_config.model_name,
        api_key=SecretStr(auth_token) if auth_token else None,
        base_url=model_config.api_base,
        model_kwargs={
            "engine": model_config.engine,
            "api_version": model_config.api_version,
            "api_type": "azure",
        },
    )


def _get_azure_model_(auth_token: str, model_config: Any) -> AzureOpenAI:
    """Get an AzureOpenAI instance"""
    return AzureOpenAI(
        model=model_config.model_name,
        api_key=SecretStr(auth_token) if auth_token else None,
        azure_endpoint=model_config.api_base,
        api_version=model_config.api_version,
        model_kwargs={
            "engine": model_config.engine,
            "api_version": model_config.api_version,
            "api_type": "azure",
        },
    )


def _get_direct_model_(model_config: Any) -> BaseLanguageModel:
    """Get a direct API model instance for any provider without EAS authentication.

    BUG-4 FIX: Replaced bare `pass` in the except block with structured
    logging.  ImportError and ValueError are expected fallback conditions
    (missing provider package or bad model name) and emit a WARNING before
    falling through to the manual provider init.  All other exceptions are
    unexpected and are re-raised as LCELModelException with the original
    traceback preserved via `raise ... from e`.
    """
    try:
        from langchain.chat_models import init_chat_model  # pylint: disable=import-outside-toplevel

        model_name = model_config.model_name
        config_dict: dict = {}

        if hasattr(model_config, "api_base") and model_config.api_base:
            config_dict["base_url"] = model_config.api_base
        if hasattr(model_config, "temperature"):
            config_dict["temperature"] = model_config.temperature

        api_key_env = getattr(model_config, "api_key_env", None)
        if api_key_env:
            api_key = os.getenv(api_key_env)
            if api_key:
                config_dict["api_key"] = api_key

        return init_chat_model(model_name, **config_dict)

    except (ImportError, ValueError) as e:
        # Expected: missing provider package or unrecognised model name.
        # Log and fall through to the manual initialisation block below.
        logger.warning(
            "init_chat_model() failed for '%s' (%s: %s); falling back to manual provider init.",
            model_config.model_name,
            type(e).__name__,
            e,
        )
    except Exception as e:  # pylint: disable=broad-except
        # Unexpected failure: preserve the original traceback.
        raise LCELModelException(
            f"Unexpected error initialising model '{model_config.model_name}': {e}"
        ) from e

    # ── Manual provider-specific initialisation fallback ──────────────────────
    _supported_providers = ("openai", "anthropic", "google", "cohere", "huggingface")
    if model_config.provider not in _supported_providers:
        # SYSTEMATIC-DEBUGGING FIX: previously the API-key lookup below ran
        # unconditionally for *any* provider, including unsupported ones. An
        # unsupported provider like "meta" would fail with a misleading
        # "API key not found in environment variable: META_API_KEY" error
        # instead of the intended "not supported" message — because this
        # provider-support check used to live only in the trailing `else`
        # branch, after the API-key check had already raised. Checking
        # provider support first restores the intended fail-fast behaviour.
        raise LCELModelException(
            f"Provider '{model_config.provider}' not supported. "
            f"Supported providers: {', '.join(_supported_providers)}"
        )

    api_key_env = getattr(model_config, "api_key_env", None)
    if not api_key_env:
        api_key_env = f"{model_config.provider.upper()}_API_KEY"

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise LCELModelException(
            f"API key not found in environment variable: {api_key_env}. "
            f"Please set it in your .env file or environment."
        )

    if model_config.provider == "openai":
        api_base = getattr(model_config, "api_base", None)
        is_azure = api_base and "openai.azure.com" in str(api_base)
        if is_azure and hasattr(model_config, "api_version"):
            return AzureOpenAI(
                model=model_config.model_name,
                api_key=SecretStr(api_key),
                azure_endpoint=api_base,
                api_version=model_config.api_version,
                model_kwargs={
                    "engine": getattr(model_config, "engine", model_config.model_name),
                    "api_version": model_config.api_version,
                    "api_type": "azure",
                },
            )
        return ChatOpenAI(
            model=model_config.model_name,
            api_key=api_key,
            base_url=api_base,
        )

    elif model_config.provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic  # pylint: disable=import-outside-toplevel

            return ChatAnthropic(
                model=model_config.model_name,
                anthropic_api_key=api_key,
                anthropic_api_url=getattr(model_config, "api_base", None),
            )
        except ImportError as exc:
            raise LCELModelException(
                "langchain-anthropic not installed. Run: pip install langchain-anthropic"
            ) from exc

    elif model_config.provider == "google":
        try:
            from langchain_google_genai import (  # pylint: disable=import-outside-toplevel
                ChatGoogleGenerativeAI,
            )

            return ChatGoogleGenerativeAI(
                model=model_config.model_name,
                google_api_key=api_key,
            )
        except ImportError as exc:
            raise LCELModelException(
                "langchain-google-genai not installed. Run: pip install langchain-google-genai"
            ) from exc

    elif model_config.provider == "cohere":
        try:
            from langchain_cohere import ChatCohere  # pylint: disable=import-outside-toplevel

            return ChatCohere(
                model=model_config.model_name,
                cohere_api_key=api_key,
            )
        except ImportError as exc:
            raise LCELModelException(
                "langchain-cohere not installed. Run: pip install langchain-cohere"
            ) from exc

    elif model_config.provider == "huggingface":
        try:
            from langchain_huggingface import (  # pylint: disable=import-outside-toplevel
                HuggingFaceEndpoint,
            )

            return HuggingFaceEndpoint(
                repo_id=model_config.model_name,
                huggingfacehub_api_token=api_key,
                endpoint_url=getattr(model_config, "api_base", None),
            )
        except ImportError as exc:
            raise LCELModelException(
                "langchain-huggingface not installed. Run: pip install langchain-huggingface"
            ) from exc

    # Unreachable: the _supported_providers guard above already rejects any
    # provider not handled by one of the branches. Raising here (rather than
    # falling off the end or returning None) keeps the BaseLanguageModel
    # return-type contract intact for mypy and any future providers added
    # to _supported_providers without a matching branch.
    raise LCELModelException(f"No initialisation branch for provider '{model_config.provider}'")
