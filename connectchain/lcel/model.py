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
import os
from typing import Any

from langchain.schema.language_model import BaseLanguageModel
from langchain_openai import AzureOpenAI, ChatOpenAI
from pydantic import SecretStr

from connectchain.utils import Config, SessionMap, get_token_from_env
from connectchain.utils.llm_proxy_wrapper import wrap_llm_with_proxy


class LCELModelException(Exception):
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

    # Check if we should use direct access (no EAS)
    needs_eas = False
    try:
        # Check if EAS is configured and not bypassed
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

    # For non-OpenAI providers, always use direct access
    if model_config.provider != "openai":
        needs_eas = False

    if needs_eas:
        # Use existing EAS flow for OpenAI
        model_instance = _get_openai_model_(index, config, model_config)
    else:
        # Use direct access for any provider
        model_instance = _get_direct_model_(model_config)

    if model_instance is None:
        raise LCELModelException("Not implemented")
    try:
        proxy_config = model_config.proxy
    except KeyError:
        # Proxy settings not required
        pass
    if proxy_config is None:
        try:
            proxy_config = config.proxy
            # Proxy settings not required
        except KeyError:
            pass
    if proxy_config is not None:
        # Convert to BaseLLM if needed for proxy wrapping
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
        # Note: SessionMap expects LLMResult but we're storing LLM instances
        session_map.new_session(model_session_key, llm)  # type: ignore[arg-type]
        return llm
    # Note: SessionMap returns LLMResult but we need BaseLanguageModel
    return session_map.get_llm(model_session_key)  # type: ignore[return-value]


def _get_chat_model_(auth_token: str, model_config: Any) -> ChatOpenAI:
    """Get a ChatOpenAI instance"""
    llm = ChatOpenAI(
        # Note: ChatOpenAI uses model parameter
        model=model_config.model_name,
        api_key=SecretStr(auth_token) if auth_token else None,
        base_url=model_config.api_base,
        model_kwargs={
            "engine": model_config.engine,
            "api_version": model_config.api_version,
            "api_type": "azure",
        },
    )
    return llm


def _get_azure_model_(auth_token: str, model_config: Any) -> AzureOpenAI:
    """Get an AzureOpenAI instance"""
    llm = AzureOpenAI(
        # Note: AzureOpenAI uses model parameter
        model=model_config.model_name,
        api_key=SecretStr(auth_token) if auth_token else None,
        azure_endpoint=model_config.api_base,
        api_version=model_config.api_version,
        # api_version is already a direct kwarg above; AzureOpenAI's pydantic validation
        # rejects it appearing a second time inside model_kwargs ("supplied twice").
        model_kwargs={
            "engine": model_config.engine,
            "api_type": "azure",
        },
    )
    return llm


def _get_direct_model_(model_config: Any) -> BaseLanguageModel:
    """Get a direct API model instance for any provider without EAS authentication"""

    try:
        # Try using LangChain's automatic model initialization
        from langchain.chat_models import init_chat_model

        # Prepare model name
        model_name = model_config.model_name

        # Prepare configuration kwargs
        config_dict = {}

        # Add custom API base if specified
        if hasattr(model_config, "api_base") and model_config.api_base:
            config_dict["base_url"] = model_config.api_base

        # Add temperature if specified. Note: getattr's default is never returned here
        # because ConfigWrapper.__getattr__ returns None (not AttributeError) for a
        # missing key, so we must check the resolved value instead of using hasattr().
        temperature = getattr(model_config, "temperature", None)
        if temperature is not None:
            config_dict["temperature"] = temperature

        # Handle custom API key environment variable
        api_key_env = getattr(model_config, "api_key_env", None)
        if api_key_env:
            api_key = os.getenv(api_key_env)
            if api_key:
                config_dict["api_key"] = api_key

        # Try automatic initialization
        return init_chat_model(model_name, **config_dict)

    except (ImportError, ValueError, Exception) as e:
        # Fallback to manual provider-specific initialization
        pass

    # Manual provider-specific initialization as fallback
    # Determine API key environment variable
    api_key_env = getattr(model_config, "api_key_env", None)
    if not api_key_env:
        # Default to {PROVIDER}_API_KEY pattern
        api_key_env = f"{model_config.provider.upper()}_API_KEY"

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise LCELModelException(
            f"API key not found in environment variable: {api_key_env}. "
            f"Please set it in your .env file or environment."
        )

    # Provider-specific instantiation
    if model_config.provider == "openai":
        # Check if this is Azure OpenAI based on api_base
        api_base = getattr(model_config, "api_base", None)
        is_azure = api_base and "openai.azure.com" in str(api_base)

        api_version = getattr(model_config, "api_version", None)
        if is_azure and api_version:
            # Azure OpenAI with direct API key. Note: getattr(..., "engine", <default>) can't
            # fall back here either, for the same ConfigWrapper.__getattr__-returns-None reason
            # as above, so the model_name fallback is applied explicitly.
            engine = getattr(model_config, "engine", None) or model_config.model_name
            return AzureOpenAI(
                model=model_config.model_name,
                api_key=SecretStr(api_key),
                azure_endpoint=api_base,
                api_version=api_version,
                # api_version is already a direct kwarg above; AzureOpenAI's pydantic
                # validation rejects it appearing a second time inside model_kwargs.
                model_kwargs={
                    "engine": engine,
                    "api_type": "azure",
                },
            )
        else:
            # Standard OpenAI
            return ChatOpenAI(
                model=model_config.model_name,
                api_key=api_key,
                base_url=api_base,  # Can be None for default OpenAI endpoint
            )

    elif model_config.provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=model_config.model_name,
                anthropic_api_key=api_key,
                anthropic_api_url=getattr(model_config, "api_base", None),
            )
        except ImportError:
            raise LCELModelException(
                "langchain-anthropic not installed. Run: pip install langchain-anthropic"
            )

    elif model_config.provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=model_config.model_name,
                google_api_key=api_key,
            )
        except ImportError:
            raise LCELModelException(
                "langchain-google-genai not installed. Run: pip install langchain-google-genai"
            )

    elif model_config.provider == "cohere":
        try:
            from langchain_cohere import ChatCohere

            return ChatCohere(
                model=model_config.model_name,
                cohere_api_key=api_key,
            )
        except ImportError:
            raise LCELModelException(
                "langchain-cohere not installed. Run: pip install langchain-cohere"
            )

    elif model_config.provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEndpoint

            return HuggingFaceEndpoint(
                repo_id=model_config.model_name,
                huggingfacehub_api_token=api_key,
                endpoint_url=getattr(model_config, "api_base", None),
            )
        except ImportError:
            raise LCELModelException(
                "langchain-huggingface not installed. Run: pip install langchain-huggingface"
            )

    else:
        raise LCELModelException(
            f"Provider '{model_config.provider}' not supported. "
            f"Supported providers: openai, anthropic, google, cohere, huggingface"
        )
