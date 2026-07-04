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
"""This module is used to keep track of the session expiration time"""
import threading
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from langchain.schema import LLMResult


class SessionMap:
    """This class is used to keep track of the session expiration time.

    BUG-2 FIX: Added existence guard in is_expired() to prevent KeyError when
    a session_id has not yet been registered (returns True so callers trigger
    a token refresh).  A threading.Lock() protects all read/write operations
    to avoid race conditions under concurrent async load.
    """

    _instance: Optional["SessionMap"] = None
    _lock: threading.Lock = threading.Lock()
    session_map: Dict[str, Tuple[datetime, LLMResult]] = {}
    expires_in: int = -1

    def __new__(cls, expires_in: int = 900) -> "SessionMap":
        if cls._instance is None:
            cls._instance = super(SessionMap, cls).__new__(cls)
            cls._instance.expires_in = expires_in
            cls._instance._lock = threading.Lock()
        return cls._instance

    def new_session(self, session_id: str, llm: LLMResult) -> None:
        """Save new session for later."""
        with self._lock:
            self.session_map[session_id] = (datetime.now(), llm)

    def is_expired(self, session_id: str) -> bool:
        """Check if the session is expired.

        Returns True (treat as expired) when session_id is not yet registered
        so that callers trigger a fresh token acquisition rather than raising
        a KeyError.
        """
        with self._lock:
            if session_id not in self.session_map:
                return True
            return (
                datetime.now() - self.session_map[session_id][0]
            ).total_seconds() > self.expires_in

    def get_llm(self, session_id: str) -> LLMResult:
        """Get the LLM instance from the session."""
        with self._lock:
            return self.session_map[session_id][1]

    @staticmethod
    def uuid_from_config(config: Any, model_config: Any) -> str:
        """Generate a UUID from the config."""
        env_id_key = None
        env_secret_key = None
        if model_config.eas:
            env_id_key = model_config.eas.id_key
            env_secret_key = model_config.eas.secret_key
        if env_id_key is None:
            env_id_key = config.eas.id_key
        if env_secret_key is None:
            env_secret_key = config.eas.secret_key
        env_uuid = f"{env_id_key}_{env_secret_key}"
        model_uuid = f"{model_config.provider}_{model_config.type}_{model_config.engine}"
        model_uuid += f"_{model_config.model_name}_{model_config.api_version}"
        return f"{env_uuid}_{model_uuid}"
