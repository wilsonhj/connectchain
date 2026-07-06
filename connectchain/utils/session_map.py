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
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from langchain.schema import LLMResult


class SessionMap:
    """This class is used to keep track of the session expiration time"""

    _instance: Optional["SessionMap"] = None
    # Each entry stores the expires_in that was active when it was cached, not the
    # singleton's current value -- a later SessionMap(different_interval) call must not
    # retroactively change the expiry policy for sessions cached under a prior interval.
    session_map: Dict[str, Tuple[datetime, int, LLMResult]] = {}
    expires_in: int = -1

    def __new__(cls, expires_in: int = 900) -> "SessionMap":
        if cls._instance is None:
            cls._instance = super(SessionMap, cls).__new__(cls)
        # Update on every call, not just the first -- this is a singleton, so a later
        # caller configuring a different expires_in must not be silently ignored for
        # sessions it caches from here on. (Already-cached sessions keep their own
        # expires_in, captured in new_session(), so this doesn't affect them.)
        cls._instance.expires_in = expires_in
        return cls._instance

    def new_session(self, session_id: str, llm: LLMResult, expires_in: Optional[int] = None) -> None:
        """Save new session for later.

        Pass `expires_in` explicitly when the caller captured it before doing
        anything slow (e.g. a network call): reading self.expires_in only at this
        point would race a concurrent SessionMap(other_interval) call made by another
        request in between, silently applying the wrong expiry to this session.
        """
        if expires_in is None:
            expires_in = self.expires_in
        self.session_map[session_id] = (datetime.now(), expires_in, llm)

    def is_expired(self, session_id: str) -> bool:
        """check if the session is expired"""
        cached_at, expires_in, _ = self.session_map[session_id]
        return (datetime.now() - cached_at).total_seconds() > expires_in

    def get_llm(self, session_id: str) -> LLMResult:
        """get the LLM instance from the session"""
        return self.session_map[session_id][2]

    @staticmethod
    def uuid_from_config(config: Any, model_config: Any) -> str:
        """generate a uuid from the config"""
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
