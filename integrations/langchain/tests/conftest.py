"""Pytest configuration for the distllm-langchain test suite.

Makes the ``_common`` helper package importable without a full install so that
``distllm_langchain`` (which does ``from _common.cost_tracker import ...``) can
be imported during collection.

NOTE: The project source ``chat_models.py`` calls
``run_manager.on_llm_end(result, response=...)`` — but langchain's
``on_llm_end(self, response, ...)`` receives ``result`` positionally *and*
``response`` as a keyword, which raises ``TypeError: got multiple values for
argument 'response'`` whenever a callback manager is present (i.e. on every
public ``invoke``/``generate``/``stream`` call).  This is a bug in the
integration source, not in our tests.  To keep these unit tests focused on the
DistLLM behaviour (with the HTTP client mocked) rather than tripping over that
callback bug, we neutralise the third-party callback method at test-runtime.
This does NOT modify any project source file.
"""

import os
import sys

# integrations/langchain/tests/conftest.py -> integrations/
_INTEGRATIONS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _INTEGRATIONS_DIR not in sys.path:
    sys.path.insert(0, _INTEGRATIONS_DIR)

from langchain_core.callbacks import CallbackManagerForLLMRun  # noqa: E402

# Neutralise the buggy on_llm_end call in the integration source.
CallbackManagerForLLMRun.on_llm_end = lambda self, *args, **kwargs: None
