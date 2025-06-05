# tests/conftest.py
import sys
import types
import pytest

# Create stub module for llm_factory_toolkit
llm_mod = types.ModuleType("llm_factory_toolkit")


class DummyLLMClient:
    def __init__(self, *args, **kwargs):
        self.calls = []
        self.response = ""

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.response, None


class DummyToolFactory:
    def __init__(self):
        self.tools = {}

    def register_tool(self, **kw):
        self.tools[kw.get("name")] = kw


class ProviderError(Exception):
    pass


class ToolError(Exception):
    pass


class LLMToolkitError(Exception):
    pass


exc_mod = types.ModuleType("llm_factory_toolkit.exceptions")
exc_mod.ProviderError = ProviderError
exc_mod.ToolError = ToolError
exc_mod.LLMToolkitError = LLMToolkitError

models_mod = types.ModuleType("llm_factory_toolkit.tools.models")


class ToolExecutionResult:
    def __init__(self, content: str, error: str | None = None):
        self.content = content
        self.error = error


models_mod.ToolExecutionResult = ToolExecutionResult

# Assemble module structure
llm_mod.LLMClient = DummyLLMClient
llm_mod.ToolFactory = DummyToolFactory
llm_mod.exceptions = exc_mod
llm_mod.tools = types.ModuleType("llm_factory_toolkit.tools")
llm_mod.tools.models = models_mod

# Register in sys.modules so imports succeed
sys.modules["llm_factory_toolkit"] = llm_mod
sys.modules["llm_factory_toolkit.exceptions"] = exc_mod
sys.modules["llm_factory_toolkit.tools"] = llm_mod.tools
sys.modules["llm_factory_toolkit.tools.models"] = models_mod


# Provide fixtures if needed
@pytest.fixture()
def dummy_llm_client(monkeypatch):
    client = DummyLLMClient()
    monkeypatch.setattr("signup_chatbot.signup.LLMClient", lambda *a, **k: client)
    return client
