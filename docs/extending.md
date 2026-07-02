# Extending the framework

Every extension point is small and local. After any change: `make lint && make test`.

## Add an LLM provider

1. If it speaks the OpenAI Chat Completions dialect, subclass `OpenAICompatibleBrain`
   in `brains/providers/concrete.py`:

   ```python
   class MyProviderBrain(OpenAICompatibleBrain):
       def __init__(self, s: Settings) -> None:
           super().__init__(
               api_key=s.myprovider_api_key,
               base_url="https://api.myprovider.com/v1",
               chat_model=s.myprovider_chat_model,
               embedding_model=s.myprovider_embedding_model,  # or None
               name="myprovider",
           )
   ```

   For a non-OpenAI dialect, subclass `Brain` directly and implement `chat` (+ optionally
   `stream`, `structured_output`, `embed`).

2. Add the enum value to `config/settings.py` (`LLMProvider` / `EmbeddingProvider`) and
   the provider's settings fields.
3. Add a branch in `brains/factory.py._build`.
4. Document defaults in `.env.example`.

That's it — `LLM_PROVIDER=myprovider` now works everywhere.

## Add a tool

```python
# tools/builtins/my_tool.py
from pydantic import BaseModel, Field
from ai_employee.tools.base import Tool, ToolResult

class WeatherTool(Tool):
    name = "weather"
    description = "Get the current weather for a city."
    category = "web"
    # approval_key = "shell_write"   # set if the tool mutates external state

    class Args(BaseModel):
        city: str = Field(description="City name.")

    async def _run(self, args: "WeatherTool.Args") -> ToolResult:
        ...  # do work
        return ToolResult(output={"city": args.city, "temp_c": 21})
```

Register it in `tools/builtins/__init__.py::build_builtin_tools`. The `Args` model is
automatically advertised to the LLM as a JSON-schema tool spec.

## Add an agent

Declarative (most cases) — add an `AgentSpec` to `agents/specs.py`:

```python
AgentSpec(
    name="translator",
    role="Translator",
    description="Translates text between languages.",
    category="language",
    tools=(),  # tools it may use; () = none, None = all
    system_prompt="You are an expert translator. Preserve tone and meaning.",
)
```

Bespoke logic — add a `BaseAgent` subclass to `agents/custom.py` and register it in
`agents/registry.py::build_agent_registry`.

## Add an API endpoint

Add a route to `api/routes.py` using the `get_employee` dependency; add request/response
models to `api/schemas.py`.

## Add a knowledge loader

Add a `_load_*` function in `knowledge/loaders.py` and wire it into the extension map in
`_load_file` (or URL handling in `load_source`). Return a `LoadedDocument`.
