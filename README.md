# Signup Chatbot

This project provides a small Python library to help build chatbots that guide users through a signup flow. It exposes utilities for storing session state, prompting an LLM to collect profile data and updating that data via a tool interface.

## Features

- **Signup handler** – `Signup` orchestrates the conversation with an LLM and manages which profile fields are still missing.
- **Memory component** – stores per user chat history and metadata in memory.
- **Edit profile tool** – allows the language model to persist profile updates via your own callback.
- **Pydantic based configuration** – prompts and behaviour can be customised using `SignupConfig`.
- **Example app** – see the `examples/` directory for a fully working interactive demo.

## Installation

```bash
pip install -e .
```

Development extras (tests and linters) can be installed with:

```bash
pip install -e .[dev]
```

## Running the tests

Execute the test suite with:

```bash
pytest tests
```

The interactive example requires an OpenAI API key and the optional `email-validator` package for `EmailStr` validation. See `examples/interactive_test.py` for details.

## License

This project is released under the [MIT License](LICENSE).