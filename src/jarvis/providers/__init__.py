"""Cloud LLM provider implementations for the hybrid router.

Each provider mirrors the three public surface functions of
``jarvis.llm`` (``call_llm_direct``, ``call_llm_streaming``,
``chat_with_messages``) so the router can swap providers transparently.

Privacy contract: provider modules are only imported when the router
mode is not ``local_only``. A 100%-local install never loads these
modules and never imports the cloud SDK.
"""
