"""Common types and result classes for tools."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ToolExecutionResult:
    """Result object for tool execution."""
    success: bool
    reply_text: Optional[str]
    error_message: Optional[str] = None
    # The policy gate declined to run this tool. Distinct from a failure
    # on purpose: a failure invites a retry, and retrying a tool that was
    # just denied is a loop the user watches and cannot stop. Callers
    # that decide whether to try again must check this first.
    refused: bool = False
    # The gate asked the user instead, and this identifies the question
    # now waiting on them. Nothing ran and nothing failed — the turn ends
    # here and the answer arrives on a later one. Kept apart from
    # ``refused`` because the two mean opposite things to a caller: a
    # refusal closes the matter, a question is waiting on someone.
    pending_id: Optional[str] = None
