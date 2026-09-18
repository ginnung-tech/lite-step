"""
Lite-STEP Compiler - Safe execution of LLM-generated scripts.

The compiler provides:
- Restricted namespace for safe script execution
- Project object extraction and validation
- Error handling with line numbers
"""

from .executor import execute_lite_step_script, ExecutionResult
from .namespace import create_namespace

__all__ = [
    "execute_lite_step_script",
    "ExecutionResult",
    "create_namespace",
]
