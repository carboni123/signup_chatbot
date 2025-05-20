# signup_chatbot/__init__.py
import logging
from .signup import Signup
from .config_models import SignupConfig
from .memory import Memory # If users of the lib might want to use/inspect Memory directly

# Setup of a default null handler for the library's root logger
# This prevents messages from being output to stderr if the consuming application doesn't configure logging
# See: https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library
library_root_logger = logging.getLogger(__name__)
if not library_root_logger.hasHandlers():
    library_root_logger.addHandler(logging.NullHandler())

__all__ = ["Signup", "SignupConfig", "Memory"]