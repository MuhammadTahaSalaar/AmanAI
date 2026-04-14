"""Authentication and authorization manager for AmanAI."""

from __future__ import annotations

import logging

import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class AuthManager:
    """Handles user authentication and authorization."""

    def __init__(self) -> None:
        self._admin_password = config.ADMIN_PASSWORD
        self._guest_user = config.GUEST_USER
        self._admin_user = config.ADMIN_USER

    def authenticate(self, username: str, password: str = "") -> tuple[bool, str]:
        """Authenticate user with username and optional password.

        Args:
            username: The username ('admin' or 'guest')
            password: The password (required for admin, ignored for guest)

        Returns:
            Tuple of (is_authenticated, error_message)
            If authenticated, error_message is empty.
            If failed, error_message contains the reason.
        """
        if username == self._guest_user:
            logger.info("Guest login successful")
            return True, ""

        if username == self._admin_user:
            if password == self._admin_password:
                logger.info("Admin login successful")
                return True, ""
            else:
                logger.warning("Admin login failed: incorrect password")
                return False, "Incorrect password"

        logger.warning("Login failed: unknown username '%s'", username)
        return False, "Unknown username. Use 'admin' or 'guest'"

    def is_admin(self, username: str) -> bool:
        """Check if user is admin.

        Args:
            username: The username to check

        Returns:
            True if the user is admin, False otherwise
        """
        return username == self._admin_user

    def can_upload_documents(self, username: str) -> bool:
        """Check if user can upload documents.

        Args:
            username: The username to check

        Returns:
            True if the user can upload documents, False otherwise
        """
        return self.is_admin(username)
