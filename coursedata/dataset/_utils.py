"""Shared utilities for the dataset CLI."""

from datetime import date
import os
import subprocess
from typing import Optional

import keyring
from loguru import logger

d8 = date.today().isoformat()


def get_password(service: str, username: str) -> Optional[str]:
    """
    Get password from macOS Keychain.

    First tries to retrieve as an internet password (more common for web services),
    then falls back to generic password if not found.
    """
    # Try internet password first
    try:
        result = subprocess.run(
            ["security", "find-internet-password", "-s", service, "-a", username, "-w"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass

    # Fall back to generic password
    return keyring.get_password(service, username)


def get_sso_credentials() -> tuple[Optional[str], Optional[str]]:
    """Get SSO username and password from environment and keychain."""
    username = os.getenv("SSO_USERNAME")
    if not username:
        logger.warning(
            "SSO_USERNAME not found in environment variables. Set it in your .env file."
        )
        username = None

    password = None
    if username:
        password = get_password("nyu-sso", username)
        if not password:
            logger.warning(
                f"Password for user '{username}' not found in macOS Keychain. "
                f"Store it with: security add-generic-password -s nyu-sso -a {username} -w YOUR_PASSWORD"
            )

    return username, password
