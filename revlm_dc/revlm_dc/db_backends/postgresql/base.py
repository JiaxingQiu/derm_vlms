import logging
import os
import subprocess

from django.db.backends.postgresql.base import DatabaseWrapper as PostgresDatabaseWrapper

logger = logging.getLogger(__name__)


class DatabaseWrapper(PostgresDatabaseWrapper):
    def get_connection_params(self):
        params = super().get_connection_params()
        password = _resolve_password()
        if password:
            params["password"] = password
        return params


def _resolve_password() -> str:
    password = os.getenv("DJANGO_DB_PASSWORD", "").strip()
    password_command = os.getenv("DJANGO_DB_PASSWORD_COMMAND", "").strip()

    if password_command:
        try:
            resolved_password = subprocess.check_output(
                password_command,
                shell=True,
                text=True,
            ).strip()
            if not resolved_password and not password:
                logger.error(
                    "DJANGO_DB_PASSWORD_COMMAND returned an empty token and no DJANGO_DB_PASSWORD fallback was provided"
                )
                raise RuntimeError(
                    "DJANGO_DB_PASSWORD_COMMAND returned an empty token and no DJANGO_DB_PASSWORD fallback was provided"
                )
            return resolved_password or password
        except subprocess.CalledProcessError as exc:
            if not password:
                logger.exception(
                    "DJANGO_DB_PASSWORD_COMMAND failed and no DJANGO_DB_PASSWORD fallback was provided"
                )
                raise RuntimeError(
                    "Failed to get DJANGO_DB_PASSWORD from DJANGO_DB_PASSWORD_COMMAND and no DJANGO_DB_PASSWORD fallback was provided"
                ) from exc

            logger.warning(
                "DJANGO_DB_PASSWORD_COMMAND failed; falling back to DJANGO_DB_PASSWORD"
            )

    if not password:
        logger.error(
            "No PostgreSQL password source configured; set DJANGO_DB_PASSWORD or DJANGO_DB_PASSWORD_COMMAND"
        )
        raise RuntimeError(
            "No PostgreSQL password source configured; set DJANGO_DB_PASSWORD or DJANGO_DB_PASSWORD_COMMAND"
        )

    return password
