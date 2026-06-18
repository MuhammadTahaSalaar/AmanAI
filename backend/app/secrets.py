"""Runtime secret resolution: fetch Supabase key from AWS Secrets Manager if needed."""
from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def resolve_supabase_key(service_key: str, secret_arn: str, region: str) -> str:
    """Return the Supabase service key.

    If *service_key* is non-empty it is returned directly.
    Otherwise, if *secret_arn* is set, the key is fetched from Secrets Manager.
    Raises RuntimeError if neither source yields a value.
    """
    if service_key:
        return service_key

    if not secret_arn:
        raise RuntimeError(
            "Supabase service key is not configured: set SUPABASE_SERVICE_KEY "
            "or SUPABASE_SECRET_ARN."
        )

    try:
        import boto3  # local import — not available in all test envs
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("boto3 is required to fetch secrets from Secrets Manager") from exc

    logger.info("Fetching Supabase key from Secrets Manager ARN: %s", secret_arn)
    client = boto3.client("secretsmanager", region_name=region)
    response = client.get_secret_value(SecretId=secret_arn)
    secret_str = response.get("SecretString", "")
    try:
        data = json.loads(secret_str)
        key = data.get("SUPABASE_SERVICE_KEY", "")
    except (json.JSONDecodeError, AttributeError):
        key = secret_str

    if not key:
        raise RuntimeError(
            f"Secret at {secret_arn} did not contain a non-empty SUPABASE_SERVICE_KEY."
        )

    return key
