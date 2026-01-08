from __future__ import annotations

"""S3 input source implementation."""

import logging
import tempfile
from pathlib import Path
from typing import Any

from .protocol import InputSource


class S3InputSource:
    """Input source for AWS S3 buckets.

    Downloads files from S3 to temporary locations for processing.
    Handles cleanup of temporary files.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize the S3 input source.

        Parameters
        ----------
        config
            Configuration dictionary with keys:
            - bucket_name: str (required) - S3 bucket name
            - prefix: str (optional) - S3 key prefix to filter files
            - aws_access_key_id: str (optional) - AWS access key
            - aws_secret_access_key: str (optional) - AWS secret key
            - region_name: str (optional) - AWS region
            - endpoint_url: str (optional) - Custom S3 endpoint (for S3-compatible services)
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.temp_files: list[Path] = []

        # Validate required config
        if "bucket_name" not in config:
            raise ValueError("bucket_name is required for S3 input source")

        self.bucket_name = config["bucket_name"]
        self.prefix = config.get("prefix", "")

        # Initialize boto3 client (lazy import)
        self._s3_client = None

    @property
    def s3_client(self):
        """Get or create S3 client (lazy initialization)."""
        if self._s3_client is None:
            try:
                import boto3
            except ImportError as e:
                raise ImportError(
                    "boto3 is required for S3 input source. "
                    "Install with: pip install boto3"
                ) from e

            # Build client kwargs
            client_kwargs = {}
            if "aws_access_key_id" in self.config:
                client_kwargs["aws_access_key_id"] = self.config["aws_access_key_id"]
            if "aws_secret_access_key" in self.config:
                client_kwargs["aws_secret_access_key"] = self.config[
                    "aws_secret_access_key"
                ]
            if "region_name" in self.config:
                client_kwargs["region_name"] = self.config["region_name"]
            if "endpoint_url" in self.config:
                client_kwargs["endpoint_url"] = self.config["endpoint_url"]

            self._s3_client = boto3.client("s3", **client_kwargs)
            self.logger.info(f"Initialized S3 client for bucket: {self.bucket_name}")

        return self._s3_client

    def list_files(self, path: str, extensions: list[str] | None = None) -> list[str]:
        """List files in S3 bucket with optional prefix and extension filtering.

        Parameters
        ----------
        path
            S3 prefix to list files from (relative to configured prefix).
        extensions
            Optional list of file extensions to filter by.

        Returns
        -------
        List of S3 URIs in format: s3://bucket/key
        """
        # Combine configured prefix with path
        full_prefix = self.prefix
        if path and path != ".":
            full_prefix = f"{self.prefix.rstrip('/')}/{path.lstrip('/')}"

        self.logger.info(f"Listing S3 objects: s3://{self.bucket_name}/{full_prefix}")

        files = []
        paginator = self.s3_client.get_paginator("list_objects_v2")

        try:
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=full_prefix):
                if "Contents" not in page:
                    continue

                for obj in page["Contents"]:
                    key = obj["Key"]

                    # Skip directories (keys ending with /)
                    if key.endswith("/"):
                        continue

                    # Filter by extensions if provided
                    if extensions is not None:
                        file_ext = Path(key).suffix.lower()
                        if file_ext not in extensions:
                            continue

                    # Return as S3 URI
                    s3_uri = f"s3://{self.bucket_name}/{key}"
                    files.append(s3_uri)

        except Exception as e:
            self.logger.error(f"Failed to list S3 objects: {e}")
            raise RuntimeError(f"Failed to list S3 objects: {e}") from e

        self.logger.info(f"Found {len(files)} file(s) in S3")
        return sorted(files)

    def get_file(self, file_id: str) -> Path:
        """Download S3 file to temporary location.

        Parameters
        ----------
        file_id
            S3 URI in format: s3://bucket/key

        Returns
        -------
        Path to the downloaded temporary file.
        """
        # Parse S3 URI
        if not file_id.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {file_id}")

        parts = file_id[5:].split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 URI format: {file_id}")

        bucket, key = parts

        # Verify bucket matches
        if bucket != self.bucket_name:
            raise ValueError(
                f"Bucket mismatch: expected {self.bucket_name}, got {bucket}"
            )

        # Create temporary file with same extension
        suffix = Path(key).suffix
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, prefix="s3_"
        )
        temp_path = Path(temp_file.name)
        temp_file.close()

        self.logger.info(f"Downloading {file_id} to {temp_path}")

        try:
            self.s3_client.download_file(bucket, key, str(temp_path))
            self.temp_files.append(temp_path)
            return temp_path
        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            self.logger.error(f"Failed to download {file_id}: {e}")
            raise RuntimeError(f"Failed to download {file_id}: {e}") from e

    def cleanup(self) -> None:
        """Clean up all temporary downloaded files."""
        self.logger.info(f"Cleaning up {len(self.temp_files)} temporary file(s)")

        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    self.logger.debug(f"Deleted temporary file: {temp_file}")
            except Exception as e:
                self.logger.warning(f"Failed to delete {temp_file}: {e}")

        self.temp_files.clear()


def create_s3_source(config: dict[str, Any]) -> InputSource:
    """Create an S3 input source.

    Parameters
    ----------
    config
        Configuration dictionary with S3 settings.

    Returns
    -------
    InputSource instance for S3.

    Raises
    ------
    ValueError
        If required configuration is missing.
    ImportError
        If boto3 is not installed.
    """
    return S3InputSource(config)
