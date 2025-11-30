"""
Storage infrastructure with Hot/Cold separation architecture.

This module provides:
- Cold Storage: Google Drive via gdrivefs for large assets and archival
- Hot Storage: LangGraph PostgresSaver for checkpointing and active state

Usage:
    # Cold Storage (Google Drive)
    from pyomnix.agents.storage import DriveManager

    drive = DriveManager()
    await drive.initialize()
    metadata = await drive.upload_asset(b"data", "file.txt")
    content = await drive.read_asset(metadata["file_id"])

    # Hot Storage (PostgreSQL Checkpointer)
    from pyomnix.agents.storage import get_checkpointer

    async with get_checkpointer() as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        # Use the graph...
"""

import asyncio
import importlib
import json
import mimetypes
import os
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any, BinaryIO, cast

from pyomnix.agents.models import Settings, get_settings
from pyomnix.omnix_logger import get_logger
from pyomnix.utils.gdrive import GoogleDriveFileSystem

_postgres_module: Any
try:  # pragma: no cover - optional dependency
    _postgres_module = importlib.import_module("langgraph.checkpoint.postgres")
except ImportError:  # pragma: no cover
    AsyncPostgresSaver = None  # type: ignore[assignment]
    PostgresSaver = None  # type: ignore[assignment]
else:
    AsyncPostgresSaver = _postgres_module.AsyncPostgresSaver  # type: ignore[attr-defined]
    PostgresSaver = _postgres_module.PostgresSaver  # type: ignore[attr-defined]

try:  # pragma: no cover - optional dependency
    _psycopg_pool = importlib.import_module("psycopg_pool")
except ImportError:  # pragma: no cover
    AsyncConnectionPool = None  # type: ignore[assignment]
else:
    AsyncConnectionPool = _psycopg_pool.AsyncConnectionPool  # type: ignore[attr-defined]

logger = get_logger(__name__)


# ===============================================================
# Part 1: Cold Storage - Google Drive via gdrivefs
# ===============================================================


class DriveManager:
    """
    Manages Google Drive interactions for cold storage using gdrivefs.

    This class handles authentication via service account and provides
    async methods for uploading and retrieving assets from Google Drive.

    Attributes:
        settings: Application settings containing service account path.
        folder_id: Target folder ID for uploads.
        fs: The gdrivefs filesystem instance.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        folder_id: str | None = None,
    ) -> None:
        """
        Initialize the DriveManager.

        Args:
            settings: Optional Settings instance. Uses get_settings() if not provided.
            folder_id: Optional folder ID override. Uses Settings.DRIVE_FOLDER_ID if not
                provided.
        """
        self.settings = settings or get_settings()
        self.folder_id = folder_id or self.settings.gdrive_folder_id or "root"
        self._initialized = False
        self._service_account: dict[str, Any] | None = None
        self._fs_cache: dict[str, GoogleDriveFileSystem] = {}
        self._drive_files_resource: Any | None = None

    async def initialize(self) -> None:
        """
        Initialize the Google Drive filesystem with service account authentication.

        Raises:
            ValueError: If credentials are missing or invalid.
        """
        if self._initialized:
            logger.debug("DriveManager already initialized, skipping.")
            return

        if not self.settings.validate_cold_storage():
            logger.raise_error(
                "GDRIVE_KEY is not configured. Update api_config.json or env vars.",
                ValueError,
            )

        self._ensure_service_account_loaded()
        await asyncio.to_thread(self._get_fs, self.folder_id)
        self._initialized = True
        logger.info("DriveManager initialized successfully with service account.")

    def _ensure_initialized(self) -> None:
        """Ensure the filesystem is initialized before operations."""
        if not self._initialized:
            logger.raise_error(
                "DriveManager not initialized. Call initialize() first.",
                RuntimeError,
            )

    def _ensure_service_account_loaded(self) -> None:
        """Load and cache the service account credentials."""
        if self._service_account is not None:
            return

        key_data = self.settings.gdrive_key
        if not key_data:
            logger.raise_error(
                "GDRIVE_KEY is not configured. Update api_config.json or env vars.",
                ValueError,
            )

        expanded = Path(os.path.expandvars(key_data)).expanduser()
        if expanded.is_file():
            raw_json = expanded.read_text(encoding="utf-8")
        else:
            raw_json = key_data

        try:
            self._service_account = json.loads(raw_json)
        except json.JSONDecodeError as exc:  # pragma: no cover - configuration error
            logger.raise_error(
                (
                    "GDRIVE_KEY must be either a JSON string or a path to a valid "
                    f"service account JSON file. Details: {exc}"
                ),
                ValueError,
            )

    def _resolve_target_folder(self, folder_id: str | None) -> str:
        """Resolve the folder that should act as the root for the operation."""
        return folder_id or self.folder_id

    def _get_fs(self, folder_id: str | None) -> GoogleDriveFileSystem:
        """
        Get or create a GoogleDriveFileSystem scoped to the specified folder ID.
        """
        target_folder = self._resolve_target_folder(folder_id)
        fs = self._fs_cache.get(target_folder)
        if fs is not None:
            return fs

        if self._service_account is None:
            self._ensure_service_account_loaded()

        fs = GoogleDriveFileSystem(
            token="service_account",
            root_file_id=target_folder,
            creds=self._service_account,
        )
        self._fs_cache[target_folder] = fs
        if self._drive_files_resource is None:
            self._drive_files_resource = fs.files
        return fs

    def _get_drive_files_resource(self) -> Any:
        """
        Return the Google Drive files resource for direct API interactions.
        """
        if self._drive_files_resource is None:
            fs = self._fs_cache.get(self.folder_id)
            if fs is None:
                fs = self._get_fs(self.folder_id)
            self._drive_files_resource = fs.files
        return self._drive_files_resource

    async def upload_asset(
        self,
        data: bytes,
        filename: str,
        folder_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Upload a file to Google Drive and return metadata.

        Args:
            data: The file content as bytes.
            filename: The name for the uploaded file.
            folder_id: Optional folder ID override. Uses instance folder_id if not provided.

        Returns:
            dict containing:
                - file_id: The Google Drive file ID
                - web_link: The web view link for the file
                - mime_type: The detected MIME type of the file

        Raises:
            RuntimeError: If DriveManager is not initialized.
            ValueError: If no folder_id is available.
        """
        self._ensure_initialized()

        target_folder = self._resolve_target_folder(folder_id)

        # Detect MIME type from filename
        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type is None:
            mime_type = "application/octet-stream"

        fs = self._get_fs(target_folder)
        drive_files = self._get_drive_files_resource()

        def _write_file() -> dict[str, Any]:
            with fs.open(filename, "wb") as file_obj:
                binary_file = cast(BinaryIO, file_obj)
                binary_file.write(data)
            info = fs.info(filename)
            drive_meta = drive_files.get(
                fileId=info.get("id", ""),
                fields="id,webViewLink,webContentLink,mimeType,size,name",
            ).execute()
            info.update(drive_meta)
            info["folder_id"] = target_folder
            return info

        file_info = await asyncio.to_thread(_write_file)

        file_id = file_info.get("id", "")
        result = {
            "file_id": file_id,
            "web_link": file_info.get("webViewLink", file_info.get("webContentLink", "")),
            "mime_type": file_info.get("mimeType", mime_type),
            "folder_id": target_folder,
            "name": file_info.get("name", filename),
            "size": int(file_info.get("size", len(data))),
        }

        logger.info(
            "Uploaded file '%s' to Drive folder '%s'. File ID: %s",
            filename,
            target_folder,
            file_id,
        )

        return result

    async def read_asset(self, file_id: str) -> bytes:
        """
        Retrieve a file's content from Google Drive by its file ID.

        Args:
            file_id: The Google Drive file ID.

        Returns:
            The file content as bytes.

        Raises:
            RuntimeError: If DriveManager is not initialized.
            FileNotFoundError: If the file doesn't exist.
        """
        self._ensure_initialized()

        def _read_file() -> bytes:
            request = self._get_drive_files_resource().get_media(fileId=file_id)
            return request.execute()

        content = await asyncio.to_thread(_read_file)
        logger.debug("Read %d bytes from file ID: %s", len(content), file_id)

        return content

    async def delete_asset(self, file_id: str) -> bool:
        """
        Delete a file from Google Drive.

        Args:
            file_id: The Google Drive file ID.

        Returns:
            True if deletion was successful.

        Raises:
            RuntimeError: If DriveManager is not initialized.
        """
        self._ensure_initialized()

        def _delete_file() -> bool:
            self._get_drive_files_resource().delete(fileId=file_id).execute()
            return True

        result = await asyncio.to_thread(_delete_file)
        logger.info("Deleted file ID: %s", file_id)

        return result

    async def list_assets(self, folder_id: str | None = None) -> list[dict[str, Any]]:
        """
        List all files in a Google Drive folder.

        Args:
            folder_id: Optional folder ID. Uses instance folder_id if not provided.

        Returns:
            List of file metadata dictionaries.
        """
        self._ensure_initialized()

        target_folder = self._resolve_target_folder(folder_id)
        fs = self._get_fs(target_folder)

        def _list_files() -> list[dict[str, Any]]:
            files = fs.ls("/", detail=True)
            return [
                {
                    "file_id": f.get("id", ""),
                    "name": f.get("name", ""),
                    "mime_type": f.get("mimeType", ""),
                    "size": f.get("size", 0),
                    "folder_id": target_folder,
                }
                for f in files
            ]

        return await asyncio.to_thread(_list_files)


class StorageCoordinator:
    """
    Coordinates cold (Google Drive) and hot (Supabase/Postgres) storage.

    Usage:
        async with StorageCoordinator() as storage:
            metadata = await storage.store_cold_memory(b"hello", "greeting.txt")
            checkpointer = storage.runtime_cache()
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        folder_id: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.drive = DriveManager(settings=self.settings, folder_id=folder_id)
        self._stack: AsyncExitStack | None = None
        self._hot_cache: Any | None = None

    async def initialize(self, *, hot_cache: bool = False) -> None:
        """
        Initialize cold storage and optionally the hot cache.
        """
        await self.drive.initialize()
        if hot_cache:
            await self.activate_hot_cache()

    async def activate_hot_cache(self) -> Any:
        """
        Open the Supabase runtime cache and return the active checkpointer.
        """
        if self._hot_cache is not None:
            return self._hot_cache

        if self._stack is None:
            self._stack = AsyncExitStack()

        self._hot_cache = await self._stack.enter_async_context(get_checkpointer(self.settings))
        return self._hot_cache

    async def release_hot_cache(self) -> None:
        """
        Close the Supabase runtime cache if it is active.
        """
        if self._stack is None:
            return

        await self._stack.aclose()
        self._stack = None
        self._hot_cache = None

    def runtime_cache(self) -> Any:
        """
        Access the active Supabase checkpointer.

        Raises:
            RuntimeError: If the cache is not activated.
        """
        if self._hot_cache is None:
            logger.raise_error(
                (
                    "Runtime cache is not active. Call activate_hot_cache() or use "
                    "'async with StorageCoordinator(...)'."
                ),
                RuntimeError,
            )
        return self._hot_cache

    async def store_cold_memory(
        self,
        data: bytes,
        filename: str,
        *,
        folder_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Persist long-term artifacts to Google Drive cold storage.
        """
        await self.drive.initialize()
        return await self.drive.upload_asset(data, filename, folder_id)

    async def fetch_cold_memory(self, file_id: str) -> bytes:
        """
        Retrieve a cold-storage artifact by Google Drive file ID.
        """
        await self.drive.initialize()
        return await self.drive.read_asset(file_id)

    async def delete_cold_memory(self, file_id: str) -> bool:
        """
        Delete an artifact from cold storage.
        """
        await self.drive.initialize()
        return await self.drive.delete_asset(file_id)

    async def list_cold_memory(self, folder_id: str | None = None) -> list[dict[str, Any]]:
        """
        List assets stored within the configured cold-storage folder.
        """
        await self.drive.initialize()
        return await self.drive.list_assets(folder_id)

    async def __aenter__(self) -> "StorageCoordinator":
        await self.initialize(hot_cache=True)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.release_hot_cache()


# =============================================================================
# Part 2: Hot Storage - LangGraph PostgreSQL Checkpointer
# =============================================================================


@asynccontextmanager
async def get_checkpointer(
    settings: Settings | None = None,
) -> AsyncGenerator[Any, None]:
    """
    Get an async PostgreSQL checkpointer for LangGraph.

    This function creates an AsyncPostgresSaver using psycopg_pool's
    AsyncConnectionPool, connected to the Supabase PostgreSQL database.

    Args:
        settings: Optional Settings instance. Uses get_settings() if not provided.

    Yields:
        AsyncPostgresSaver: The initialized checkpointer ready for use.

    Raises:
        ValueError: If SUPABASE_DSN is not configured.
        ImportError: If required packages are not installed.

    Usage:
        async with get_checkpointer() as checkpointer:
            graph = builder.compile(checkpointer=checkpointer)
            result = await graph.ainvoke(state, config)
    """
    settings = settings or get_settings()

    if not settings.validate_hot_storage():
        logger.raise_error(
            "SUPABASE_DSN is not configured. Set it in environment or .env file.",
            ValueError,
        )

    if AsyncPostgresSaver is None or AsyncConnectionPool is None:  # pragma: no cover
        logger.raise_error(
            "Required packages not installed. Install with: "
            "pip install 'psycopg[binary,pool]' langgraph-checkpoint-postgres",
            ImportError,
        )

    conninfo = str(settings.supabase_dsn)

    pool = AsyncConnectionPool(
        conninfo=conninfo,
        min_size=settings.postgres_pool_min_size,
        max_size=settings.postgres_pool_max_size,
        open=False,  # Don't open immediately
    )

    try:
        await pool.open(wait=True)
        logger.debug(
            "Opened connection pool with %d-%d connections.",
            settings.postgres_pool_min_size,
            settings.postgres_pool_max_size,
        )

        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        logger.info("PostgreSQL checkpointer initialized and tables verified.")

        yield checkpointer

    finally:
        await pool.close()
        logger.debug("Closed connection pool.")


async def setup_checkpoint_tables(settings: Settings | None = None) -> None:
    """
    Initialize the checkpoint tables in PostgreSQL.

    This is a convenience function to set up tables without creating a full checkpointer.
    Useful for initial database setup.

    Args:
        settings: Optional Settings instance. Uses get_settings() if not provided.
    """
    async with get_checkpointer(settings) as _:
        # Tables are already created in get_checkpointer via setup()
        logger.info("Checkpoint tables have been set up successfully.")


# Helper function for sync contexts
def get_sync_checkpointer(settings: Settings | None = None) -> Any:
    """
    Get a synchronous PostgreSQL checkpointer for LangGraph.

    This is a convenience function for synchronous code paths.
    For async code, prefer get_checkpointer().

    Args:
        settings: Optional Settings instance. Uses get_settings() if not provided.

    Returns:
        PostgresSaver: The initialized synchronous checkpointer.

    Usage:
        with get_sync_checkpointer() as checkpointer:
            graph = builder.compile(checkpointer=checkpointer)
            result = graph.invoke(state, config)
    """
    settings = settings or get_settings()

    if not settings.validate_hot_storage():
        logger.raise_error(
            "SUPABASE_DSN is not configured. Set it in environment or .env file.",
            ValueError,
        )

    if PostgresSaver is None:  # pragma: no cover
        logger.raise_error(
            "langgraph-checkpoint-postgres is required for synchronous checkpoints.",
            ImportError,
        )

    return PostgresSaver.from_conn_string(str(settings.supabase_dsn))


__all__ = [
    "DriveManager",
    "StorageCoordinator",
    "get_checkpointer",
    "get_sync_checkpointer",
    "setup_checkpoint_tables",
]
