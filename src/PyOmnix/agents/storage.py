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
import mimetypes
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any, BinaryIO, Literal, cast

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.postgres.aio import Conn as PostgresConn
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from psycopg import AsyncConnection
from psycopg import conninfo as psycopg_conninfo
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from pyomnix.agents.models_settings import Settings, get_settings
from pyomnix.omnix_logger import get_logger
from pyomnix.utils.gdrive import GoogleDriveFileSystem

logger = get_logger(__name__)


def _resolve_supabase_ssl_cert(settings: Settings) -> Path | None:
    """
    Determine the SSL certificate path for Supabase connections, if available.
    """
    if settings.supabase_ssl_cert:
        cert_path = settings.supabase_ssl_cert
        if not cert_path.exists():
            logger.raise_error(
                f"Supabase SSL certificate not found at {cert_path}",
                FileNotFoundError,
            )
        return cert_path


def _build_supabase_conninfo(settings: Settings) -> str:
    """
    Construct the Supabase connection string with SSL parameters if available.
    """
    base_conninfo = str(settings.supabase_dsn)
    cert_path = _resolve_supabase_ssl_cert(settings)
    if cert_path is None:
        return base_conninfo

    logger.debug("Using Supabase SSL certificate at %s", cert_path)
    return psycopg_conninfo.make_conninfo(
        base_conninfo,
        sslmode="verify-full",
        sslrootcert=str(cert_path),
    )


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
        self._gdrive_key: dict[str, Any] | None = None
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

        self._ensure_gdrive_key_loaded()
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

    def _ensure_gdrive_key_loaded(self) -> None:
        """Load and cache the service account credentials."""
        if self._gdrive_key is not None:
            return

        key_data = self.settings.gdrive_key
        if key_data is None:
            logger.raise_error(
                "GDRIVE_KEY is not configured. Update api_config.json or env vars.",
                ValueError,
            )
        self._gdrive_key = key_data

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

        if self._gdrive_key is None:
            self._ensure_gdrive_key_loaded()

        assert self._gdrive_key is not None

        if "type" in self._gdrive_key.keys() and self._gdrive_key["type"] == "service_account":
            token_type = "service_account"
            creds = self._gdrive_key
        else:
            token_type = "browser"
            creds = None

        fs = GoogleDriveFileSystem(
            token=token_type,
            root_file_id=target_folder,
            creds=creds,
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
    sql_type: Literal["supabase", "sqlite", "memory"] = "supabase",
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

    if sql_type == "supabase":
        if AsyncPostgresSaver is None or AsyncConnectionPool is None:  # pragma: no cover
            logger.raise_error(
                "Required packages not installed. Install with: "
                "pip install 'psycopg[binary,pool]' langgraph-checkpoint-postgres",
                ImportError,
            )

        conninfo = _build_supabase_conninfo(settings)

        pool: PostgresConn = AsyncConnectionPool(
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

            # Run setup with a separate autocommit connection since
            # CREATE INDEX CONCURRENTLY cannot run inside a transaction block
            setup_conn = await AsyncConnection.connect(conninfo, autocommit=True)
            setup_conn.row_factory = dict_row  # type: ignore[assignment]
            async with setup_conn:
                await AsyncPostgresSaver(setup_conn).setup()  # type: ignore[arg-type]
            logger.info("PostgreSQL checkpointer initialized and tables verified.")

            yield checkpointer

        finally:
            await pool.close()
            logger.debug("Closed connection pool.")
    elif sql_type == "memory":
        checkpointer = MemorySaver()
        logger.info("In-memory checkpointer initialized.")
        yield checkpointer
        logger.debug("In-memory checkpointer closed.")
    else:
        if AsyncSqliteSaver is None:  # pragma: no cover
            logger.raise_error(
                "AsyncSqliteSaver not available. Install langgraph-checkpoint-sqlite.",
                ImportError,
            )
        db_path = settings.sqlite_db_path
        if db_path is None:
            logger.raise_error(
                "SQLite database path is not configured. Update api_config.json or env vars.",
                ValueError,
            )
        async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
            await checkpointer.setup()
            logger.info("SQLite checkpointer initialized at %s.", db_path)
            yield checkpointer
        logger.debug("SQLite checkpointer closed.")


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

    if PostgresSaver is None:  # pragma: no cover
        logger.raise_error(
            "langgraph-checkpoint-postgres is required for synchronous checkpoints.",
            ImportError,
        )

    return PostgresSaver.from_conn_string(_build_supabase_conninfo(settings))


__all__ = [
    "DriveManager",
    "StorageCoordinator",
    "get_checkpointer",
    "get_sync_checkpointer",
    "setup_checkpoint_tables",
]


if __name__ == "__main__":

    async def _demo() -> None:
        """
        Demonstrate how to combine Google Drive cold storage with Supabase cache.
        """
        storage = StorageCoordinator()

        logger.info("Uploading demo artifact to Google Drive cold storage...")
        metadata = await storage.store_cold_memory(b"hello, cold storage!", "storage-demo.txt")
        logger.info("Uploaded artifact info: %s", metadata)

        logger.info("Activating Supabase runtime cache...")
        await storage.activate_hot_cache()
        cache = storage.runtime_cache()
        logger.info("Runtime cache ready: %s", cache)
        await storage.release_hot_cache()

    asyncio.run(_demo())
