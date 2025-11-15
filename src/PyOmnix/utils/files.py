import os
import re
import shutil
import stat
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import NamedTuple

from pyomnix.omnix_logger import get_logger

logger = get_logger(__name__)

_BasePath = type(Path())


class GrepHit(NamedTuple):
    path: Path
    line_no: int
    line: str
    match_span: tuple | None  # (start, end)


class ShPath(_BasePath):
    """
    A path class compatible with Shell operations:
    - cd/chdir/pwd
    - ls/ll
    - cat/head/tail/grep/wc
    - cp/mv/rm/rm_rf
    - mkdir_p/touch/chmod
    - ln_s/ln_h/readlink
    - find/du
    """

    # Constructor and Converter
    def __new__(cls, *args):
        return super().__new__(cls, *args)

    @classmethod
    def from_path(cls, p: str | os.PathLike | Path) -> "ShPath":
        return cls(p)

    def to_path(self) -> Path:
        return Path(self)

    def __truediv__(self, other: str | os.PathLike | Path) -> "ShPath":
        """
        automatically removes leading slashes or backslashes from the other argument to avoid unexpected behavior caused by absolute path concatenation
        """
        if isinstance(other, (str, Path)):
            other_str = str(other)
            if other_str.startswith(("/", "\\")):
                other = other_str.lstrip("/\\")
        return type(self)(super().__truediv__(other))

    # _flavour = type(Path())._flavour

    # Shell operations
    def cd(self, path: str | os.PathLike | Path) -> "ShPath":
        return self / path

    def chdir_to_here(self) -> None:
        """
        change the current working directory to the path of the current object
        """
        os.chdir(self)

    # List operations
    def ls(
        self,
        pattern: str | None = None,
        *,
        recursive: bool = False,
        files: bool = False,
        dirs: bool = False,
        hidden: bool | None = None,
        sort: str = "name",  # name | size | mtime
        reverse: bool = False,
    ) -> list["ShPath"]:
        if not self.exists():
            return []

        it: Iterable[Path]
        if pattern:
            it = self.rglob(pattern) if recursive else self.glob(pattern)
        else:
            it = self.rglob("*") if recursive else self.iterdir()

        out: list[ShPath] = []
        for p in it:
            if files and not p.is_file():
                continue
            if dirs and not p.is_dir():
                continue
            if hidden is not None:
                is_hidden = p.name.startswith(".")
                if hidden != is_hidden:
                    continue
            out.append(type(self)(p))

        # sort keys
        if sort == "name":
            keyfunc = lambda x: x.name.lower()
        elif sort == "size":
            keyfunc = lambda x: (x.stat().st_size if x.exists() and x.is_file() else -1)
        elif sort == "mtime":
            keyfunc = lambda x: x.stat().st_mtime if x.exists() else 0
        else:
            raise ValueError("sort must be one of: name | size | mtime")

        out.sort(key=keyfunc, reverse=reverse)
        return out

    def ll(self) -> list[tuple]:
        rows = []
        for p in self.iterdir():
            try:
                st = p.stat()
                rows.append((self._mode_string(st.st_mode), st.st_size, st.st_mtime, p.name))
            except Exception:
                continue
        rows.sort(key=lambda r: r[3].lower())
        return rows

    @staticmethod
    def _mode_string(mode: int) -> str:
        is_dir = "d" if stat.S_ISDIR(mode) else "-"

        def bit(m, r, w, x):
            return ("r" if m & r else "") + ("w" if m & w else "") + ("x" if m & x else "")

        u = bit(mode, stat.S_IRUSR, stat.S_IWUSR, stat.S_IXUSR)
        g = bit(mode, stat.S_IRGRP, stat.S_IWGRP, stat.S_IXGRP)
        o = bit(mode, stat.S_IROTH, stat.S_IWOTH, stat.S_IXOTH)
        return is_dir + f"{u:->3}{g:->3}{o:->3}"

    # View content operations
    def cat(
        self,
        *files: str | os.PathLike | Path,
        encoding: str = "utf-8",
        errors: str = "strict",
        binary: bool = False,
        head: int | None = None,
        tail: int | None = None,
        sep: str = "",
    ) -> str | bytes | None:
        targets: list[Path]
        if files:
            targets = [self / f for f in files]
        else:
            targets = [self]

        for p in targets:
            if p.is_dir():
                logger.warning(f"cat: {p} is a directory")
                return

        if binary:
            bufs = []
            for f in targets:
                with open(f, "rb") as fh:
                    bufs.append(fh.read())
            return b"".join(bufs)

        def read_text_portion(p: Path) -> str:
            text = p.read_text(encoding=encoding, errors=errors)
            if head is not None:
                return "".join(text.splitlines(True)[:head])
            if tail is not None:
                return "".join(text.splitlines(True)[-tail:])
            return text

        return sep.join(read_text_portion(p) for p in targets)

    def head(
        self,
        *files: str | os.PathLike | Path,
        n: int = 3,
        encoding: str = "utf-8",
        errors: str = "ignore",
    ) -> str | bytes | None:
        return self.cat(*files, encoding=encoding, errors=errors, head=n)

    def tail(
        self,
        *files: str | os.PathLike | Path,
        n: int = 3,
        encoding: str = "utf-8",
        errors: str = "ignore",
    ) -> str | bytes | None:
        return self.cat(*files, encoding=encoding, errors=errors, tail=n)

    # grep operations
    def grep(
        self,
        pattern: str | re.Pattern,
        glob: str = "**/*",
        *,
        flags: int = 0,
        fixed_string: bool = False,
        include_binary: bool = False,
        encoding: str = "utf-8",
        errors: str = "ignore",
        max_matches_per_file: int | None = None,
    ) -> Iterator[GrepHit]:
        if not fixed_string:
            rx = re.compile(pattern if isinstance(pattern, str) else pattern.pattern, flags)
        else:
            rx = None

        for p in self.rglob(glob):
            if not p.is_file():
                continue
            # rough binary detection
            if not include_binary:
                try:
                    with open(p, "rb") as fh:
                        if b"\x00" in fh.read(4096):
                            continue
                except Exception:
                    continue

            hits = 0
            try:
                with open(p, encoding=encoding, errors=errors) as fh:
                    for i, line in enumerate(fh, start=1):
                        if rx:
                            m = rx.search(line)
                            if m:
                                yield GrepHit(p, i, line.rstrip("\n"), m.span())
                                hits += 1
                        else:
                            s = str(pattern)
                            idx = line.find(s)
                            if idx != -1:
                                yield GrepHit(p, i, line.rstrip("\n"), (idx, idx + len(s)))
                                hits += 1
                        if max_matches_per_file and hits >= max_matches_per_file:
                            break
            except Exception:
                continue

    # Copy/move/delete operations
    def mk_dir(self, folder_name: str, mode: int = 755, exist_ok: bool = False) -> "ShPath":
        (self / folder_name).mkdir(parents=True, exist_ok=exist_ok, mode=mode)
        return self

    def touch(
        self,
        file_name: str | os.PathLike | Path,
        mode: int = 755,
        exist_ok: bool = False,
    ) -> "ShPath":
        (self / file_name).touch(mode=mode, exist_ok=exist_ok)
        return self

    def rm(self, file_name: str) -> None:
        """Similar to `rm`: file/symlink unlink; empty directory rmdir; non-empty directory error."""
        if (self / file_name).is_dir() and not (self / file_name).is_symlink():
            os.rmdir(self / file_name)  # only empty directory
        else:
            os.unlink(self / file_name)

    def cp(
        self,
        source: str | os.PathLike | Path,
        dst: str | os.PathLike | Path,
        recursive: bool = False,
        preserve: bool = True,
    ) -> "ShPath":
        """Similar to `cp`/`cp -r`:
        - file -> use copy2(preserve timestamp/permissions) or copy
        - directory -> need recursive=True; use copytree(dirs_exist_ok=True)
        return target path (ShPath).
        """
        sourcep = self / source
        dstp = self / dst
        if sourcep.is_dir() and not sourcep.is_symlink():
            if not recursive:
                raise IsADirectoryError("cp: omitting directory (use recursive=True)")
            # Python 3.8+ supports dirs_exist_ok
            shutil.copytree(sourcep, dstp, dirs_exist_ok=True)
        elif preserve:
            shutil.copy2(sourcep, dstp)
        else:
            shutil.copy(sourcep, dstp)
        return dstp

    def mv(self, source: str | os.PathLike | Path, dst: str | os.PathLike | Path) -> "ShPath":
        sourcep = self / source
        dstp = self / dst
        shutil.move(os.fspath(sourcep), os.fspath(dstp))
        return dstp

    # Link operations
    def ln_s(self, source: str | os.PathLike | Path, target: str | os.PathLike | Path) -> "ShPath":
        """Create a symbolic link at self location, pointing to target (corresponding to ln -s target linkname)"""
        sourcep = self / source
        if not sourcep.exists():
            raise FileNotFoundError(f"ln_s: {sourcep} does not exist")
        targetp = self / target
        sourcep.symlink_to(targetp)
        return sourcep

    def ln_h(self, source: str | os.PathLike | Path, target: str | os.PathLike | Path) -> "ShPath":
        """Create a hard link at self location, pointing to target"""
        sourcep = self / source
        targetp = self / target
        if not sourcep.exists():
            raise FileNotFoundError(f"ln_h: {sourcep} does not exist")
        os.link(os.fspath(sourcep), os.fspath(targetp))
        return self

    def readlink(self) -> "ShPath":
        """If it is a symbolic link, return its target path (not resolved to an absolute path)."""
        if not self.is_symlink():
            raise OSError("not a symlink")
        return type(self)(os.readlink(self))

    # Find/du operations
    def find(
        self,
        pattern: str = "*",
        *,
        type: str | None = None,  # file | dir | symlink
        max_depth: int | None = None,
        follow_symlinks: bool = False,
    ) -> list["ShPath"] | None:
        """Similar to `find`: recursively generate matching paths by glob pattern.
        - type: filter type
        - max_depth: limit depth (0 means only itself; 1 means sub-items)
        - follow_symlinks: whether to follow directory symlinks
        """
        if max_depth is not None and max_depth < 0:
            return
        base_depth = len(self.resolve().parts)
        lst = []
        for p in self.rglob(pattern):
            if max_depth is not None:
                depth = len(p.resolve().parts) - base_depth
                if depth > max_depth:
                    continue
            lst.append(ShPath(p))
        return lst

    def du(self, max_depth: int = 1) -> list[tuple["ShPath", int]]:
        """Similar to `du -d <depth>`: return a list of (path, size) tuples, sorted by size in descending order."""
        results: list[tuple[ShPath, int]] = []
        base_depth = len(self.resolve().parts)

        def dir_size(p: Path) -> int:
            total = 0
            for root, dirs, files in os.walk(p, onerror=lambda e: None):
                for fn in files:
                    fp = Path(root) / fn
                    try:
                        total += fp.stat().st_size
                    except Exception:
                        pass
            return total

        # self itself
        size_self = (
            dir_size(self) if self.is_dir() else (self.stat().st_size if self.exists() else 0)
        )
        results.append((type(self)(self), size_self))

        # sub-levels
        for p in self.rglob("*"):
            try:
                depth = len(p.resolve().parts) - base_depth
                if max_depth is not None and depth > max_depth:
                    continue
                if p.is_dir():
                    size = dir_size(p)
                elif p.exists():
                    size = p.stat().st_size
                else:
                    size = 0
                results.append((type(self)(p), size))
            except Exception:
                continue

        results.sort(key=lambda x: x[1], reverse=True)
        return results
