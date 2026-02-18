import ast
import base64
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain.tools import tool

from pyomnix.agents.models_settings import get_settings

MAX_READ_BYTES = 2_000_000
MAX_PREVIEW_CHARS = 20_000
MAX_LIST_ITEMS = 500
MAX_SEARCH_RESULTS = 500
MAX_EXEC_TIMEOUT_SECONDS = 30

BLOCKED_IMPORTS = {
    "os",
    "subprocess",
    "socket",
    "pathlib",
    "shutil",
    "multiprocessing",
    "ctypes",
}
BLOCKED_BUILTINS = {
    "open",
    "exec",
    "eval",
    "compile",
    "input",
    "__import__",
    "breakpoint",
}

ALLOWED_MATH_FUNCS = {
    name: getattr(math, name)
    for name in (
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "sqrt",
        "log",
        "log10",
        "exp",
        "fabs",
        "ceil",
        "floor",
    )
}
ALLOWED_MATH_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


def handle_tool_error(error: Exception) -> str:
    """
    Custom error handling logic.
    Return a friendly message to the LLM instead of a Python Traceback.
    """
    return (
        f"Error: {error!r}. Please try again with different parameters or handle the missing data."
    )


def _json_dump(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _tool_root() -> Path:
    root = Path(os.getenv("PYOMNIX_TOOL_ROOT", ".")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_safe_path(path_str: str) -> Path:
    root = _tool_root()
    candidate = Path(path_str).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Path is outside tool root: {resolved}")
    return resolved


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[truncated, total={len(text)} chars]"


def _validate_python_code(code: str) -> None:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in BLOCKED_IMPORTS:
                    raise ValueError(f"Import not allowed: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in BLOCKED_IMPORTS:
                raise ValueError(f"Import not allowed: {node.module}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_BUILTINS:
                raise ValueError(f"Builtin not allowed: {node.func.id}")
            if isinstance(node.func, ast.Attribute) and node.func.attr in {"system", "popen"}:
                raise ValueError(f"Function not allowed: .{node.func.attr}")

def _normalize_tavily_results(raw_results: list[dict]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "title": str(item.get("title", "")).strip(),
                "link": str(item.get("url", "") or item.get("link", "")).strip(),
                "snippet": str(
                    item.get("content")
                    or item.get("snippet")
                    or item.get("description")
                    or item.get("body")
                    or ""
                ).strip(),
            }
        )
    return normalized


def _format_tavily_results_text(query: str, results: list[dict[str, str]]) -> str:
    if not results:
        return f"No search results found for: {query}"

    lines = [f"Tavily results for: {query}"]
    for idx, item in enumerate(results, start=1):
        title = item.get("title", "") or "(no title)"
        link = item.get("link", "") or "(no link)"
        snippet = item.get("snippet", "")
        lines.append(f"{idx}. {title}")
        lines.append(f"   URL: {link}")
        if snippet:
            lines.append(f"   Snippet: {snippet}")
    return "\n".join(lines)


def _search_with_tavily(
    query: str,
    num_results: int,
    search_depth: str,
    include_answer: bool,
) -> dict:
    api_key = get_settings().tavily_api_key
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": search_depth,
        "max_results": num_results,
        "include_answer": include_answer,
    }
    response = requests.post("https://api.tavily.com/search", json=payload, timeout=20)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("Unexpected Tavily response format")
    return data


@tool
def tavily_search(
    search_query: str,
    num_results: int = 5,
    search_depth: str = "basic",
    include_answer: bool = False,
    output_format: str = "text",
) -> str:
    """
    Search the web using Tavily Search API.

    Args:
        search_query: The query to search for.
        num_results: Number of results to return (1-10).
        search_depth: Search depth, "basic" or "advanced".
        include_answer: Whether Tavily should include a concise answer.
        output_format: Output format, either "text" or "json".

    Returns:
        str: Search results in text or JSON format.
    """
    try:
        query = search_query.strip()
        if not query:
            raise ValueError("search_query must not be empty")

        num_results = min(max(1, num_results), 10)
        search_depth = search_depth.strip().lower()
        if search_depth not in {"basic", "advanced"}:
            raise ValueError("search_depth must be one of: basic, advanced")
        output_format = output_format.lower().strip()
        if output_format not in {"text", "json"}:
            raise ValueError("output_format must be one of: text, json")

        data = _search_with_tavily(
            query=query,
            num_results=num_results,
            search_depth=search_depth,
            include_answer=include_answer,
        )
        raw_results = data.get("results", [])
        results = _normalize_tavily_results(raw_results)

        if output_format == "json":
            return _json_dump(
                {
                    "query": query,
                    "answer": data.get("answer"),
                    "count": len(results),
                    "results": results,
                }
            )

        formatted = _format_tavily_results_text(query, results)
        if include_answer and data.get("answer"):
            return f"Answer: {data.get('answer')}\n\n{formatted}"
        return formatted
    except Exception as e:
        return handle_tool_error(e)


@tool
def google_search(
    search_query: str,
    num_results: int = 5,
    output_format: str = "text",
) -> str:
    """
    Backward-compatible alias for Tavily search.

    Notes:
        This function keeps the historical tool name `google_search`
        but executes search using Tavily under the hood.
    """
    return tavily_search.func(
        search_query=search_query,
        num_results=num_results,
        output_format=output_format,
    )


@tool
def get_current_time() -> str:
    """
    Get the current date and time.

    Returns:
        str: Current date and time in ISO format.
    """
    return datetime.now().isoformat()


@tool
def read_file(
    file_path: str,
    binary: bool = False,
    max_bytes: int = MAX_READ_BYTES,
) -> str:
    """
    Read a file from the configured tool root.

    Args:
        file_path: Relative or absolute file path under tool root.
        binary: If true, return base64-encoded bytes.
        max_bytes: Maximum bytes to read.

    Returns:
        str: File content or JSON metadata with base64 payload.
    """
    try:
        path = _resolve_safe_path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        data = path.read_bytes()
        if len(data) > max_bytes:
            raise ValueError(f"File too large: {len(data)} bytes (max={max_bytes})")

        if binary:
            return _json_dump(
                {
                    "path": str(path),
                    "size_bytes": len(data),
                    "content_base64": base64.b64encode(data).decode("ascii"),
                }
            )
        return data.decode("utf-8", errors="replace")
    except Exception as e:
        return handle_tool_error(e)


@tool
def write_file(
    file_path: str,
    content: str,
    binary_base64: bool = False,
    overwrite: bool = True,
) -> str:
    """
    Write content to a file under tool root.

    Args:
        file_path: Relative or absolute file path under tool root.
        content: Text content or base64 payload when binary_base64 is true.
        binary_base64: Interpret content as base64-encoded bytes.
        overwrite: Whether to overwrite existing file.

    Returns:
        str: JSON metadata of the write result.
    """
    try:
        path = _resolve_safe_path(file_path)
        existed_before = path.exists()
        if existed_before and not overwrite:
            raise FileExistsError(f"File exists and overwrite=False: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = (
            base64.b64decode(content.encode("ascii"), validate=True)
            if binary_base64
            else content.encode("utf-8")
        )
        path.write_bytes(data)
        return _json_dump(
            {"path": str(path), "size_bytes": len(data), "overwritten": existed_before}
        )
    except Exception as e:
        return handle_tool_error(e)


@tool
def list_directory(
    directory: str = ".",
    recursive: bool = False,
    max_items: int = 200,
) -> str:
    """
    List files and folders under a directory.

    Args:
        directory: Directory path under tool root.
        recursive: Whether to recurse subdirectories.
        max_items: Max entries to return.

    Returns:
        str: JSON list of directory entries.
    """
    try:
        max_items = min(max(1, max_items), MAX_LIST_ITEMS)
        root = _resolve_safe_path(directory)
        if not root.exists():
            raise FileNotFoundError(f"Directory not found: {root}")
        if not root.is_dir():
            raise ValueError(f"Path is not a directory: {root}")

        iterator = root.rglob("*") if recursive else root.iterdir()
        items: list[dict[str, object]] = []
        for p in iterator:
            stat = p.stat()
            items.append(
                {
                    "name": p.name,
                    "path": str(p),
                    "is_dir": p.is_dir(),
                    "size_bytes": stat.st_size if p.is_file() else None,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
            )
            if len(items) >= max_items:
                break

        return _json_dump(
            {"directory": str(root), "recursive": recursive, "count": len(items), "items": items}
        )
    except Exception as e:
        return handle_tool_error(e)


@tool
def file_search(
    pattern: str,
    directory: str = ".",
    recursive: bool = True,
    max_results: int = 200,
) -> str:
    """
    Search files by glob pattern.

    Args:
        pattern: Glob pattern, e.g. "*.py" or "*report*".
        directory: Directory path under tool root.
        recursive: Whether to search recursively.
        max_results: Max results to return.

    Returns:
        str: JSON with matched file paths.
    """
    try:
        max_results = min(max(1, max_results), MAX_SEARCH_RESULTS)
        root = _resolve_safe_path(directory)
        if not root.exists() or not root.is_dir():
            raise ValueError(f"Invalid directory: {root}")

        iterator = root.rglob(pattern) if recursive else root.glob(pattern)
        matches: list[str] = []
        for p in iterator:
            if p.is_file():
                matches.append(str(p))
                if len(matches) >= max_results:
                    break
        return _json_dump(
            {
                "directory": str(root),
                "pattern": pattern,
                "recursive": recursive,
                "count": len(matches),
                "matches": matches,
            }
        )
    except Exception as e:
        return handle_tool_error(e)


@tool
def execute_python_code(code: str, timeout_seconds: int = 10) -> str:
    """
    Execute restricted Python code in a subprocess.

    Args:
        code: Python source code to execute.
        timeout_seconds: Timeout in seconds (max 30).

    Returns:
        str: JSON with stdout, stderr, and return code.
    """
    try:
        timeout_seconds = min(max(1, timeout_seconds), MAX_EXEC_TIMEOUT_SECONDS)
        _validate_python_code(code)
        child_env = os.environ.copy()
        child_env.update({"PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"})
        result = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=str(_tool_root()),
            env=child_env,
            check=False,
        )
        return _json_dump(
            {
                "return_code": result.returncode,
                "stdout": _truncate(result.stdout, MAX_PREVIEW_CHARS),
                "stderr": _truncate(result.stderr, MAX_PREVIEW_CHARS),
                "timeout_seconds": timeout_seconds,
            }
        )
    except subprocess.TimeoutExpired:
        return _json_dump({"error": f"Execution timed out after {timeout_seconds} seconds"})
    except Exception as e:
        return handle_tool_error(e)


def _validate_math_expression(expression: str) -> None:
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
    )
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Unsupported syntax in expression: {type(node).__name__}")
        if isinstance(node, ast.Name):
            if node.id not in ALLOWED_MATH_FUNCS and node.id not in ALLOWED_MATH_CONSTS:
                raise ValueError(f"Name not allowed: {node.id}")


@tool
def calculator(expression: str) -> str:
    """
    Safely evaluate a math expression.

    Args:
        expression: Mathematical expression, e.g. "sqrt(9) + sin(pi/2)".

    Returns:
        str: Calculation result.
    """
    try:
        _validate_math_expression(expression)
        result = eval(
            compile(ast.parse(expression, mode="eval"), "<calculator>", "eval"),
            {"__builtins__": {}},
            {**ALLOWED_MATH_FUNCS, **ALLOWED_MATH_CONSTS},
        )
        return str(result)
    except Exception as e:
        return handle_tool_error(e)


@tool
def fetch_url(
    url: str,
    method: str = "GET",
    payload_json: str = "",
    timeout_seconds: int = 20,
) -> str:
    """
    Fetch content from an HTTP endpoint.

    Args:
        url: HTTP(S) URL.
        method: GET or POST.
        payload_json: JSON object string for POST body.
        timeout_seconds: Request timeout.

    Returns:
        str: JSON response metadata and body preview.
    """
    try:
        method = method.upper().strip()
        if method not in {"GET", "POST"}:
            raise ValueError("Only GET and POST methods are supported")
        if not re.match(r"^https?://", url):
            raise ValueError("URL must start with http:// or https://")
        payload = json.loads(payload_json) if payload_json else None
        response = requests.request(method, url, json=payload, timeout=timeout_seconds)
        content_type = response.headers.get("content-type", "")

        if "application/json" in content_type:
            body = _json_dump(response.json())
        else:
            body = response.text

        return _json_dump(
            {
                "url": response.url,
                "status_code": response.status_code,
                "ok": response.ok,
                "content_type": content_type,
                "body_preview": _truncate(body, MAX_PREVIEW_CHARS),
            }
        )
    except Exception as e:
        return handle_tool_error(e)


@tool
def extract_text_from_url(url: str, timeout_seconds: int = 20) -> str:
    """
    Extract readable text from a web page.

    Args:
        url: HTTP(S) URL.
        timeout_seconds: Request timeout.

    Returns:
        str: JSON with page title and extracted text preview.
    """
    try:
        if not re.match(r"^https?://", url):
            raise ValueError("URL must start with http:// or https://")
        response = requests.get(url, timeout=timeout_seconds)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        title = soup.title.get_text(strip=True) if soup.title else ""
        text = " ".join(soup.stripped_strings)
        text = re.sub(r"\s+", " ", text).strip()
        return _json_dump(
            {
                "url": response.url,
                "title": title,
                "text_preview": _truncate(text, MAX_PREVIEW_CHARS),
            }
        )
    except Exception as e:
        return handle_tool_error(e)


@tool
def read_csv(file_path: str, preview_rows: int = 20) -> str:
    """
    Read a CSV file and return summary plus preview.

    Args:
        file_path: CSV file path under tool root.
        preview_rows: Number of rows to include in preview.

    Returns:
        str: JSON with schema and preview records.
    """
    try:
        path = _resolve_safe_path(file_path)
        df = pd.read_csv(path)
        preview_rows = min(max(1, preview_rows), 200)
        preview = df.head(preview_rows).to_dict(orient="records")
        return _json_dump(
            {
                "path": str(path),
                "rows": int(df.shape[0]),
                "columns": list(df.columns),
                "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
                "preview": preview,
            }
        )
    except Exception as e:
        return handle_tool_error(e)


@tool
def write_csv(file_path: str, rows_json: str, overwrite: bool = True) -> str:
    """
    Write rows to a CSV file.

    Args:
        file_path: CSV output path under tool root.
        rows_json: JSON list of row objects.
        overwrite: Whether to overwrite existing file.

    Returns:
        str: JSON metadata about the write operation.
    """
    try:
        path = _resolve_safe_path(file_path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"File exists and overwrite=False: {path}")
        rows = json.loads(rows_json)
        if not isinstance(rows, list):
            raise ValueError("rows_json must be a JSON list of objects")
        if rows and not isinstance(rows[0], dict):
            raise ValueError("rows_json elements must be JSON objects")
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({k for row in rows for k in row.keys()}) if rows else []
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return _json_dump({"path": str(path), "rows_written": len(rows), "columns": fieldnames})
    except Exception as e:
        return handle_tool_error(e)


@tool
def read_json(file_path: str, max_chars: int = MAX_PREVIEW_CHARS) -> str:
    """
    Read and parse a JSON file.

    Args:
        file_path: JSON file path under tool root.
        max_chars: Maximum response length.

    Returns:
        str: Pretty-printed JSON content.
    """
    try:
        path = _resolve_safe_path(file_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return _truncate(_json_dump(data), max_chars)
    except Exception as e:
        return handle_tool_error(e)


@tool
def write_json(file_path: str, json_content: str, overwrite: bool = True) -> str:
    """
    Write JSON content to file.

    Args:
        file_path: JSON output path under tool root.
        json_content: JSON object/array string.
        overwrite: Whether to overwrite existing file.

    Returns:
        str: JSON metadata of the write operation.
    """
    try:
        path = _resolve_safe_path(file_path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"File exists and overwrite=False: {path}")
        data = json.loads(json_content)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_json_dump(data), encoding="utf-8")
        return _json_dump({"path": str(path), "type": type(data).__name__})
    except Exception as e:
        return handle_tool_error(e)


@tool
def translate_text(
    text: str,
    target_lang: str = "en",
    source_lang: str = "auto",
    timeout_seconds: int = 15,
) -> str:
    """
    Translate text via MyMemory public API.

    Args:
        text: Source text.
        target_lang: Target language code, e.g. "en", "zh-CN".
        source_lang: Source language code or "auto".
        timeout_seconds: Request timeout.

    Returns:
        str: Translated text.
    """
    try:
        if not text.strip():
            raise ValueError("text must not be empty")
        lang_pair = f"{source_lang}|{target_lang}"
        url = (
            "https://api.mymemory.translated.net/get?"
            f"q={quote_plus(text)}&langpair={quote_plus(lang_pair)}"
        )
        response = requests.get(url, timeout=timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        translated = payload.get("responseData", {}).get("translatedText", "")
        if not translated:
            raise ValueError("Translation service returned empty result")
        return translated
    except Exception as e:
        return handle_tool_error(e)


@tool
def get_weather(city: str, timeout_seconds: int = 15) -> str:
    """
    Get real-time weather from wttr.in.

    Args:
        city: City name.
        timeout_seconds: Request timeout.

    Returns:
        str: JSON weather summary.
    """
    try:
        if not city.strip():
            raise ValueError("city must not be empty")
        url = f"https://wttr.in/{quote_plus(city)}?format=j1"
        response = requests.get(url, timeout=timeout_seconds)
        response.raise_for_status()
        data = response.json()
        current = data["current_condition"][0]
        area = data.get("nearest_area", [{}])[0]
        return _json_dump(
            {
                "city": city,
                "resolved_area": area.get("areaName", [{}])[0].get("value"),
                "country": area.get("country", [{}])[0].get("value"),
                "temperature_c": current.get("temp_C"),
                "temperature_f": current.get("temp_F"),
                "humidity": current.get("humidity"),
                "weather_desc": current.get("weatherDesc", [{}])[0].get("value"),
                "wind_kmph": current.get("windspeedKmph"),
                "feels_like_c": current.get("FeelsLikeC"),
            }
        )
    except Exception as e:
        return handle_tool_error(e)


@tool
def base64_encode(text: str) -> str:
    """
    Encode text to base64.

    Args:
        text: Input text.

    Returns:
        str: Base64-encoded string.
    """
    try:
        return base64.b64encode(text.encode("utf-8")).decode("ascii")
    except Exception as e:
        return handle_tool_error(e)


@tool
def base64_decode(encoded_text: str) -> str:
    """
    Decode base64 string to text.

    Args:
        encoded_text: Base64 text.

    Returns:
        str: Decoded UTF-8 text.
    """
    try:
        data = base64.b64decode(encoded_text.encode("ascii"), validate=True)
        return data.decode("utf-8", errors="replace")
    except Exception as e:
        return handle_tool_error(e)


@tool
def hash_text(text: str, algorithm: str = "sha256") -> str:
    """
    Hash text with a selected algorithm.

    Args:
        text: Input text.
        algorithm: Hash algorithm, e.g. md5, sha1, sha256.

    Returns:
        str: Hex digest.
    """
    try:
        algorithm = algorithm.lower().replace("-", "")
        if algorithm not in hashlib.algorithms_available:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        hasher = hashlib.new(algorithm)
        hasher.update(text.encode("utf-8"))
        return hasher.hexdigest()
    except Exception as e:
        return handle_tool_error(e)


@tool
def dummy_weather_tool(city: str) -> str:
    """
    Dummy weather tool for testing.

    Args:
        city: The city to query the weather for.

    Returns:
        str: Hardcoded weather string for deterministic testing.
    """
    return f"The weather in {city} is sunny and -271.15C."


FILE_TOOLS = [read_file, write_file, list_directory, file_search]
CODE_TOOLS = [execute_python_code]
MATH_TOOLS = [calculator]
WEB_TOOLS = [tavily_search, google_search, fetch_url, extract_text_from_url]
DATA_TOOLS = [read_csv, write_csv, read_json, write_json]
UTILITY_TOOLS = [
    get_current_time,
    translate_text,
    get_weather,
    base64_encode,
    base64_decode,
    hash_text,
    dummy_weather_tool,
]
ALL_TOOLS = FILE_TOOLS + CODE_TOOLS + MATH_TOOLS + WEB_TOOLS + DATA_TOOLS + UTILITY_TOOLS

TEST_TOOL = [dummy_weather_tool]
