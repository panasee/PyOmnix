# PyOmnix API Reference

Detailed reference for PyOmnix modules. Use when you need specific signatures, options, or implementation details.

---

## Agents Module

### graphs.py

**build_chat_graph(model, \*, enable_multimodal=False, ...)**
- Returns: `StateGraph` (not compiled)
- Flow: conversation → (summarize or END)
- Optional: multimodal ingest node before conversation

**build_tool_agent_graph(model, tools=None, \*, enable_multimodal=False, ...)**
- Returns: `StateGraph`
- Flow: chat ↔ tools (ReAct loop)

**build_self_correction_graph(model, \*, enable_multimodal=False, ...)**
- Returns: `StateGraph` with human_review interrupt
- Flow: chat → critic → human_review → (continue: chat, end: END)
- Requires checkpointer for `Command(resume="continue"|"end")`

**GraphSession**
- `__init__(graph, thread_id=None, config_dict=None)`
- `ainvoke(input_data)`, `astream(input_data)`, `astream_pretty(input_data)`
- `get_state()`, `update_state(values, as_node)`, `inspect_state()`
- `get_state_history(limit=10)`, `print_state_history(limit=10)`
- `export_history(target: Path | DriveManager, filename=None)`

### models_settings.py

**ModelConfig** (singleton)
- `setup_model_factory(factory_fullname: str | list[str], **user_kwargs)` → dict of model factories
- `get_api_config(provider)` → (api_key, base_url, provider_kwargs)
- `list_providers()` → list of provider names
- `setup_langsmith(full_sample=False)`

**Provider format**: `"provider-api"` (e.g. `"siliconflow-deepseek"`). Provider = config key; api = LangChain integration.

**get_settings()** → cached `Settings` from `api_config.json`

### schemas.py

**ConversationState** (extends MessagesState)
- `messages`, `summary`, `user_profile`, `structured_memory`, `retrieved_docs`
- `dialogue_turn_count`, `current_intent`
- Optional: `images`, `audio_files`, `video_files`, `pdf_files`
- Optional: `image_context`, `audio_transcripts`, etc.
- Optional: `multimodal_model_parts`, `multimodal_notes`

**GraphContext**: `thread_id`, `max_history_messages`

### nodes.py

- `create_chat_node(chain)` — generic chat node
- `create_named_chat_node(node_name, chain)` — chat node with name
- `create_tools_node(tools)` — ToolNode
- `create_standard_tools_node(tools)` — ToolNode with standard tools
- `create_tavily_search_tools_node()` — ToolNode with Tavily only
- `create_summarize_node(model)` — summarization node
- `create_human_review_node(destinations, continue_to)` — interrupt for human decision
- `create_multimodal_ingest_node(processing_mode, preprocessor, ...)` — multimodal preprocessing

### runnables.py

- `create_chat_chain(model, tools=None, system_prompt=None)` — LCEL chat chain
- `create_structured_output_chain(model, schema)` — structured output
- `create_multimodal_chain(model, ...)` — multimodal chain

### storage.py

**get_checkpointer(settings=None)** — async context manager
- Returns PostgresSaver (Supabase) or SqliteSaver based on `api_config.json`
- `async with get_checkpointer() as cp: graph.compile(checkpointer=cp)`

**DriveManager**
- `__init__(settings=None, folder_id=None)`
- `async initialize()`
- `async upload_asset(data: bytes, filename: str)` → metadata dict
- `async read_asset(file_id: str)` → bytes

### tools.py

All tools use `@tool` decorator. Key tools:

- `tavily_search(query: str)` — requires `tavily_api_key` in config
- `google_search(query: str)` — backward-compatible alias for `tavily_search`
- `get_current_time()` — current date and time in ISO format
- `read_file(file_path: str)` — sandboxed under `PYOMNIX_TOOL_ROOT`
- `write_file(file_path: str, content: str)`
- `execute_python_code(code: str, timeout_seconds=10)` — blocked: os, subprocess, open, exec, eval
- `calculator(expression: str)`
- `read_csv(file_path: str, preview_rows=20)`
- `write_csv(file_path: str, rows_json: str, overwrite=True)`

---

## Data Process Module

### data_manipulator.py

**DataManipulator**

Constructor: `DataManipulator(*dims, plot_params=None, usetex=False, usepgf=False, data_fill_value=None)`

Data methods:
- `load_dfs(loc, data_in, ...)` — load from file/path
- `get_datas(loc)`, `extend_dims(*dims)`
- `add_label(label, loc)`

Plotting:
- `live_plot_init(loc=0)` — init Plotly live plot
- `live_plot_update(loc=0, idx=...)` — update live plot
- `create_plotly_figure(loc=0, ...)` — create go.Figure
- `create_dash(figure, ...)` — create Dash app

Dash: `get_dash_port()`, `get_dash_url()` — after running Dash server

### data_splitter.py

**DataSplitter**
- `detect_sweep_segments(df, field_col, *, min_points=10)` → list[SweepSegment]
- `split_dataframe_by_segments(df, segments)` → list[pd.DataFrame]
- `create_sweep_filenames(base_name, *, index, direction, ...)` → str
- `estimate_param_value(series)` → float | None

**SweepSegment**: `start_index`, `end_index`, `direction`

---

## Utils Module

### data.py

**ObjectArray(*dims, fill_value=None, unique=False)**
- `extend(*dims)`, `_flatten()`, `_create_objects()`
- Getter supports flattened indexing, setter does not

**CacheArray** — cached array with invalidation

Functions: `difference`, `loop_diff`, `match_with_tolerance`, `rename_duplicates`, `symmetrize`, `identify_direction`, `extract_longest_monotonic_segment`, `aggregate_by_distance`, `smooth_irregular_series`, `sph_to_cart`, `scatter_to_candle`

### math.py

Constants: `HPLANCK`, `HBAR`, `HBAR_THZ`, `KB`, `CM_TO_INCH`, `UNIT_FACTOR_FROM_SI`, `UNIT_FACTOR_TO_SI`, `SWITCH_DICT`

- `factor(unit, mode="from_SI"|"to_SI")` — SI unit conversion
- `convert_unit(value, from_unit, to_unit)`
- `get_unit_factor_and_texname(unit)`
- `split_no_str(s)` — split "1.5k" → (1.5, "k")
- `gen_seq()`, `time_generator()`, `timestr_convert()`, `combined_generator_list()`, `constant_generator()`, `next_lst_gen()`

### plot.py

**PlotParam(*dims)** — extends ObjectArray, fill_value=copy of DEFAULT_PLOT_DICT

- `hex_to_rgb(hex_str)`, `combine_cmap()`, `truncate_cmap()`, `hsv_analyze()`
- `print_progress_bar(iteration, total, prefix="", suffix="", ...)`
- `add_watermark(input_image_path, output_image_path, watermark_text, ...)`

### env.py

- `is_notebook()` → bool
- `set_envs()` — set default env vars

---

## omnix_logger.py

**setup_logger(\*, name, log_level, console_level, file_level, log_file, log_format, date_format, propagate, add_trace_level, rotation, max_size, backup_count, interval)**
- `rotation`: "size" | "time" | None

**get_logger(name=None, \*)** — returns existing or creates new

**TempLogLevel(logger, level)** — context manager

**close_logger(name=None)** — close handlers

**OmnixLogger** — extends logging.Logger with `trace()`, `raise_error()`, `validate()`

---

## consts.py

- `OMNIX_PATH`, `LOG_FILE_PATH` — set by `set_paths()` or env
- `set_paths(omnix_path=None)` — reads `OMNIX_PATH` or `PYLAB_DB_LOCAL`
- `enable_rich_traceback()`

---

## pltconfig

- `color_preset.py` — color presets
- `pan-colors.json` — package data
- `plot_config_tex.py`, `plot_config_notex.py`, `plot_config_tex_pgf.py`, `matplotlib_config_tex_pgf.py` — Matplotlib configs
