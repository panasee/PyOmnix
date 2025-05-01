from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph.state import CompiledGraph
from pathlib import Path


def save_graph_as_png(app: CompiledGraph, output_file_path: Path) -> None:
    png_image = app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    with open(output_file_path, "wb") as f:
        f.write(png_image)