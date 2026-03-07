"""
query.py — CLI interface for querying the memory system.
Usage:
    python query.py chat              # interactive chat
    python query.py search "term"     # one-shot search
    python query.py timeline          # show all memories
    python query.py stats             # storage stats
"""

import os
import sys
import vertexai
from vertexai.generative_models import GenerativeModel
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from config import VERTEX_PROJECT, VERTEX_LOCATION, GEMINI_FLASH_MODEL
from storage import init_db, get_all_memories, search_by_embedding, get_stats
from processor import get_embedding

vertexai.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)
gemini_model = GenerativeModel(GEMINI_FLASH_MODEL)
console = Console()


def synthesize_answer(query: str, memories: list[dict]) -> str:
    context = "\n\n".join([
        f"Memory {i+1} (importance: {m['importance']:.2f}):\n{m['summary']}"
        for i, m in enumerate(memories)
    ])
    prompt = f"""Based on these memories, answer the question naturally and concisely.

Memories:
{context}

Question: {query}

Answer directly. If memories don't have enough info, say so."""

    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Could not synthesize answer: {e}"


def show_memory(memory: dict):
    imp = memory["importance"]
    bar = "█" * int(imp * 10) + "░" * (10 - int(imp * 10))
    ts = memory.get("timestamp", "")[:19].replace("T", " ")
    clips    = memory.get("clip_paths", [])
    segments = memory.get("segments", [])

    content = f"""[bold]{memory['summary']}[/bold]

[dim]Time:[/dim] {ts}  |  [dim]Source:[/dim] {memory.get('source', '?')}
[dim]Importance:[/dim] {bar} {imp:.2f}  |  [dim]Level:[/dim] {memory.get('detail_level', '?')}
[dim]Activity:[/dim] {memory.get('activity', '—')}  |  [dim]Tags:[/dim] {', '.join(memory.get('tags', [])) or '—'}
[dim]Clips:[/dim] {len(clips)} saved{f'  → {", ".join(os.path.basename(c) for c in clips)}' if clips else '  (none)'}"""

    if segments:
        seg_str = "  ".join([f"{s['start']:.0f}s–{s['end']:.0f}s" for s in segments])
        content += f"\n[dim]Segments:[/dim] {seg_str}"

    color = "red" if imp >= 0.8 else "yellow" if imp >= 0.5 else "green" if imp >= 0.2 else "dim"
    console.print(Panel(content, border_style=color))


def cmd_search(query: str, reconstruct: bool = False):
    init_db()
    console.print(f"\n[bold cyan]Searching:[/bold cyan] {query}\n")

    embedding = get_embedding(query)
    results = search_by_embedding(embedding, top_k=5)

    if not results:
        console.print("[dim]No memories found.[/dim]")
        return

    answer = synthesize_answer(query, results)
    console.print(Panel(f"[bold]{answer}[/bold]", title="Answer", border_style="cyan"))

    console.print(f"\n[dim]Top {len(results)} matching memories:[/dim]")
    for m in results:
        show_memory(m)
        if reconstruct and not m.get("keyframe_paths"):
            console.print("[cyan]  Reconstructing with NanoBanana 2...[/cyan]")
            path = reconstruct_image(m)
            if path:
                console.print(f"  [green]Saved to:[/green] {path}")


def cmd_timeline():
    init_db()
    memories = get_all_memories()
    if not memories:
        console.print("[dim]No memories yet. Run: python ingest.py <youtube_url>[/dim]")
        return

    table = Table(title=f"Memory Timeline ({len(memories)} memories)", box=box.ROUNDED)
    table.add_column("Time", style="dim", width=19)
    table.add_column("Imp", width=14)
    table.add_column("Summary", ratio=True)
    table.add_column("Tags", width=20)
    table.add_column("KF", width=4)

    for m in memories:
        ts = m.get("timestamp", "")[:19].replace("T", " ")
        imp = m["importance"]
        bar = f"{'█'*int(imp*8)}{'░'*(8-int(imp*8))} {imp:.2f}"
        style = "red" if imp >= 0.8 else "yellow" if imp >= 0.5 else "green" if imp >= 0.2 else "dim"
        table.add_row(ts, bar, m["summary"][:80], ", ".join(m.get("tags", [])[:2]), str(len(m.get("keyframe_paths", []))), style=style)

    console.print(table)


def cmd_stats():
    init_db()
    stats = get_stats()
    table = Table(title="Memory System Stats", box=box.SIMPLE)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")
    table.add_row("Total memories", str(stats["total_memories"]))
    table.add_row("High importance (≥0.5)", str(stats["high_importance"]))
    table.add_row("Low importance (<0.5)", str(stats["low_importance"]))
    table.add_row("Avg importance score", str(stats["avg_importance"]))
    table.add_row("DB size", f"{stats['db_size_kb']} KB")
    table.add_row("Keyframes size", f"{stats['keyframes_size_kb']} KB")
    console.print(table)


def cmd_chat():
    init_db()
    console.print(Panel(
        "[bold]Memory Chat[/bold]\nAsk anything about your memories.\nType [cyan]quit[/cyan] to exit | [cyan]timeline[/cyan] | [cyan]stats[/cyan]",
        border_style="cyan"
    ))
    while True:
        try:
            query = console.input("\n[bold cyan]>[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break
        if query.lower() == "timeline":
            cmd_timeline()
            continue
        if query.lower() == "stats":
            cmd_stats()
            continue
        cmd_search(query, reconstruct=query.lower().startswith("show "))


def main():
    if len(sys.argv) < 2:
        cmd_chat()
        return
    cmd = sys.argv[1].lower()
    if cmd == "chat":
        cmd_chat()
    elif cmd == "search":
        cmd_search(" ".join(sys.argv[2:]), reconstruct=True)
    elif cmd == "timeline":
        cmd_timeline()
    elif cmd == "stats":
        cmd_stats()
    else:
        cmd_search(" ".join(sys.argv[1:]), reconstruct=True)


if __name__ == "__main__":
    main()
