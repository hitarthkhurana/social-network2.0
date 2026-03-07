"""
query.py — CLI for querying the person-indexed memory system.
Usage:
    python query.py face <image.jpg>                    # find person, show memories
    python query.py face <image.jpg> "question"         # find person, answer question
    python query.py face <image.jpg> --name "Alice"     # label a person
    python query.py persons                             # list all known persons
    python query.py search "term"                       # text search across all memories
    python query.py chat                                # interactive chat
    python query.py timeline                            # show all memories
    python query.py stats                               # storage stats
"""

import sys
import argparse
from google import genai
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from config import GEMINI_API_KEY, GEMINI_FLASH_MODEL
from storage import (
    init_db, get_all_memories, search_by_embedding, get_stats,
    find_matching_person, get_memories_for_person, get_all_persons,
    update_person_label,
)
from processor import get_embedding
from faces import init_face_app, detect_faces

client = genai.Client(api_key=GEMINI_API_KEY)
console = Console()


def synthesize_answer(query: str, memories: list) -> str:
    context = "\n\n".join([
        f"Memory {i+1} (importance: {m['importance']:.2f}):\n{m['summary']}"
        + (f"\nTranscript: {m['transcript']}" if m.get('transcript') else "")
        for i, m in enumerate(memories)
    ])
    prompt = f"""Based on these memories, answer the question naturally and concisely.

Memories:
{context}

Question: {query}

Answer directly. If memories don't have enough info, say so."""

    try:
        response = client.models.generate_content(
            model=GEMINI_FLASH_MODEL,
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Could not synthesize answer: {e}"


def show_memory(memory: dict):
    imp = memory["importance"]
    bar = "█" * int(imp * 10) + "░" * (10 - int(imp * 10))
    ts = memory.get("timestamp", "")[:19].replace("T", " ")

    transcript = memory.get("transcript") or ""
    transcript_display = f"\n[dim]Transcript:[/dim] {transcript[:200]}{'...' if len(transcript) > 200 else ''}" if transcript else ""

    content = f"""[bold]{memory['summary']}[/bold]

[dim]Time:[/dim] {ts}  |  [dim]Source:[/dim] {memory.get('source', '?')}
[dim]Importance:[/dim] {bar} {imp:.2f}  |  [dim]Level:[/dim] {memory.get('detail_level', '?')}
[dim]Activity:[/dim] {memory.get('activity', '—')}  |  [dim]Tags:[/dim] {', '.join(memory.get('tags', [])) or '—'}{transcript_display}"""

    color = "red" if imp >= 0.8 else "yellow" if imp >= 0.5 else "green" if imp >= 0.2 else "dim"
    console.print(Panel(content, border_style=color))


def cmd_face(image_path: str, question: str = None, name: str = None):
    init_db()
    console.print(f"\n[cyan]Detecting face in:[/cyan] {image_path}")

    face_app = init_face_app()
    faces = detect_faces(face_app, image_path)

    if not faces:
        console.print("[red]No face detected in the image.[/red]")
        return

    face_embedding = faces[0]["embedding"]
    match = find_matching_person(face_embedding, threshold=0.45)

    if not match:
        console.print("[yellow]Person not recognized. Not in the memory system.[/yellow]")
        return

    person_id, similarity = match
    persons = get_all_persons()
    person = next((p for p in persons if p["id"] == person_id), None)
    label = person["label"] if person and person.get("label") else f"Person #{person_id}"

    console.print(f"\n[bold green]Match found:[/bold green] {label} (similarity: {similarity:.4f})")

    if name:
        update_person_label(person_id, name)
        label = name
        console.print(f"[green]Updated name to:[/green] {name}")

    memories = get_memories_for_person(person_id)

    if not memories:
        console.print(f"[dim]No memories associated with {label}.[/dim]")
        return

    console.print(f"\n[bold]{len(memories)} memories for {label}:[/bold]")
    for m in memories:
        show_memory(m)

    if question:
        console.print(f"\n[cyan]Answering:[/cyan] {question}")
        answer = synthesize_answer(question, memories)
        console.print(Panel(f"[bold]{answer}[/bold]", title="Answer", border_style="cyan"))


def cmd_persons():
    init_db()
    persons = get_all_persons()
    if not persons:
        console.print("[dim]No persons stored yet. Run: python ingest.py <video.mp4>[/dim]")
        return

    table = Table(title=f"Known Persons ({len(persons)})", box=box.ROUNDED)
    table.add_column("ID", width=4)
    table.add_column("Name", width=20)
    table.add_column("Memories", width=10)
    table.add_column("Thumbnail", ratio=True)
    table.add_column("Created", width=19)

    for p in persons:
        memories = get_memories_for_person(p["id"])
        label = p.get("label") or "—"
        thumb = p.get("thumbnail_path") or "—"
        created = (p.get("created_at") or "")[:19]
        table.add_row(str(p["id"]), label, str(len(memories)), thumb, created)

    console.print(table)


def cmd_search(query: str):
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


def cmd_timeline():
    init_db()
    memories = get_all_memories()
    if not memories:
        console.print("[dim]No memories yet. Run: python ingest.py <video.mp4>[/dim]")
        return

    table = Table(title=f"Memory Timeline ({len(memories)} memories)", box=box.ROUNDED)
    table.add_column("Time", style="dim", width=19)
    table.add_column("Imp", width=14)
    table.add_column("Summary", ratio=True)
    table.add_column("Tags", width=20)

    for m in memories:
        ts = m.get("timestamp", "")[:19].replace("T", " ")
        imp = m["importance"]
        bar = f"{'█'*int(imp*8)}{'░'*(8-int(imp*8))} {imp:.2f}"
        style = "red" if imp >= 0.8 else "yellow" if imp >= 0.5 else "green" if imp >= 0.2 else "dim"
        table.add_row(ts, bar, m["summary"][:80], ", ".join(m.get("tags", [])[:2]), style=style)

    console.print(table)


def cmd_stats():
    init_db()
    stats = get_stats()
    table = Table(title="Memory System Stats", box=box.SIMPLE)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")
    table.add_row("Total memories", str(stats["total_memories"]))
    table.add_row("High importance (>=0.5)", str(stats["high_importance"]))
    table.add_row("Low importance (<0.5)", str(stats["low_importance"]))
    table.add_row("Avg importance score", str(stats["avg_importance"]))
    table.add_row("Known persons", str(stats["persons"]))
    table.add_row("DB size", f"{stats['db_size_kb']} KB")
    table.add_row("Keyframes size", f"{stats['keyframes_size_kb']} KB")
    console.print(table)


def cmd_chat():
    init_db()
    console.print(Panel(
        "[bold]Memory Chat[/bold]\nAsk anything about your memories.\n"
        "Type [cyan]quit[/cyan] to exit | [cyan]timeline[/cyan] | [cyan]stats[/cyan] | [cyan]persons[/cyan]\n"
        "Use [cyan]face <path>[/cyan] to search by face",
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
        if query.lower() == "persons":
            cmd_persons()
            continue
        if query.lower().startswith("face "):
            parts = query.split(None, 2)
            img = parts[1] if len(parts) > 1 else ""
            q = parts[2] if len(parts) > 2 else None
            cmd_face(img, question=q)
            continue
        cmd_search(query)


def main():
    parser = argparse.ArgumentParser(description="Query the person-indexed memory system")
    subparsers = parser.add_subparsers(dest="command")

    face_parser = subparsers.add_parser("face", help="Search by face image")
    face_parser.add_argument("image", help="Path to face image")
    face_parser.add_argument("question", nargs="?", help="Optional question about this person")
    face_parser.add_argument("--name", help="Assign a name to this person")

    subparsers.add_parser("persons", help="List all known persons")

    search_parser = subparsers.add_parser("search", help="Text search across memories")
    search_parser.add_argument("query", nargs="+", help="Search query")

    subparsers.add_parser("timeline", help="Show all memories")
    subparsers.add_parser("stats", help="Show storage stats")
    subparsers.add_parser("chat", help="Interactive chat")

    args = parser.parse_args()

    if args.command == "face":
        cmd_face(args.image, question=args.question, name=args.name)
    elif args.command == "persons":
        cmd_persons()
    elif args.command == "search":
        cmd_search(" ".join(args.query))
    elif args.command == "timeline":
        cmd_timeline()
    elif args.command == "stats":
        cmd_stats()
    elif args.command == "chat" or args.command is None:
        cmd_chat()


if __name__ == "__main__":
    main()
