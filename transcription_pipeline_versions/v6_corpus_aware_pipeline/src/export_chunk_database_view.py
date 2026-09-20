from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from database import DEFAULT_DATABASE, initialize_database


VERSION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = VERSION_DIR / "reports" / "chunk_database_view.png"
DEFAULT_JSON = VERSION_DIR / "reports" / "chunk_database_view.json"


def load_chunk_view(
    connection: sqlite3.Connection,
    video_id: str | None = None,
    chunk_index: int = 0,
) -> dict[str, Any]:
    columns = [
        row["name"] for row in connection.execute("PRAGMA table_info(transcript_chunks)")
    ]
    if not columns:
        raise RuntimeError("transcript_chunks table does not exist")

    parameters: list[Any] = [chunk_index]
    video_filter = ""
    if video_id is not None:
        video_filter = "AND v.video_id = ?"
        parameters.append(video_id)
    row = connection.execute(
        f"""
        SELECT c.*, v.video_id, v.title, v.canonical_url
        FROM transcript_chunks AS c
        JOIN transcripts AS t ON t.transcript_id = c.transcript_id
        JOIN videos AS v ON v.video_id = t.video_id
        WHERE c.chunk_index = ? {video_filter}
        ORDER BY c.chunk_id
        LIMIT 1
        """,
        parameters,
    ).fetchone()
    if row is None:
        raise RuntimeError("No matching stored chunk was found")
    return {
        "table": "transcript_chunks",
        "video": {
            "video_id": row["video_id"],
            "title": row["title"],
            "canonical_url": row["canonical_url"],
        },
        "columns": columns,
        "row": {column: row[column] for column in columns},
    }


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    filename = "segoeuib.ttf" if bold else "segoeui.ttf"
    path = Path("C:/Windows/Fonts") / filename
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _wrap_pixels(
    draw: ImageDraw.ImageDraw,
    value: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    paragraphs = value.splitlines() or [""]
    lines: list[str] = []
    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if draw.textlength(candidate, font=font) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def render_chunk_view(view: dict[str, Any], output_path: Path) -> None:
    width = 1800
    label_width = 330
    margin = 60
    value_width = width - (margin * 2) - label_width - 40
    title_font = _font(42, bold=True)
    subtitle_font = _font(24)
    header_font = _font(24, bold=True)
    body_font = _font(22)
    line_height = 31

    sizing_image = Image.new("RGB", (width, 100), "white")
    sizing_draw = ImageDraw.Draw(sizing_image)
    prepared: list[tuple[str, list[str], int]] = []
    for column in view["columns"]:
        value = view["row"][column]
        displayed = "NULL" if value is None else str(value)
        lines = _wrap_pixels(sizing_draw, displayed, body_font, value_width)
        row_height = max(54, len(lines) * line_height + 24)
        prepared.append((column, lines, row_height))

    header_height = 205
    table_header_height = 58
    footer_height = 70
    height = header_height + table_header_height + sum(item[2] for item in prepared) + footer_height
    image = Image.new("RGB", (width, height), "#f5f7fa")
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, 0, width, header_height), fill="#17324d")
    draw.text((margin, 38), "V6 Transcript Chunk Database", font=title_font, fill="white")
    draw.text(
        (margin, 98),
        "One real stored row — displayed vertically so every column is readable",
        font=subtitle_font,
        fill="#dbe8f5",
    )
    draw.text(
        (margin, 145),
        f"Video: {view['video']['video_id']}  |  {view['video']['title']}",
        font=subtitle_font,
        fill="#dbe8f5",
    )

    y = header_height
    draw.rectangle((margin, y, width - margin, y + table_header_height), fill="#2f6f9f")
    draw.text((margin + 16, y + 12), "Column name", font=header_font, fill="white")
    draw.text(
        (margin + label_width + 20, y + 12),
        "Stored value for chunk row",
        font=header_font,
        fill="white",
    )
    y += table_header_height

    for index, (column, lines, row_height) in enumerate(prepared):
        fill = "#ffffff" if index % 2 == 0 else "#eaf0f5"
        draw.rectangle((margin, y, width - margin, y + row_height), fill=fill)
        draw.line(
            (margin + label_width, y, margin + label_width, y + row_height),
            fill="#b9c8d5",
            width=2,
        )
        draw.text((margin + 16, y + 14), column, font=header_font, fill="#17324d")
        for line_index, line in enumerate(lines):
            draw.text(
                (margin + label_width + 20, y + 12 + line_index * line_height),
                line,
                font=body_font,
                fill="#18222c",
            )
        draw.line((margin, y + row_height, width - margin, y + row_height), fill="#c9d4de")
        y += row_height

    draw.text(
        (margin, y + 24),
        "Source: pipeline_v6.sqlite / transcript_chunks. Full row is also saved as JSON.",
        font=subtitle_font,
        fill="#526474",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)


def write_json(view: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(view, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export transcript_chunks columns and one stored row."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--video-id")
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    connection = initialize_database(args.db)
    try:
        view = load_chunk_view(connection, args.video_id, args.chunk_index)
    finally:
        connection.close()
    write_json(view, args.json)
    render_chunk_view(view, args.image)
    print(f"Video: {view['video']['video_id']}")
    print(f"Columns shown: {len(view['columns'])}")
    print(f"Chunk row: {view['row']['chunk_id']}")
    print(f"Image: {args.image.resolve()}")
    print(f"JSON: {args.json.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

