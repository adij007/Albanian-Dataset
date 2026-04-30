from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

from convert_schema import SYSTEM_PROMPT, convert_entry


DEFINITION_TASK_TYPES = {"definition", "dictionary_definition", "lexical_definition"}
SENTENCE_SPLIT_RE = re.compile(r"(.+?\.)($|\s)", re.DOTALL)


def load_json(path: Path) -> list[Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return [data]


def ensure_new_schema(entries: list[Any], source_default: str, dialect_default: str, confidence_default: float) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        converted.append(convert_entry(item, source_default, dialect_default, confidence_default))
    return converted


def get_assistant_text(entry: dict[str, Any]) -> str:
    for message in entry.get("messages", []):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def set_assistant_text(entry: dict[str, Any], text: str) -> None:
    for message in entry.get("messages", []):
        if message.get("role") == "assistant":
            message["content"] = text
            return
    entry.setdefault("messages", []).append({"role": "assistant", "content": text})


def is_definition_entry(entry: dict[str, Any]) -> bool:
    metadata = entry.get("metadata", {})
    task_type = str(metadata.get("task_type", "")).casefold()
    if task_type in DEFINITION_TASK_TYPES:
        return True

    user_text = ""
    for message in entry.get("messages", []):
        if message.get("role") == "user":
            user_text = str(message.get("content", ""))
            break
    return "përkufizo" in user_text.casefold() or "përkufizon fjalën" in user_text.casefold()


def word_count(text: str) -> int:
    return len(text.split())


def first_sentence(text: str) -> str:
    match = SENTENCE_SPLIT_RE.search(text.strip())
    if match:
        return match.group(1).strip()
    return text.strip()


def constrain_to_twenty_words(text: str) -> str:
    words = text.split()
    if len(words) <= 20:
        sentence = " ".join(words).strip()
        return sentence if sentence.endswith(".") else f"{sentence}."
    shortened = " ".join(words[:20]).rstrip(",;:- ")
    return f"{shortened}."


def build_short_form(entry: dict[str, Any]) -> dict[str, Any]:
    short_entry = copy.deepcopy(entry)
    short_text = constrain_to_twenty_words(first_sentence(get_assistant_text(entry)))
    set_assistant_text(short_entry, short_text)
    metadata = dict(short_entry.get("metadata", {}))
    metadata["short_form"] = True
    metadata.setdefault("task_type", "definition")
    metadata.setdefault("source", "fjalori-1980")
    metadata.setdefault("dialect", "standard")
    metadata.setdefault("confidence", 0.95)
    short_entry["metadata"] = metadata
    short_entry.setdefault("system", SYSTEM_PROMPT)
    return short_entry


def generate_short_forms(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    generated: list[dict[str, Any]] = []
    for entry in entries:
        generated.append(entry)
        assistant_text = get_assistant_text(entry)
        if is_definition_entry(entry) and word_count(assistant_text) > 100:
            generated.append(build_short_form(entry))
    return generated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate short-form definition variants.")
    parser.add_argument("input_path", help="Path to the source JSON file.")
    parser.add_argument(
        "-o",
        "--output-path",
        help="Path to the output JSON file. Defaults to <input>_shortforms.json",
    )
    parser.add_argument("--source-default", default="fjalori-1980")
    parser.add_argument("--dialect-default", default="standard")
    parser.add_argument("--confidence-default", type=float, default=0.95)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path) if args.output_path else input_path.with_name(f"{input_path.stem}_shortforms.json")

    raw_entries = load_json(input_path)
    schema_entries = ensure_new_schema(
        raw_entries,
        source_default=args.source_default,
        dialect_default=args.dialect_default,
        confidence_default=args.confidence_default,
    )
    combined_entries = generate_short_forms(schema_entries)
    output_path.write_text(json.dumps(combined_entries, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
