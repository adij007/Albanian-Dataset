from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "Jeni një asistent gjuhësor shqiptar i specializuar. "
    "Përgjigjuni gjithmonë në shqip standard, me saktësi dhe qartësi."
)


def infer_task_type(instruction: str, input_text: str, output_text: str, row: dict[str, Any]) -> str:
    joined = f"{instruction}\n{input_text}".casefold()
    output_lower = output_text.casefold()
    source = str(row.get("source", "")).casefold()

    if "_____" in joined or "plotëso" in joined:
        return "fill_in_blank"
    if "lakimin e plotë" in joined or "lakimin" in joined or "declension" in joined:
        return "morphological_inflection"
    if "gjej dhe korrigjo gabimin" in joined or "korrigjo gabimin" in joined:
        return "error_correction"
    if "rishkruaje" in joined or "gjuhë të përditshme" in joined or "thjeshtë" in joined:
        return "register_transform"
    if output_lower.strip() == "nuk e di" or "mund ta sqaroni" in output_lower or "mund të sqaroni" in output_lower:
        return "graceful_refusal"
    if "përkufizon fjalën" in joined or "përkufizo" in joined:
        return "definition"
    if "sinonim" in joined:
        return "synonym_selection"
    if "artikull" in joined or "çfarë informacioni pritet" in joined or source.startswith("http"):
        return "article_qa"
    return "general_qa"


def parse_formatted_text_blob(text: str) -> tuple[str, str, str]:
    instruction = ""
    input_text = ""
    output_text = text.strip()

    instruction_match = re.search(r"###\s*Udhëzim:\s*(.*?)(?:\n###|\Z)", text, re.DOTALL | re.IGNORECASE)
    input_match = re.search(r"###\s*Hyrja:\s*(.*?)(?:\n###|\Z)", text, re.DOTALL | re.IGNORECASE)
    output_match = re.search(r"###\s*Përgjigja:\s*(.*)\Z", text, re.DOTALL | re.IGNORECASE)

    if instruction_match:
        instruction = instruction_match.group(1).strip()
    if input_match:
        input_text = input_match.group(1).strip()
    if output_match:
        output_text = output_match.group(1).strip()

    return instruction, input_text, output_text


def convert_entry(
    row: dict[str, Any],
    source_default: str,
    dialect_default: str,
    confidence_default: float,
) -> dict[str, Any]:
    if "messages" in row and "metadata" in row:
        converted = dict(row)
        metadata = dict(converted.get("metadata", {}))
        metadata.setdefault("source", source_default)
        metadata.setdefault("dialect", dialect_default)
        metadata.setdefault("confidence", confidence_default)
        if "task_type" not in metadata:
            user_message = next(
                (message.get("content", "") for message in converted["messages"] if message.get("role") == "user"),
                "",
            )
            assistant_message = next(
                (message.get("content", "") for message in converted["messages"] if message.get("role") == "assistant"),
                "",
            )
            parts = user_message.split("\n\n", 1)
            instruction = parts[0] if parts else user_message
            input_text = parts[1] if len(parts) == 2 else ""
            metadata["task_type"] = infer_task_type(instruction, input_text, assistant_message, row)
        converted["metadata"] = metadata
        converted.setdefault("system", SYSTEM_PROMPT)
        return converted

    instruction = str(row.get("instruction", "")).strip()
    input_text = str(row.get("input", "")).strip()
    output_text = str(row.get("output", "")).strip()

    if not instruction and "text" in row:
        instruction, input_text, output_text = parse_formatted_text_blob(str(row["text"]))

    if not instruction and not input_text and not output_text:
        raise ValueError("Unsupported entry format; expected instruction/input/output or text.")

    user_content = instruction
    if input_text:
        user_content = f"{instruction}\n\n{input_text}" if instruction else input_text

    task_type = infer_task_type(instruction, input_text, output_text, row)
    metadata = {
        "task_type": task_type,
        "source": row.get("source", source_default),
        "dialect": row.get("dialect", dialect_default),
        "confidence": float(row.get("confidence", row.get("confidence_score", confidence_default))),
    }

    if "short_form" in row:
        metadata["short_form"] = bool(row["short_form"])

    return {
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": user_content.strip()},
            {"role": "assistant", "content": output_text},
        ],
        "metadata": metadata,
    }


def load_json(path: Path) -> list[Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return [data]


def convert_file(
    input_path: Path,
    output_path: Path,
    source_default: str,
    dialect_default: str,
    confidence_default: float,
) -> None:
    data = load_json(input_path)
    converted = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Entry {index} in {input_path.name} is not a JSON object.")
        converted.append(convert_entry(item, source_default, dialect_default, confidence_default))

    output_path.write_text(
        json.dumps(converted, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Albanian training data to the chat-style schema.")
    parser.add_argument("input_path", help="Path to the input JSON file.")
    parser.add_argument(
        "-o",
        "--output-path",
        help="Path to the output JSON file. Defaults to <input>_converted.json",
    )
    parser.add_argument("--source-default", default="fjalori-1980")
    parser.add_argument("--dialect-default", default="standard")
    parser.add_argument("--confidence-default", type=float, default=0.95)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_converted.json")

    convert_file(
        input_path=input_path,
        output_path=output_path,
        source_default=args.source_default,
        dialect_default=args.dialect_default,
        confidence_default=args.confidence_default,
    )


if __name__ == "__main__":
    main()
