from __future__ import annotations

import csv
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PRIMARY_DATA_DIR = BASE_DIR / "data"
FALLBACK_DATA_DIR = BASE_DIR / "csv"
OUTPUT_FILE = PRIMARY_DATA_DIR / "merged_dataset.csv"
REPORT_FILE = PRIMARY_DATA_DIR / "dedup_report.txt"
SKIP_OUTPUTS = {"merged_dataset.csv"}
TOKEN_RE = re.compile(r"[A-Za-zÇËçëÂÊÎÔÛâêîôû]+(?:[-'][A-Za-zÇËçëÂÊÎÔÛâêîôû]+)*")


def maximize_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def resolve_input_dir() -> Path:
    if PRIMARY_DATA_DIR.exists() and any(
        path.is_file() and path.name not in SKIP_OUTPUTS for path in PRIMARY_DATA_DIR.glob("*.csv")
    ):
        return PRIMARY_DATA_DIR
    if FALLBACK_DATA_DIR.exists():
        return FALLBACK_DATA_DIR
    raise FileNotFoundError("Neither ./data nor ./csv exists.")


def iter_csv_files(data_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in data_dir.glob("*.csv")
        if path.is_file() and path.name not in SKIP_OUTPUTS
    )


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def row_text_for_dialect(row: dict[str, object]) -> str:
    preferred_fields = [
        "cleaned_text",
        "definition",
        "word",
        "headword",
        "synonym",
        "synonyms",
        "entry_word",
        "raw_text",
    ]
    fragments = [normalize_text(row.get(field, "")) for field in preferred_fields if row.get(field)]
    if not fragments:
        fragments = [
            normalize_text(value)
            for key, value in row.items()
            if key not in {"source_file", "dialect", "confidence_score", "register"}
            and normalize_text(value)
        ]
    return " ".join(fragments)


def detect_dialect(text: str) -> str:
    lowered = text.casefold()
    tokens = [token.casefold() for token in TOKEN_RE.findall(text)]

    strong_gheg_markers = (
        "[âêîôû]" in lowered
        or any(
            marker in lowered
            for marker in (
                " asht ",
                " nji ",
                " qysh ",
                " kqyr ",
                " mue ",
                " due ",
                " kena ",
                " jena ",
                " shpi ",
                " katund ",
            )
        )
        or any(token.endswith("ue") or token.endswith("uem") for token in tokens)
    )
    if strong_gheg_markers:
        return "gegë"

    final_schwa_ratio = 0.0
    if tokens:
        final_schwa_ratio = sum(1 for token in tokens if len(token) > 3 and token.endswith("ë")) / len(tokens)

    strong_tosk_markers = any(
        marker in lowered
        for marker in (
            " çupë ",
            " moj ",
            " këndej ",
            " andej ",
            " vallë ",
            " kështu ",
        )
    )

    if strong_tosk_markers or final_schwa_ratio >= 0.18:
        return "toskë"

    return "standard"


def infer_confidence_score(source_file: str, fieldnames: set[str]) -> float:
    lowered = source_file.casefold()
    if lowered == "kushtetuta.csv":
        return 0.99
    if "enhanced" in lowered or "final" in lowered:
        return 0.95
    if {"raw_text", "cleaned_text"}.intersection(fieldnames) or "ocr" in lowered:
        return 0.75
    return 0.85


def infer_register(source_file: str) -> str:
    return "legal" if source_file.casefold() == "kushtetuta.csv" else "general"


def dedup_signature(row: dict[str, object]) -> tuple[str, str]:
    def has_values(*fields: str) -> bool:
        return all(normalize_text(row.get(field, "")) for field in fields)

    if has_values("word", "definition"):
        payload = f"{normalize_text(row['word']).casefold()}||{normalize_text(row['definition']).casefold()}"
        return "word_definition", payload

    if has_values("headword", "synonym"):
        payload = f"{normalize_text(row['headword']).casefold()}||{normalize_text(row['synonym']).casefold()}"
        return "headword_synonym", payload

    if has_values("entry_word", "synonyms"):
        payload = f"{normalize_text(row['entry_word']).casefold()}||{normalize_text(row['synonyms']).casefold()}"
        return "headword_synonym_alias", payload

    if has_values("filename", "cleaned_text"):
        payload = f"{normalize_text(row['filename']).casefold()}||{normalize_text(row['cleaned_text']).casefold()}"
        return "ocr_text", payload

    relevant_items = []
    for key, value in row.items():
        if key in {"source_file", "dialect", "confidence_score", "register"}:
            continue
        cleaned = normalize_text(value)
        if cleaned:
            relevant_items.append((key, cleaned.casefold()))
    payload = "||".join(f"{key}={value}" for key, value in sorted(relevant_items))
    return "fallback", payload


def hash_signature(kind: str, payload: str) -> str:
    return hashlib.sha256(f"{kind}::{payload}".encode("utf-8")).hexdigest()


def load_rows(csv_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return rows
        fieldnames = {field for field in reader.fieldnames if field}
        confidence = infer_confidence_score(csv_path.name, fieldnames)
        register = infer_register(csv_path.name)
        for raw_row in reader:
            row = {key: normalize_text(value) for key, value in raw_row.items() if key}
            row["source_file"] = csv_path.name
            row["dialect"] = detect_dialect(row_text_for_dialect(row))
            row["confidence_score"] = confidence
            row["register"] = register
            rows.append(row)
    return rows


def deduplicate_rows(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    kept: dict[str, dict[str, object]] = {}
    dedup_kind_counter: Counter[str] = Counter()
    collisions = 0

    for row in rows:
        kind, payload = dedup_signature(row)
        signature = hash_signature(kind, payload)
        dedup_kind_counter[kind] += 1
        current = kept.get(signature)
        if current is None or float(row["confidence_score"]) > float(current["confidence_score"]):
            if current is not None:
                collisions += 1
            kept[signature] = row
        else:
            collisions += 1

    sorted_rows = sorted(
        kept.values(),
        key=lambda item: (
            normalize_text(item.get("source_file", "")).casefold(),
            normalize_text(item.get("word", item.get("headword", item.get("entry_word", item.get("filename", ""))))).casefold(),
            normalize_text(item.get("definition", item.get("synonym", item.get("synonyms", item.get("cleaned_text", ""))))).casefold(),
        ),
    )
    report = {
        "total_rows": len(rows),
        "kept_rows": len(sorted_rows),
        "dropped_rows": len(rows) - len(sorted_rows),
        "collisions": collisions,
        "dedup_kind_counter": dedup_kind_counter,
    }
    return sorted_rows, report


def collect_fieldnames(rows: list[dict[str, object]]) -> list[str]:
    preferred_order = [
        "word",
        "definition",
        "headword",
        "synonym",
        "entry_word",
        "synonyms",
        "filename",
        "cleaned_text",
        "raw_text",
        "timestamp",
        "source_file",
        "dialect",
        "confidence_score",
        "register",
    ]
    discovered = []
    seen = set()
    for field in preferred_order:
        if any(field in row for row in rows):
            discovered.append(field)
            seen.add(field)
    for row in rows:
        for field in row.keys():
            if field not in seen:
                discovered.append(field)
                seen.add(field)
    return discovered


def write_merged_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = collect_fieldnames(rows)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serializable = dict(row)
            serializable["confidence_score"] = f"{float(serializable['confidence_score']):.2f}"
            writer.writerow({field: serializable.get(field, "") for field in fieldnames})


def write_report(path: Path, input_dir: Path, csv_files: list[Path], file_counts: Counter[str], report: dict[str, object]) -> None:
    lines = [
        f"Input directory: {input_dir}",
        f"CSV files loaded: {len(csv_files)}",
        f"Rows loaded: {report['total_rows']}",
        f"Rows kept after deduplication: {report['kept_rows']}",
        f"Rows dropped: {report['dropped_rows']}",
        f"Replacement collisions resolved by highest confidence: {report['collisions']}",
        "",
        "Rows loaded per file:",
    ]
    for file_name, count in sorted(file_counts.items()):
        lines.append(f"- {file_name}: {count}")

    lines.extend(["", "Dedup signature usage:"])
    for kind, count in sorted(report["dedup_kind_counter"].items()):
        lines.append(f"- {kind}: {count}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    maximize_csv_field_limit()
    input_dir = resolve_input_dir()
    PRIMARY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_files = iter_csv_files(input_dir)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    all_rows: list[dict[str, object]] = []
    file_counts: Counter[str] = Counter()
    for csv_path in csv_files:
        rows = load_rows(csv_path)
        file_counts[csv_path.name] = len(rows)
        all_rows.extend(rows)

    deduped_rows, report = deduplicate_rows(all_rows)
    write_merged_csv(OUTPUT_FILE, deduped_rows)
    write_report(REPORT_FILE, input_dir, csv_files, file_counts, report)


if __name__ == "__main__":
    main()
