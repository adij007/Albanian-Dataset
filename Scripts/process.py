from __future__ import annotations

import csv
import hashlib
import logging
import os
import re
import shutil
import time
import unicodedata
from pathlib import Path
from typing import Iterable

import pytesseract
from pdf2image import convert_from_path
from pdfminer.pdfpage import PDFPage

try:
    import ftfy
except ImportError:  # pragma: no cover - fallback for environments without ftfy
    class _FtfyShim:
        @staticmethod
        def fix_text(text: str) -> str:
            return text

    ftfy = _FtfyShim()


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = BASE_DIR / "scanned pages"
OUTPUT_FILE = BASE_DIR / "albanian_dictionary_dataset.csv"
DEBUG_OUTPUT_FILE = BASE_DIR / "debug_raw.csv"
TESSERACT_BINARY = Path(r"D:\Tesseract-OCR\tesseract.exe")
TESSDATA_PREFIX = Path(r"D:\Tesseract-OCR\tessdata")
POPLER_BIN_PATH = Path(r"D:\poppler-25.12.0\Library\bin")
OCR_CONFIG = r"--oem 3 --psm 6 -l sqi"

pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_BINARY)
os.environ.setdefault("TESSDATA_PREFIX", str(TESSDATA_PREFIX))

LOGGER = logging.getLogger("albanian_ocr_process")
TOKEN_RE = re.compile(r"[A-Za-zÇËçë]+(?:[-'][A-Za-zÇËçë]+)*", re.UNICODE)

CHAR_REPLACEMENTS = {
    "\ufeff": "",
    "\u200b": "",
    "\u200c": "",
    "\u200d": "",
    "\u2060": "",
    "\u00ad": "",
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u2018": "'",
    "\u2019": "'",
    "\u201b": "'",
    "\u2032": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2033": '"',
    "\u00a0": " ",
    "\u202f": " ",
    "\t": " ",
    "€": "ë",
    "¢": "ç",
    "©": "ç",
    "®": "ë",
    "¬": "ë",
    "°": "ë",
    "¸": "ç",
    "¨": "ë",
}

WHOLE_WORD_CORRECTIONS = {
    "nje": "një",
    "eshte": "është",
    "qe": "që",
    "per": "për",
    "gjate": "gjatë",
    "mbi te gjitha": "mbi të gjitha",
    "nen": "nën",
    "nder": "ndër",
    "cfare": "çfarë",
    "cfaredo": "çfarëdo",
    "cdo": "çdo",
    "cdonjeri": "çdonjëri",
    "cuditem": "çuditem",
    "cudi": "çudi",
    "cmim": "çmim",
    "cmuar": "çmuar",
    "ceshtje": "çështje",
    "ceshtjet": "çështjet",
    "celes": "çelës",
    "celesi": "çelësi",
    "celje": "çelje",
    "coj": "çoj",
    "cojne": "çojnë",
    "corape": "çorape",
    "corodit": "çorodit",
    "cun": "çun",
    "cupe": "çupë",
    "caj": "çaj",
    "cast": "çast",
    "cati": "çati",
    "cirak": "çirak",
    "cmime": "çmime",
    "cmimeve": "çmimeve",
    "cemtj": "çemtj",
    "shtepi": "shtëpi",
    "shtepia": "shtëpia",
    "shtepise": "shtëpisë",
    "femije": "fëmijë",
    "femijet": "fëmijët",
    "femer": "femër",
    "zemer": "zemër",
    "nene": "nënë",
    "vella": "vëlla",
    "moter": "motër",
    "kenge": "këngë",
    "fjale": "fjalë",
    "fjales": "fjalës",
    "here": "herë",
    "bere": "bërë",
    "bejne": "bëjnë",
    "bej": "bëj",
    "behet": "bëhet",
    "beri": "bëri",
    "doreshkrim": "dorëshkrim",
    "kembengulje": "këmbëngulje",
    "kenga": "kënga",
    "shqiperi": "Shqipëri",
    "shqiptare": "shqiptare",
    "shqiptaret": "shqiptarët",
    "shqiperise": "Shqipërisë",
}

WHOLE_PHRASE_CORRECTIONS = {
    "me qellim": "me qëllim",
    "ne menyre": "në mënyrë",
    "ne qofte se": "në qoftë se",
    "per shembull": "për shembull",
    "me te madhe": "më të madhe",
    "me te mire": "më të mirë",
    "e cila": "e cila",
    "i cili": "i cili",
}

REGEX_CORRECTIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\r\n?"), "\n"),
    (re.compile(r"([A-Za-zÇËçë])[\u00ad‐‑‒–—-]+\s*\n\s*([A-Za-zÇËçë])"), r"\1\2"),
    (re.compile(r"([A-Za-zÇËçë])\s*¬\s*\n\s*([A-Za-zÇËçë])"), r"\1\2"),
    (re.compile(r"([A-Za-zÇËçë])\s*\n\s*([jJhHlLrR])"), r"\1\2"),
    (re.compile(r"[ \t]+"), " "),
    (re.compile(r"\n{3,}"), "\n\n"),
    (re.compile(r"([,:;.!?])([^\s])"), r"\1 \2"),
    (re.compile(r"([^\s])([,:;.!?])"), r"\1\2"),
]

DIGRAPH_FIXES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?<=\w)g[\s\|/\\`´'’‘-]{1,3}j(?=\w)", re.IGNORECASE), "gj"),
    (re.compile(r"(?<=\w)n[\s\|/\\`´'’‘-]{1,3}j(?=\w)", re.IGNORECASE), "nj"),
    (re.compile(r"(?<=\w)s[\s\|/\\`´'’‘-]{1,3}h(?=\w)", re.IGNORECASE), "sh"),
    (re.compile(r"(?<=\w)x[\s\|/\\`´'’‘-]{1,3}h(?=\w)", re.IGNORECASE), "xh"),
    (re.compile(r"(?<=\w)z[\s\|/\\`´'’‘-]{1,3}h(?=\w)", re.IGNORECASE), "zh"),
    (re.compile(r"(?<=\w)t[\s\|/\\`´'’‘-]{1,3}h(?=\w)", re.IGNORECASE), "th"),
    (re.compile(r"(?<=\w)l[\s\|/\\`´'’‘-]{1,3}l(?=\w)", re.IGNORECASE), "ll"),
    (re.compile(r"(?<=\w)r[\s\|/\\`´'’‘-]{1,3}r(?=\w)", re.IGNORECASE), "rr"),
    (re.compile(r"(?<=[aeiouyëAEIOUYË])l[1I|](?=[aeiouyëAEIOUYË])"), "ll"),
    (re.compile(r"(?<=[aeiouyëAEIOUYË])r[1I|](?=[aeiouyëAEIOUYË])"), "rr"),
    (re.compile(r"5h"), "sh"),
    (re.compile(r"Sh\b"), "Sh"),
    (re.compile(r"Xh\b"), "Xh"),
    (re.compile(r"Zh\b"), "Zh"),
    (re.compile(r"Th\b"), "Th"),
]

SUSPICIOUS_SEQUENCES = (
    "0",
    "1",
    "|",
    "5h",
    "tli",
    "xli",
    "zli",
    "g|",
    "n|",
    "rn",
    "vv",
    "cf",
    "cd",
    "qe",
    "nje",
    "eshte",
)

COMMON_ALBANIAN_WORDS = {
    "ai",
    "ajo",
    "ata",
    "ato",
    "ardhje",
    "bëri",
    "bëhet",
    "bëj",
    "bëjnë",
    "botë",
    "çdo",
    "çfarë",
    "dhe",
    "ditë",
    "drejtë",
    "duke",
    "emër",
    "fjalë",
    "fëmijë",
    "gjatë",
    "gjuhë",
    "i",
    "jam",
    "janë",
    "jo",
    "jetë",
    "ka",
    "kam",
    "kanë",
    "këngë",
    "kjo",
    "krye",
    "ku",
    "kur",
    "ligj",
    "mal",
    "me",
    "më",
    "mirë",
    "mund",
    "në",
    "një",
    "njeri",
    "për",
    "që",
    "rrugë",
    "shtëpi",
    "shqip",
    "shqiptar",
    "të",
    "u",
    "ujë",
    "vend",
    "vëlla",
    "zemër",
    "është",
}


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_pdf_page_count(file_path: Path) -> int:
    try:
        with file_path.open("rb") as handle:
            return sum(1 for _ in PDFPage.get_pages(handle))
    except Exception as exc:  # pragma: no cover - pdf errors depend on file state
        LOGGER.warning("Could not read page count for %s: %s", file_path.name, exc)
        return 0


def preserve_case(source: str, target: str) -> str:
    if source.isupper():
        return target.upper()
    if source[:1].isupper():
        return target[:1].upper() + target[1:]
    return target


def replace_whole_words(text: str, replacements: dict[str, str]) -> str:
    for source, target in replacements.items():
        pattern = re.compile(rf"\b{re.escape(source)}\b", re.IGNORECASE)
        text = pattern.sub(lambda match: preserve_case(match.group(0), target), text)
    return text


def apply_char_replacements(text: str) -> str:
    for source, target in CHAR_REPLACEMENTS.items():
        text = text.replace(source, target)
    return text


def repair_digraphs(text: str) -> str:
    for pattern, replacement in DIGRAPH_FIXES:
        def _replace(match: re.Match[str]) -> str:
            return preserve_case(match.group(0), replacement)

        text = pattern.sub(_replace, text)
    return text


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFC", text)
    text = apply_char_replacements(text)

    for pattern, replacement in REGEX_CORRECTIONS:
        text = pattern.sub(replacement, text)

    text = repair_digraphs(text)
    text = replace_whole_words(text, WHOLE_WORD_CORRECTIONS)
    text = replace_whole_words(text, WHOLE_PHRASE_CORRECTIONS)

    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = unicodedata.normalize("NFC", text)
    return text.strip()


def extract_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def normalize_lookup_token(token: str) -> str:
    return unicodedata.normalize("NFC", token).strip("'-").lower()


def load_wordlist() -> set[str]:
    words = set(COMMON_ALBANIAN_WORDS)
    candidate_dirs = [BASE_DIR / "csv", BASE_DIR / "data"]

    for directory in candidate_dirs:
        if not directory.exists():
            continue
        for csv_path in directory.glob("*.csv"):
            try:
                with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                    reader = csv.DictReader(handle)
                    if reader.fieldnames:
                        for row in reader:
                            for value in row.values():
                                if not value:
                                    continue
                                for token in extract_tokens(str(value)):
                                    words.add(normalize_lookup_token(token))
                    else:
                        handle.seek(0)
                        for raw_line in handle:
                            for token in extract_tokens(raw_line):
                                words.add(normalize_lookup_token(token))
            except Exception as exc:  # pragma: no cover - depends on external files
                LOGGER.warning("Skipping wordlist source %s: %s", csv_path.name, exc)
    return {word for word in words if word}


def toggle_candidates(token: str) -> set[str]:
    lowered = normalize_lookup_token(token)
    candidates = {lowered}

    for source, target in (
        ("e", "ë"),
        ("c", "ç"),
        ("qe", "që"),
        ("nje", "një"),
        ("per", "për"),
        ("gjate", "gjatë"),
        ("shtepi", "shtëpi"),
        ("femije", "fëmijë"),
        ("vella", "vëlla"),
    ):
        if source in lowered:
            candidates.add(lowered.replace(source, target))

    digraph_candidates = {
        lowered.replace("g j", "gj").replace("g|", "gj"),
        lowered.replace("n j", "nj").replace("n|", "nj"),
        lowered.replace("5h", "sh").replace("s h", "sh"),
        lowered.replace("x h", "xh"),
        lowered.replace("z h", "zh"),
        lowered.replace("t h", "th"),
        lowered.replace("l l", "ll"),
        lowered.replace("r r", "rr"),
    }
    candidates.update(digraph_candidates)
    return {candidate for candidate in candidates if candidate}


def looks_like_ocr_error(token: str, wordlist: set[str]) -> bool:
    normalized = normalize_lookup_token(token)
    if not normalized or len(normalized) < 3 or normalized in wordlist:
        return False

    if re.search(r"[0-9]", token):
        return True

    if re.search(r"[^A-Za-zÇËçë'-]", token):
        return True

    if any(sequence in normalized for sequence in SUSPICIOUS_SEQUENCES):
        if any(candidate in wordlist for candidate in toggle_candidates(token)):
            return True

    return any(candidate in wordlist for candidate in toggle_candidates(token) if candidate != normalized)


def flag_suspected_ocr_errors(text: str, wordlist: set[str]) -> str:
    flagged = []
    seen: set[str] = set()
    for token in extract_tokens(text):
        normalized = normalize_lookup_token(token)
        if normalized in seen:
            continue
        if looks_like_ocr_error(token, wordlist):
            flagged.append(token)
            seen.add(normalized)
    return ", ".join(flagged)


def discover_poppler_path() -> str | None:
    if POPLER_BIN_PATH.exists():
        return str(POPLER_BIN_PATH)
    if shutil.which("pdftoppm"):
        return None
    raise FileNotFoundError(
        f"Poppler was not found at {POPLER_BIN_PATH} and pdftoppm is not available on PATH."
    )


def process_pdf(file_path: Path, poppler_path: str | None, wordlist: set[str]) -> dict[str, str] | None:
    page_count = get_pdf_page_count(file_path)
    if page_count <= 0:
        return None

    raw_pages: list[str] = []
    start_time = time.time()
    LOGGER.info("Processing %s (%s pages)", file_path.name, page_count)

    for page_number in range(1, page_count + 1):
        try:
            kwargs = {
                "dpi": 300,
                "first_page": page_number,
                "last_page": page_number,
            }
            if poppler_path:
                kwargs["poppler_path"] = poppler_path

            images = convert_from_path(str(file_path), **kwargs)
            if not images:
                continue

            raw_text = pytesseract.image_to_string(images[0], config=OCR_CONFIG)
            raw_pages.append(raw_text)
            LOGGER.info("  Page %s/%s complete", page_number, page_count)
        except Exception as exc:  # pragma: no cover - OCR depends on local binaries/files
            LOGGER.error("  OCR failed on %s page %s: %s", file_path.name, page_number, exc)
            break

    full_text = "\n".join(part for part in raw_pages if part).strip()
    if not full_text:
        LOGGER.warning("No OCR text extracted from %s", file_path.name)
        return None

    cleaned_text = clean_text(full_text)
    suspected_errors = flag_suspected_ocr_errors(cleaned_text, wordlist)
    LOGGER.info("Finished %s in %.2fs", file_path.name, time.time() - start_time)

    return {
        "filename": file_path.name,
        "raw_text": full_text,
        "cleaned_text": cleaned_text,
        "suspected_ocr_errors": suspected_errors,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def deduplicate_records(records: Iterable[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    seen_hashes: set[str] = set()
    unique_records: list[dict[str, str]] = []
    dropped = 0

    for record in records:
        cleaned_text = record["cleaned_text"].strip()
        digest = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()
        if digest in seen_hashes:
            dropped += 1
            continue
        seen_hashes.add(digest)
        record["cleaned_text_hash"] = digest
        unique_records.append(record)

    return unique_records, dropped


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def process_dictionary(folder_path: Path) -> list[dict[str, str]]:
    pdf_files = sorted(path for path in folder_path.iterdir() if path.suffix.lower() == ".pdf")
    if not pdf_files:
        LOGGER.error("No PDF files found in %s", folder_path)
        return []

    poppler_path = discover_poppler_path()
    wordlist = load_wordlist()
    LOGGER.info("Environment ready. Processing %s PDFs.", len(pdf_files))

    records: list[dict[str, str]] = []
    for pdf_file in pdf_files:
        record = process_pdf(pdf_file, poppler_path, wordlist)
        if record:
            records.append(record)

    deduped_records, dropped = deduplicate_records(records)
    LOGGER.info("Exact duplicate cleaned_text rows dropped across files: %s", dropped)
    return deduped_records


def main() -> None:
    configure_logging()
    target_folder = DEFAULT_INPUT_DIR

    if not target_folder.exists():
        LOGGER.info("Creating input folder at %s", target_folder)
        target_folder.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Place PDF files in %s and run the script again.", target_folder)
        return

    records = process_dictionary(target_folder)
    if not records:
        LOGGER.error("No rows were produced.")
        return

    write_csv(
        OUTPUT_FILE,
        ["filename", "cleaned_text", "suspected_ocr_errors", "timestamp"],
        records,
    )
    write_csv(
        DEBUG_OUTPUT_FILE,
        ["filename", "raw_text", "cleaned_text_hash", "timestamp"],
        records,
    )

    LOGGER.info("Saved cleaned dataset to %s", OUTPUT_FILE)
    LOGGER.info("Saved raw OCR debug output to %s", DEBUG_OUTPUT_FILE)


if __name__ == "__main__":
    main()
