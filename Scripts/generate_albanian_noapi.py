#!/usr/bin/env python3
"""
Albanian Q&A Dataset Generator  —  NO API REQUIRED
====================================================
Generates 10 000 instruction/input/output entries from:
  1. Albanian_Synonym_Dataset (1).csv  — word–synonym pairs
  2. kushtetuta.csv                    — Albanian Constitution articles

Q&A types produced
-------------------
  A  Forward synonym  : "Cilat janë sinonim/et e fjalës 'X'?"
  B  Single synonym   : "Jep një sinonim për fjalën 'X'."
  C  Reverse lookup   : "Cila fjalë ka si sinonim 'Y'?"
  D  Yes / No check   : "A është 'Y' sinonim i 'X'?"
  E  Multiple choice  : "Cila nga këto fjalë është sinonim i 'X'?"
  F  Synonym list     : "Rendit sinonim/et e fjalës 'X'."
  G  Constitution Q   : "Çfarë parashikon Neni X i Kushtetutës?"
  H  Constitution TF  : "Sipas Kushtetutës, vlerëso nëse kjo pohim është e saktë."

Usage
-----
  python generate_albanian_noapi.py [--target 10000] [--out albanian_qa.json]
"""

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
# Locate data relative to the repository root (one level up from Scripts/)
BASE = Path(__file__).parent.parent / "csv"
SYNONYM_CSV  = BASE / "Albanian_Synonym_Dataset (1).csv"
KUSHTETUTA   = BASE / "kushtetuta.csv"

EXISTING_JSON = Path(__file__).parent.parent / "JSON" / "albanian_vocabulary_dataset.json"

# ── Instruction variants (cycled per type for diversity) ──────────────────────
INSTR_FORWARD = [
    "Cilat janë sinonim/et e fjalës shqipe të mëposhtme?",
    "Jep sinonim/et e fjalës shqipe.",
    "Rendit fjalët që kanë kuptim të ngjashëm me fjalën e dhënë.",
    "Çfarë fjalësh shqipe kanë kuptim të njëjtë ose të ngjashëm me këtë fjalë?",
    "Shëno sinonim/et e fjalës shqipe.",
]
INSTR_SINGLE = [
    "Jep një sinonim për fjalën shqipe të mëposhtme.",
    "Cila fjalë shqipe mund të zëvendësojë fjalën e dhënë?",
    "Shkruaj një fjalë shqipe me kuptim të ngjashëm.",
    "Çfarë fjale tjetër shqipe ka kuptim të njëjtë me këtë?",
]
INSTR_REVERSE = [
    "Cila fjalë shqipe ka si sinonim fjalën e mëposhtme?",
    "Gjej fjalën shqipe kryesore që lidhet me sinonimin e dhënë.",
    "Sinonimi i dhënë i përket cilës fjalë shqipe?",
    "Për cilën fjalë shqipe shërben si sinonim fjala e mëposhtme?",
]
INSTR_YESNO = [
    "A janë këto dy fjalë sinonime në gjuhën shqipe?",
    "A kanë kuptim të ngjashëm këto dy fjalë shqipe?",
    "Vlerëso: a mund të zëvendësohen këto dy fjalë me njëra-tjetrën?",
    "Thuaj po ose jo: a janë këto fjalë sinonime?",
]
INSTR_MCQ = [
    "Cila nga fjalët e mëposhtme është sinonim i fjalës së dhënë?",
    "Zgjidh sinonimin e saktë për fjalën shqipe.",
    "Nga opsionet e dhëna, cila fjalë ka kuptim të njëjtë me fjalën e shënuar?",
    "Identifiko sinonimin e saktë.",
]
INSTR_LIST = [
    "Rendit të gjithë sinonim/et e njohur të fjalës shqipe.",
    "Bëj një listë të sinonimeve të fjalës shqipe të mëposhtme.",
    "Shkruaj sa më shumë sinonime për fjalën shqipe.",
]
INSTR_KUSH = [
    "Çfarë parashikon Neni {n} i Kushtetutës së Republikës së Shqipërisë?",
    "Shpjego përmbajtjen e Nenit {n} të Kushtetutës shqiptare.",
    "Çfarë thuhet në Nenin {n} të Kushtetutës?",
    "Cili është teksti i Nenit {n} të Kushtetutës së Shqipërisë?",
]
INSTR_NONSYNONYM = [
    "Cila nga fjalët e dhëna NUK është sinonim i fjalës shqipe?",
    "Gjej fjalën që nuk lidhet me kuptimin e fjalës shqipe.",
    "Identifiko fjalën që është e ndryshme nga të tjerat për nga kuptimi.",
    "Cila fjalë NUK mund të zëvendësojë fjalën e dhënë?",
]
INSTR_KUSH_TF = [
    "Sipas Kushtetutës shqiptare, vlerëso nëse pohimi i mëposhtëm është i saktë.",
    "Bazuar te Kushtetuta e Shqipërisë, a është e vërtetë kjo?",
    "Sipas Nenit {n} të Kushtetutës, pohimi i mëposhtëm është i saktë apo i gabuar?",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _alb(s: str) -> bool:
    """Quick check: plausible Albanian word (no OCR junk)."""
    return bool(re.match(
        r'^[a-zA-Z\u00eb\u00e7\u00cb\u00c7][a-zA-Z\u00eb\u00e7\u00cb\u00c7\s\-]{1,30}$',
        s
    )) and 'shih' not in s.lower() and len(s) > 2

def _clean_syns(raw: str) -> list:
    parts = re.split(r'[,;:]', raw)
    return [p.strip().strip('.,;:()\'\"').strip() for p in parts
            if _alb(p.strip().strip('.,;:()\'\"').strip())]

def _pick(lst: list, exclude=None):
    pool = [x for x in lst if x != exclude]
    return random.choice(pool) if pool else None

def _e(instruction: str, word: str, output: str) -> dict:
    return {"instruction": instruction, "input": word, "output": output}


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_synonyms() -> list:
    """Returns list of dicts: {word, synonyms: [str]}"""
    rows = []
    try:
        with open(SYNONYM_CSV, encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                word = (r.get('entry_word') or '').strip()
                if not _alb(word):
                    continue
                syns = _clean_syns(r.get('synonyms') or '')
                if syns:
                    rows.append({'word': word, 'synonyms': syns})
    except FileNotFoundError:
        sys.exit(f"[ERROR] Cannot find: {SYNONYM_CSV}")
    return rows


def load_constitution() -> list:
    """Returns list of dicts: {neni: str, text: str}"""
    try:
        import pandas as pd
        df = pd.read_csv(KUSHTETUTA)
        raw = df['cleaned_text'].dropna().iloc[0]
    except Exception as e:
        print(f"[WARN] Could not load kushtetuta: {e}")
        return []

    articles, cur_n, cur_t = [], None, []
    for line in raw.split('\n'):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^Neni\s+(\d+[a-z]*)', line, re.IGNORECASE)
        if m:
            if cur_n and cur_t:
                articles.append({'neni': cur_n, 'text': ' '.join(cur_t)})
            cur_n, cur_t = m.group(1), []
        elif cur_n and len(line) > 3 and not re.match(r'^\d+$', line):
            cur_t.append(line)
    if cur_n and cur_t:
        articles.append({'neni': cur_n, 'text': ' '.join(cur_t)})
    return articles


def load_existing() -> list:
    """Load pre-existing 360-entry JSON so we don't discard it."""
    if EXISTING_JSON.exists():
        with open(EXISTING_JSON, encoding='utf-8') as f:
            return json.load(f)
    return []


# ── Generators ────────────────────────────────────────────────────────────────

def gen_A_forward(rows: list) -> list:
    """Cilat janë sinonim/et e fjalës X?"""
    out = []
    for r in rows:
        w, syns = r['word'], r['synonyms']
        syn_str = ', '.join(syns)
        instr = random.choice(INSTR_FORWARD)
        answer = (
            f"Sinonim{'et' if len(syns) > 1 else 'i'} e fjalës '{w}' "
            f"{'janë' if len(syns) > 1 else 'është'}: {syn_str}."
        )
        out.append(_e(instr, w, answer))
    return out


def gen_B_single(rows: list) -> list:
    """Jep NJË sinonim për fjalën X."""
    out = []
    for r in rows:
        w, syns = r['word'], r['synonyms']
        best = syns[0]
        instr = random.choice(INSTR_SINGLE)
        answer = f"Një sinonim i fjalës '{w}' është '{best}'."
        out.append(_e(instr, w, answer))
    return out


def gen_C_reverse(rows: list) -> list:
    """Cila fjalë ka si sinonim Y? — one entry per (synonym, headword) pair."""
    out = []
    for r in rows:
        w, syns = r['word'], r['synonyms']
        for syn in syns:
            instr = random.choice(INSTR_REVERSE)
            answer = f"Fjala '{syn}' shërben si sinonim i fjalës '{w}'."
            out.append(_e(instr, syn, answer))
    return out


def gen_D_yesno(rows: list, negatives_ratio: float = 0.4) -> list:
    """A është Y sinonim i X? — mix of Yes and No answers."""
    out = []
    all_words = [r['word'] for r in rows]
    for r in rows:
        w, syns = r['word'], r['synonyms']
        # Positive
        syn = random.choice(syns)
        instr = random.choice(INSTR_YESNO)
        inp = f"{w} / {syn}"
        out.append(_e(instr, inp,
            f"Po, '{syn}' dhe '{w}' janë sinonime. Ato kanë kuptim të njëjtë ose shumë të ngjashëm."))
        # Negative (random word from pool that is NOT a known synonym)
        neg = _pick(all_words, exclude=w)
        if neg:
            inp_n = f"{w} / {neg}"
            out.append(_e(instr, inp_n,
                f"Jo, '{neg}' nuk është sinonim i '{w}'. Këto dy fjalë kanë kuptime të ndryshme."))
    return out


def gen_E_mcq(rows: list) -> list:
    """Multiple-choice: cila nga A/B/C/D është sinonim i X?"""
    out = []
    all_words = [r['word'] for r in rows]
    for r in rows:
        w, syns = r['word'], r['synonyms']
        correct = random.choice(syns)
        # 3 wrong distractors from the full word pool
        wrong_pool = [ww for ww in all_words if ww != w and ww not in syns]
        if len(wrong_pool) < 3:
            continue
        distractors = random.sample(wrong_pool, 3)
        options = distractors + [correct]
        random.shuffle(options)
        labels = ['A', 'B', 'C', 'D']
        opt_str = '  '.join(f"{labels[i]}. {options[i]}" for i in range(4))
        correct_label = labels[options.index(correct)]
        instr = random.choice(INSTR_MCQ)
        answer = (
            f"Përgjigja e saktë është {correct_label}. '{correct}' është sinonim i '{w}'. "
            f"Opsionet e tjera ({', '.join(distractors)}) kanë kuptime të ndryshme."
        )
        out.append(_e(instr, f"{w}\n{opt_str}", answer))
    return out


def gen_F_list(rows: list) -> list:
    """Rendit të gjithë sinonim/et e njohur të fjalës X."""
    out = []
    for r in rows:
        w, syns = r['word'], r['synonyms']
        if len(syns) < 2:   # skip singles — already covered by gen_B
            continue
        numbered = '  '.join(f"{i+1}. {s}" for i, s in enumerate(syns))
        instr = random.choice(INSTR_LIST)
        answer = (
            f"Sinonim/et e fjalës '{w}' ({len(syns)} gjithsej):\n"
            f"{numbered}"
        )
        out.append(_e(instr, w, answer))
    return out


def gen_I_nonsynonym(rows: list) -> list:
    """Cila fjalë NUK është sinonim i X? — 3 valid synonyms + 1 intruder."""
    out = []
    all_words = [r['word'] for r in rows]
    for r in rows:
        w, syns = r['word'], r['synonyms']
        if len(syns) < 3:
            continue
        correct_syns = random.sample(syns, min(3, len(syns)))
        intruder_pool = [ww for ww in all_words if ww not in syns and ww != w]
        if not intruder_pool:
            continue
        intruder = random.choice(intruder_pool)
        options = correct_syns + [intruder]
        random.shuffle(options)
        labels = ['A', 'B', 'C', 'D']
        opt_str = '  '.join(f"{labels[i]}. {options[i]}" for i in range(4))
        intruder_label = labels[options.index(intruder)]
        instr = random.choice(INSTR_NONSYNONYM)
        answer = (
            f"Përgjigja e saktë është {intruder_label}. '{intruder}' nuk është sinonim i '{w}'. "
            f"Sinonim/et e vërteta të '{w}' janë: {', '.join(correct_syns)}."
        )
        out.append(_e(instr, f"{w}\n{opt_str}", answer))
    return out


def gen_G_kush(articles: list) -> list:
    """Çfarë parashikon Neni X i Kushtetutës?"""
    out = []
    for a in articles:
        n, text = a['neni'], a['text']
        if len(text) < 30:
            continue
        # Clean minor OCR artifacts (stray digits, 'ë' → '3' fixups already in CSV)
        text = re.sub(r'\s+', ' ', text).strip()
        instr = random.choice(INSTR_KUSH).format(n=n)
        answer = (
            f"Neni {n} i Kushtetutës së Republikës së Shqipërisë parashikon:\n"
            f"{text}"
        )
        out.append(_e(instr, f"Neni {n}", answer))
    return out


def gen_H_kush_tf(articles: list) -> list:
    """True/False about constitutional articles."""
    out = []
    for a in articles:
        n, text = a['neni'], a['text']
        if len(text) < 40:
            continue
        text = re.sub(r'\s+', ' ', text).strip()

        # Build a plausible TRUE claim from first clause
        first_clause = text[:180].rstrip(',').strip()
        instr_t = random.choice(INSTR_KUSH_TF).format(n=n)
        out.append(_e(instr_t,
            f"Pohimi: \"{first_clause}...\"",
            f"E SAKTË. Sipas Nenit {n} të Kushtetutës: {text[:300].strip()}"
        ))

        # Build a FALSE claim by swapping a keyword
        swaps = [
            ('parlamentare', 'presidenciale'),
            ('popullit', 'qeverisë'),
            ('ligjvënës', 'ekzekutiv'),
            ('Kushtetuta', 'ligji'),
            ('drejtpërsëdrejti', 'indirektisht'),
        ]
        swapped = text
        for orig, repl in swaps:
            if orig in swapped:
                swapped = swapped.replace(orig, repl, 1)
                fake_claim = swapped[:180].rstrip(',').strip()
                out.append(_e(instr_t,
                    f"Pohimi: \"{fake_claim}...\"",
                    f"E GABUAR. Teksti i saktë i Nenit {n} thotë: \"{text[:250].strip()}\""
                ))
                break
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=int, default=10_000)
    parser.add_argument('--out',    type=str, default='albanian_qa_dataset.json')
    parser.add_argument('--seed',   type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("[1/6] Loading synonym data...")
    rows = load_synonyms()
    print(f"      {len(rows)} clean word-synonym pairs loaded.")

    print("[2/6] Loading constitution...")
    articles = load_constitution()
    print(f"      {len(articles)} articles loaded.")

    print("[3/6] Loading existing JSON entries...")
    existing = load_existing()
    print(f"      {len(existing)} pre-existing entries.")

    print("[4/6] Generating Q&A entries...")
    all_entries = list(existing)   # start from existing 360

    generators = [
        ("A - Forward synonym",  gen_A_forward,  rows),
        ("B - Single synonym",   gen_B_single,   rows),
        ("C - Reverse lookup",   gen_C_reverse,  rows),
        ("D - Yes/No check",     gen_D_yesno,    rows),
        ("E - Multiple choice",  gen_E_mcq,      rows),
        ("F - Full list",        gen_F_list,      rows),
        ("G - Constitution Q",   gen_G_kush,      articles),
        ("H - Constitution T/F", gen_H_kush_tf,   articles),
        ("I - Non-synonym MCQ",  gen_I_nonsynonym, rows),
    ]

    counts = {}
    for label, fn, data in generators:
        batch = fn(data)
        random.shuffle(batch)
        counts[label] = len(batch)
        all_entries.extend(batch)
        print(f"      {label}: {len(batch)} entries  (running total: {len(all_entries)})")

    print("[5/6] Deduplicating and shuffling...")
    # Deduplicate by (instruction, input) key
    seen = set()
    deduped = []
    for e in all_entries:
        key = (e['instruction'].strip(), e['input'].strip())
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    random.shuffle(deduped)
    print(f"      {len(deduped)} unique entries after dedup.")

    # Cap or report
    final = deduped[:args.target] if len(deduped) > args.target else deduped
    if len(final) < args.target:
        print(f"[WARN] Only {len(final)} entries generated (target was {args.target}).")
        print("       The synonym CSV doesn't have enough variety to reach 10 000 alone.")
        print("       Consider augmenting with additional word lists.")
    else:
        print(f"[OK]  Capping to {args.target} entries.")

    print(f"[6/6] Writing to {args.out}...")
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    # ── Summary ──
    print()
    print("=" * 58)
    print(f"  OUTPUT FILE : {args.out}")
    print(f"  TOTAL       : {len(final):,} entries")
    print()
    print("  Breakdown by type:")
    for label, cnt in counts.items():
        print(f"    {label:30s}  {cnt:>5}")
    print(f"    {'Existing JSON (vocabulary)':30s}  {len(existing):>5}")
    print("=" * 58)

    # Preview
    print()
    print("  Sample entries:")
    for e in random.sample(final, min(3, len(final))):
        print(f"  instruction : {e['instruction']}")
        print(f"  input       : {e['input']}")
        print(f"  output      : {e['output'][:120]}...")
        print()


if __name__ == '__main__':
    main()
