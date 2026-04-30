#!/usr/bin/env python3
"""
Albanian Article Q&A Dataset Generator
========================================
Scrapes Albanian text from:
  1. Albanian Wikipedia API  (sq.wikipedia.org)  — bulk, clean, structured
  2. RSS feeds               (news24.al, bota.al, shqiptarja.com, vizionplus.tv)
  3. Full article scrapers   (top-channel.tv, rtsh.al, panorama.com.al)

Generates instruction/input/output Q&A entries without any AI API.

Q&A types produced
-------------------
  A  Summary          : "Përmblidh këtë artikull shqip."
  B  Topic            : "Cila është tema kryesore e këtij teksti?"
  C  True/False       : Claim extracted from article — correct or negated
  D  Fill-in-blank    : Sentence with key noun masked
  E  Headline ↔ Body  : "Cili është titulli i përshtatshëm për këtë artikull?"
  F  Comprehension    : "Çfarë thuhet në tekst për [entity]?"
  G  Continuation     : "Vazhdo tekstin shqip duke ruajtur frymën e autorit."
  H  Source domain    : "Ky tekst vjen nga: lajme / kulturë / sport / politikë?"

Usage
-----
  pip install requests beautifulsoup4 feedparser
  python generate_albanian_articles.py --target 10000 --out albanian_articles_qa.json

Resume: re-running appends to existing output and skips already-fetched articles.
"""

import argparse
import json
import random
import re
import time
import os
from pathlib import Path
from urllib.parse import quote

# ── optional deps (graceful fallback if missing) ──────────────────────────────
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("[WARN] requests not installed. Run: pip install requests beautifulsoup4 feedparser")

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DEFAULT     = Path("albanian_articles_qa.json")
CHECKPOINT      = Path("articles_checkpoint.json")
MIN_ARTICLE_LEN = 200    # characters — skip stubs shorter than this
MAX_ARTICLE_LEN = 8000   # characters — truncate very long articles
SLEEP_BETWEEN   = 1.2    # seconds between HTTP requests (be polite)
USER_AGENT      = "AlbanianNLPDatasetBot/1.0 (research; contact: research@example.com)"

HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "sq,en;q=0.5"}

# ── RSS feeds ─────────────────────────────────────────────────────────────────
RSS_FEEDS = [
    {"url": "https://news24.al/feed",          "domain": "lajme",    "name": "News24"},
    {"url": "https://bota.al/feed",            "domain": "kulturë",  "name": "Bota.al"},
    {"url": "https://vizionplus.tv/feed",       "domain": "lajme",    "name": "VizionPlus"},
    {"url": "https://lajme.al/feed",           "domain": "lajme",    "name": "Lajme.al"},
    {"url": "https://shqiptarja.com/rss.xml",  "domain": "politikë", "name": "Shqiptarja"},
]

# ── Wikipedia categories to pull from ─────────────────────────────────────────
WIKI_CATEGORIES = [
    "Historia_e_Shqipërisë",
    "Kultura_shqiptare",
    "Gjeografia_e_Shqipërisë",
    "Personalitete_shqiptare",
    "Shkenca",
    "Teknologjia",
    "Sporti_në_Shqipëri",
    "Letërsia_shqipe",
    "Politika_e_Shqipërisë",
    "Ekonomia_e_Shqipërisë",
]

# ── Instruction variants per type ─────────────────────────────────────────────
INSTR_SUMMARY = [
    "Përmblidh artikullin shqip të mëposhtëm në 2-3 fjali.",
    "Shkruaj një përmbledhje të shkurtër të këtij teksti shqip.",
    "Jep thelbin e këtij artikulli shqip me fjalët tuaja.",
    "Cila është mesazhi kryesor i këtij teksti? Përmblidhe shkurt.",
]
INSTR_TOPIC = [
    "Cila është tema kryesore e këtij teksti shqip?",
    "Për çfarë flet ky artikull? Identifiko temën.",
    "Çfarë subjekti trajton ky tekst shqip?",
    "Identifiko fushën dhe temën e këtij artikulli shqip.",
]
INSTR_TF = [
    "Bazuar në tekstin shqip, a është e saktë pohimi i mëposhtëm? Shpjego.",
    "Sipas artikullit, vlerëso nëse ky pohim është i vërtetë apo i gabuar.",
    "Lexo tekstin dhe trego nëse pohimi përputhet me informacionin e dhënë.",
    "Sipas tekstit shqip, pohimi i mëposhtëm është i saktë apo jo? Argumento.",
]
INSTR_FILL = [
    "Plotëso fjalinë shqipe duke gjetur fjalën që mungon.",
    "Cila fjalë mungon në fjalinë e mëposhtme shqipe?",
    "Gjej fjalën e duhur shqipe për të plotësuar fjalinë.",
    "Plotëso boshllëkun në fjalinë shqipe me fjalën e përshtatshme.",
]
INSTR_HEADLINE = [
    "Cili do të ishte titulli i përshtatshëm për këtë artikull shqip?",
    "Propozoni një titull gazetaresk shqip për tekstin e mëposhtëm.",
    "Shkruaj titullin më të mirë shqip për këtë artikull.",
    "Formuloni titullin e duhur shqip për këtë tekst.",
]
INSTR_COMPREHENSION = [
    "Çfarë thuhet në tekst lidhur me '{entity}'?",
    "Si përshkruhet '{entity}' në artikullin shqip?",
    "Cila informacion jep teksti shqip për '{entity}'?",
    "Shpjego rolin e '{entity}' sipas këtij teksti shqip.",
]
INSTR_CONTINUATION = [
    "Vazhdo tekstin shqip duke ruajtur frymën dhe stilin e autorit.",
    "Shkruaj vazhdimin e natyrshëm të këtij fragmenti shqip.",
    "Si do të vazhdonte ky tekst shqip? Shkruaj 2-3 fjali.",
    "Plotëso tekstin shqip duke ruajtur tonin origjinal.",
]
INSTR_DOMAIN = [
    "Në cilën kategori bën pjesë ky tekst shqip: lajme, kulturë, sport, politikë, ekonomi, shkencë apo histori?",
    "Identifiko fushën e këtij artikulli shqip.",
    "Klasifiko këtë tekst shqip sipas temës: lajme / kulturë / sport / politikë / tjetër.",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_session() -> "requests.Session":
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update(HEADERS)
    return s


def clean_text(raw: str) -> str:
    """Strip HTML tags, normalize whitespace, remove edit markers."""
    if HAS_BS4:
        raw = BeautifulSoup(raw, "html.parser").get_text(separator=" ")
    raw = re.sub(r'\[.*?\]', '', raw)          # wiki refs like [1]
    raw = re.sub(r'\{\{.*?\}\}', '', raw)      # wiki templates
    raw = re.sub(r'==+[^=]+==+', '', raw)      # wiki section headers
    raw = re.sub(r'\s+', ' ', raw).strip()
    return raw


def extract_sentences(text: str, min_len: int = 40) -> list:
    """Split text into sentences of reasonable length."""
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if len(p.strip()) >= min_len]


def extract_nouns_approx(sentence: str) -> list:
    """
    Rough noun extraction for Albanian without NLP libs:
    capitalized words that aren't sentence-starters.
    """
    words = sentence.split()
    nouns = []
    for i, w in enumerate(words):
        clean = re.sub(r'[^a-zA-ZëçËÇ]', '', w)
        if i > 0 and clean and clean[0].isupper() and len(clean) > 3:
            nouns.append(clean)
    return nouns


def _e(instruction: str, input_text: str, output: str, source: str = "") -> dict:
    entry = {"instruction": instruction, "input": input_text, "output": output}
    if source:
        entry["source"] = source
    return entry

# ── Wikipedia scraper ─────────────────────────────────────────────────────────

def fetch_wiki_article_list(session, category: str, limit: int = 50) -> list:
    """Get page titles from a Wikipedia category."""
    url = "https://sq.wikipedia.org/w/api.php"
    params = {
        "action": "query", "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": limit, "cmnamespace": 0, "format": "json"
    }
    try:
        r = session.get(url, params=params, timeout=10)
        data = r.json()
        return [m["title"] for m in data.get("query", {}).get("categorymembers", [])]
    except Exception as e:
        print(f"  [WARN] Wiki category {category}: {e}")
        return []


def fetch_wiki_random(session, count: int = 20) -> list:
    """Fetch random Albanian Wikipedia article titles."""
    url = "https://sq.wikipedia.org/w/api.php"
    params = {"action": "query", "list": "random", "rnnamespace": 0,
              "rnlimit": count, "format": "json"}
    try:
        r = session.get(url, params=params, timeout=10)
        return [p["title"] for p in r.json().get("query", {}).get("random", [])]
    except Exception as e:
        print(f"  [WARN] Wiki random: {e}")
        return []


def fetch_wiki_text(session, title: str) -> dict | None:
    """Fetch plain text extract of a Wikipedia article."""
    url = "https://sq.wikipedia.org/w/api.php"
    params = {
        "action": "query", "titles": title,
        "prop": "extracts", "explaintext": True,
        "exsectionformat": "plain", "format": "json"
    }
    try:
        r = session.get(url, params=params, timeout=15)
        pages = r.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()))
        text = page.get("extract", "")
        text = clean_text(text)
        if len(text) < MIN_ARTICLE_LEN:
            return None
        return {
            "title": title,
            "text": text[:MAX_ARTICLE_LEN],
            "domain": "enciklopedi",
            "source": f"https://sq.wikipedia.org/wiki/{quote(title)}"
        }
    except Exception as e:
        print(f"  [WARN] Wiki fetch '{title}': {e}")
        return None

# ── RSS scraper ───────────────────────────────────────────────────────────────

def fetch_rss_articles(session, feed: dict) -> list:
    """Parse an RSS feed and return list of article dicts."""
    if not HAS_FEEDPARSER:
        return []
    articles = []
    try:
        f = feedparser.parse(feed["url"])
        for entry in f.entries:
            title = entry.get("title", "").strip()
            summary = clean_text(entry.get("summary", entry.get("description", "")))
            if len(summary) < MIN_ARTICLE_LEN:
                continue
            articles.append({
                "title": title,
                "text": summary[:MAX_ARTICLE_LEN],
                "domain": feed["domain"],
                "source": feed["name"]
            })
    except Exception as e:
        print(f"  [WARN] RSS {feed['name']}: {e}")
    return articles

# ── Full article scrapers ─────────────────────────────────────────────────────

def scrape_topchannel(session, max_articles: int = 30) -> list:
    """Scrape Top Channel news articles."""
    if not HAS_BS4:
        return []
    articles = []
    try:
        r = session.get("https://top-channel.tv/lajme/", timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        links = []
        for a in soup.select("a[href]"):
            href = a["href"]
            if "/lajme/" in href and href.startswith("https://top-channel.tv/lajme/") and len(href) > 35:
                links.append(href)
        links = list(dict.fromkeys(links))[:max_articles]

        for link in links:
            try:
                ar = session.get(link, timeout=15)
                asoup = BeautifulSoup(ar.text, "html.parser")
                title_tag = asoup.find("h1")
                title = title_tag.get_text(strip=True) if title_tag else ""
                body_tags = asoup.select("div.article-content p, div.entry-content p, article p")
                body = " ".join(t.get_text(strip=True) for t in body_tags)
                body = clean_text(body)
                if len(body) >= MIN_ARTICLE_LEN:
                    articles.append({
                        "title": title, "text": body[:MAX_ARTICLE_LEN],
                        "domain": "lajme", "source": "Top Channel"
                    })
                time.sleep(SLEEP_BETWEEN)
            except Exception:
                continue
    except Exception as e:
        print(f"  [WARN] Top Channel: {e}")
    return articles


def scrape_rtsh(session, max_articles: int = 30) -> list:
    """Scrape RTSH news articles."""
    if not HAS_BS4:
        return []
    articles = []
    try:
        r = session.get("https://www.rtsh.al/lajme/", timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        links = []
        for a in soup.select("a[href]"):
            href = a["href"]
            if "rtsh.al" in href and "/lajme/" in href and len(href) > 25:
                if not href.startswith("http"):
                    href = "https://www.rtsh.al" + href
                links.append(href)
        links = list(dict.fromkeys(links))[:max_articles]

        for link in links:
            try:
                ar = session.get(link, timeout=15)
                asoup = BeautifulSoup(ar.text, "html.parser")
                title_tag = asoup.find("h1")
                title = title_tag.get_text(strip=True) if title_tag else ""
                body_tags = asoup.select(".article-body p, .post-content p, article p")
                body = " ".join(t.get_text(strip=True) for t in body_tags)
                body = clean_text(body)
                if len(body) >= MIN_ARTICLE_LEN:
                    articles.append({
                        "title": title, "text": body[:MAX_ARTICLE_LEN],
                        "domain": "lajme", "source": "RTSH"
                    })
                time.sleep(SLEEP_BETWEEN)
            except Exception:
                continue
    except Exception as e:
        print(f"  [WARN] RTSH: {e}")
    return articles


def scrape_panorama(session, max_articles: int = 30) -> list:
    """Scrape Panorama news articles."""
    if not HAS_BS4:
        return []
    articles = []
    try:
        r = session.get("https://www.panorama.com.al/", timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        links = []
        for a in soup.select("a[href]"):
            href = a["href"]
            if "panorama.com.al" in href and len(href) > 30:
                links.append(href)
        links = list(dict.fromkeys(links))[:max_articles]

        for link in links:
            try:
                ar = session.get(link, timeout=15)
                asoup = BeautifulSoup(ar.text, "html.parser")
                title_tag = asoup.find("h1")
                title = title_tag.get_text(strip=True) if title_tag else ""
                body_tags = asoup.select(".article-text p, .entry-content p, article p")
                body = " ".join(t.get_text(strip=True) for t in body_tags)
                body = clean_text(body)
                if len(body) >= MIN_ARTICLE_LEN:
                    articles.append({
                        "title": title, "text": body[:MAX_ARTICLE_LEN],
                        "domain": "lajme", "source": "Panorama"
                    })
                time.sleep(SLEEP_BETWEEN)
            except Exception:
                continue
    except Exception as e:
        print(f"  [WARN] Panorama: {e}")
    return articles

# ── Q&A generators ────────────────────────────────────────────────────────────

def gen_A_summary(articles: list) -> list:
    out = []
    for a in articles:
        text = a["text"]
        sents = extract_sentences(text)
        if not sents:
            continue
        # Output = first 3 sentences as the "summary"
        summary = " ".join(sents[:3])
        instr = random.choice(INSTR_SUMMARY)
        out.append(_e(instr, text[:1500], summary, a["source"]))
    return out


def gen_B_topic(articles: list) -> list:
    out = []
    for a in articles:
        if not a.get("title"):
            continue
        domain_map = {
            "lajme": "Lajme dhe aktualitet",
            "kulturë": "Kulturë dhe arte",
            "sport": "Sport",
            "politikë": "Politikë",
            "ekonomi": "Ekonomi",
            "enciklopedi": "Informacion enciklopedik",
        }
        domain_label = domain_map.get(a["domain"], "Informacion i përgjithshëm")
        output = (
            f"Tema kryesore e këtij teksti është '{a['title']}'. "
            f"Bën pjesë në kategorinë: {domain_label}."
        )
        instr = random.choice(INSTR_TOPIC)
        out.append(_e(instr, a["text"][:1500], output, a["source"]))
    return out


def gen_C_truefalse(articles: list) -> list:
    out = []
    for a in articles:
        sents = extract_sentences(a["text"])
        if len(sents) < 2:
            continue

        # TRUE claim — pick a sentence from middle of article
        true_sent = random.choice(sents[1:min(6, len(sents))])
        instr = random.choice(INSTR_TF)
        out.append(_e(
            instr,
            f"Teksti:\n{a['text'][:1200]}\n\nPohimi: \"{true_sent}\"",
            f"E SAKTË. Ky pohim përputhet me informacionin e dhënë në tekst: \"{true_sent}\"",
            a["source"]
        ))

        # FALSE claim — swap a number or keyword with something plausible but wrong
        false_sent = true_sent
        swaps = [("është", "nuk është"), ("ka", "nuk ka"), ("do të", "nuk do të"),
                 ("shqiptar", "grek"), ("Tiranë", "Durrës"), ("2024", "2019"),
                 ("rritje", "rënie"), ("fitoi", "humbi"), ("u hap", "u mbyll")]
        for orig, repl in swaps:
            if orig in false_sent:
                false_sent = false_sent.replace(orig, repl, 1)
                out.append(_e(
                    instr,
                    f"Teksti:\n{a['text'][:1200]}\n\nPohimi: \"{false_sent}\"",
                    f"E GABUAR. Teksti origjinal thotë: \"{true_sent}\". "
                    f"Pohimi i modifikuar nuk përputhet me informacionin e dhënë.",
                    a["source"]
                ))
                break

    return out


def gen_D_fillinblank(articles: list) -> list:
    out = []
    for a in articles:
        sents = extract_sentences(a["text"])
        for sent in sents[1:6]:
            nouns = extract_nouns_approx(sent)
            if not nouns:
                continue
            target = nouns[0]
            masked = sent.replace(target, "______", 1)
            if masked == sent:
                continue
            instr = random.choice(INSTR_FILL)
            out.append(_e(
                instr,
                f"Fjalia: \"{masked}\"\n\nKonteksti: {a['text'][:600]}",
                f"Fjala që mungon është: '{target}'.\nFjalia e plotë: \"{sent}\"",
                a["source"]
            ))
            break
    return out


def gen_E_headline(articles: list) -> list:
    out = []
    for a in articles:
        if not a.get("title") or len(a["title"]) < 10:
            continue
        instr = random.choice(INSTR_HEADLINE)
        # Give only the body, ask for headline
        out.append(_e(
            instr,
            a["text"][:1000],
            f"Titulli i përshtatshëm: \"{a['title']}\"",
            a["source"]
        ))
        # Reverse: give headline, summarize what article might be about
        out.append(_e(
            "Duke u bazuar në titullin e mëposhtëm shqip, çfarë informacioni pritet të gjesh në artikull?",
            a["title"],
            f"Bazuar në titullin '{a['title']}', artikulli trajton: {a['text'][:300]}...",
            a["source"]
        ))
    return out


def gen_F_comprehension(articles: list) -> list:
    out = []
    for a in articles:
        sents = extract_sentences(a["text"])
        for sent in sents[:8]:
            nouns = extract_nouns_approx(sent)
            if not nouns:
                continue
            entity = nouns[0]
            # Find all sentences mentioning this entity
            related = [s for s in sents if entity in s]
            if not related:
                continue
            instr = random.choice(INSTR_COMPREHENSION).format(entity=entity)
            answer = " ".join(related[:2])
            out.append(_e(
                instr,
                a["text"][:1500],
                f"Sipas tekstit, lidhur me '{entity}': {answer}",
                a["source"]
            ))
            break
    return out


def gen_G_continuation(articles: list) -> list:
    out = []
    for a in articles:
        sents = extract_sentences(a["text"])
        if len(sents) < 4:
            continue
        # Give first half, expect the second half
        cut = len(sents) // 2
        fragment = " ".join(sents[:cut])
        continuation = " ".join(sents[cut:cut + 3])
        instr = random.choice(INSTR_CONTINUATION)
        out.append(_e(instr, fragment, continuation, a["source"]))
    return out


def gen_H_domain(articles: list) -> list:
    domain_descriptions = {
        "lajme":      "Lajme dhe aktualitet — teksti raporton ngjarje të fundit.",
        "kulturë":    "Kulturë dhe arte — teksti trajton aspekte kulturore ose artistike.",
        "sport":      "Sport — teksti lidhet me ngjarje sportive ose atletë.",
        "politikë":   "Politikë — teksti diskuton zhvillime politike.",
        "ekonomi":    "Ekonomi — teksti trajton çështje ekonomike ose financiare.",
        "enciklopedi":"Informacion enciklopedik — teksti ofron njohuri të përgjithshme.",
    }
    out = []
    for a in articles:
        label = domain_descriptions.get(a["domain"], "Tjetër")
        instr = random.choice(INSTR_DOMAIN)
        out.append(_e(instr, a["text"][:800], label, a["source"]))
    return out


# ── Checkpoint ────────────────────────────────────────────────────────────────

def load_checkpoint() -> set:
    if CHECKPOINT.exists():
        with open(CHECKPOINT, encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_checkpoint(seen: set):
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(list(seen), f, ensure_ascii=False)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",      type=int, default=10_000)
    parser.add_argument("--out",         type=str, default=str(OUT_DEFAULT))
    parser.add_argument("--wiki-limit",  type=int, default=500,
                        help="Max Wikipedia articles to fetch")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--skip-scrape", action="store_true",
                        help="Skip live news scrapers, use only Wikipedia + RSS")
    args = parser.parse_args()

    random.seed(args.seed)

    if not HAS_REQUESTS:
        print("[ERROR] Install dependencies first:")
        print("  pip install requests beautifulsoup4 feedparser")
        return

    out_path = Path(args.out)
    existing_entries = []
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            existing_entries = json.load(f)
        print(f"[INFO] Resuming — {len(existing_entries)} entries already saved.")

    seen_titles = load_checkpoint()
    session = make_session()

    # ── 1. Collect articles ────────────────────────────────────────────────────
    print("\n[1/3] Collecting articles...")
    all_articles = []

    # Wikipedia — categories
    for cat in WIKI_CATEGORIES:
        print(f"  Wikipedia category: {cat}")
        titles = fetch_wiki_article_list(session, cat, limit=50)
        for title in titles:
            if title in seen_titles:
                continue
            time.sleep(SLEEP_BETWEEN)
            art = fetch_wiki_text(session, title)
            if art:
                all_articles.append(art)
                seen_titles.add(title)
        if len(all_articles) >= args.wiki_limit // 2:
            break

    # Wikipedia — random articles to fill diversity
    print(f"  Wikipedia random articles...")
    needed = args.wiki_limit - len(all_articles)
    batches = (needed // 20) + 1
    for _ in range(batches):
        titles = fetch_wiki_random(session, count=20)
        for title in titles:
            if title in seen_titles:
                continue
            time.sleep(SLEEP_BETWEEN)
            art = fetch_wiki_text(session, title)
            if art:
                all_articles.append(art)
                seen_titles.add(title)
        if len(all_articles) >= args.wiki_limit:
            break

    print(f"  → {len(all_articles)} Wikipedia articles collected.")

    # RSS feeds
    print("  Fetching RSS feeds...")
    for feed in RSS_FEEDS:
        rss_arts = fetch_rss_articles(session, feed)
        new = [a for a in rss_arts if a["title"] not in seen_titles]
        for a in new:
            seen_titles.add(a["title"])
        all_articles.extend(new)
        print(f"    {feed['name']}: {len(new)} articles")
        time.sleep(SLEEP_BETWEEN)

    # Live scrapers (skippable)
    if not args.skip_scrape:
        print("  Scraping Top Channel...")
        tc = scrape_topchannel(session, max_articles=40)
        all_articles.extend([a for a in tc if a["title"] not in seen_titles])

        print("  Scraping RTSH...")
        rtsh = scrape_rtsh(session, max_articles=40)
        all_articles.extend([a for a in rtsh if a["title"] not in seen_titles])

        print("  Scraping Panorama...")
        pan = scrape_panorama(session, max_articles=40)
        all_articles.extend([a for a in pan if a["title"] not in seen_titles])

    print(f"\n  TOTAL articles collected: {len(all_articles)}")

    # ── 2. Generate Q&A entries ────────────────────────────────────────────────
    print("\n[2/3] Generating Q&A entries...")
    generators = [
        ("A - Summary",       gen_A_summary,       all_articles),
        ("B - Topic ID",      gen_B_topic,          all_articles),
        ("C - True/False",    gen_C_truefalse,      all_articles),
        ("D - Fill-in-blank", gen_D_fillinblank,    all_articles),
        ("E - Headline",      gen_E_headline,       all_articles),
        ("F - Comprehension", gen_F_comprehension,  all_articles),
        ("G - Continuation",  gen_G_continuation,   all_articles),
        ("H - Domain class",  gen_H_domain,         all_articles),
    ]

    new_entries = []
    counts = {}
    for label, fn, data in generators:
        batch = fn(data)
        random.shuffle(batch)
        counts[label] = len(batch)
        new_entries.extend(batch)
        print(f"  {label:28s}: {len(batch):>5} entries")

    # ── 3. Merge, dedup, cap ───────────────────────────────────────────────────
    print("\n[3/3] Deduplicating and saving...")
    all_entries = list(existing_entries) + new_entries
    seen_keys = set()
    deduped = []
    for e in all_entries:
        key = (e["instruction"][:80], e["input"][:80])
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(e)

    random.shuffle(deduped)
    final = deduped[:args.target]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    save_checkpoint(seen_titles)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 58)
    print(f"  OUTPUT      : {out_path}")
    print(f"  TOTAL       : {len(final):,} entries  (target: {args.target:,})")
    print(f"  ARTICLES    : {len(all_articles)} unique articles scraped")
    print()
    print("  Entries by type:")
    for label, cnt in counts.items():
        print(f"    {label:30s}  {cnt:>5}")
    print("=" * 58)

    if len(final) < args.target:
        shortfall = args.target - len(final)
        print(f"\n[NOTE] {shortfall} entries short of target.")
        print("  Options to increase coverage:")
        print("  1. Re-run — Wikipedia random gives different articles each time")
        print("  2. Increase --wiki-limit (default 500)")
        print("  3. Combine with albanian_qa_dataset.json from the synonym generator")
        print("  4. Add more RSS feeds to RSS_FEEDS list in the script")

    print("\nSample entry:")
    sample = random.choice(final)
    print(f"  instruction : {sample['instruction']}")
    print(f"  input       : {sample['input'][:120]}...")
    print(f"  output      : {sample['output'][:120]}...")
    print(f"  source      : {sample.get('source', 'N/A')}")


if __name__ == "__main__":
    main()
