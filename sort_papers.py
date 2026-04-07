#!/usr/bin/env python3
"""sort_papers.py — Sort science PDFs into topic folders.

Usage:
    python sort_papers.py ./my-papers
    python sort_papers.py ./my-papers --output ./sorted
    python sort_papers.py ./my-papers --categories 12
    python sort_papers.py ./my-papers --dry-run

Requirements:
    pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from rich import box
from rich.console import Console
from rich.progress import (BarColumn, Progress, SpinnerColumn,
                            TaskProgressColumn, TextColumn, TimeRemainingColumn)
from rich.table import Table

import bio_tags as bt

console = Console()

# ── Constants ─────────────────────────────────────────────────────────────────

_MAX_TEXT_CHARS   = 40000  # full text stored per paper (for tag detection)
_EMBED_TEXT_CHARS = 6000   # truncated text fed to embedder
_MAX_PAGES        = 12     # pages to read per PDF (captures methods section)
_EMBED_MODEL      = "all-MiniLM-L6-v2"

_STOP_WORDS = {
    "the","a","an","and","or","of","in","for","to","with","on","at","by",
    "from","is","are","was","were","be","been","this","that","we","our",
    "their","which","also","paper","show","shows","propose","proposed",
    "method","results","data","based","using","used","can","approach",
    "model","models","work","study","new","two","one","three","use",
    "uses","task","tasks","set","learning","deep","machine","neural",
}

_YEAR_RE  = re.compile(r"\b(19[89]\d|20[0-2]\d)\b")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_ID_RE    = re.compile(r"^\d+$|^[a-z0-9]{6,}\d{4,}|npgrj|biorxiv|doi", re.I)

_MIN_TEXT_CHARS = 200   # skip PDFs shorter than this (icons, toolbar images)


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class PaperMeta:
    path: Path
    raw_text: str       = ""
    extraction_ok: bool = True

    title:             str | None = None
    first_author_last: str        = "Unknown"
    year:              str        = "XXXX"
    abstract:          str | None = None
    keywords:          list[str]  = field(default_factory=list)

    # Biology tags (filled by extract_meta -> bt.tag)
    omics:        list[str] = field(default_factory=list)
    organism:     str       = ""
    article_type: str       = ""
    diseases:     list[str] = field(default_factory=list)
    is_trial:     bool      = False
    tissues:      list[str] = field(default_factory=list)

    # Set after clustering
    category:   str        = ""
    confidence: float      = 0.0
    new_path:   Path | None = None


# ── 1. PDF discovery ──────────────────────────────────────────────────────────

def find_pdfs(root: Path) -> list[Path]:
    return sorted(root.rglob("*.pdf"))


# ── 2. Text extraction ────────────────────────────────────────────────────────

def extract_text(pdf: Path) -> tuple[str, bool]:
    text, ok = _try_pdfplumber(pdf)
    if not text.strip():
        text, ok = _try_pypdf(pdf)
    return text[:_MAX_TEXT_CHARS], bool(text.strip())


def _try_pdfplumber(pdf: Path) -> tuple[str, bool]:
    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(str(pdf)) as doc:
            parts = [p.extract_text() or "" for p in doc.pages[:_MAX_PAGES]]
        text = "\n\n".join(p for p in parts if p.strip())
        return text, bool(text.strip())
    except Exception:
        return "", False


def _try_pypdf(pdf: Path) -> tuple[str, bool]:
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(str(pdf))
        parts = [page.extract_text() or "" for page in reader.pages[:_MAX_PAGES]]
        text = "\n\n".join(p for p in parts if p.strip())
        return text, bool(text.strip())
    except Exception:
        return "", False


# ── 3. Metadata extraction ────────────────────────────────────────────────────

def extract_meta(meta: PaperMeta) -> None:
    _fill_from_pdf_info(meta)
    text = meta.raw_text

    if not meta.title:
        meta.title = _find_title(text)
    if not meta.abstract:
        meta.abstract = _find_abstract(text)
    if meta.first_author_last == "Unknown":
        authors = _find_authors(text)
        if authors:
            meta.first_author_last = _last_name(authors[0])
    if meta.year == "XXXX":
        y = _find_year(text)
        if y:
            meta.year = str(y)
    if not meta.keywords:
        meta.keywords = _find_keywords(text)

    tags = bt.tag(meta.title or "", meta.abstract or "",
                  meta.keywords, meta.raw_text)
    meta.omics        = tags["omics"]
    meta.organism     = tags["organism"]
    meta.article_type = tags["article_type"]
    meta.diseases     = tags["diseases"]
    meta.is_trial     = tags["is_trial"]
    meta.tissues      = tags["tissues"]


def _fill_from_pdf_info(meta: PaperMeta) -> None:
    try:
        from pypdf import PdfReader  # type: ignore
        info = PdfReader(str(meta.path)).metadata
        if not info:
            return
        if info.get("/Title") and not meta.title:
            meta.title = str(info["/Title"]).strip() or None
        if info.get("/Author") and meta.first_author_last == "Unknown":
            parts = re.split(r"[,;]", str(info["/Author"]))
            if parts:
                meta.first_author_last = _last_name(parts[0].strip())
        if info.get("/Keywords") and not meta.keywords:
            meta.keywords = [k.strip() for k in
                             re.split(r"[,;]", str(info["/Keywords"])) if k.strip()]
    except Exception:
        pass


def _find_title(text: str) -> str | None:
    for line in text.strip().split("\n")[:30]:
        line = line.strip()
        if not line or _EMAIL_RE.search(line):
            continue
        words = line.split()
        if 4 <= len(words) <= 20 and not line.endswith(","):
            return line
    return None


def _find_abstract(text: str) -> str | None:
    m = re.search(
        r"(?:abstract|ABSTRACT)[—:\s]*\n?(.*?)"
        r"(?=\n\s*(?:\d\.?\s+\w|\w+\s*\n[-=]{3,}|Keywords?:|$))",
        text, re.DOTALL | re.IGNORECASE,
    )
    if m:
        a = re.sub(r"\s+", " ", m.group(1).strip())
        if len(a) > 50:
            return a[:2000]
    for para in re.split(r"\n\s*\n", text[:3000])[1:6]:
        para = para.strip()
        if len(para) > 200 and len(para.split()) > 30:
            return re.sub(r"\s+", " ", para)[:2000]
    return None


def _find_authors(text: str) -> list[str]:
    candidates: list[str] = []
    for line in text.strip().split("\n")[1:15]:
        line = line.strip()
        if not line or len(line) > 120 or _EMAIL_RE.search(line):
            continue
        if re.search(r"abstract|introduction|university|department|institute",
                     line, re.I):
            break
        words = line.split()
        if (1 < len(words) <= 12 and
                sum(1 for w in words if w[0].isupper()) >= len(words) * 0.5):
            candidates.extend(a.strip() for a in re.split(r"[,;]", line)
                              if a.strip())
            if len(candidates) >= 2:
                break
    return candidates[:10]


def _find_year(text: str) -> int | None:
    m = _YEAR_RE.search(text[:2000])
    return int(m.group(1)) if m else None


def _find_keywords(text: str) -> list[str]:
    m = re.search(r"keywords?[:\s—]+([^\n]{10,200})", text, re.I)
    if m:
        return [k.strip().rstrip(".") for k in
                re.split(r"[,;·•]", m.group(1)) if k.strip()][:15]
    return []


def _last_name(full: str) -> str:
    full = full.strip()
    if "," in full:
        return full.split(",")[0].strip().split()[-1]
    parts = full.split()
    return parts[-1] if parts else "Unknown"


# ── 4. Embedding + clustering ─────────────────────────────────────────────────

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError:
            console.print("[red]Missing:[/] sentence-transformers — "
                          "run [bold]pip install -r requirements.txt[/bold]")
            sys.exit(1)
        console.print(f"[dim]Loading {_EMBED_MODEL}…[/dim]")
        _embedder = SentenceTransformer(_EMBED_MODEL)
    return _embedder


def _paper_corpus(metas: list[PaperMeta]) -> list[str]:
    texts = []
    for m in metas:
        parts = [p for p in [m.abstract, m.title,
                              " ".join(m.keywords)] if p]
        texts.append((" ".join(parts) or m.raw_text)[:_EMBED_TEXT_CHARS])
    return texts


def build_taxonomy(metas: list[PaperMeta],
                   max_cats: int) -> tuple[list[str], np.ndarray]:
    try:
        from sklearn.cluster import AgglomerativeClustering, KMeans  # type: ignore
        from sklearn.feature_extraction.text import TfidfVectorizer   # type: ignore
    except ImportError:
        console.print("[red]Missing:[/] scikit-learn — "
                      "run [bold]pip install -r requirements.txt[/bold]")
        sys.exit(1)

    corpus = _paper_corpus(metas)
    console.print(f"[dim]Embedding {len(corpus)} papers…[/dim]")
    embeddings = _get_embedder().encode(
        corpus, convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False)

    if len(metas) < 2:
        name, _, _ = _name_cluster(corpus, TfidfVectorizer)
        return [name], embeddings

    labels = _cluster(embeddings, max_cats, AgglomerativeClustering, KMeans)
    category_names, centroids = [], []
    for label in sorted(set(labels)):
        mask = np.array(labels) == label
        centroids.append(embeddings[mask].mean(axis=0))
        name, _, _ = _name_cluster(
            [corpus[i] for i, m in enumerate(mask) if m], TfidfVectorizer)
        category_names.append(name)
    return category_names, np.vstack(centroids)


def _cluster(embeddings, max_cats, AgglomerativeClustering, KMeans,
             n_trials=12):
    n = len(embeddings)
    target_max, target_min = min(max_cats, n), max(2, min(3, n))
    lo, hi, best = 0.1, 2.0, list(range(n))
    for _ in range(n_trials):
        mid = (lo + hi) / 2
        labels = AgglomerativeClustering(
            n_clusters=None, distance_threshold=mid,
            linkage="ward", metric="euclidean"
        ).fit_predict(embeddings).tolist()
        k = len(set(labels))
        if k <= target_max:
            best, hi = labels, mid
            if k >= target_min:
                break
        else:
            lo = mid
    if len(set(best)) > target_max:
        best = KMeans(n_clusters=target_max, random_state=42,
                      n_init="auto").fit_predict(embeddings).tolist()
    return best


def _name_cluster(texts, TfidfVectorizer, n_kw=4):
    if not texts:
        return "Uncategorized", [], ""
    try:
        vec = TfidfVectorizer(max_features=300, stop_words="english",
                              ngram_range=(1, 2), min_df=1)
        tfidf = vec.fit_transform(texts)
        names = vec.get_feature_names_out()
        scores = tfidf.mean(axis=0).A1
        keywords, seen = [], set()
        for idx in scores.argsort()[::-1]:
            term = names[idx]
            words = set(term.lower().split())
            if (len(term) > 2 and term.lower() not in _STOP_WORDS
                    and not _ID_RE.search(term)
                    and not words.issubset(seen)):
                keywords.append(term)
                seen.update(words)
            if len(keywords) >= n_kw:
                break
    except Exception:
        keywords = []
    if not keywords:
        return "Uncategorized", [], ""
    return " ".join(w.title() for w in keywords), keywords, ""


# ── 5. Category assignment ────────────────────────────────────────────────────

def assign(meta: PaperMeta, category_names: list[str],
           centroids: np.ndarray) -> None:
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    emb = _get_embedder().encode(
        _paper_corpus([meta]), convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False)
    sims = cosine_similarity(emb, centroids)[0]
    best = int(np.argmax(sims))
    meta.category   = category_names[best] if best < len(category_names) \
                      else category_names[0]
    meta.confidence = float(np.clip(sims[best], 0.0, 1.0))


# ── 6. Rename ─────────────────────────────────────────────────────────────────

_TITLE_STOP = {
    "a","an","the","of","in","on","at","to","for","and","or","with",
    "is","are","from","by","as","via","using","toward","towards","based",
}


def make_filename(meta: PaperMeta, original: Path) -> str:
    author = re.sub(r"[^\w]", "", meta.first_author_last)[:20] or "Unknown"
    year   = meta.year or "XXXX"
    if meta.title:
        words = re.sub(r"[^\w\s]", " ", meta.title).split()
        slug  = "".join(w.capitalize() for w in words
                        if w.lower() not in _TITLE_STOP and len(w) > 1)[:40]
    else:
        slug = re.sub(r"[^\w]", "", original.stem)[:30]
    return re.sub(r"[^\w.\-]", "_", f"{author}{year}_{slug}{original.suffix}")


# ── 7. Move ───────────────────────────────────────────────────────────────────

def move_paper(src: Path, dest_dir: Path, new_name: str,
               dry_run: bool) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = _unique_path(dest_dir, new_name)
    if not dry_run:
        shutil.move(str(src), str(dest))
    return dest


def _unique_path(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem, suffix = Path(filename).stem, Path(filename).suffix
    for i in range(2, 9999):
        candidate = directory / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
    return directory / filename


def _safe_folder(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip(".")[:120] or "Uncategorized"


# ── 8. CSV report ─────────────────────────────────────────────────────────────

def save_report(results: list[PaperMeta], output_dir: Path,
                dry_run: bool) -> Path:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"sort_report_{ts}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["original_path","new_path","category","confidence",
                    "title","first_author","year",
                    "article_type","is_trial","organism",
                    "omics","diseases","tissues"])
        for m in results:
            w.writerow([
                str(m.path),
                str(m.new_path) if m.new_path
                    else ("(dry run)" if dry_run else "(failed)"),
                m.category, f"{m.confidence:.3f}",
                m.title or "", m.first_author_last, m.year,
                m.article_type, "Yes" if m.is_trial else "No", m.organism,
                "; ".join(m.omics),
                "; ".join(m.diseases),
                "; ".join(m.tissues),
            ])
    return path


# ── 9. Terminal output ────────────────────────────────────────────────────────

def print_summary(results: list[PaperMeta], failed: list[Path],
                  dry_run: bool, report_path: Path) -> None:
    cat_confs: dict[str, list[float]] = {}
    for m in results:
        cat_confs.setdefault(m.category, []).append(m.confidence)

    table = Table(title="[bold]Sort Summary[/]", box=box.ROUNDED,
                  header_style="bold magenta", border_style="blue")
    table.add_column("Category", style="cyan", min_width=30)
    table.add_column("Papers", justify="center")
    table.add_column("Confidence", justify="left")

    for cat, confs in sorted(cat_confs.items(), key=lambda x: -len(x[1])):
        avg = sum(confs) / len(confs)
        table.add_row(cat, str(len(confs)), _conf_bar(avg))
    if failed:
        table.add_row("[red]✗ Skipped/failed[/]", str(len(failed)), "—")
    console.print(table)

    mode = "[yellow](dry run — no files moved)[/]" if dry_run else ""
    console.print(
        f"\n[bold green]✓ {len(results)} paper(s) sorted[/]  ·  "
        f"[red]✗ {len(failed)} skipped[/]  {mode}\n"
        f"[dim]Report → {report_path}[/dim]"
    )


def _conf_bar(v: float) -> str:
    filled = round(v * 10)
    bar    = "█" * filled + "░" * (10 - filled)
    color  = "green" if v >= 0.7 else "yellow" if v >= 0.4 else "red"
    return f"[{color}]{bar}[/]  {v:.2f}"


def make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(), TaskProgressColumn(), TimeRemainingColumn(),
        console=console,
    )


# ── 10. Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sort_papers.py",
        description="Sort science PDFs into topic folders (no API key needed).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input_dir", type=Path,
                        help="Folder containing PDFs (searched recursively)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Destination folder (default: same as input_dir)")
    parser.add_argument("--categories", "-c", type=int, default=15,
                        help="Max topic categories (default: 15)")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Preview only — no files moved")
    args = parser.parse_args()

    input_dir  = args.input_dir.resolve()
    output_dir = (args.output or args.input_dir).resolve()
    max_cats   = max(2, min(args.categories, 50))
    dry_run    = args.dry_run

    if not input_dir.is_dir():
        console.print(f"[red]Error:[/] '{input_dir}' is not a directory.")
        sys.exit(1)

    pdfs = find_pdfs(input_dir)
    if not pdfs:
        console.print(f"[yellow]No PDFs found in {input_dir}[/]")
        sys.exit(0)

    mode_tag = "[bold yellow]DRY RUN[/]" if dry_run else "[bold green]LIVE[/]"
    console.print(
        f"\n[bold]Paper Sorter[/] — {mode_tag}\n"
        f"  Input:  [cyan]{input_dir}[/]\n"
        f"  Output: [cyan]{output_dir}[/]\n"
        f"  Found:  [cyan]{len(pdfs)}[/cyan] PDF(s)\n"
    )

    metas: list[PaperMeta] = []
    failed: list[Path]     = []

    with make_progress() as prog:
        task = prog.add_task("Extracting text & metadata…", total=len(pdfs))
        for pdf in pdfs:
            text, ok = extract_text(pdf)
            m = PaperMeta(path=pdf, raw_text=text, extraction_ok=ok)
            if ok and len(text.strip()) >= _MIN_TEXT_CHARS:
                extract_meta(m)
                metas.append(m)
            else:
                failed.append(pdf)
                msg = ("too short, likely not a paper"
                       if ok else "could not extract text")
                console.print(f"  [yellow]⚠ Skipped ({msg}):[/] {pdf.name}")
            prog.advance(task)

    if not metas:
        console.print("[red]No papers could be processed.[/]")
        sys.exit(1)

    console.print(f"\n[dim]Clustering {len(metas)} papers into up to "
                  f"{max_cats} topics…[/dim]")
    category_names, centroids = build_taxonomy(metas, max_cats)

    console.print(f"\n[bold]Discovered {len(category_names)} topic(s):[/]")
    for name in category_names:
        console.print(f"  [cyan]·[/] {name}")

    with make_progress() as prog:
        task = prog.add_task("Sorting papers…", total=len(metas))
        for m in metas:
            assign(m, category_names, centroids)
            new_name = make_filename(m, m.path)
            dest_dir = output_dir / _safe_folder(m.category)
            m.new_path = move_paper(m.path, dest_dir, new_name, dry_run)
            prog.advance(task)

    report_path = save_report(metas, output_dir, dry_run)
    print_summary(metas, failed, dry_run, report_path)


if __name__ == "__main__":
    main()
