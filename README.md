# gt-rag — course knowledge base as an MCP server

Turn a folder of markdown course notes into a searchable, self-healing
knowledge base that any MCP-capable agent (Claude Code, Claude Desktop, ...)
can query. Built for a Russian-language Smart Money / ICT trading course, but
the pipeline is content-agnostic: bring your own markdown.

- **Hybrid search** — ChromaDB vector search (local multilingual embeddings,
  no GPU/torch/API needed) + rapidfuzz keyword layer. Russian and English
  queries, typo-tolerant.
- **MCP server** — 8 tools over stdio, with built-in usage instructions the
  agent receives automatically.
- **Self-diagnosis** — `gt_health` / `gt_doctor` let the agent detect and fix
  a stale or broken index on its own.
- **Image-aware** — chart screenshots live next to the text with captions and
  vision descriptions indexed, so search matches chart *content* and the agent
  can open the actual image.
- **Transcript pipeline** — optional `transcribe.py` turns Google Drive video
  recordings into timestamped, searchable markdown via faster-whisper.

> **Note:** this repo ships the tooling only. `data/` (course texts, images,
> transcripts) is not included — the original content is copyrighted course
> material. See [Data format](#data-format) to populate your own.

## Layout

```
gt-rag/
  data/raw/*.md         # YOUR content: markdown + YAML frontmatter (gitignored)
  data/images/<slug>/   # optional archived images referenced from pages
  data/chroma/          # persistent ChromaDB store (created by ingest)
  data/index_state.json # file hashes for incremental ingest
  gt_rag/
    common.py           # config, frontmatter, chunking (1800 chars, 200 overlap)
    store.py            # ChromaDB collection + fastembed embedding function
    search.py           # hybrid vector + fuzzy ranking
    indexer.py          # reusable ingest core (used by CLI and server)
    doctor.py           # health checks + safe self-repair
  ingest.py             # CLI: (re)index data/raw into ChromaDB
  query.py              # CLI: search check
  server.py             # MCP stdio server
  transcribe.py         # optional: video -> timestamped transcript markdown
```

## Quickstart

```powershell
git clone https://github.com/Miha21222/gt-rag
cd gt-rag
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# add your content to data/raw/*.md (see Data format), then:
.venv\Scripts\python ingest.py       # first run downloads the embedding model (~470 MB)
.venv\Scripts\python query.py "что такое FVG и почему цена его заполняет"
```

Register the MCP server with Claude Code:

```powershell
claude mcp add gt-database -- "<abs-path>\gt-rag\.venv\Scripts\python.exe" "<abs-path>\gt-rag\server.py"
```

(Linux/macOS: same commands with `.venv/bin/python`.)

## Embedding in another application

The server is meant to be shippable: an app can carry the code, the content and
a prebuilt index and hand each of them a different directory.

| Variable | What it moves | Written to? |
|---|---|---|
| `GT_RAG_DATA` | the whole data directory (default `<repo>/data`) | yes |
| `GT_RAG_RAW` | the source markdown only | no |
| `GT_RAG_IMAGES` | the image archive only | no |

The split exists because **ChromaDB opens its sqlite store read-write even to
answer a query**, and an incremental re-index rewrites `index_state.json`: the
store must live somewhere writable, while `raw/` and `images/` can stay in a
read-only installation directory. `manifest.json` and `style-overrides.json` are
read next to the code and need no variable.

Set `PYTHONIOENCODING=utf-8` when you spawn the server on Windows. The content is
Russian and the tools answer in JSON on stdout; a console default of cp1251 turns
the first «→» in a hit into a `UnicodeEncodeError` mid-answer.

`server.py` runs on either MCP backend: it uses **fastmcp** when the environment
has one and falls back to `mcp>=2` (what `requirements.txt` installs) otherwise.
Both expose the same eight tools. This is what lets a host that already runs a
fastmcp server — which pins `mcp<2` — put gt-rag in the same environment instead
of building a second one for it.

## Data format

One markdown file per page in `data/raw/`, YAML frontmatter required:

```markdown
---
title: "Dealing Range | Premium & Discount"
notion_url: "https://..."        # optional source link, any URL or ""
block: 3                          # optional module/section number, or omit
type: "lesson"                    # lesson | guide | conference | qa |
                                  # homework | terminology | info |
                                  # transcript | your own
trading_style: "Для интрадей"     # Для свинга | Для интрадей |
                                  # Для интрадей и свинга
course: "Cryptology Flow"         # optional source/course grouping
course_level: "Junior"            # optional package/module grouping
market: "general"                 # optional: general | crypto | forex
slug: "b3-dealing-range"          # must match the filename stem
crawled_at: "2026-08-27"          # optional
---

### Section heading

Body text. Chunking splits on ##/### headings, then windows long sections.
```

Optional image convention (what `gt_image_path` and the health checks
understand): store files as `data/images/<slug>/img-NN.png` and reference them
from the page as:

```markdown
![img-01](../images/b3-dealing-range/img-01.png)

[Изображение: original caption]

[Описание графика: vision-model description of what the chart shows]
```

Caption and description are plain text inside the chunk, so retrieval matches
chart content; the link lets an agent open the pixels.

How the original dataset was produced (blueprint, all via Claude Code):
Notion pages crawled to markdown by agents; conference videos transcribed
with `transcribe.py`; images downloaded within Notion's ~5-minute signed-URL
window and described by a vision pass.

## MCP tools

Content:

| Tool | Purpose |
|---|---|
| `gt_search(query, top_k, block, type, trading_style)` | hybrid search; returns scored chunks with slug, section, text |
| `gt_get_page(slug)` | full markdown of one page |
| `gt_list_pages()` | inventory: title, block, type, slug, indexed status |
| `gt_image_path(link)` | resolve a markdown image link to an absolute path |

Filtering by `Для свинга` or `Для интрадей` also includes audited shared
sections tagged `Для интрадей и свинга`. Section-level cross-style decisions
and their semantic evidence are stored in the local `style-overrides.json`
(start from `style-overrides.example.json`). Changes to that file automatically
make only affected pages stale for incremental ingest. Corpus-specific
overrides and audit reports are gitignored because they can contain course text.

Maintenance:

| Tool | Purpose |
|---|---|
| `gt_stats()` | chunk counts by type and block — quick "is it populated" check |
| `gt_health()` | read-only diagnosis: raw files, collection, index freshness, images, search smoke test |
| `gt_doctor(fix=False)` | diagnose; `fix=True` applies safe repairs (incremental re-index, orphan pruning) |
| `gt_reindex(rebuild=False)` | explicit re-index; `rebuild=True` wipes and re-embeds everything |

## Agent guide

The server sends these usage instructions to the agent automatically; summary:

1. `gt_search` first for anything conceptual. Filter `type="transcript"` for
   "what was said/asked live", plain lessons are the canon. Filter `block=N`
   to stay inside one module.
2. `gt_get_page(slug)` when a hit needs full context.
3. Image link in a hit → `gt_image_path(link)` → open the returned absolute
   path with file/vision tools to see the actual chart.
4. Anything misbehaves → `gt_health()` → `gt_doctor(fix=True)` →
   `gt_reindex(rebuild=True)` as last resort.

Typical self-repair loop after content edits:

```
gt_health()                # index_freshness: stale_pages: [b3-dealing-range]
gt_doctor(fix=True)        # re-indexed 1 page; status ok
gt_search("...")           # fresh results
```

## Updating content

Edit or add files in `data/raw/`, then either run `ingest.py` (incremental,
hash-based) or just ask the agent — `gt_doctor(fix=True)` does the same from
inside the MCP session. `ingest.py --only <slug>` forces one page;
`--rebuild` re-embeds everything (needed only after chunking/embedding
changes).

## Transcription pipeline (optional)

`transcribe.py` scans `data/raw` for pages of type `conference`/`qa`
containing a Google Drive link, then per recording: download → transcribe
(faster-whisper large-v3, CUDA if available, int8 fallback on CPU) → write
`<slug>-transcript.md` with `### [hh:mm:ss]` paragraph timestamps → delete
the video. Resumable: skips sources whose transcript exists.

```powershell
.venv\Scripts\pip install faster-whisper gdown nvidia-cublas-cu12 nvidia-cudnn-cu12
.venv\Scripts\python transcribe.py                # all pending
.venv\Scripts\python transcribe.py --only <slug> --model medium --device cpu
```

Hard-won details already handled in the script: downloads run in a killable
subprocess with timeout/retries (gdown has no socket timeout), UTF-8 is
forced on Windows (cp1251 mangles non-ASCII filenames), and
`SetThreadExecutionState` keeps the laptop awake for multi-hour runs.

## Troubleshooting

| Symptom | Fix |
|---|---|
| search returns nothing | `gt_stats()` — if 0 chunks, run `ingest.py` / `gt_doctor(fix=True)` |
| edited a page, search shows old text | `gt_doctor(fix=True)` or `ingest.py` |
| ChromaDB fails to open / schema errors | `gt_reindex(rebuild=True)` (or delete `data/chroma/` and re-ingest) |
| first ingest very slow | one-time embedding model download (~470 MB) |
| broken image links in `gt_health` | source page needs a re-fetch of its images; search still works via captions |
| server won't start | run `.venv\Scripts\python -c "import server"` to see the import error; requires `mcp>=2` (`mcp.server.mcpserver`) |

## License

MIT (code). Course content is not part of this repository.
