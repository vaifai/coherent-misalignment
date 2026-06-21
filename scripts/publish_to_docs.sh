#!/usr/bin/env bash
# Sync workingDocs/REPORT_DRAFT.md to docs/index.md (the GitHub Pages source)
# and regenerate the DOCX export.
#
# Usage: bash scripts/publish_to_docs.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO_ROOT/workingDocs/REPORT_DRAFT.md"
DEST="$REPO_ROOT/docs/index.md"
DOCX="$REPO_ROOT/workingDocs/REPORT_DRAFT.docx"

if [[ ! -f "$SRC" ]]; then
  echo "Source not found: $SRC" >&2
  exit 1
fi

# Sync figures
mkdir -p "$REPO_ROOT/docs/figures"
cp "$REPO_ROOT/figures/"*.png "$REPO_ROOT/docs/figures/"

# Generate index.md: prepend Jekyll front matter, rewrite ../figures/ -> figures/
{
  printf -- "---\nlayout: default\ntitle: \"Behavior Grounded Honesty Specs Eliminate the Inverted Persona Effect on Forced Choice Probes\"\n---\n\n"
  sed 's|\.\./figures/|figures/|g' "$SRC"
} > "$DEST"

echo "Synced: $SRC -> $DEST"

# Regenerate DOCX if pandoc is available
if command -v pandoc >/dev/null 2>&1; then
  cd "$REPO_ROOT/workingDocs"
  pandoc REPORT_DRAFT.md \
    --from gfm+footnotes \
    --to docx \
    --resource-path=.:..:../figures \
    --toc \
    --toc-depth=2 \
    -o REPORT_DRAFT.docx
  echo "DOCX:    $DOCX"
else
  echo "pandoc not installed; skipped DOCX regen. Install with: brew install pandoc"
fi
