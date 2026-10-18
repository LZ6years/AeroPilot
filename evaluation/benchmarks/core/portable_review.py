"""Portable human review export, preserving original images and place archives.

This reads generated documents only; it never rebuilds historical model evidence.
"""
import base64
import html
from io import BytesIO
import mimetypes
import os
from pathlib import Path
import re
from urllib.parse import quote, unquote, urlsplit

from PIL import Image


def _embed_document_images(text: str, directory: Path) -> tuple[str, int]:
    """Resolve images against their own document before embedding that document."""
    refs = re.findall(r'<img\b[^>]*\bsrc="([^"]+)"', text)
    replacements = {}
    for ref in dict.fromkeys(refs):
        if ref.startswith('data:image/'):
            continue
        if urlsplit(ref).scheme:
            raise ValueError(f"Expected local report image: {ref[:120]}")
        path = (directory / unquote(html.unescape(ref))).resolve()
        payload = path.read_bytes()
        with Image.open(BytesIO(payload)) as image:
            image.verify()
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        replacements[ref] = f"data:{mime};base64," + base64.b64encode(payload).decode("ascii")
    # Replace image source and corresponding click-through URL together.
    text = re.sub(r'\b(src|href)="([^"]+)"',
                  lambda match: f'{match[1]}="{replacements.get(match[2], match[2])}"', text)
    return text.replace('loading="lazy"', 'loading="eager"'), len(refs)


def place_archive_section(archive: Path, *, review_directory: Path | None = None) -> str:
    """Scope archive anchors/styles; embed exports or rebase live review links."""
    body = archive.read_text(encoding='utf-8')
    if review_directory is None:
        body, _ = _embed_document_images(body, archive.parent)
    else:
        def rebase(match):
            ref = html.unescape(match[2])
            if ref.startswith('#') or urlsplit(ref).scheme:
                return match[0]
            target = (archive.parent / unquote(ref)).resolve()
            relative = Path(os.path.relpath(target, review_directory)).as_posix()
            return f'{match[1]}="{html.escape(quote(relative, safe="/."), quote=True)}"'
        body = re.sub(r'\b(src|href)="([^"]+)"', rebase, body)

    def scoped_style(style):
        def rule(match):
            selectors = ['.portable-place-archive' if item.strip() == 'body'
                         else '.portable-place-archive ' + item.strip()
                         for item in match[1].split(',')]
            return ','.join(selectors) + '{'
        return '<style>' + re.sub(r'([^{}]+)\{', rule, style[1]) + '</style>'
    body = re.sub(r'<style>(.*?)</style>', scoped_style, body, flags=re.S)
    body = re.sub(r'\bid="([^"]+)"', lambda m: 'id="place-archive-' + m[1] + '"', body)
    body = re.sub(r'\bhref="#([^"]+)"', lambda m: 'href="#place-archive-' + m[1] + '"', body)
    body = re.sub(r'<!doctype[^>]*>|</?html[^>]*>|</?head[^>]*>|</?body[^>]*>|<meta[^>]*>|<title>.*?</title>', '', body, flags=re.I | re.S)
    return ('<details id="place-memory"><summary>地点拓扑与存档：展开完整地图、照片和动作</summary>'
            '<div class="portable-place-archive">' + body + '</div></details>')


def attach_place_archive(page: Path, run_directory: Path) -> None:
    """Attach a task's existing archive without iframe or duplicate media files."""
    from evaluation.benchmarks.core.report_io import atomic_write_text
    archives = list((run_directory / 'contexts/place_memory').glob('*/index.html'))
    if not archives:
        return
    if len(archives) != 1:
        raise ValueError('Expected one task-local place archive')
    text = page.read_text(encoding='utf-8')
    if 'id="place-memory"' in text:
        return
    section = place_archive_section(archives[0], review_directory=page.parent)
    marker = '</body>' if '</body>' in text else '</html>'
    atomic_write_text(page, text.replace(marker, section + marker) if marker in text else text + section)


def embed_images(source: Path) -> tuple[str, int]:
    """Export process images and the complete task-local place archive inline.

    A detached HTML file has no base directory for the old archive iframe.
    Read that archive at its original location, embed its photos, scope its CSS
    and anchors, then expose it with native details/summary (no script or server).
    Source documents and graph snapshots are never rewritten or regenerated.
    """
    text = source.read_text(encoding="utf-8")

    def archive_section(match):
        iframe = re.search(r'<iframe\b[^>]*\bsrc="([^"]+)"', match[0])
        if iframe is None:
            raise ValueError("place archive section has no local archive source")
        ref = html.unescape(iframe[1])
        if urlsplit(ref).scheme:
            raise ValueError("expected a local place archive")
        archive = (source.parent / unquote(ref)).resolve()
        return place_archive_section(archive)

    text = re.sub(r'<section\b[^>]*\bid="place-memory"[^>]*>.*?</section>', archive_section, text, flags=re.S)
    text, count = _embed_document_images(text, source.parent)
    # Historical appenders sometimes placed the section after </html>.
    if "</html>" in text:
        text = text.replace("</html>", "") + "\n</html>"
    return text, count
