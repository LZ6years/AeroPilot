"""Portable human review export, preserving original images and place archives.

This reads generated documents only; it never rebuilds historical model evidence.
"""
import base64
import html
from io import BytesIO
import mimetypes
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from PIL import Image


def _embed_document_images(text: str, directory: Path) -> tuple[str, int]:
    """Resolve images against their own document before embedding that document."""
    refs = re.findall(r'<img\b[^>]*\bsrc="([^"]+)"', text)
    replacements = {}
    for ref in dict.fromkeys(refs):
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


def embed_images(source: Path) -> tuple[str, int]:
    """Export process images and the complete task-local place archive inline.

    A detached HTML file has no base directory for the old archive iframe.
    Read that archive at its original location, embed its photos, scope its CSS
    and anchors, then expose it with native details/summary (no script or server).
    Source documents and graph snapshots are never rewritten or regenerated.
    """
    text = source.read_text(encoding="utf-8")
    extra_images = 0

    def archive_section(match):
        nonlocal extra_images
        iframe = re.search(r'<iframe\b[^>]*\bsrc="([^"]+)"', match[0])
        if iframe is None:
            raise ValueError("place archive section has no local archive source")
        ref = html.unescape(iframe[1])
        if urlsplit(ref).scheme:
            raise ValueError("expected a local place archive")
        archive = (source.parent / unquote(ref)).resolve()
        body, count = _embed_document_images(archive.read_text(encoding="utf-8"), archive.parent)
        extra_images += count
        scope = ".portable-place-archive"

        def scoped_style(style):
            def rule(rule_match):
                selectors = [scope if item.strip() == "body" else scope + " " + item.strip()
                             for item in rule_match[1].split(",")]
                return ",".join(selectors) + "{"
            return "<style>" + re.sub(r"([^{}]+)\{", rule, style[1]) + "</style>"

        body = re.sub(r"<style>(.*?)</style>", scoped_style, body, flags=re.S)
        body = re.sub(r'\bid="([^"]+)"', lambda m: 'id="place-archive-' + m[1] + '"', body)
        body = re.sub(r'\bhref="#([^"]+)"', lambda m: 'href="#place-archive-' + m[1] + '"', body)
        body = re.sub(r"<!doctype[^>]*>|</?html[^>]*>|</?head[^>]*>|</?body[^>]*>|<meta[^>]*>|<title>.*?</title>", "", body, flags=re.I | re.S)
        return ('<details id="place-memory"><summary>地点拓扑与存档：展开完整地图、照片和动作</summary>'
                '<div class="portable-place-archive">' + body + '</div></details>')

    text = re.sub(r'<section\b[^>]*\bid="place-memory"[^>]*>.*?</section>', archive_section, text, flags=re.S)
    # Archive images are already data URLs; embed only the original process
    # portion to avoid decoding or replacing the same large payload repeatedly.
    archive_match = re.search(r'<details id="place-memory">.*?</div></details>', text, flags=re.S)
    archive_text = archive_match[0] if archive_match else ""
    if archive_match:
        text = text[:archive_match.start()] + "<!--portable-place-archive-->" + text[archive_match.end():]
    text, count = _embed_document_images(text, source.parent)
    text = text.replace("<!--portable-place-archive-->", archive_text)
    # Historical appenders sometimes placed the section after </html>.
    if "</html>" in text:
        text = text.replace("</html>", "") + "\n</html>"
    return text, count + extra_images


