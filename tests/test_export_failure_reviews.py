"""Offline review images must remain usable after separating HTML from assets."""
import base64
import re

from PIL import Image
import pytest

from evaluation.benchmarks.core.portable_review import embed_images, attach_place_archive


def test_embeds_encoded_paths_and_click_links_without_changing_source(tmp_path):
    image = tmp_path / "sample image.png"
    Image.new("RGB", (3, 2), "blue").save(image)
    source = tmp_path / "review.html"
    original = '<a href="sample%20image.png"><img loading="lazy" src="sample%20image.png"></a>'
    source.write_text(original, encoding="utf-8")
    text, count = embed_images(source)
    data = re.search(r'src="data:image/png;base64,([^"]+)"', text)[1]
    assert base64.b64decode(data) == image.read_bytes()
    assert 'href="data:image/png;base64,' in text
    assert 'loading="eager"' in text
    assert source.read_text(encoding="utf-8") == original
    assert count == 1


def test_missing_image_is_not_silently_exported(tmp_path):
    source = tmp_path / "review.html"
    source.write_text('<img src="missing.png">', encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        embed_images(source)


def test_place_archive_is_embedded_with_scoped_anchors_and_no_external_iframe(tmp_path):
    archive = tmp_path / "run/places"
    archive.mkdir(parents=True)
    Image.new("RGB", (8, 6), "green").save(archive / "map.png")
    Image.new("RGB", (4, 4), "red").save(archive / "photo.png")
    original_archive = ('<!doctype html><meta charset="utf-8"><title>map</title>'
        '<style>body{color:red}.map>img{width:100%}article img,pre{margin:2px}</style>'
        '<div class="map"><img src="map.png"><a href="#P001">地点</a></div>'
        '<article id="P001"><img src="photo.png"><details><summary>实际执行动作</summary><pre>move</pre></details></article>')
    (archive / "index.html").write_text(original_archive, encoding="utf-8")
    source = tmp_path / "review.html"
    original = ('<html><h1>过程</h1></html><section id="place-memory"><h2>地点拓扑与存档</h2>'
        '<a href="run/places/index.html">打开</a><iframe src="run/places/index.html"></iframe></section>')
    source.write_text(original, encoding="utf-8")
    exported, count = embed_images(source)
    assert count == 2 and exported.count('src="data:image/png;base64,') == 2
    assert "<iframe" not in exported and "run/places/index.html" not in exported
    assert 'href="#place-archive-P001"' in exported and 'id="place-archive-P001"' in exported
    assert '.portable-place-archive article img,.portable-place-archive pre' in exported
    assert 'id="place-memory"' in exported and '实际执行动作' in exported
    assert exported.index('id="place-memory"') < exported.index('</html>')
    assert source.read_text(encoding="utf-8") == original
    assert (archive / "index.html").read_text(encoding="utf-8") == original_archive


def test_live_archive_uses_rebased_assets_and_remains_portable(tmp_path):
    run = tmp_path / 'runs/task'
    archive = run / 'contexts/place_memory/instruction'
    archive.mkdir(parents=True)
    Image.new('RGB', (3, 2), 'blue').save(archive / 'map.png')
    (archive / 'index.html').write_text('<img src="map.png"><a href="#P1">place</a><article id="P1">move</article>')
    page = tmp_path / 'evaluation/cases/task/review.html'
    page.parent.mkdir(parents=True)
    page.write_text('<html><body>process</body></html>')
    attach_place_archive(page, run)
    original = page.read_text(encoding='utf-8')
    assert '<iframe' not in original and 'data:image' not in original
    assert 'href="#place-archive-P1"' in original
    attach_place_archive(page, run)
    assert page.read_text(encoding='utf-8') == original
    exported, count = embed_images(page)
    assert count == 1 and 'src="data:image/png;base64,' in exported
