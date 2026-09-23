# 透明背景编辑

使用内置 image_gen；原图保留，输出 history_context_projection_transparent.png。只请求移除外部白色画布，保留模块内部底色。

```text
Use case: background-extraction. Remove ONLY the outer white canvas background from this exact scientific figure, make it genuinely TRANSPARENT with PNG alpha (not white and not a painted checkerboard). Keep every foreground element unchanged and complete: all black text including title and footer, formulas, thin outlines, arrows, plot axes, curve and points, pastel colored panels and their interior fills including the white chart panel. Preserve original positions, wording, proportions and colors. Do not redraw, reinterpret, crop or omit anything. Smooth clean antialiased cutout edges, no white fringe. The spaces outside/between panels and behind free-floating title/footer text must have alpha zero, while the filled module cards remain intact.
```
