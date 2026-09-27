# 批量测试的 HTML 审阅要求

默认交付可查看过程图片的时间线和完整审计页。文件存在不等于浏览器能加载：
Windows Chrome、Edge 的本地文件读取在完整路径达到 260 字符时可能失败。
审阅导出将目录缩短为案例编号，并将全部资源路径限制在 240 字符以内。
页面保留真实案例 ID、实际请求图像、工具记录和原始审计数据。

## 本轮记录离线导出

```powershell
python tests/manual/export_benchmark_review.py --batch-root tests/manual_outputs/benchmark_20_flash_20260923_v2 --output-dir tests/manual_outputs/reviews/flash_20260923
python tests/manual/review_browser_probe.py --review-root tests/manual_outputs/reviews/flash_20260923
```

打开导出目录的 `index.html`。每例提供 `review.html` 和 `audit_review.html`，
原失败尝试单独保留并标记“无最终评分”。复制或归档时应保留整个导出目录。
离线导出不运行模拟器、不调用模型、不覆盖原始实验文件。

## 下一次运行默认行为

`benchmark_20_case_probe.py run` 自动在批次旁的 `reviews/<批次名>` 生成短路径审阅包，
也可通过 `--review-root` 指定更短的独立目录。每例结束后校验图片、本地链接及路径长度。
导出失败会记录 `review_failure.json` 并暂停调度，已完成的导航评分仍然保留。
修复导出问题后续跑只修复审阅产物，已完成案例不会为了补 HTML 而重飞。

浏览器探针对 AerialVLN、TravelUAV 的代表案例分别检查简版及完整审计版：
让图片实际解码、滚动到决策图片、记录加载失败数，并保存截图。
本地文件检查通过后，还必须通过浏览器检查才能宣称过程图片可显示。
