#!/usr/bin/env python3
"""Build the local reading site and the complete Markdown translation (stdlib only)."""
from pathlib import Path
import html
import json
import re

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "webapp"
SOURCE = "https://christianjmills.com/posts/cuda-mode-notes/lecture-007/"
CHAPTERS = [('ch00', '导论', 'GPU MODE 第 7 讲：进阶量化', 'Introduction', '讲者背景、全文目录与原始演讲资源。', ['contents', 'introduction']),
 ('ch01', '第 1 章', '量化概览', 'Overview of Quantization', '动态量化、仅权重量化、量化粒度与精度取舍。', ['overview-of-quantization']),
 ('ch02',
  '第 2 章',
  '动态量化内核',
  'Dynamic Quantization Kernels',
  '缩放因子、INT32 中间结果与 Triton 内核融合。',
  ['dynamic-quantization-kernels']),
 ('ch03',
  '第 3 章',
  '仅权重量化内核',
  'Weight-only Quantization Kernels',
  '从朴素矩阵乘法到逐元素乘法加归约。',
  ['weight-only-quantization-kernels']),
 ('ch04',
  '第 4 章',
  'INT4 仅权重量化内核',
  'INT4 Weight-only Quantization Kernels',
  '低位宽数据的打包、解包，以及 CUDA 与 Triton 的对比。',
  ['int4-weight-only-quantization-kernels']),
 ('ch05',
  '第 5 章',
  'Triton 的优势与局限',
  'Strengths and Limitations of Triton',
  '易用性、融合、数据类型与内核配置的边界。',
  ['strengths-and-limitations-of-triton']),
 ('ch06',
  '第 6 章',
  '问答环节',
  'Q&A Session',
  '15 个专题：量化生态、硬件、累加、GPTQ 与推测解码。',
  ['qa-session',
   'motivation-and-comparison-with-other-libraries',
   'reasons-for-developing-native-pytorch-quantization',
   'comparison-with-bitsandbytes',
   'comparison-with-other-quantization-libraries',
   'hardware-support-and-integration',
   'nvidia-kernels-and-data-types',
   'amd-support',
   'mac-support',
   'triton-usage-and-development',
   'triton-installation',
   'tritons-openness-and-future',
   'technical-deep-dive-into-quantization',
   'accumulation-data-type-selection',
   'one-bit-quantization',
   'weight-distribution-analysis',
   'quantization-for-speculative-decoding',
   'future-directions-and-insights',
   'scaling-vs.-optimization',
   'layer-specific-optimization',
   'accuracy-metrics']),
 ('references',
  '附录',
  '作者介绍、术语与来源',
  'Author, Terminology & Sources',
  '原网页作者附记、术语对照及本地底本记录。',
  ['about-author', 'terminology', 'source-and-scope', 'translation-notes', 'attribution-and-files'])]


def anchored(md, ids):
    headings = re.findall(r"^#{2,4} .+$", md, re.M)
    if len(headings) != len(ids):
        raise ValueError(f"Heading count mismatch: {len(headings)} != {len(ids)}")
    anchors = iter(ids)
    return re.sub(r"^#{2,4} .+$", lambda m: f'<a id="{next(anchors)}"></a>\n\n{m[0]}', md, flags=re.M)


def page(title, md, nav, prefix):
    chapter_links = "".join(
        f'<li><a href="{c[0]}.html">{c[1]} · {c[2]}</a></li>' for c in CHAPTERS
    )
    return f'''<!DOCTYPE html>
<html lang="zh-Hans">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} — 进阶量化 · GPU MODE</title>
<link rel="icon" href="data:,">
<link rel="stylesheet" href="{prefix}assets/style.css"></head>
<body>
<header class="topbar"><a class="brand" href="{prefix}index.html">进阶量化 · GPU MODE<span>Triton 与 CUDA</span></a>
<nav class="chapnav"><a href="{prefix}all.html">全文阅读</a><a href="{SOURCE}">英文原文 ↗</a></nav></header>
<div class="layout"><aside class="sidebar"><nav class="toc" id="toc"><h2>章节导航</h2>
<ol class="book-toc">{chapter_links if prefix else chapter_links.replace('href="', 'href="chapters/')}</ol><h2>本页目录</h2></nav></aside>
<main class="main"><article id="content" class="prose"></article>
<noscript>此页需要 JavaScript 排版。也可直接阅读目录中的完整译文 Markdown 文件。</noscript>
<nav class="pager">{nav}</nav>
<footer class="pagefoot">原作者 Christian Mills · 2024-09-10 · 非官方中文译本 ·
依据用户提供的 HTML 制作 · 保留原文署名与权利说明</footer></main></div>
<script type="application/json" id="chapter-data">{json.dumps(md, ensure_ascii=False).replace('<', chr(92) + 'u003c')}</script>
<script src="{prefix}assets/marked.min.js"></script><script src="{prefix}assets/app.js"></script>
</body></html>
'''


def main():
    (WEB / "chapters").mkdir(parents=True, exist_ok=True)
    translations = []
    for i, c in enumerate(CHAPTERS):
        name, number, title, _, _, ids = c
        md = anchored((ROOT / "content" / f"{name}.md").read_text(), ids)
        translations.append(md)
        links = []
        for offset, label, cls in [(-1, "← 上一章", "prev"), (1, "下一章 →", "next")]:
            j = i + offset
            if 0 <= j < len(CHAPTERS):
                other = CHAPTERS[j]
                links.append(f'<a class="{cls}" href="{other[0]}.html"><span class="dir">{label}</span><div class="ttl">{other[1]}　{other[2]}</div></a>')
        rendered = md.replace("../webapp/assets/", "../assets/")
        (WEB / "chapters" / f"{name}.html").write_text(page(f"{number}　{title}", rendered, "".join(links), "../"))

    combined = "\n\n---\n\n".join(translations)
    combined = re.sub(r"\((?:ch\d+|references)\.html#", "(#", combined)
    (ROOT / "advanced_quantization_zh.md").write_text(combined.replace("../webapp/assets/", "webapp/assets/"))
    (WEB / "all.html").write_text(page("完整译文", combined.replace("../webapp/assets/", "assets/"), "", ""))

    cards = "\n".join(f'''<a class="card" href="chapters/{c[0]}.html"><span class="no">{c[1]}</span><span class="zh">{c[2]}</span><span class="en">{c[3]}</span><span class="desc">{c[4]}</span></a>''' for c in CHAPTERS)
    (WEB / "index.html").write_text(f'''<!DOCTYPE html>
<html lang="zh-Hans"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>进阶量化 · GPU MODE — Triton 与 CUDA</title><link rel="icon" href="data:,">
<link rel="stylesheet" href="assets/style.css"></head>
<body><section class="hero"><div class="eyebrow">简体中文全译本 · 本地阅读版</div>
<h1>GPU MODE 第 7 讲<br>进阶量化</h1>
<p class="sub">Advanced Quantization<br>Christian Mills 讲座笔记 · 讲者 Charles Hernandez · 2024 年 9 月 10 日</p>
<p class="sub">从动态量化与仅权重量化，到 INT4 打包、内核融合，以及 Triton 和 CUDA 的性能取舍。完整保留问答环节的 15 个专题。</p>
<p class="reading-actions"><a class="primary-link" href="all.html">阅读完整译文 →</a><a href="chapters/ch06.html#qa-session">直达问答环节 →</a></p>
<div class="meta"><div><b>7</b>主要章节</div><div><b>15</b>问答专题</div><div><b>3</b>原始插图</div><div><b>27</b>原文标题</div></div></section>
<div class="grid-wrap"><h2>章节与参考资料</h2><div class="chapter-grid">{cards}</div></div>
<footer class="foot">非官方中文译本 · 原作者 <a href="{SOURCE}">Christian Mills</a> · 基于用户提供的 HTML · 本地中文阅读版<br>阅读界面沿用 Ultra-Scale Playbook 中文版的布局与样式。性能结论保留 2024 年语境。</footer>
</body></html>''')
    print(f"Built {len(CHAPTERS)} chapter/resource pages, index.html, all.html and advanced_quantization_zh.md")


if __name__ == "__main__":
    main()
