# GPU MODE 第 7 讲：进阶量化 — 简体中文全译本

> 原作者 Christian Mills，讲者 Charles Hernandez，原文发表于 2024 年 9 月 10 日。
> [原文网页](https://christianjmills.com/posts/cuda-mode-notes/lecture-007/) · 本译本以用户提供的完整 HTML 文件为底本。

沿用相邻 `slaying-ooms-zh-cn` 和 `ultrascale-playbook-zh-tw` 的格式：逐章 Markdown、静态阅读页、侧栏目录、前后章导航、原图及双语标题。

## 阅读

- [完整 Markdown 译文](advanced_quantization_zh.md)
- [阅读首页](webapp/index.html)
- [全文 HTML](webapp/all.html)
- [问答环节](webapp/chapters/ch06.html#qa-session)

可以直接用浏览器打开 `webapp/index.html`，也可以启动本地服务：

```bash
bash run.sh
```

访问 <http://127.0.0.1:8645>。使用 `bash run.sh 8650` 可更换端口，默认仅监听本机。

## 内容范围

全部翻译下载文件中的文章内容：7 个主要章节、问答的 5 个分类与 15 个专题，共 27 个原标题；还包括简介、系列说明、资料入口和文末作者介绍。

3 张插图使用下载文件中的本地图像，图题译为中文，图内英文保留。13 处行内代码均保留，包括公式、配置项和表达式。原文没有完整可运行的代码块。新增术语表与译者注均与正文区分。

## 目录

```text
├── content/                       # 7 个主要章节 + 附录
├── webapp/                        # 可离线阅读的静态页面
│   ├── index.html                 # 目录首页
│   ├── all.html                   # 完整译文
│   ├── chapters/                  # 分章阅读
│   └── assets/                    # 样式、渲染器和 3 张原图
├── src/                           # 用户提供的底本与来源清单
├── licenses/                      # 模板和 Markdown 渲染器许可
├── qa/                            # 页面检查记录及截图
├── advanced_quantization_zh.md    # 完整译文（构建生成）
├── build.py                       # 标准库构建脚本
├── run.sh                         # 本地阅读服务
├── NOTICE.md                      # 署名和修改说明
└── VALIDATION.md                  # 核对结果与验证边界
```

修改 `content/*.md` 后运行 `python3 build.py` 重建。阅读页的样式、脚本和插图均为本地资源；原文链接、视频、演示文稿和外部参考资料需要联网。

## 说明

保留 2024 年的讲座语境，没有将原文限制或性能数字改写成当前版本结论。本次未进行 GPU 性能复现。H100 / FP4 表述另依据 NVIDIA 官方资料添加译者注。原文和图片权利归原权利人，不套用上一篇历史底本的内容许可。
