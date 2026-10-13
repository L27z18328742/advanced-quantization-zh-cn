# 附录　作者介绍、术语与来源

## 关于作者（原网页附记）

> 以下为原作者自述的翻译。

我是 Christian Mills，一名应用 AI 顾问和教育者。

无论是撰写深入的教程，还是分享详细的笔记，我的目标始终相同：把复杂话题讲清楚，并从中找到实用、有价值的洞见。

如果你的 AI 项目需要一位具备这种深入研究能力和系统思考方式的战略合作伙伴，我愿意提供帮助。我们可以一起讨论如何降低项目路线图的风险，构建能在现实中发挥作用的解决方案。

欢迎通过我的[快速 AI 项目评估](https://docs.google.com/forms/d/e/1FAIpQLScKDKPJF9Be47LA3nrEDXTVpzH2UMLz8SzHMHM9hWT5qlvjkw/viewform?usp=sf_link)开启交流，或进一步了解[我的工作方法](https://christianjmills.com/about.html)。

## 常用术语（Terminology）

| 英文 | 中文 |
|---|---|
| quantization / dequantization | 量化／反量化 |
| dynamic quantization | 动态量化 |
| weight-only quantization | 仅权重量化 |
| weights / activations | 权重／激活 |
| compute-bound | 计算受限 |
| memory-bound | 内存带宽受限 |
| scale / rescale | 缩放因子／重新缩放 |
| accumulation | 累加 |
| materialization | 实体化存储，即显式产生并保存中间结果 |
| fusion | 融合 |
| packing / unpacking | 打包／解包 |
| per-tensor / per-channel | 逐张量／逐通道 |
| per-token / per-block | 逐词元／逐块 |
| quantization-aware training / QAT | 量化感知训练 |
| outlier | 离群值 |
| speculative decoding | 推测解码 |
| perplexity | 困惑度 |
| kernel / Tensor Core | 内核／Tensor Core |

## 底本与翻译范围（Source and Scope）

- **原作者**：Christian Mills。
- **讲者**：Charles Hernandez，PyTorch AO 团队。
- **原文标题**：GPU MODE Lecture 7: Advanced Quantization。
- **原文发布日期**：2024 年 9 月 10 日。
- **原网页**：[Lecture 007](https://christianjmills.com/posts/cuda-mode-notes/lecture-007/)。
- **本次底本**：用户提供的 `GPU MODE Lecture 7_ Advanced Quantization – Christian Mills.html` 及其配套资源文件夹。
- **翻译日期**：2026 年 9 月 20 日。

译文覆盖下载页面中的全部文章内容：简介、系列说明、资料入口、7 个主要章节、问答环节的 5 个分类与 15 个专题，以及末尾作者介绍。站点导航、搜索界面和评论组件不属于文章正文。

3 张正文插图使用用户下载的原图，图题译为中文，图片内英文标签保留。原文中的公式、代码表达式、配置项和链接均保留。页面新增术语表和清楚标记的译者注，便于阅读；没有添加原文没有的性能测试结果。

## 翻译与核对说明（Translation Notes）

- 使用简体中文，排版沿用上一篇与 `ultrascale-playbook-zh-tw` 的阅读界面。
- 原文所有 27 个正文标题均有对应译文，并保留英文锚点。
- 原文使用行内代码，没有完整可运行的代码块。译文保留原有表达式，不把它们扩充为未经验证的程序。
- 数字“约 4 倍”“约 14%”以及相关性能判断按原文保留，没有进行 GPU 实测。
- 对 H100 / FP4 混淆、FastINT8 命名、Torch.io 拼写和部分口语化技术概括，分别添加译者注。
- H100 / Blackwell 的数据类型说明另核对了 [NVIDIA 官方 Transformer Engine 文档](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.11/user-guide/examples/fp8_primer.html)，其他补充不代表对原文全部技术结论进行了独立复现。
- 讲者关于未来生态、硬件计划和库性能的判断均保留当时的语境，不作为当前状态报告。

## 署名与文件记录（Attribution and Files）

本项目是依据用户提供内容制作的本地非官方中文译本。正文、原图和演讲资料的权利归原权利人；此处不重新授予其许可，也不沿用上一篇历史底本的 CC BY-NC-SA 许可。

原始 HTML、下载资源和来源清单位于 `src/`，原文与图片的 SHA-256 校验值保存在 `src/source.json`。界面样式来自 Twinkle AI Community 的 Ultra-Scale Playbook 中文版，模板许可为 Apache 2.0；marked v12.0.2 采用 MIT 许可。相应声明保存在 `NOTICE.md` 与 `licenses/`。
