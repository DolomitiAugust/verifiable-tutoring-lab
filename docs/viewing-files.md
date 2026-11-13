# Viewing and downloading repository files

[English README](../README.md) | [中文首页](../README.zh-CN.md)

## Start here

- [English figure gallery](../reports/figures.md)
- [中文图表总览](../reports/figures.zh-CN.md)
- [English report](../reports/research-report.en.pdf)
- [中文报告](../reports/research-report.zh-CN.pdf)

The galleries embed the existing PNG images directly in Markdown. SVG and PDF files remain available as vector sources. A failed GitHub preview does not establish that a file is missing or damaged; compare the file with `release-manifest.json` or run `python -B scripts/verify_release.py` after downloading the repository.

| Format | Recommended way to view |
|---|---|
| PNG | Use the figure gallery or open the image; download if the file viewer fails. |
| SVG | Use the corresponding PNG for online reading. Download SVG for vector editing; a valid SVG can still fail in a particular web preview. |
| PDF | Try GitHub's preview; use **Download raw file** to open it in a local PDF reader if rendering fails. |
| JSON / JSONL | Small files can be read in **Code**. Large records may be truncated or unavailable in the web viewer; use **Raw**, download, or clone. JSONL contains one JSON record per line. |
| SQLite | Download the database and use SQLite tools or Python's built-in `sqlite3`; a GitHub text preview is not expected. |

GitHub's [official format documentation](https://docs.github.com/en/repositories/working-with-files/using-files/working-with-non-code-files) describes its supported renderers and limitations. In particular, CSV/TSV rich rendering is limited to 512 KB. Repository visibility remains private; viewing and downloading require an authorized signed-in account.

To download everything with GitHub CLI:

```bash
gh repo clone DolomitiAugust/verifiable-tutoring-lab
cd verifiable-tutoring-lab
python -B scripts/verify_release.py
```

## 中文说明

优先打开中文图表总览，页面会直接显示PNG，不依赖单个SVG文件的预览器。SVG和PDF矢量图仍完整保留；预览报错时可下载后用本机程序打开。

较大的JSON/JSONL可能无法完整在线展示，SQLite数据库也不适合当作网页文本查看。它们仍属于已上传的实验记录，可使用Raw、下载或上面的克隆命令取得完整文件。`release-manifest.json`和核验脚本用于检查下载内容，而不是证明模型教学效果。

私有仓库需要已登录且有权限的账户；不能为了修复预览而把仓库改成公开。
