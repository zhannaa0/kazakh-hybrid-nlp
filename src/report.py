"""Методичка, табл. 8, этап 8 «Creating a report».

Отчёт собирается из данных, а не пишется руками, поэтому цифры в нём всегда
совпадают с результатами. Один и тот же отчёт сохраняется в двух видах:
  report.md   — для переноса в диссертацию (таблицы Markdown, ссылки на рисунки);
  report.html — для просмотра: один файл, рисунки встроены, открывается где угодно.
"""

import base64
import html
from pathlib import Path

import pandas as pd


class Report:
    def __init__(self, title):
        self.title = title
        self.blocks = []          # (kind, content)

    def h(self, text):
        self.blocks.append(("h", text))

    def p(self, text):
        self.blocks.append(("p", text))

    def table(self, df, floatfmt="{:,.3f}", index=True):
        self.blocks.append(("table", (df, floatfmt, index)))

    def figure(self, path, caption):
        self.blocks.append(("fig", (Path(path), caption)))

    # ------------------------------------------------------------ вывод

    @staticmethod
    def _fmt(df, floatfmt):
        out = df.copy()
        def one(v):
            if isinstance(v, bool):
                return "yes" if v else "no"
            if isinstance(v, int):
                return f"{v:,}"
            if isinstance(v, float):
                if pd.isna(v):
                    return ""
                return f"{int(v):,}" if v.is_integer() and abs(v) >= 1 else floatfmt.format(v)
            try:                                   # numpy-числа
                import numpy as np
                if isinstance(v, np.integer):
                    return f"{int(v):,}"
                if isinstance(v, np.floating):
                    return one(float(v))
                if isinstance(v, np.bool_):
                    return one(bool(v))
            except ImportError:
                pass
            return v
        return out.apply(lambda col: col.map(one))

    def to_markdown(self, path):
        path = Path(path)
        lines, fig_no = [f"# {self.title}", ""], 0
        for kind, c in self.blocks:
            if kind == "h":
                lines += [f"## {c}", ""]
            elif kind == "p":
                lines += [c, ""]
            elif kind == "table":
                df, ff, index = c
                d = self._fmt(df, ff)
                if index:
                    d = d.reset_index()
                cols = [str(x) for x in d.columns]
                lines.append("| " + " | ".join(cols) + " |")
                lines.append("|" + "---|" * len(cols))
                for row in d.itertuples(index=False):
                    lines.append("| " + " | ".join(str(v) for v in row) + " |")
                lines.append("")
            elif kind == "fig":
                fig_no += 1
                fp, cap = c
                rel = Path("figures") / fp.name
                lines += [f"![Figure {fig_no}. {cap}]({rel.as_posix()})", "",
                          f"*Figure {fig_no}. {cap}*", ""]
        path.write_text("\n".join(lines), encoding="utf-8")

    def to_html(self, path):
        parts, fig_no = [f"<h1>{html.escape(self.title)}</h1>"], 0
        for kind, c in self.blocks:
            if kind == "h":
                parts.append(f"<h2>{html.escape(c)}</h2>")
            elif kind == "p":
                parts.append(f"<p>{html.escape(c)}</p>")
            elif kind == "table":
                df, ff, index = c
                d = self._fmt(df, ff)
                if index:
                    d = d.reset_index()
                parts.append(d.to_html(index=False, border=0, escape=True))
            elif kind == "fig":
                fig_no += 1
                fp, cap = c
                if fp.exists():
                    data = fp.read_bytes()
                    width = int.from_bytes(data[16:20], "big") // 3      # 300 dpi → экранный размер
                    b64 = base64.b64encode(data).decode()
                    parts.append(f'<figure><img src="data:image/png;base64,{b64}" style="width:{width}px" '
                                 f'alt="{html.escape(cap)}">'
                                 f"<figcaption>Figure {fig_no}. {html.escape(cap)}</figcaption></figure>")
        style = """<style>
body{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 16px;color:#1a1a19;line-height:1.55}
h1{font-size:1.6rem}h2{font-size:1.2rem;margin-top:2.2rem;border-bottom:1px solid #e4e3df;padding-bottom:.3rem}
table{border-collapse:collapse;font-size:.85rem;margin:.6rem 0 1rem;display:block;overflow-x:auto}
th,td{padding:4px 10px;border-bottom:1px solid #e4e3df;text-align:right}th{background:#f6f5f2}
td:first-child,th:first-child{text-align:left}img{max-width:100%}figcaption{font-size:.85rem;color:#52514e}
</style>"""
        Path(path).write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(self.title)}"
                              f"</title>{style}</head><body>{''.join(parts)}</body></html>", encoding="utf-8")
