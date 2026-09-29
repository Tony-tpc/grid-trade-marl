"""检查仓库说明文档的本地链接/锚点/源码行号及算法、训练函数注释覆盖。"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]


def anchors(text: str) -> set[str]:
    """输入 Markdown，返回显式 HTML 锚点和 GitHub 风格标题锚点。"""
    result = set(re.findall(r'<a id="([^"]+)"', text))
    for heading in re.findall(r"^#{1,6}\s+(.+)$", text, re.M):
        slug = re.sub(r"[^\w\-\s]", "", heading.lower()).replace(" ", "-")
        result.add(slug)
    return result


def main() -> None:
    """无参数；逐个检查本地链接和目标函数 docstring，发现错误以非零状态退出。"""
    failures: list[str] = []
    link_count = 0
    pages = sorted(
        {
            ROOT / "README.md",
            *(
                page
                for folder in ("doc", "docs", "examples", "marl", "benchmarks")
                for page in (ROOT / folder).rglob("*.md")
            ),
        }
    )
    for page in pages:
        content = page.read_text(encoding="utf-8")
        # 代码块中的示意语法不是导航链接。
        plain = re.sub(r"```.*?```", "", content, flags=re.S)
        for target in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", plain):
            if re.match(r"[a-zA-Z]+://", target) or target.startswith("mailto:"):
                continue
            link_count += 1
            address, _, fragment = unquote(target).partition("#")
            destination = (page.parent / address).resolve() if address else page
            label = f"{page.relative_to(ROOT)} -> {target}"
            if not destination.exists():
                failures.append("Missing: " + label)
                continue
            if not fragment or destination.is_dir():
                continue
            if destination.suffix == ".md":
                if fragment not in anchors(destination.read_text(encoding="utf-8")):
                    failures.append("Anchor missing: " + label)
            elif re.fullmatch(r"L\d+", fragment):
                line = int(fragment[1:])
                lines = destination.read_text(encoding="utf-8").splitlines()
                if not 1 <= line <= len(lines):
                    failures.append("Line out of range: " + label)
                elif destination.suffix == ".py" and not re.match(
                    r"\s*(async )?(def|class)\s", lines[line - 1]
                ):
                    failures.append("Line is not symbol start: " + label)
    functions = 0
    for folder in ("algorithms", "training"):
        for path in (ROOT / "marl" / folder).glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                functions += 1
                doc = ast.get_docstring(node) or ""
                if "Returns:" not in doc or not ("Args:" in doc or "Inputs:" in doc):
                    failures.append(f"Incomplete docstring: {path.name}:{node.lineno} {node.name}")
                arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
                if node.args.vararg:
                    arguments.append(node.args.vararg)
                if node.args.kwarg:
                    arguments.append(node.args.kwarg)
                for arg in arguments:
                    if arg.arg not in ("self", "cls") and not re.search(
                        rf"^\s*{re.escape(arg.arg)}:", doc, re.M
                    ):
                        failures.append(
                            f"Parameter undocumented: {path.name}:{node.name}:{arg.arg}"
                        )
    if failures:
        raise SystemExit("\n".join(failures))
    print(
        f"Checked {len(pages)} Markdown pages, {link_count} links, {functions} function contracts"
    )


if __name__ == "__main__":
    main()
