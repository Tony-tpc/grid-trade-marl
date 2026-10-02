"""独立复现实验的命令行配置和依赖方向回归。"""

import ast
import importlib
from pathlib import Path

import pytest

from marl.config import load_algorithm_config


@pytest.mark.parametrize("module_name, config_name", [
    ("train_sn_mappo", "sn_mappo.yaml"),
    ("train_demand_response", "mappo_continuous.yaml"),
])
@pytest.mark.parametrize("explicit", [False, True])
def test_training_cli_resolves_configs_outside_repository(
    monkeypatch, tmp_path, module_name, config_name, explicit,
):
    module = importlib.import_module(f"reproduction.sn_mappo.{module_name}")
    default = Path(module.__file__).parent / "configs" / config_name
    expected = load_algorithm_config(default)
    argv = [module_name, "--seeds", "17", "--steps", "1"]
    if explicit:
        custom = tmp_path / "custom.yaml"
        custom.write_text(default.read_text(encoding="utf-8"), encoding="utf-8")
        argv.extend(["--algorithm", "custom.yaml"])

    seen = []

    def capture(args, seed):
        assert load_algorithm_config(args.algorithm) == expected
        assert Path(args.algorithm) == (Path("custom.yaml") if explicit else default)
        seen.append(seed)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(module, "run", capture)
    monkeypatch.setattr(module.torch, "set_num_threads", lambda _: None)
    module.main()
    assert seen == [17]


def test_algorithm_library_does_not_import_reproduction():
    root = Path(__file__).parents[1] / "marl"
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(name.split(".")[0] == "reproduction" for name in names), path
