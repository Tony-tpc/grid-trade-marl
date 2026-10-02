"""从 AST 生成可跳转参考；仅使用标准库，不导入库或执行训练。"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "doc/api"

# 手写补充用于核心算法/训练以外尚未逐参数展开 docstring 的接口。
# key 是完整符号名；原有 docstring 优先，签名与字段始终来自当前 AST。
DESCRIPTIONS = {
    "InputEncoder.__init__": (
        "记录 input_dim/output_dim，初始化 nn.Module；不分配学习参数，不返回特征。"
    ),
    "IdentityEncoder.__init__": "输入 input_dim，建立输入输出同宽的无参数编码器。",
    "IdentityEncoderConfig.build": (
        "输入 input_dim 与可选 layout，返回新 IdentityEncoder；不使用布局，不创建参数。"
    ),
    "HistoryEncoder.__init__": (
        "输入 input_dim、HistoryLayout 和 temporal Config；验证索引完整覆盖后创建时序网络，"
        "注册历史/当前索引 buffer，输出宽度为时序隐藏宽度加当前特征数。返回 None。"
    ),
    "HistoryEncoderConfig.build": (
        "输入 input_dim 和 HistoryLayout，返回新 HistoryEncoder；layout=None 报错。"
        "创建独立参数，每次 forward 从零状态处理完整窗口。"
    ),
    "HistoryEncoderConfig.__post_init__": (
        "验证 temporal 是 GRU/LSTM/Transformer Config；非法类型抛 TypeError，返回 None。"
    ),
    "ObservationEncoder.__init__": (
        "输入 EnvironmentSpec、encoder 与可选 graph Config；创建 N 个独立局部编码器，"
        "可选共享图层及邻接 buffer。forward 接收 [...,N,O]，输出 [...,N,E]。"
    ),
    "JointActionEncoder.__init__": (
        "输入 spec、encoder 与可选 graph Config；装配联合观测编码器，保留全部动作分支。"
        "输入按全部观测后接全部动作排列，宽度 N*(O+A)，输出宽度 N*(E+A)。"
    ),
    "StateEncoder.__init__": (
        "输入 spec、encoder 与可选 graph Config；要求 state_layout，创建节点间共享的局部"
        "编码器及可选图层，注册节点/当前索引与邻接 buffer。输出宽度 N*E+G。"
    ),
    "QMixerConfig.build": (
        "输入 EnvironmentSpec，创建 state 编码、权重超网络及 value 分支并返回 QMixer；"
        "局部 Q 数量来自 N，状态维度来自 S，不增加共享的全局注册状态。"
    ),
    "QMixer.forward": (
        "输入 agent_q_values [...,N] 和同批维 state [...,S]，编码 state 后生成非负混合权重，"
        "返回团队 Q [...,1]；校验批维和末维，保留计算图但不更新参数。"
    ),
    "config_to_dict.encode": (
        "递归把 dataclass、Enum、dtype、外部配置等转成可序列化值，返回规"
        "范化数据。"
    ),
    "config_to_dict.deep_settings": "递归编码外部组件 settings，返回独立的可序列化字典。",
    "TensorReplayBuffer.__init__.allocate": (
        "按容量及尾部 shape 分配指定 dtype 的 CPU replay 存储，返回 Tenso"
        "r。"
    ),
    (
        "TensorReplayBuffer.sample.take"
    ): (
        "选择已抽中的 replay 索引并迁移到目标设备，返回一个字段的批 Tensor。"
    ),
    "MARLBatch.to.move": "迁移可选 Tensor 到目标设备，None 保持 None；返回迁移结果。",
    "MARLBatch.pin_memory.pin": "对可选 CPU Tensor 执行锁页，返回 Tensor 或 None。",
    "transitions_to_batch.stack": (
        "按字段名堆叠 current 或 next 的 NumPy 数组，返回目标设备的 float"
        "32 Tensor。"
    ),
    "_mapping": "检查输入为字符串键 Mapping，返回原映射；location 用于错误定位。",
    "_backbone": (
        "读取 kind 并严格解析 MLP/GRU/LSTM 配置；未知 kind 报错，返回对应"
        " BackboneConfig。"
    ),
    "_component": "选择内置组件配置或显式 catalog 的外部组件；返回 Buildable，尚不实例化网络。",
    "algorithm_config_from_dict": (
        "输入算法字典和可选 catalog，校验版本/字段/类型后返回五种算法 Con"
        "fig 之一。"
    ),
    "load_algorithm_config": (
        "从路径安全加载 YAML，然后返回经过校验的算法 Config；catalog 仅用"
        "于外部组件。"
    ),
    "_off_policy": (
        "按离策略配置构造网络、独立命名 optimizer、target 更新器和 replay"
        " trainer，返回 Experiment。"
    ),
    "_freeze": "将嵌套环境配置递归转为只读 mapping/tuple，返回不可变数据快照。",
    "Buildable.kind": "返回组件选择名称，供配置序列化或 YAML 选择使用。",
    "Buildable.build": "输入 EnvironmentSpec，返回该配置定义的组件；结构化能力协议不提供实现。",
    (
        "ExternalConfig.build"
    ): (
        "输入 spec，调用加载配置时绑定的 create，返回已校验兼容性的外部组件。"
    ),
    "ExtensionCatalog.register": (
        "登记 category/name 对应的 parse/build 和动作/奖励兼容集合；重复"
        "名称报错，返回 None。"
    ),
    "ExtensionCatalog.configure": (
        "解析外部 settings 并绑定构造闭包，返回 ExternalConfig；网络稍后"
        "由 build(spec) 创建。"
    ),
    (
        "ExtensionCatalog.register.configure"
    ): (
        "用注册的 parse 解析 settings，返回绑定解析结果的网络构造闭包。"
    ),
    (
        "ExtensionCatalog.configure.build"
    ): (
        "校验 spec 的动作/奖励兼容性后实例化 nn.Module，验证能力接口并返回组件。"
    ),
    "EnvironmentStep.done": (
        "返回 terminated or truncated；仅用于 episode 控制，不可代替 boot"
        "strap 的 terminated。"
    ),
    "EnergyTradingEnv.ownership": "输入设备名称 name，返回逐家庭是否拥有该设备的 bool [N] 数组。",
    "EnergyTradingEnv.ev_available": "返回当前时刻各家庭 EV 是否接入的布尔数组；不涉及策略网络。",
    "EnergyTradingEnv.sa_can_start": "返回各家庭可移时负荷是否能在当前时刻启动的布尔数组。",
    (
        "EnergyProfiles.validate"
    ): (
        "依据配置验证外部能源曲线形状、范围和有限性；无返回，非法数据提前报错。"
    ),
    (
        "EnergyTradingAdapter._action_mask"
    ): (
        "根据当前设备可行性返回离散合法动作 bool [N,A]；连续动作返回 None。"
    ),
    "EnergyTradingAdapter._snapshot": (
        "打包能源环境当前观测、state、收益和终止信息，返回经过 validate_s"
        "tep 的 EnvironmentStep。"
    ),
    (
        "MemoryCueAdapter._step_result"
    ): (
        "根据当前时间和历史提示构造观测/查询标记，附加输入奖励并返回统一环境结果。"
    ),
    (
        "MPE2SimpleAdversaryAdapter._flat_dim"
    ): (
        "检查空间 shape 并计算展平维度；返回 int，异常空间显式拒绝。"
    ),
    (
        "MPE2SimpleAdversaryAdapter._validate_action_spaces"
    ): "检查所有玩家的动作空间与配置动作类型一致，返回 None。",
    (
        "MPE2SimpleAdversaryAdapter._observations"
    ): (
        "按固定 agent 顺序整理并补齐观测，返回 float32 [N,O]。"
    ),
    "MPE2SimpleAdversaryAdapter._snapshot": (
        "将 MPE 原始输出打包为固定玩家顺序的 EnvironmentStep，并校验统一"
        "接口。"
    ),
    (
        "ParallelMPEEnvironment.observation_space"
    ): (
        "输入 agent 名称，返回底层 MPE 该玩家的观测空间对象。"
    ),
    "ParallelMPEEnvironment.action_space": "输入 agent 名称，返回底层 MPE 该玩家的动作空间对象。",
    "ParallelMPEEnvironment.state": "无输入，返回底层环境的全局 state NumPy 数组。",
    "Actor._attach_hidden_state": (
        "将 backbone 输出的新状态写入动作输出的 hidden_state，返回该动作"
        "输出。"
    ),
    "Actor._encode": (
        "校验输入，依次调用 encoder 与 backbone，返回带 features 和跨步隐藏状态的"
        " BackboneOutput；窗口 encoder 本身不保存跨调用状态。"
    ),
    "QEnsemble.critics": "返回逐智能体独立 critic 序列，供算法仅对选定玩家动作保留梯度。",
    "TwinQEnsemble.first": "返回第一组独立集中式 critic。",
    "TwinQEnsemble.second": "返回第二组独立集中式 critic，与第一组参数独立。",
    "IndependentDiscretePolicy._agent_state": (
        "从 [K,B,N,H] 隐藏状态中选择一个玩家，返回 [K,B,H]；LSTM 同时选择"
        " h/c。"
    ),
    (
        "IndependentDiscretePolicy._agent_state.select"
    ): "验证单个 h/c 张量维度并选取目标玩家，返回连续 [K,B,H] 张量。",
    "IndependentDiscretePolicy._mask_for": (
        "从联合 action_mask 选一个玩家，返回 [...,A]；输入缺失则为 None。"
    ),
    "_scalar": "检查 Tensor 只有一个元素并 reshape 为零维标量，保留计算图；否则抛 ValueError。",
    "LossBundle.combine": (
        "输入非空 ObjectiveResult 序列，求和 loss 并合并唯一命名指标，返"
        "回 LossBundle。"
    ),
    "PPOClipObjective.__call__": (
        "输入同形 [...,N] 新旧 log-prob/advantage，返回裁剪策略目标及 KL"
        "、clip_fraction。"
    ),
    "ValueMSEObjective.__call__": (
        "输入同形 [...,N] value/return，返回系数加权 MSE 和未加权 value_l"
        "oss 指标。"
    ),
    "ClippedValueObjective.__call__": (
        "输入 value/old_value/return 和可选尺度，取裁剪与未裁剪误差较大者"
        "，返回目标及裁剪率。"
    ),
    "EntropyObjective.__call__": "输入逐智能体熵，返回负系数乘平均熵的目标和平均 entropy 指标。",
    "TDLossObjective.__call__": (
        "输入同形 prediction/target，返回加权 TD MSE 和 name 指定的原始 M"
        "SE 指标。"
    ),
    "CounterfactualPolicyObjective.__call__": (
        "输入 [...,A] logits/Q，计算 detached 反事实 advantage 的 score-f"
        "unction 目标，返回 ObjectiveResult。"
    ),
    "DeterministicPolicyObjective.__call__": (
        "输入逐智能体 Q，返回 -mean(Q) 及 actor_loss 指标；梯度隔离由调用"
        "算法完成。"
    ),
    "SACEntropyObjective.alpha": "返回 exp(log_alpha) 的正温度 Tensor [N]，该属性本身保留梯度。",
    "SACEntropyObjective.actor": (
        "输入 [...,N] log_prob/min_q，返回 mean(detach(alpha)*log_prob-mi"
        "n_q) 策略目标。"
    ),
    "SACEntropyObjective.temperature": (
        "输入 [...,N] log_prob，返回只对 log_alpha 求导的熵目标与温度指标"
        "。"
    ),
    "TD0Estimator.estimate": (
        "输入同形 reward/next_value/terminated，返回 r+gamma*(1-terminate"
        "d)*next_value；不自行 detach。"
    ),
    (
        "ValueTargetEstimator.estimate"
    ): (
        "声明同形逐智能体 TD target 估计能力；不接收整个时间序列的 GAE 状态。"
    ),
    "GAEEstimator.estimate": (
        "输入时间优先 [T,...,N] 奖励/value/终止及末步 value [...,N]，返回"
        "同形 advantage/return；只标准化 advantage。"
    ),
    "AdvantageEstimator.estimate": (
        "声明时间优先优势估计接口，输出与 rewards 同形的 (advantages,retu"
        "rns)。"
    ),
    "SoftTargetUpdate.step": (
        "输入 (target,online) 对；no_grad 下参数按 tau 做 Polyak 平均，bu"
        "ffers 完整复制，返回 None。"
    ),
    "HardTargetUpdate.step": (
        "输入 (target,online) 对；完整复制 state_dict，返回 None。间隔判"
        "定由调用方完成。"
    ),
    "HardTargetUpdate.should_step": "输入算法 update_count，返回是否被 interval 整除的 bool。",
    "TargetUpdate.step": "声明目标网络原地同步接口，输入 target/online 对，返回 None。",
    "TargetScale.update": (
        "输入 [...,N] target，在启用时更新逐智能体 Welford 统计；无梯度、"
        "返回 None。"
    ),
    "TargetScale.forward": (
        "输入 [...,N] 数值，启用时减逐智能体均值再除以尺度；返回同形 Tens"
        "or，禁用时原样返回。"
    ),
    "TargetScale.scale": "无输入，返回 [N] 尺度；启用时为最小值 1 的标准差，禁用时为全 1。",
    "SyncVectorEnv.reset": (
        "输入每个 adapter 对应的 seed，顺序 reset，返回 EnvironmentStep "
        "元组。"
    ),
    "SyncVectorEnv.step": "输入 NumPy [B,N] 或 [B,N,A]，依次推进环境并返回结果元组；不是多进程。",
    "TensorReplayBuffer.__len__": "无输入，返回当前有效 transition 数，而非配置容量。",
    "TensorReplayBuffer.add": "输入 Transition，写入环形存储并更新位置/大小；返回 None。",
    "TensorReplayBuffer.sample": (
        "输入 batch_size、NumPy RNG 和目标 device，随机抽样并返回 MARLBat"
        "ch，不消费存储。"
    ),
    "TensorReplayBuffer.load_state_dict": (
        "输入 replay 状态快照，校验 spec/capacity 与存储数据后恢复位置、"
        "大小和张量；返回 None。"
    ),
    "read_settings": (
        "解析 tanh_value 外部 settings，拒绝未知字段/非整数宽度，返回 Tan"
        "hValueConfig。"
    ),
}

FORWARDS = {
    "MLPBackbone": (
        "逐位置编码 [...,input_dim] 为 [...,output_dim]；返回 BackboneOut"
        "put，无循环状态。"
    ),
    "GRUBackbone": (
        "处理 [B,T,O] 序列（单步由 Actor 补时间维），返回 features 与 [K,"
        "B,H] 新隐藏状态。"
    ),
    "LSTMBackbone": (
        "处理 [B,T,O] 序列（单步由 Actor 补时间维），返回 features 与新 ("
        "h,c)，两者 [K,B,H]。"
    ),
    "GNNBackbone": (
        "输入节点特征和图邻接信息，执行消息聚合，返回节点 BackboneOutput"
        "；调用方管理图语义。"
    ),
    (
        "TransformerBackbone"
    ): (
        "编码输入序列为 BackboneOutput；不能据存在该类推断已支持因果时序训练。"
    ),
    "DiscreteActionHead": (
        "输入 features、deterministic 与可选 mask，返回采样动作及其 logit"
        "s/log_prob/entropy。"
    ),
    "DeterministicActionHead": "输入 features，返回 tanh 限幅连续动作的 ActionHeadOutput。",
    "GaussianActionHead": (
        "输入 features 与 deterministic，返回 tanh-Gaussian 动作、修正 lo"
        "g-prob 和熵估计。"
    ),
    "ValueCritic": "输入 [...,O] 特征，返回 squeeze(-1) 后的标量 value Tensor [...]。",
    "CentralizedCritic": (
        "输入全局 state [...,S] 和可选循环状态，返回 [...,N] value；retur"
        "n_state=True 时同时返回状态。"
    ),
    "IndependentCentralizedCritics": "输入集中式联合特征，返回各独立玩家 Q [...,N]。",
    (
        "TwinIndependentCentralizedCritics"
    ): (
        "输入集中式联合特征，返回两组独立 Q Tensor，各为 [...,N]。"
    ),
    "AttentionCritic": "输入联合观测和联合离散动作，返回每个玩家候选动作 Q 表 [...,N,A]。",
    "VDNMixer": "输入 [...,N] 局部 Q，沿 N 求和返回 [...,1]；忽略可选 state。",
    "QMixer": "输入 [...,N] Q 与 [...,S] state，使用非负超网络权重混合，返回 [...,1] 团队 Q。",
}


def symbols(tree: ast.AST):
    """输入 AST；产出包含嵌套函数的 (限定名,节点,所属类)；重载保留最后实现。"""
    found = {}

    def walk(node, prefix="", owner=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                name = prefix + child.name
                found[name] = (child, child.name)
                walk(child, name + ".", child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + child.name
                found[name] = (child, owner)
                walk(child, name + ".", owner)
            else:
                walk(child, prefix, owner)

    walk(tree)
    return [(name, *entry) for name, entry in found.items()]


def anchor(name: str) -> str:
    """输入限定符号名，返回稳定 ASCII 锚点；保留私有函数下划线。"""
    return name.lower().replace(".", "-")


def describe(name: str, node, owner: str) -> str:
    """优先返回源码 docstring；其他接口使用审阅后的语义补充。"""
    doc = ast.get_docstring(node)
    if doc:
        return doc
    if name in DESCRIPTIONS:
        return DESCRIPTIONS[name]
    method = node.name
    if isinstance(node, ast.ClassDef):
        return "数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。"
    if method == "__init__":
        return f"构造 {owner}，按下方参数初始化网络子模块或运行状态；返回 None。"
    if method == "__post_init__":
        return f"在 {owner} 数据类构造后校验字段范围/一致性；返回 None，非法配置立即报错。"
    if method == "build":
        return f"从 {owner} 的静态参数构造目标对象；输入与具体返回类型见签名，不执行训练。"
    if method in ("forward", "__call__"):
        if owner in FORWARDS:
            return FORWARDS[owner]
        protocols = {
            "ValueNetwork": "输入 state [...,S]，返回逐智能体 value [...,N]。",
            "RecurrentValueNetwork": (
                "输入 state 与可选隐藏状态，按 return_state 返回 value 或 (value,"
                "新状态)。"
            ),
            "AttentionQNetwork": "输入联合观测和动作，返回候选动作 Q [...,N,A]。",
            "TwinQEnsemble": "输入联合特征，返回 (Q1,Q2)，每个 [...,N]。",
            "MixingNetwork": "输入局部 Q [...,N] 与 state [...,S]，返回团队 Q [...,1]。",
        }
        if owner in protocols:
            return protocols[owner]
    if method == "parameters":
        return "返回可遍历的 nn.Parameter；recurse 决定是否包含子模块，用于优化器参数装配。"
    if method == "initial_state":
        return (
            "根据 batch 尺寸创建零循环状态，继承参数设备/dtype；MLP 返回 None"
            "，GRU 返回 h，LSTM 返回 (h,c)。"
        )
    if method in ("_validate", "_validate_input"):
        return "检查输入张量尺寸、mask 或循环布局与实例规格一致；返回 None，错误早报。"
    if method == "spec":
        return "无输入；返回静态 EnvironmentSpec，作为所有算法尺寸与动作/奖励语义的唯一来源。"
    if method == "reset":
        return (
            "按可选 seed 重置环境并返回初始结果；adapter 返回统一 Environment"
            "Step，底层协议返回原生格式。"
        )
    if method == "step":
        return (
            "执行输入动作并推进环境；adapter 返回下一 EnvironmentStep，底层协"
            "议返回原生观测/奖励/终止信息。"
        )
    if method == "close":
        return "释放底层环境资源，返回 None；无 optimizer 或模型副作用。"
    if method == "act":
        return (
            "输入逐智能体观测和可选 mask，返回 MARLModelOutput；动作形状 [..."
            ",N] 或 [...,N,A]，随机性由策略实现决定。"
        )
    if method in ("logits", "logits_for_agent"):
        return "输入观测/mask，输出离散 logits；联合 [...,N,A] 或指定玩家 [...,A]，不采样动作。"
    if method == "evaluate":
        return (
            "对输入观测下已经给定的 actions 计算 log_prob/entropy 并返回 MARL"
            "ModelOutput；不重新选择动作。"
        )
    if method == "q_values":
        return "输入观测 [...,N,O]，返回局部离散 Q Tensor [...,N,A]；不混合团队价值。"
    if method == "main":
        return "读取命令行参数或示例配置，执行该脚本描述的演示并打印结果；返回 None。"
    raise ValueError(f"需要人工补充说明: {name}")


def render(path: Path) -> str:
    """输入一个源码路径，返回带目录/字段/函数签名和源码链接的 Markdown。"""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    relative = path.relative_to(ROOT).as_posix()
    entries = symbols(tree)
    result = [
        f"# {relative}",
        "",
        "[手册首页](../README.md) · [全部 API](README.md)",
        "",
        f"[打开源码](../../{relative})",
        "",
        ast.get_docstring(tree) or "模块职责见类与函数说明。",
        "",
        "本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。",
        "",
    ]
    if not entries:
        result += [(
            "本文件定义公共导出/数据别名，没有手写函数。请通过源码查看导入的具体模块。"
        ), (
            ""
        )]
    else:
        result += ["## 符号目录", ""]
        result += [f"- [{name}](#{anchor(name)})" for name, _, _ in entries]
        result += [""]
    for name, node, owner in entries:
        result += [
            f'<a id="{anchor(name)}"></a>',
            "",
            f"## {name}",
            "",
            f"[源码位置](../../{relative}#L{node.lineno}) · [页内目录](#符号目录)",
            "",
        ]
        if isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(base) for base in node.bases)
            result += [f"`class {node.name}({bases})`", "", describe(name, node, owner), ""]
            fields = [item for item in node.body if isinstance(item, ast.AnnAssign)]
            if fields:
                result += [(
                    "声明字段（dataclass 的 init 字段同时也是构造输入）："
                ), (
                    ""
                ), (
                    "```python"
                )]
                result += [ast.unparse(item) for item in fields]
                result += ["```", ""]
        else:
            signature = f"def {node.name}({ast.unparse(node.args)})"
            if node.returns:
                signature += " -> " + ast.unparse(node.returns)
            result += ["```python", signature, "```", "", describe(name, node, owner), ""]
            # 保留 Args/Returns 缩进，用文本代码块展示张量形状。
            if "Args:" in result[-2] or "Returns:" in result[-2]:
                result[-2] = "```text\n" + result[-2] + "\n```"
    return "\n".join(line.rstrip() for line in "\n".join(result).splitlines()).rstrip() + "\n"


def main() -> None:
    """--check 只检查参考是否过期；默认生成，失败返回非零退出码。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    paths = sorted(
        [*ROOT.glob("marl/**/*.py"), *ROOT.glob("examples/*.py"),
         *ROOT.glob("reproduction/**/*.py"), *ROOT.glob("doc/examples/*.py")]
    )
    outputs = {}
    index = [
        "# 全库函数索引",
        "",
        "[手册首页](../README.md) · [目录职责](../architecture.md)",
        "",
        "覆盖全部 marl 源文件、复现实验、通用示例和手册教学脚本；包含私有/嵌套函数，",
        "overload 在同一锚点展示最终实现。测试按专题见 [验证导航](../training.md#verification)，",
        "基准实验入口与内部组织见 [基准说明](../../benchmarks/README.md)。",
        "",
        "| 源文件 | 类/函数条目数 |",
        "|---|---:|",
    ]
    for path in paths:
        relative = path.relative_to(ROOT)
        filename = "-".join(relative.with_suffix("").parts) + ".md"
        outputs[filename] = render(path)
        count = len(symbols(ast.parse(path.read_text(encoding="utf-8"))))
        index.append(f"| [{relative.as_posix()}]({filename}) | {count} |")
    outputs["README.md"] = "\n".join(index) + "\n"
    if args.check:
        stale = [
            name
            for name, content in outputs.items()
            if not (DEST / name).exists() or (DEST / name).read_text(encoding="utf-8") != content
        ]
        if stale:
            raise SystemExit("API 参考过期: " + ", ".join(stale))
    else:
        DEST.mkdir(exist_ok=True)
        for name, content in outputs.items():
            (DEST / name).write_text(content, encoding="utf-8")
    print(f"API {'checked' if args.check else 'generated'}: {len(paths)} modules")


if __name__ == "__main__":
    main()
