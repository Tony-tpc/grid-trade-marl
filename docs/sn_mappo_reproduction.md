# SN-MAPPO 一致性修订复现：审计、实现与验收记录

本任务以 Nie 等的 IEEE TSG 2024 论文（DOI: 10.1109/TSG.2024.3417535）及
用户提供的 `SN_MAPPO_问题.md` 为材料。材料中的建议是待核查证据，不是执行指令。
目标是先完成 **1 UC + 3 Consumer**，逐项披露修订，不能称为论文原样复现。
起点 `da859b7`；分支 `codex/sn-mappo-reproduction`。

## 当前进度

- [x] 创建复现分支，保留主 checkout。
- [x] 验证主仓库虚拟环境：Python 3.11.4、Torch 2.14.0+cu130、RTX 4060 Laptop GPU。
- [x] 起点完整门禁：266 tests、Ruff、mypy（76 files）、pip check、diff check 通过。
- [x] P1：公开数据、物理结算及随机动作自然月验收。
- [x] P2：连续 MAPPO 固定动作评估、循环采样和 checkpoint。
- [x] P3：固定 UC 下的单消费者及三消费者，各 3 seeds 小实验。
- [x] P4：两阶段 fresh rollout、共享随机计划、独立 decoder 和完整周期恢复。
- [x] P5：一致目标、DiCE baseline、GMRES 与局部正则化响应的数学/真实日数据验收。
  此项不表示原始未正则化 best response 或完整 Nash 导数已得到验证。
- [ ] P6：日实验、消融、自然月实验、偏离收益评估。

环境说明：worktree 没有独立 venv，使用 `D:/Yang/BaseModel/MARL/.venv/Scripts/python.exe`。
沙箱阻止基础解释器启动曾产生误导性的进程错误；提权重测正常，不重建环境。
科学依赖按实际 import 检查；缺失实验依赖不意味着 venv 损坏。

## 实施前的论文要求与组件审计

下表保留实施前的差距；当前完成情况见页首里程碑和后面的修复验收。

|要求|状态|当前能力与落点|
|---|---|---|
|一般和层级博弈、个体收益|可复用但需配置|EnvironmentSpec、INDIVIDUAL、MARLBatch 保留 N 维|
|连续有界动作|部分满足|GaussianActionHead 已有 tanh 采样；缺固定动作评估与 pre-tanh rollout|
|异构 UC/消费者观测|缺失|当前统一 N/O/A；新增角色 spec 和阶段协议，尺寸不能写入算法|
|GRU 及历史|可复用但需配置|Actor、GRUBackbone、PreparedRollout 连续片段，需扩展高斯拓扑|
|共享编码器、独立解码器|缺失|现有 independent 策略不能表达随机聚合计划和条件解码器|
|PPO/GAE/值函数/梯度裁剪|已满足基础机制|复用 objectives、returns、BaseMARLAlgorithm.optimize|
|连续 PPO|缺失|MAPPO 限离散，不能直接以 MASAC 冒充|
|Stackelberg/Nash 两阶段|缺失|新鲜阶段数据、冻结边界、版本标识及完整周期恢复需实现|
|二阶响应导数|缺失|普通 detached PPO 不提供跨策略 Hessian；需联合随机计算图估计|
|经验优先级|缺失|仅允许本轮新鲜片段，IS 修正；不能复用离策略 replay|
|电力市场与 ESS 物理|缺失|energy_trading 是另一环境，不能改名作为论文环境|
|终止/截断|已满足基础机制|GAE 分开处理终止 bootstrap 和 episode 边界|
|实验及恢复|可复用但需扩展|schema 5、配置快照、optimizer/RNG；另存阶段版本及数据 SHA|

## 已选定的修订及明确假设

1. 采用一致性修订，尽量保留原经济目标与 signed DR。功率守恒采用方案 A：
   `pc=pd+pe; pg=pa+pd+sum(pr); ps=sum(pu); pb+po=pe+sum(pu)`，因此
   `pb+pg+po=pa+pc+ps+sum(pr)`。消费者 `pu+pr+delta=baseline`。
2. `dt=0.25 h`，`E_next=(1-rho)E+eta_c*pc*dt-po*dt/eta_o`；
   `eta_c=.8, eta_o=.9, initial_SOC=.5, rho=0`。论文 eta_SOC 的含义不充分，
   不默认为每 15 分钟损失 5%。充电函数作为上限；不允许同时充放电。
3. 正 shift 延后用能、负 shift 偿还；债务以 kWh 计，每日结清。只用当前负荷和
   训练集确定的偿还容量构造可行域，不能偷看未来实测负荷。
4. 原目标中的 `Ts*D^2/pm-Tr*D`、`Tc*delta_i^2/pu_i-Tr*delta_i` 保留。
   分母和分子同为零取零；非零调整且分母为零不可行。非零 DR 的正域下界明确设定，
   不能偷偷加 epsilon 改写目标。下界训练尺度的 1e-3，做 0.1/10 倍敏感性。
5. UC 使用四个独立请求：储能净调度、充电来源比例、DER 提供比例、DR 目标；
   论文冗余的八条物理流由结算推导。消费者请求 DER/PG/削减/移时。
   确定储能模式后用 SciPy SLSQP 投影至可行约束，核查残差；失败明确报错。
   概率密度对应策略原始请求，不能用投影后动作重算 old log-prob。
6. 价格按论文 Table II，第一档峰 .078（20–22）、高 .068（09–15）、平 .048；
   第二档 .067/.059/.041，第三档 .060/.053/.037；阈值 2.88e7/4.8e7 kWh。
   小规模通常只落第一档，但阈值仍单测。买电价、DER 仿射价格参数等缺失值标为假设。
7. 共享编码器产生随机计划 z（4 维），各消费者根据本地观测和广播 z 独立行动。
   计划与实际聚合动作的误差记录，不将隐变量误称为满足物理约束的总动作。
   存在公共聚合信息通信，不能宣称完全无通信/隐私保证。
8. 初始诊断 MLP，核心实验宽度 32，UC GRU 两层、编码器/解码器 GRU 一层。
   使用固定训练集归一化；无 batch normalization。循环片段长度 32，无 burn-in。

## 数据协议

使用 [OPSD household_data 2020-04-15](https://data.open-power-system-data.org/household_data/2020-04-15/)
（CC-BY-4.0）的 15 分钟累计 kWh：

|角色|列|
|---|---|
|C1|DE_KN_residential2_grid_import|
|C2|DE_KN_residential5_grid_import|
|C3|DE_KN_industrial3_area_offices|
|UC DER|DE_KN_residential3_pv|

差分除以 .25 得平均 kW，保留窗口之前一个表点、UTC、Europe/Berlin 本地日期和原始
插补标记。不把累计读数当功率，不零填缺测。训练 2016-06/07，验证 08，测试 09。
容量/归一化只由训练集确定；原始文件 SHA、列、窗口、缺测/负差分/插补统计存 manifest。
先前候选 residential3/4/6 的 import+PV-export 存在负差分，已淘汰。
这些计量列是公开可用的代理负荷，并非作者未公开的原始实验数据。

## 两阶段与数学验收

每个周期先冻结全部采样策略，采集 DS；在任何参数变动前计算 UC 与编码器梯度，
更新 critic、UC 和编码器，消费者解码器冻结。丢弃 DS，再以新版本采集 DN；
冻结 UC 和编码器，更新各消费者 decoder/critic。只在完整周期边界保存 checkpoint。
编码器目标明确是消费者成本之和；个体 Nash 回报仍逐消费者保留。

采用 DiCE 联合策略 score-function 估计，随机节点包含 UC、z、消费者及因果前缀。
样本动作、old log-prob/GAE 均 detached，不通过黑箱环境求导。

`g_u = grad_u L_u - B_cu^T (H_cc + mu I)^(-T) grad_c L_u`

H 和 B 来自消费者目标 L_c；不能用 UC loss 的交叉块代替 B。
响应仅针对固定 decoder 时的共享编码器，不等价于完整消费者 NE 的导数。
复用 PyTorch HVP 和 SciPy LinearOperator/GMRES，mu 初值 1e-3，实际相对残差
1e-4，每次最多 100 次 Arnoldi 内迭代、3 个阻尼尝试（0.001/0.01/0.1）。
失败记录并跳过 UC actor，不能静默置零。每次重试和未加阻尼残差均记录。
DS 默认同一外部场景的 8 条独立 fresh 日轨迹；baseline 排除本轨迹。
上下层经济目标都除以 `sum(gamma**t)`，UC 直接梯度不再混入 PPO 归一化 advantage。
每周期 8*96 DS + 96 DN = 864 个实际训练步。日实验使用完整 96 步；
高阶月估计的分块/截断仍未实现。CUDA 二阶 RNN 必要时禁用 cuDNN。

必须验证：解析双层例子 u=.3 的导数 3.4（错误公式 1.4）、有限差分、离散联合枚举、
零耦合、非正定 Hessian、求解失败、阶段梯度隔离、旧经验拒绝、连续动作饱和、
GRU/LSTM 状态及完整恢复。优先级采样最后接入，限当前 fresh 片段并带 IS 权重；
高阶估计保持均匀全序列。

## 实施顺序和验收目标

1. P1：数据下载/差分/manifest；物理与经济公式固定例子、随机可行性、日结债务。
2. P2：复用 GaussianActionHead、Actor、MAPPO、RolloutBuffer；新增固定动作评估，
   保存 pre-tanh 值以处理 float 饱和；离散及离策略数值回归。
3. P3：固定 UC，先 1 Consumer 后 3 Consumer；各 3 seeds、约 20k 真实步，
   验证可学习性、独立回报和 checkpoint，不称为均衡证明。
4. P4：阶段协议、角色 spec、随机共享计划、双阶段 trainer/算法薄装配；fresh 版本测试。
5. P5：DiCE、隐式响应、GMRES、梯度代理接入已有 optimize；数学验收先于训练。
6. P6：日实验 no-DR、规则、同步连续 MAPPO、顺序一阶、修订 SN-MAPPO；
   学习方法各 5 seeds、约 200k 物理步。implicit/GRU/PER 消融各 3 seeds 同预算。
   再自然月顺序一阶/完整方法各 3 seeds、约 200k 步。

预算包含 DS+DN 的所有环境交互，完整周期结束可能超额，必须记录实际数。
每次 update 记录诊断，每约 5000 物理步独立评估；原始 JSONL、逐 seed 曲线、配置与
数据 SHA、checkpoint 放入忽略目录 `runs/sn_mappo/`。曲线不得补造或插值伪装为采样。
指标包括 UC 利润、三人成本、弃电、PG 功率、PAR/DAP/TV、债务/SOC、分离 actor/critic
损失、KL/entropy/梯度、HVP/求解残差。偏离收益使用受限 best-response 搜索；
UC 偏离重新求 follower response，结果只能称“找到的偏离收益”，不证明 Stackelberg/NE。

## 复用证据与边界

- [官方 MAPPO](https://github.com/marlbenchmark/on-policy/tree/de66d7a4b23fac2513f56f96f73b3f5cb96695ac)：
  核对 PPO/value clipping 与循环片段；优先沿用本库现有实现。
- [StackelbergPOMDP](https://github.com/GlcBrero/StackelbergPOMDP/tree/799ad1eacb91d7a7566bf48254b21f96b979fe4e)：
  参考 follower-response/reward phase、数据排除及 policy cache；不是 SN-MAPPO 的实现。
  旧 gym/SB3 依赖不整体引入。
- [Stackelberg actor-critic](https://github.com/LeoZhengZLY/stackelberg-actor-critic-algos/tree/180adf0fd34637a74de0ac0719e62ffa1f12d414)：
  参考 HVP/线性求解组织，不复用其 actor-critic 博弈目标冒充 UC/consumer 博弈。
- [DiCE/LOLA](https://github.com/alshedivat/lola/tree/0e8b13e480d6483007549ddf1f660c2f45460fda)：
  参考 MagicBox 及因果依赖，不引入旧 TensorFlow 栈。
- [SciPy GMRES](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html)：
  复用 Arnoldi 正交化与显式残差停止，不自行实现 Krylov 求解器；MINRES 为早期诊断路径。

## 完成门禁

每个完整成果同时包含源码、配置、数学/状态/端到端测试和说明；通过 pytest、Ruff、
mypy、pip check、diff check 后才提交。提交用中文，不推送远端，不提交数据和训练产物。
后续实际结果追加于本文件，未完成项保持未勾选，不能将计划当成验证证据。

## 2026-10-02 修复前执行结果（保留原始证据）

### 已落地的代码

- `marl/envs/demand_response_data.py`：累计 kWh 差分、真实 15 分钟功率、本地月分割、
  两端点插补标记、训练集尺度和 SHA manifest。测试额外检查验证/测试负荷放大十倍
  不会改变训练尺度。
- `marl/envs/demand_response.py`：SLSQP 可行清算、ESS、费用分项、日结债务、阶段承诺，
  固定 UC 和训练日抽样 adapter。1+3 同步观测统一填充到 14 维；顺序消费者追加 4 维计划。
- 连续 MAPPO：Gaussian 固定动作评估、pre-tanh 保存、当前策略熵 MC 估计、GRU/LSTM，
  复用原 RolloutBuffer、GAE、PreparedRollout 和 MAPPO optimizer 循环。
- `SNMAPPO` 直接继承基类并组合三个 MAPPO 角色；`SequentialTrainer` 不导入算法类。
  checkpoint 限完整周期，包含数据/环境上下文、策略版本、角色 optimizer 和 NumPy/Torch RNG。
- `marl/training/implicit.py`：DiCE、autograd HVP、SciPy MINRES、真实残差验收及梯度代理。
  解析例子得到 3.4，错误使用上层交叉块会得到 1.4；联合离散枚举一/二阶导数相符。

### 实际数据与预算

原始 SHA256：`eca7dee5ac6e53373cd0209bc05ba3383cd860bfa5d5e995f74fa78db8857c58`。
训练/验证/测试步数分别为 5,856 / 2,976 / 2,880，均无缺测及显著负差分。
训练插补区间数为 C1=114、C2=236、C3=0、DER=0；验证 DER=84；测试所有列为 0。
与早期按 UTC 月统计不同，这里使用**区间开始的本地月份**，并合并两个表点的插补标记。
95% 训练功率尺度为 `[0.800, 0.680, 2.776, 3.116] kW`。

|实际实验|种子|每 seed 实际训练物理步|结果|
|---|---|---:|---|
|固定 UC + 1 Consumer、MLP|0/1/2|20,064|采样、更新、日志、checkpoint 全完成|
|固定 UC + 3 Consumer、MLP|0/1/2|20,064|个体成本分别保存，训练全完成|
|顺序一阶 + GRU、1+3|0/1/2|20,160|两个阶段均训练，并完成完整 9 月 2,880 步评估|
|默认隐式响应 + GRU、1+3|0/1/2|192|三次阻尼重试仍不达标，记录并停止长训练|

固定 UC 的同一个验证日，三种子平均总成本从 0.24332→0.19624（1 Consumer）、
1.60568→1.32360（3 Consumer）。终点是**最后已记录的验证点 15,264 步**，不是训练
终点的伪造评估。个别消费者成本可能变差；这些短程结果不证明收敛或博弈均衡。
一阶顺序实验的自然月 UC 回报为 7.6957 / 8.8788 / 6.7842；其阶段、网络与预算不同，
不能与固定 UC 的单日数据直接排优劣。测试月是按日训练策略的外推评估，不是月度训练验收。

完整测试月随机动作（seed=8）2,880 步最大约束残差约 `6.0e-14`。
一阶顺序策略的月评估最大残差不超过 `1.5e-13`，终日移峰债务均为零。

### 真实二阶验收没有通过，未自动扩大训练

默认 `mu=1e-3`、倍率 10、3 次重试、每次 100 次 MINRES 后：

|seed|最终 mu|总迭代|实际相对残差|UC 更新|
|---|---:|---:|---:|---|
|0|0.1|300|0.0439419|跳过|
|1|0.1|300|0.0260783|跳过|
|2|0.1|300|0.0408882|跳过|

残差门槛为 `1e-4`。倍率改为 100、最终 mu=10 的独立敏感性实验仍失败。
强阻尼 mu=1000 的独立诊断可在 4/4/5 次迭代达到
`1.87e-6 / 2.43e-7 / 7.21e-7`，说明强正则化的线性系统可求解，
**不能据此将默认值偷偷改成 1000，更不能声称原目标的响应导数已经得到验证**。
这些证据提示默认样本 Hessian 的条件性/方差需要进一步研究，并非 venv 运行故障。
同一 seed=0 日轨迹改用 float64 重算，最终残差仍为 0.0379708（mu=0.1，300 次迭代），
单纯提高浮点精度没有解决问题；记录在 `solver_probe/float64_seed0.json`。

因此 P5 的真实数据门槛保持未完成，P6 的 200k 长实验、PER、消融、best-response 搜索
和月度训练没有启动。无样本的曲线没有补造；没有计算或宣称 Nash/Stackelberg gap。

### 2026-10-02：二阶残差失败的原因定位

本次只固定模型/轨迹做诊断，没有执行 optimizer 更新或修改正式算法。入口和原始
JSON 在本地忽略目录 `runs/sn_mappo/diagnostics/`；`analyze_implicit.py` 支持
`--double` 和 `--sampling-only`。模型 seed=0，日窗口 seed=1826701614，
coordinator actor 有 4,744 个参数；联合 log-prob 为 `[1,96,5]`，其中 batch=1。

求解的是 `(H+mu I)x=b`，其中 H 是消费者总成本的 DiCE 样本 Hessian，
b 是 UC 目标对 coordinator 参数的梯度。验收量为 `norm(Ax-b)/norm(b)`。

|同一 seed=0 轨迹的对照|实际迭代数|MINRES info|实际相对残差|
|---|---:|---:|---:|
|原始 float32 默认三次尝试，最后 mu=0.1|100+100+100|原日志未记录|0.0439419|
|float64，mu=0.1，rtol=1e-5，上限 100|100|100|0.0379708|
|float64，mu=0.1，rtol=1e-5，上限 1,000|517|0|0.0157318|
|float64，mu=0.1，rtol=1e-10，上限 2,000|2,000|2,000|0.00762475|

1. **默认求解预算不足，且停止标准没有对齐。** 默认的 300 次是三个不同阻尼值
   各从零开始的 100 次，不是同一个系统连续求解 300 次。实际安装的 SciPy 1.17.1
   使用 `norm(r)/(estimated_norm(A)*norm(x))`、最小二乘条件等停止，不能将
   `rtol=1e-5` 等同于本项目的右端项相对残差 `1e-4`。上表 info=0 但实际残差
   仍为 0.0157 是直接证据。显式残差验收应保留，不能只接收 info=0。
   [SciPy 对应版本源码](https://github.com/scipy/scipy/blob/v1.17.1/scipy/sparse/linalg/_isolve/minres.py)。
2. **样本 Hessian 有不定曲率和冗余参数方向，默认阻尼没有充分改善尺度。**
   SciPy eigsh 测得最小/最大特征值约 -54.7365 / 613.521；极值特征对相对残差
   低于 1e-6。GRU 的 reset/update gate 两组 bias 存在相反变化互相抵消的方向，
   实测该方向 Hv=0。mu=0.1 时条件数至少约 6,136，但这不是完整条件数测量。
   该零方向的 b 投影也近零，所以不能把它单独当成残差无法下降的原因。
   MINRES 本身允许不定/奇异矩阵，负特征值不意味着选错了求解器。
   mu=1000 将这批数据的极值移动到约 945 / 1614，条件数约 1.71，解释了强阻尼
   为何容易求解；但它也明显改变了所求响应，不能据此认定原隐式导数正确。
3. **单轨迹、无 baseline 的高阶估计波动显著。** 固定参数和同一天外部负荷，
   只改变动作随机种子 0–7；沿固定的原始归一化 b 方向，曲率估计落在
   -32.18 到 85.52，正负号均出现。原始轨迹在此方向为 329.55。
   这是有限样本的波动证据，不是对真实期望 Hessian 的谱估计。
   当前实现只平均一条轨迹，因果前缀累积到 96 步，也没有 DiCE baseline。
   [DiCE 原论文 §4.2/§5](https://proceedings.mlr.press/v80/foerster18a/foerster18a.pdf)
   给出了保持梯度估计期望的 baseline 及降低方差的实验；估计器数学正确并不保证
   一条轨迹就足以稳定估计 Hessian，更不保证其逆是稳定的响应估计。
4. **没有发现本次 HVP 因随机重采样或明显非对称而失效。** 同向量重复计算一致；
   float64 的随机方向对称误差约 1.9e-15、线性误差约 1.4e-15。
   这类抽查不能替代全部正确性证明，但与 float64/2,000 次对照一起说明：
   venv 和单纯浮点精度不是当前问题的主要解释。

还有两个与残差失败不同层面的复现问题。其一，第一次 DS 就在初始策略上求逆，
没有验证下层已接近驻点；原始样本下层梯度范数约 14.95（该值本身也有采样噪声），
不能直接解释为已达到 best response。其二，实际 UC 直接项使用归一化 GAE 的 PPO
surrogate 和 entropy，Hessian/交叉项使用未归一化折扣经济回报。两者并非同一个
目标的完整隐式导数；该样本两种直接梯度范数为 0.355 / 0.824，余弦约 -0.125。
这些问题不会直接改变当前已构造 A 的线性残差，但会影响求解成功后的更新含义。

后续修复优先级：先统一目标/尺度并明确正则化响应的定义；用同一冻结策略采集多条
fresh 日轨迹并引入符合 DiCE 高阶依赖的 baseline，测量方差随样本量的变化；
再对齐求解器停止条件、记录每次重试的 info/真实残差、评估合适的预条件与阻尼。
保留小网络可精确比较的基准及真实 GRU 验收。单纯增加 max_iterations、改用
float64 或把 mu 改到 1000 均不构成完整修复；本次没有改这些正式默认值。

### 修复后的目标与响应定义

DS 的上层目标为 `Lu=-E[sum(gamma**t*r_uc)]/sum(gamma**t)+beta*KL_uc`，
下层为 `Lc=-E[sum(gamma**t*sum_i r_i)]/sum(gamma**t)+beta*KL_c`。
UC 直接梯度、Hessian、交叉导数和 coordinator 的梯度下降使用同一对目标。
DN 保留逐消费者奖励的 PPO/GAE；它不使用消费者均值奖励替换个人目标。

第 b 条轨迹的控制变量 `baseline[b,t]=mean(reward[j,t], j!=b)`；全部动作仍是
独立随机样本。共同外部负荷条件下，其他轨迹不受本轨迹动作影响。使用
`MagicBox(prefix)*(reward-baseline)+baseline` 保留前向回报以及一阶、二阶估计的期望。
不能将本轨迹的 value/GAE 直接塞入这条全前缀公式；它们依赖本轨迹过去的动作。
基于两个独立两步条件动作轨迹的穷举测试覆盖所有交叉二阶项，常数成本测试验证
baseline 可以消除对应的 score 噪声；没有把重复同一动作轨迹当独立样本。

在当前点 `(u0,c0)`，固定 `g0=stopgrad(grad_c Lc(u0,c0))`，响应定义为局部方程
`grad_c Lc(u,c)+mu*(c-c0)-g0=0` 的敏感度。因此在可逆条件下得到上述
`(H+mu I)^-1` 公式，不需要谎称初始下层梯度为零。这是**局部正则化响应近似**，
不是求解 `grad_c Lc=0` 的未正则化 best response，也不是完整消费者 NE 的导数。
非驻点解析测试验证该局部定义的有限差分；日志保留 `lower_gradient_norm` 和
`response_undamped_residual`，将线性求解精度与正则化影响分开报告。

首周期默认 GRU 真实数据验收：

|seed|DS 轨迹数|实际训练步|接受 mu|总内迭代|加阻尼相对残差|未加阻尼相对残差|
|---|---:|---:|---:|---:|---:|---:|
|0|8|864|0.01|115|3.43029e-5|0.81165|
|1|8|864|0.01|116|2.42013e-5|1.32918|
|2|8|864|0.01|159|2.99220e-5|4.08536|

所有值直接来自 `runs/sn_mappo/repair_acceptance/` 的 JSONL。
显式验收门槛保持 `1e-4`，没有升到 mu=1000。未加阻尼残差仍大，说明不能把
成功的正则化求解冒充原始 Hessian 逆。旧失败记录和曲线保留，不覆盖历史。

配对方差诊断固定同一初始策略、同一外部负荷和同一随机单位向量，共 8 个独立批次、
每批 8 条轨迹。对每一批相同数据分别启用/关闭 baseline，HVP 向量各分量的样本方差
之和从 `9.17789e-4` 降至 `8.52859e-6`，比值 `0.009293`。两个分支使用相同归一化
和 KL；该下降不是目标缩放造成。这里只是一个固定场景的小样本诊断，不能外推为
所有场景的方差保证。证据为 `diagnostics/repair_variance.json`。

真实 checkpoint 恢复：seed=0 在 20,736 步的 checkpoint，两次恢复后执行同一个
后续周期至 21,600 步，全部模型张量及全部指标完全一致；后续求解残差 `4.39550e-5`。
这两次诊断不写回训练历史，原训练结果保持 20,736 步，证据为
`diagnostics/repair_resume.json`。旧配置 checkpoint 因目标/轨迹协议变化拒绝精确续训。

### 修复后的三种子短训练验收

`runs/sn_mappo/repair_short/` 使用默认修复配置，每种子目标 20,000 步，在完整周期
结束后实际达到 20,736 步。全部 24 次 DS 更新均有原始诊断，没有丢弃失败批次重跑。
每约 2,000 步独立验证，实际周期对齐为每 2,592 步一次；每种子 8 个验证观测点，
并单独执行 2,880 步测试月评估。每种子的 3,648 个评估步不计入训练预算。

|seed|实际训练步|通过的 DS 求解|最大加阻尼残差|mu=0.001/0.01/0.1 次数|最大未加阻尼残差|
|---|---:|---:|---:|---|---:|
|0|20,736|24/24|7.86122e-5|2/21/1|8.0024|
|1|20,736|24/24|4.72924e-5|1/22/1|3.7373|
|2|20,736|24/24|6.28087e-5|0/22/2|90.6865|

三种子合计 62,208 训练步、72/72 求解通过；各项原始训练指标均有限。
测试月最大物理投影残差分别约 `9.88e-14 / 2.16e-14 / 6.22e-14`。
每个目录均保存配置/数据来源、逐次日志、独立评估、checkpoint 及 `diagnostics.png`。
汇总为 `repair_short/aggregate.json`，跨 seed 的每次实际残差和正则化影响图为
`repair_short/response_acceptance.png`。未加阻尼残差可能很大，这些结果只验证
局部正则化系统的数值求解和训练生命周期，不证明原始最优响应、策略收敛或博弈均衡。

### 当前实现相对于目标计划的明确边界

- UC 直接项与响应项已统一为 DiCE 目标，但正则化响应仍为局部近似。
  基于有限随机样本的 Hessian 逆不保证是期望 Hessian 逆的无偏估计。
- DS actor 每批只更新一次；DN 复用四 epoch PPO。DS 不使用 PPO clip/entropy，
  KL 在旧策略点的一阶导数通常为零，但进入二阶曲率；没有实现步后 KL 回退或 trust region。
- 三个角色的 critic 独立；消费者内部沿用 N 输出的集中式 value 网络，参数共享，
  reward/return 没有跨消费者平均。若论文要求每人独立 critic，还需拓扑消融。
- 当前高阶路径完整展开一个事件；只验收了日窗口。不支持月高阶的 96 步分块和
  时间截断 bootstrap，collector 对非 true termination 明确拒绝。
- DR 开启时采用显式正域下界；存在负荷/债务的消费者保留少量 UC 购电，
  因而不覆盖“DR 开启但零调整时恰好全 DER”的全部边界。关闭 DR 的配置保留该边界。
- 4 kWh/2 kW 电池、taper=.8/.5、购电价 .04、DER=.025+.001*pg、弃电价 .01
  均是公开的小规模假设。无实时价格噪声、无额外终端 SOC 等式；月评估记录末态能量。
- 优先级采样尚未实现。新增计划请求误差诊断只适用于后续采样；既有实验未记录该字段，
  不从终点模型倒填历史。

### 复运行与证据目录

以下命令在 worktree 根目录执行；解释器可替换为已验证的主仓库 `.venv` 绝对路径。
实验目录若已有日志会拒绝覆盖，重跑时请用新的 `--output`。

```powershell
python -m examples.prepare_demand_response
python -m examples.train_demand_response --consumers 1 --steps 20000
python -m examples.train_demand_response --consumers 3 --steps 20000
python -m examples.train_sn_mappo --first-order --steps 20000
python -m examples.train_sn_mappo --steps 864 --output runs/sn_mappo/new_acceptance
python -m examples.train_sn_mappo --steps 20000 --eval-every 2000 --output runs/sn_mappo/new_short
python -m examples.plot_demand_response
```

每个实验包含 `manifest.json`、`training.jsonl`、`evaluation.jsonl`、`summary.json`、
`checkpoint.pt`、`diagnostics.png`；目录为 `runs/sn_mappo/fixed_uc/` 和
`runs/sn_mappo/sequential/`。强阻尼诊断位于 `runs/sn_mappo/solver_probe/`。
图中独立评估只显示真实散点；高频训练曲线显示原始更新及明确标注的尾随均值。

修复后最终工程门禁：`297 passed in 39.66s`；Ruff、mypy（90 files）、pip check、
`git diff --check` 均通过；API 生成覆盖 65 个模块，文档检查通过 83 页、2,583 个链接、
183 个函数契约。工程门禁与正则化数值验收均通过；均衡/完整复现仍是独立的后续目标。
代码、配置、测试和说明构成同一可验证成果；原始数据、诊断脚本、图和 checkpoint
均在忽略目录中，不提交训练产物，不推送远端。
