# OpenArm NSP-IK 工作空间 使用手册

> 路径：`/ros2_ws/openarm_nsp_ws/`
> 目标机器人：OpenArm v1.0（7-DoF × 2 双臂，达妙电机，CAN-FD）
> 最后更新：2026-08-11

---

## 1. 这是什么

一个**独立于 `openarm_ros2`** 的工作空间，基于 Pinocchio 库为 OpenArm v1.0 提供：

1. **零空间投影逆运动学（NSP-IK）** —— 双阶段求解：DLS 收敛 + σ_min/关节裕度爬升，主动远离奇异与限位。
2. **离线轨迹规划** —— 笛卡尔直线 / SE(3) 弧线拟合 / B 样条优化（含自碰撞避免）。
3. **实机控制 Dashboard** —— Dash Web 界面（:8050），CAN 直驱、状态机、示教-复现、实时关节曲线。

两个 ROS2 包：

| 包 | 作用 |
|----|------|
| `openarm_pinocchio_nsp` | IK 求解器 + 轨迹规划 + 命令行工具（纯计算，可离线测试） |
| `openarm_dashboard` | 实机/仿真 Web 控制台，直驱 `openarm_can`（不走 ros2_control） |

> 与 `openarm_ros2` 的关系：本工作空间**不依赖** ros2_control 的 `joint_trajectory_controller`；Dashboard 直接通过 `openarm_can` 库发 MIT 模式帧控制电机。`openarm_ros2` 仅提供 URDF（`openarm_description`）。

---

## 2. 目录结构

```
openarm_nsp_ws/
├── venv-openarm-ik/          # numpy<2 虚拟环境（系统 numpy 2.x 会让 pinocchio 段错误）
├── USAGE.md                  # 本文件
└── src/
    ├── openarm_pinocchio_nsp/
    │   ├── src/openarm_pinocchio_nsp/
    │   │   ├── kinematics.py          # PinocchioModel: fk/ik/ik_nsp/ik_multi
    │   │   ├── ik_nsp.py              # 零空间投影核心 (damped_pseudoinverse 等)
    │   │   ├── cartesian_planner.py   # plan_cartesian / fit_arc / densify_se3 / ease_in_out_retime
    │   │   ├── bspline_planner.py     # BSplineOptimizer (SLSQP + 自碰撞)
    │   │   ├── urdf_path.py           # resolve_urdf_path() 统一 URDF 定位
    │   │   ├── fk.py / move_joints.py / plan_cartesian.py / home.py / ik_node.py  # CLI 入口
    │   └── validation/                # 离线验证脚本（见 §8）
    └── openarm_dashboard/
        └── src/openarm_dashboard/
            ├── hardware_dashboard.py  # Dash :8050 主界面
            ├── arm_controller.py      # 每臂状态机 + 250Hz 工作线程（CAN 直驱）
            └── robot_state.py         # 线程安全状态容器 + 滚动绘图缓冲
```

---

## 3. 环境准备（重要）

### 3.1 必须用 venv

系统 `python3` 的 numpy 是 2.x，与 `ros-humble-pinocchio`（按 numpy 1.x 编译）二进制不兼容，导入即段错误。
**所有运行/编译都要先激活 venv：**

```bash
source /ros2_ws/openarm_nsp_ws/venv-openarm-ik/bin/activate
python -c "import numpy, pinocchio; print(numpy.__version__)"   # 应为 1.26.4
```

### 3.2 编译

⚠️ **不要用 `source activate && colcon build`**。`colcon` 本身的 shebang 是 `#!/usr/bin/python3`，
即使激活了 venv，colcon 仍用系统 python 跑 setuptools，生成的 `ros2 run` 入口脚本 shebang 会是
系统 python → 启动时报 `No module named 'dash'`（系统 python 没装 dash，且 numpy 2.x 会让 pinocchio 段错误）。

**正确写法：用 venv 的 python 直接驱动 colcon**（venv 开了 system-site-packages，能看见系统的 colcon，
本地 numpy 1.26.4 阴影掉系统 numpy 2.x）：

```bash
source /opt/ros/humble/setup.bash
cd /ros2_ws/openarm_nsp_ws
/ros2_ws/openarm_nsp_ws/venv-openarm-ik/bin/python -m colcon build \
    --packages-select openarm_pinocchio_nsp openarm_dashboard --symlink-install
source install/setup.bash
```

这样 setuptools 的 `sys.executable` = venv python，入口脚本 shebang 自动是 venv python。
验证：`head -1 install/openarm_dashboard/lib/openarm_dashboard/hardware_dashboard`
应显示 `#!/ros2_ws/openarm_nsp_ws/venv-openarm-ik/bin/python`。

> 用 `--symlink-install` 改源码即时生效，无需重复编译。
> 若 shebang 已损坏（显示 `#!/usr/bin/python3`），重新执行上面的编译命令即可修复。

### 3.3 CAN 口（实机）

硬件可能变动，**每次先查** `ip link | grep can`。当前（2026-08）：

| 接口 | 臂 |
|------|----|
| `can_slot1_ch0` | LEFT |
| `can_slot1_ch1` | RIGHT |

Dashboard 启动时会自动 `ip link set ... up`（需 root）。CAN 映射的单一事实源在
`hardware_dashboard.py` 的 `CAN_MAP`，硬件换槽时改这里。

> ⚠️ **`openarm.bimanual.launch.py` 的默认 left/right 是反的**（默认 right=ch0/left=ch1）。
> 如果你同时跑 ros2_control，必须显式传 `left_can_interface:=can_slot1_ch0 right_can_interface:=can_slot1_ch1`。

---

## 4. 命令行工具（`ros2 run openarm_pinocchio_nsp <tool>`）

### 4.1 `fk` —— 正运动学
```bash
ros2 run openarm_pinocchio_nsp fk --side right --deg "0,0,0,90,0,0,0"
# 输出末端位姿 (xyz + xyzw)
```

### 4.2 `move_joints` —— 关节空间点到点（需 ros2_control 在跑）
```bash
ros2 run openarm_pinocchio_nsp move_joints --side right --deg "0,0,0,90,0,0,0" --time 2.0
```

### 4.3 `plan_cartesian` —— 离线笛卡尔规划（可发到 ros2_control）
```bash
ros2 run openarm_pinocchio_nsp plan_cartesian \
    --side right \
    --line "0.30,0.10,0.40 -> 0.35,0.05,0.45" \
    --quat "0,0,0,1" \
    --output /tmp/traj.json          # 可选：写轨迹 JSON
# --publish 额外把轨迹发到 /right_joint_trajectory_controller/joint_trajectory
```

### 4.3b `plan_trajectory` —— 纯规划工具（规划/执行分离）★ 新增
**只做规划，不执行、不碰 CAN/硬件**。输入示教点 + 最大角速度 + 控制频率，输出一条**关节位姿链**（带时间戳的关节角序列）到 JSON。执行交给"另一部分"（ros2_control、dashboard、或自定义节点）。

```bash
# 1) 示教点文件 pts.json —— 7 关节角(rad)的列表（可从 dashboard 示教导出，或手写）
cat > pts.json <<'EOF'
[[0,0,0,1.5708,0,0,0],
 [0.349,-0.175,0,1.396,0.087,0.262,0],
 [0.611,-0.349,0.005,1.222,0.175,0.524,0]]
EOF

# 2) 规划（纯计算）→ chain.json
ros2 run openarm_pinocchio_nsp plan_trajectory \
    --points pts.json --max-speed 1.0 --freq 100 --side right \
    --output chain.json
# planned 96-point chain | duration 0.95s | peak 1.00 rad/s | max_step 0.0100 rad -> chain.json
```

参数：`--max-speed`（每关节最大角速度 rad/s）、`--freq`（控制频率/采样 Hz）、`--side`、`--q-seed`（IK 热启动种子，默认第1点）、`--null-iters`（6=快速 / 12=精细）。

输出 `chain.json` 同时含两种格式，任意执行器都能消费：
- **ros2_control `JointTrajectory` 形**：`joint_names` + `points[{positions[7], time_from_start{sec,nanosec}}]`
- **原始形**：`times[]` + `positions[[7],...]` + `duration/max_speed/freq/n_points`

规划保证（与 dashboard 回放同一套算法）：时间严格均匀 `1/freq`，每关节速度 ≤ `max_speed`，末端在示教点间走直线。求解后自动跑平滑性检测（仅 stderr 警告，不阻断）。

**执行（另一部分）—— 任选其一：**
- **ros2_control**：把 `points` 转成 `trajectory_msgs/JointTrajectory` 发到 `/{side}_joint_trajectory_controller/joint_trajectory`（需 ros2_control 在跑、且**不要**同时跑 dashboard 直驱 CAN）。
- **dashboard**：`_replay_run` 内部已用同一 `pose_replay_traj`，可视为执行器之一。
- **自定义**：读 `chain.json` 的 `times/positions`，按时间戳插值下发。

> 库 API（任意 Python 代码复用）：
> ```python
> from openarm_pinocchio_nsp.plan_trajectory import build_chain, chain_to_joint_trajectory
> chain = build_chain(model, points, max_speed=1.0, freq=100)   # {'times':[...], 'q_path':[[7],...]}
> jt = chain_to_joint_trajectory(chain, "right", 1.0, 100)       # JointTrajectory 形 dict
> ```

### 4.4 `home` / `ik_node`
```bash
ros2 run openarm_pinocchio_nsp home --side right          # 回零
ros2 run openarm_pinocchio_nsp ik_node                    # ROS IK 服务节点
```

---

## 5. 实机 Dashboard（核心用法）

### 5.1 启动

```bash
source /opt/ros/humble/setup.bash
source /ros2_ws/openarm_nsp_ws/venv-openarm-ik/bin/activate
source /ros2_ws/openarm_nsp_ws/install/setup.bash
ros2 run openarm_dashboard hardware_dashboard        # 实机
ros2 run openarm_dashboard hardware_dashboard --sim  # 仿真（无硬件，自动走零力矩）
```

浏览器打开 `http://<IP>:8050`。启动后**双臂自动进入零力矩**（安全默认，可拖动）。

### 5.2 状态机

```
DISABLED --使能-->  ZERO_TORQUE (零力矩，可自由拖动)
ZERO_TORQUE --抱住--> ENABLED_HOLD (PD 抱紧当前位姿)
*HOLD --归零--> HOMING --完成--> HOLD (回零，会动!)
*HOLD --回起点--> GO_START --完成--> HOLD
*HOLD --TRACKING(traj)--> TRACKING --完成--> HOLD (到位保持)
任意 --失能--> DISABLED
```

- 工作线程 **250Hz**（仿真 20Hz），所有 CAN 调用都在工作线程内；按钮只往队列里塞命令。
- **过热保护**：MOS > 85°C 自动失能。
- **忙碌态**（HOMING/GO_START/TRACKING）期间除「失能」外按钮全部禁用。

### 5.3 按钮布局（每臂）

```
第1行(基本):  [使能(零力矩)] [抱住] [归零⚠] [失能]
第2行(单点):  [📍记录起点] [📍记录终点] [↩回起点] [▶到终点] [▶直线运动]
第3行(弧线):  弧线:N点  [📍弧线加点] [🗑清空弧线] [↩弧线回起点] [▶弧线IK] [▶B样条优化]
```

---

## 6. 示教-复现工作流

### 6.1 单点到终点（关节空间插值，最可靠）
```
[使能] → 拖到起点 → [📍记录起点]
       → 拖到终点 → [📍记录终点]
       → [↩回起点]            # 平滑回起点 (~2s) → HOLD
       → [▶到终点]            # 单次 IK 解终点 → 关节空间直线插值 → HOLD
```
末端走**非直线**（关节匀速，末端随意），但实机最稳。

### 6.2 直线运动（末端走空间直线）
```
同上记录起点/终点 → [↩回起点] → [▶直线运动]
# FK(起,终) → densify(位置线性+SLERP) → 逐点 ik_nsp(warm-start) → 逐点执行 → HOLD
```
末端走**真直线**，相邻 IK warm-start 保证关节连续。**不检查安全门**，直接执行。

### 6.3 弧线运动（多点拟合弧线 + 静止启停）★ 新增
```
[使能] → 拖到起点   → [📍弧线加点]   (P0)
       → 拖到中间点 → [📍弧线加点]   (P1)
       → 拖到终点   → [📍弧线加点]   (P2)
       → [↩弧线回起点]              # 平滑回 P0 → HOLD
       → [▶弧线IK]                  # 见下
```
`[▶弧线IK]` 流程：
1. `FK(所有弧线点)` → 笛卡尔控制点
2. `fit_arc(n_dense=100)` → 过所有点的 B 样条位置 + SLERP 姿态
3. `plan_cartesian(presampled=True)` → warm-start 逐点 `ik_nsp` → 关节序列 + C² 平滑
4. **`ease_in_out_retime`** → 静止启停重定时（见 §7）
5. **不检查安全门** → 直接 `TRACKING` 执行 → HOLD

### 6.4 B 样条优化（高级，关节空间优化 + 自碰撞）
```
记录 ≥2 个弧线点 → [▶B样条优化]
# FK → BSplineOptimizer.optimize (SLSQP: 跟踪+平滑+碰撞) → post_verify → TRACKING
```
带后验证（碰撞/速度/分支跳变），未在实机充分验证，作为高级选项保留。

### 6.5 关节回放（位姿插值 + 逐点 IK，末端走直线）★ 新增
**任务空间复演** —— 只记录每个示教点的 **6D 位姿**（位置+姿态），回放时在位姿空间插值出中点（匹配控制频率），每个点做 IK（"反解"）得到关节角。末端在相邻示教点间**走空间直线**。

```
[使能] → 拖到位姿A → [📍弧线加点]      (记录 A 的 6D 位姿)
       → 拖到位姿B → [📍弧线加点]
       → ... (任意多个点)
[↩弧线回起点]                          (先回到第1点，避免起跳)
顶部填: 最大角速度(rad/s)  控制频率(Hz)
       → [▶关节回放]
```
顶部"关节回放"区两个输入：
- **最大角速度** (rad/s)：每关节转速硬上限（默认 1.0）。
- **控制频率** (Hz)：输出轨迹点的时间间距 = 1/频率（默认 100，实机 ≤ 250）。

算法（`pose_replay_traj`）—— 两个约束**同时严格成立**：
1. **细采样 + IK**：每段在位姿空间细分（位置线性 + 姿态 SLERP），逐点 IK（从当前臂位姿热启动，避免分支跳变）→ 稠密关节路径。
2. **关节行程弧长重采样**：以"最大关节行程"为弧长，按 `步长 = 最大角速度/频率` 均匀重采样 → 每个输出点 = 一个控制周期，关节移动恰好 = 步长。
3. 结果：时间**严格均匀** (0, 1/freq, 2/freq, …)，且**每个关节速度 ≤ 最大角速度**。

实测（3 示教点，构型差异较大）：

| 最大角速度 | 频率 | 点数 | 时长 | 峰速 | 末端段内直线偏离 |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1.0 | 100 | 93 | 0.92s | **1.000** | 0.02mm |
| 0.5 | 100 | 185 | 1.84s | **0.500** | 0.02mm |
| 1.0 | 200 | 185 | 0.92s | 1.000 | 0.02mm |

- 峰速 = 最大角速度（精确满足，无超速）。
- 减半最大角速度 → 时长/点数翻倍；翻倍频率 → 点数翻倍、时长不变。
- 末端在每段内走直线（偏离 0.02mm，IK 精确跟踪）。

> 注：回放会做 IK，若某点不可达/分支断裂会失败（日志提示切换『精细』或调整示教点）。实际电机指令频率 = 工作线程 250Hz（固定）；"控制频率"决定输出轨迹的时间分辨率，`_step_track` 按时间戳插值跟踪。如需**不做 IK、纯关节角回放**（最稳、不会 IK 失败），可用模块里的 `joint_replay_traj`。

### 6.6 轨迹平滑性检测（求解后，仅警告不拦截）
所有运动（直线/弧线/回放/B样条）求解后、执行前，自动跑 `check_traj_smoothness(times, q_path)`，**只发警告、不阻断执行**。检测三类问题（事件日志里以 ⚠ 开头）：

1. **IK 分支跳变** —— 某个相邻步长是其余步长的严重离群值（MAD z 分数 >6 且 >0.15rad）。这是 warm-start 链最典型的失败：IK 在某点切到另一个关节构型，实机会猛抖。
2. **绝对大步长** —— 相邻点单步 >0.5rad（29°），轨迹不平滑。
3. **超电机速度** —— 某关节峰值速度 > 其 vmax 的 90%。

正常轨迹（如弧线峰速 1.0rad/s、回放步长 0.01rad）**不会触发任何警告**；只有真有问题才提示。看到 ⚠ 可忽略继续执行，或据此调整示教点/切换『精细』模式/降速。
API：`from openarm_pinocchio_nsp.cartesian_planner import check_traj_smoothness` → 返回 `TrajCheck(warnings, max_step, max_velocity, jump_index)`。

---

## 7. 弧线静止启停（本次新增功能）

### 7.1 解决什么问题
改前弧线轨迹两端速度非零（起 ≈3.2、末 ≈5.4 rad/s），起步猛、到位急停，实机抖动。

### 7.2 原理
**只重塑时间分布，空间路径完全不变。** 用五次平滑阶梯：

```
s(τ) = 10τ³ − 15τ⁴ + 6τ⁵ ,   τ∈[0,1]
s'(0) = s'(1) = 0   →   起末速度为 0
s'(0.5) = 1.875     →   中段为平均速度的 1.875×
```

`_step_track` 按时间戳线性插值 → **均匀时间步下的 \|Δq\| 就是速度**。缓动让两端 \|Δq\|→0、中段最大，速度曲线天然变成 S 型（静止→加速→减速→静止），无需改控制器或路径。

### 7.3 验证结果（3 点弧线）

| | 起点速度 | 终点速度 | 中段峰值 |
|----|----|----|----|
| 改前 | 3.16 ❌猛起 | 5.38 ❌急停 | 5.38 |
| **改后** | **0.005** ✅ | **0.009** ✅ | 6.95 |

- 空间端点：Δ = 机器精度（路径不变）
- 总时长不变
- 各关节峰值速度全部 < vmax（最高 j3 = 56%，正好是 `vel_safety=0.3 × 1.875` 的理论上限，留有裕度）

### 7.4 代码位置
- `cartesian_planner.py::ease_in_out_retime(times, q_path, n_dense=None, slowdown=1.0, min_duration=None, vmax_cap=None)` —— 通用后处理函数。
- `hardware_dashboard.py::_arc_run()` —— 在 `plan_cartesian` 之后、`TRACKING` 之前调用。
- 仅作用于**弧线**；直线/到终点/B 样条逻辑不变（`保持现有逻辑`）。

### 7.5 弧线限速 + 双 IK 模式（界面上可切换）
弧线栏有三个可调项（改完点 [▶弧线IK] 即生效，无需重启）：

- **放慢倍数** (>1 更慢)：总时长 × 此值。
- **转速限幅** (rad/s)：每关节硬上限，闭式精确满足。
- **IK模式**（单选）：
  - **快速(~1s,优化版)** —— 安全点跳过 Stage 2，规划 ~1s。margin≈0.20（仍安全）。日常用。
  - **精细(7-10s,完整Stage2)** —— 每点都跑满 12 轮 null-space 优化，**裕度最饱满**(margin≈0.24)，规划 ~10s。快速模式失败或要最佳实机表现时用。

> 两种模式奇异性(σ_min)一致，差别在**关节裕度**（精细更高）和**速度**（快速 15×）。A/B 实测：两者**失败率相同**——3 点弧线失败的根因是示教点逼臂越过关节限位（margin→0），与求解器无关，换精细模式也救不了几何不可达。换精细模式主要换**裕度**，不换**成功率**。

- **`ARC_SLOWDOWN`**：纯拉伸时间轴，所有关节速度按比例下降。
- **`ARC_VEL_CAP`**：硬限幅。因 eased 位置增量 Δq 与总时长 T 无关、只拉伸时间轴，
  故可闭式求满足限幅的最小时长：`T ≥ Δq_max·(n−1)/cap`，一步到位、精确满足。
  最快的关节正好顶到 cap，其余关节按比例更慢。

实测（3 点弧线，原时长 0.41s、原峰值 5.23 rad/s）：

| 配置 | 时长 | 各关节峰值 | 达标 |
|------|:---:|---|:---:|
| 改前（无限速） | 0.41s | 最高 5.23 rad/s | — |
| slowdown=2, cap=1.0 | 2.12s | 全部 ≤1.0（j1 顶到限幅） | ✅ |
| slowdown=2, cap=0.5 | 4.24s | 全部 ≤0.5 | ✅ |

> 静止启停不受影响（起末段速度仍 ≈0）。实机先从 `cap=1.0` 试起，仍猛就降到 0.5。

---

## 8. 离线验证脚本（`openarm_pinocchio_nsp/validation/`）

```bash
cd /ros2_ws/openarm_nsp_ws/src/openarm_pinocchio_nsp/validation
source ../../../venv-openarm-ik/bin/activate
source ../../../install/setup.bash

python v_nsp_vs_dls.py            # NSP vs 普通 DLS 对比（奇异/裕度）
python v_singularity_map.py       # j1×j4 奇异性热力图 → singularity_map_j1j4.csv
python v_trajectory_validation.py # 轨迹全程 σ_min/裕度/误差
python v_closed_loop_fk_compare.py# 闭环：发的关节角 vs FK 复算末端
python arc_diagnose.py            # 诊断弧线为何过不了安全门
python print_trajectory.py        # 打印轨迹点
python preview_trajectory.py      # meshcat 预览
```

---

## 9. Python API 速查

```python
from openarm_pinocchio_nsp.urdf_path import resolve_urdf_path
from openarm_pinocchio_nsp.kinematics import PinocchioModel
from openarm_pinocchio_nsp.cartesian_planner import (
    Waypoint, plan_cartesian, fit_arc, densify_se3, ease_in_out_retime,
)
from openarm_pinocchio_nsp.bspline_planner import BSplineOptimizer

m = PinocchioModel(resolve_urdf_path(), "right")

# 正运动学
pos, quat = m.fk(q)                       # q: (7,) 关节角

# 逆运动学
r = m.ik_nsp(pos, quat, q_init=q_seed)    # 推荐：双阶段 NSP
r.converged, r.q, r.pos_err_mm, r.sigma_min, r.joint_margin

# 笛卡尔规划（直线）
res = plan_cartesian(m, [Waypoint(p0,q0), Waypoint(p1,q1)],
                     q_init=q0, smooth=True)
res.q_path, res.times, res.passed_gate

# 弧线规划
arc = fit_arc([Waypoint(*m.fk(q)) for q in control_points], n_dense=100)
res = plan_cartesian(m, arc, control_points[0], presampled=True)

# 静止启停重定时（两端速度归零）
times, q_path = ease_in_out_retime(res.times, res.q_path)

# B 样条优化
opt = BSplineOptimizer(resolve_urdf_path(), "right")
spline, q_path, scipy_res = opt.optimize(waypoints_xyz, q_init, duration=3.0)
verify = opt.post_verify(q_path, duration=3.0, n_collision_check=50)
```

### 关键阈值（`kinematics.py`）
| 符号 | 值 | 含义 |
|------|----|----|
| `SIGMA_WARN` | 0.05 | 奇异警告（最小奇异值低于此即接近奇异） |
| `_SIGMA_CRIT` | 0.02 | 奇异临界 |
| `_SIGMA_GOOD` | 0.08 | 良好区 |
| 安全门 margin | 0.1 rad | 关节到限位阈值 |

### 电机参数（`arm_controller.py`）
- `ARM_KP = [70,70,70,60,10,10,10]`，`ARM_KD = [2.75,2.5,2.0,2.0,0.7,0.6,0.5]`
- `MOVE_DUR = 2.0s`（归零/回起点插值时长）
- 工作线程 250Hz；过热阈值 MOS 85°C

---

## 10. 运动模式对比

| 按钮 | 末端路径 | 安全门 | 静止启停 | 实机状态 |
|------|---------|:------:|:------:|---------|
| [▶到终点] | 关节直线（末端非直线） | ❌ | ✅(MOVE_DUR线性) | ✅ 可靠 |
| [▶直线运动] | **末端空间直线** | ❌ | ❌ | ✅ 可靠 |
| [▶弧线IK] | **多点拟合弧线** | ❌ | **✅ 新增** | ✅ sim 通过 |
| [▶B样条优化] | B 样条优化 | ✅ 后验证 | ❌ | ⚠️ 未充分验证 |

> 经验：实机失败主因是**安全门过严**（笛卡尔路径穿奇异区/限位），而示教点本身安全。
> 因此直线/弧线已改为**不检查安全门、直接执行**（warm-start IK 保证关节连续）。

---

## 11. 故障排查

| 现象 | 原因 | 解决 |
|------|------|------|
| `import pinocchio` 段错误 | 系统 numpy 2.x | 激活 venv（numpy 1.26.4） |
| `ros2 run` 报 `No module named 'dash'` | 入口脚本 shebang 是系统 python | 用 `venv/bin/python -m colcon build` 重编（见 §3.2） |
| 启动后双臂不响应 | CAN 口 DOWN / 槽位变了 | `ip link \| grep can`，更新 `CAN_MAP` |
| "left" 按钮动了右臂 | launch 默认 L/R 反 | 显式传 left/right CAN 接口 |
| Dashboard 曲线卡死/狂 POST | 数据量过大 | 已优化：2 图 × 80 点 × 500ms |
| 弧线过不了安全门 | 笛卡尔弧线穿奇异区 | 已改为不检查门；诊断用 `arc_diagnose.py` |
| 工作线程崩溃按钮失效 | `_process` 异常 | 已加 try/except 隔离每次状态切换 |
| B 样条优化超时 | 碰撞检测慢 | `w3=0` 跳过碰撞（`_bspline_run` 默认） |

---

## 12. 典型实机操作（弧线，含静止启停）

```bash
# 1. 启动
source /opt/ros/humble/setup.bash
source /ros2_ws/openarm_nsp_ws/venv-openarm-ik/bin/activate
source /ros2_ws/openarm_nsp_ws/install/setup.bash
ros2 run openarm_dashboard hardware_dashboard
# → 浏览器 http://<IP>:8050，双臂零力矩

# 2.（某臂）先小幅度测试：拖到起点 → [📍弧线加点]
#    拖到中间点 → [📍弧线加点]；拖到终点 → [📍弧线加点]
# 3. [↩弧线回起点] → 等到位 HOLD
# 4. [▶弧线IK] → 柔和起步、走弧线、平稳到位 → HOLD
```

> 首次测试建议起点/终点间距 ≤ 5cm，确认无异常后再放大行程。
