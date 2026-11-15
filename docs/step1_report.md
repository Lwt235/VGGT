# Step 1 实际执行结果 · 2026-11-15

已完成独立初始化、Builder-First 规则、主计划和 Step 1 的实际执行。可微训练链路与三场景过拟合收敛已经跑通；稳定的位姿精化增益、几何选择性注意力和空间局部因果作用尚未建立。未启动 15k–20k 正式长训练或组件消融。V6 留在 Step 3。

## 项目与只读复用

新仓库 `/home/liwt/VGGT` 独立初始化，根提交 `5ae42a4`，实现提交 `dcd9288`，没有继承旧仓库历史。旧仓库仅做了基础设施盘点，没有读取旧实验报告/业务代码或复制旧规则、文档、实验产物，没有修改旧仓库、共享资产、历史或状态文件。未恢复任何历史会话，未终止任何进程，没有系统 GLIBC 改动。GPU 任务均在启动前实时选卡并检查现存作业，单卡分配上限约 20 GiB，任务现已退出。

实际复用路径与状态：

| 资产 | 绝对路径 | 实际验证 |
|---|---|---|
| Python | `/home/liwt/miniColmap/.venv/bin/python` | Python 3.11、PyTorch 2.6.0+cu124、GLIBC 2.27；实际 CUDA 前后向与 gsplat 渲染成功 |
| 备选 Python | `/home/liwt/anaconda3/bin/python` | 成功导入 PyTorch，完成 A6000 最小矩阵运算 |
| gsplat | `/home/liwt/miniColmap/.venv/lib/python3.11/site-packages/gsplat` | 1.5.3，成功导入并完成真实渲染/梯度测试 |
| 官方 VGGT 权重 | `/mnt/f70f6709-366e-49a0-861f-497645d98975/users/liwt/data/GRoPE/gaussian_memory/model.safetensors` | 约 5.0 GB，1797 个张量；使用的 aggregator/point_head 严格匹配加载成功 |
| DINOv2 | 同一 `model.safetensors` 的 `aggregator.patch_embed.*` | 344 个编码器张量已加载，训练时冻结；独立原始 DINOv2 文件未确认 |
| 官方依赖源码 | `/home/liwt/GRoPE/third_party/vggt` | 只读导入与原生 API 对照通过，不使用旧项目业务代码 |
| Sun3D C5 | `/home/liwt/GRoPE/data/round5_sun3d/harvard_c5/hv_c5_1` | 54 组 RGB-D，深度/内外参读取与跨视图检查成功 |
| Sun3D BM1 | `/home/liwt/GRoPE/data/round6_sun3d/brown_bm_1/brown_bm_1` | 7 组 RGB-D，同上 |
| Sun3D C6 | `/home/liwt/GRoPE/data/round5_sun3d/harvard_c6/hv_c6_1` | 17 组 RGB-D，同上 |
| 主线 val | `/home/liwt/GRoPE/data/round5_sun3d/harvard_c11/hv_c11_2` | 52 组 RGB-D，和上述 train 场景分离，本次未做主线验证 |

用户提供的 StreamVGGT 权重文件存在，未加载或用于本轮。ScanNet 未在限定盘点中找到，也不是本轮前置条件。小型补充依赖仅安装到新项目 `.deps`，既有环境没有被修改。字节码、HF/Torch/编译/绘图缓存及全部输出均写入新项目。没有下载或复制大权重、数据集。

## 几何与模块验证

`tests/test_geometry_conventions.py`：**11 项真实通过，最终无跳过**。覆盖官方 VGGT 反投影对照、非平凡 c2w/w2c 互逆、OpenCV 轴向、camera-z 深度、整数/半整数像素中心、14 倍 patch K 变换、gsplat WXYZ、受控世界坐标/深度期望、透明度与遮挡混合、深度方差及实际渲染梯度。gsplat 测试使用符合浮点/光栅特性的容差，未要求任意光栅结果逐位相同。

数据、空间矩匹配、SPD、零门控开启后梯度、活跃读取预算、擦除不修改公共记忆、原生交互层适配和 Pose API 等另 **9 项通过**。先完成 12 步真实完整模型接入检查，再执行下面的正式冒烟。

Sun3D 深度按官方 uint16 右旋 3 位后除 1000 得米制 z；原始 MATLAB 1 起始 K 主点减 1，统一成 OpenCV 整数中心，再转 gsplat 边缘坐标。外参 c2w 按 frame ID 对应，世界 gauge 固定到首个建图相机。三场景初始跨视角有效点的 z 残差中位数约 2.0 / 2.4 / 7.1 cm，包含遮挡和实际测量误差。数据检查图已打开，未把 GT/GT 叠合当作模型 V1。

## 1000 步真实过拟合

运行：`results/smoke_seed17_v1/`。seed=17，输入 112×140，patch 14，O5 训练网格 8×10；三个场景轮换更新 334 / 333 / 333 次。每场景建图索引 `[0,2,4]`，训练查询 `[1,3,5]`，同场景留出查询 `[6]`，不混用。这些索引和主线场景划分已固定在 `configs/splits.yaml`。

实际模型为官方冻结 DINOv2、Writer/空间融合、O5/O6、官方 VGGT 第 4 号 global 接收块、原生 DPT 和两个 Pose 头，约 5045 万可训练参数。其余预训练交互块冻结参数，但保留回传到接收层的计算图。每步从历史训练观测重建可微记忆，返回的记忆只有 `(mu,Sigma,alpha,z)`；查询接口仅接收 RGB、K 和记忆。建图 RGB-D/外参明确作为允许的建图输入，查询真值只进入 loss/evaluator。DPT 相机系输出以**预测** T1 在模型内转成世界点，没有预测后的配准、尺度拟合、去均值或 ICP。

前后 30 步窗口：**Loss 1.812 → 0.198；世界点 EPE 2.168 → 0.281 m**。下表是最终固定权重对每场景全部 3 个训练查询的平均，而非训练过程中不同权重的窗口均值：

| 场景 | World L1 m | World EPE m | R0 → R1 ° | t0 → t1 m |
|---|---:|---:|---:|---:|
| harvard_c5 | 0.066 | 0.136 | 7.07 → 7.11 | 0.008 → 0.011 |
| brown_bm_1 | 0.124 | 0.260 | 6.95 → 6.99 | 0.111 → 0.111 |
| harvard_c6 | 0.060 | 0.131 | 2.39 → 2.32 | 0.039 → 0.043 |

没有 NaN/Inf；最小协方差特征值约 `7.3e-5 m²`，保持正定。门控打开后 Writer、O5、O6、接收块、DPT、两个 Pose 头均出现有限梯度和实际参数更新。第 31–1000 步中 mu/alpha/z 均有梯度；Sigma 与 O6 在 964 步有梯度，另外 6 步均是困难走廊查询的活跃集为空，不是持续断梯度。零门控初期的内部零梯度没有被误判为失效。

走廊第 3193 帧相对首建图相机约转 180°、平移 8.3 m，未被删除。最终固定模型在该训练查询上 EPE 0.485 m、旋转约 15.5°、平移约 0.272 m。第 3330 帧是同场景留出查询，EPE **5.55 m**，重定位仍失败；C5/C6 同场景留出 EPE 为 0.408 / 0.316 m。这些不是跨场景 val 结果。

## 已实际打开的 V1–V5

三个场景共 15 张 V1–V5 和训练曲线已生成并打开查看。初版保留在 `vis_initial/`；最终图在 `vis/`，权重始终相同。V3 显示用同一记忆在 112×140 重渲染以便看清边缘，图中标明训练读取仍为 8×10；显示的是几何 z 和不透明度，不是潜特征伪造的 RGB。

| 场景 | V1 世界叠合/剖面 | V2 椭球 | V3 自举校准 | V4 注意力 | V5 擦除 |
|---|---|---|---|---|---|
| C5 | [查看](/home/liwt/VGGT/results/smoke_seed17_v1/vis/harvard_c5/V1_world_overlap.png) | [查看](/home/liwt/VGGT/results/smoke_seed17_v1/vis/harvard_c5/V2_ellipsoid_shells.png) | [查看](/home/liwt/VGGT/results/smoke_seed17_v1/vis/harvard_c5/V3_bootstrap_calibration.png) | [查看](/home/liwt/VGGT/results/smoke_seed17_v1/vis/harvard_c5/V4_attention_rays.png) | [查看](/home/liwt/VGGT/results/smoke_seed17_v1/vis/harvard_c5/V5_local_erasure.png) |
| BM1 | [查看](/home/liwt/VGGT/results/smoke_seed17_v1/vis/brown_bm_1/V1_world_overlap.png) | [查看](/home/liwt/VGGT/results/smoke_seed17_v1/vis/brown_bm_1/V2_ellipsoid_shells.png) | [查看](/home/liwt/VGGT/results/smoke_seed17_v1/vis/brown_bm_1/V3_bootstrap_calibration.png) | [查看](/home/liwt/VGGT/results/smoke_seed17_v1/vis/brown_bm_1/V4_attention_rays.png) | [查看](/home/liwt/VGGT/results/smoke_seed17_v1/vis/brown_bm_1/V5_local_erasure.png) |
| C6 | [查看](/home/liwt/VGGT/results/smoke_seed17_v1/vis/harvard_c6/V1_world_overlap.png) | [查看](/home/liwt/VGGT/results/smoke_seed17_v1/vis/harvard_c6/V2_ellipsoid_shells.png) | [查看](/home/liwt/VGGT/results/smoke_seed17_v1/vis/harvard_c6/V3_bootstrap_calibration.png) | [查看](/home/liwt/VGGT/results/smoke_seed17_v1/vis/harvard_c6/V4_attention_rays.png) | [查看](/home/liwt/VGGT/results/smoke_seed17_v1/vis/harvard_c6/V5_local_erasure.png) |

实际观察：

- **V1**：桌椅、房间边缘和走廊轮廓已在同一世界系形成对应，但橙色预测仍有弯曲、偏移和厚度。固定参考垂直结构 ROI 的法向 P90–P10，参考/预测分别为 C5 `0.132/0.247 m`、BM1 `0.097/0.173 m`、C6 `0.181/0.434 m`。ROI 在图中和 evaluation.json 标出，仅是该 ROI 的剖面，不当作整个场景的纯墙厚度。
- **V2**：近处细面元、远处大面元和 1.5σ 薄片外壳可见；法向颜色偏集中，多个外壳仍明显重叠。公共基元数为 `198/220/210`，对应所画查询活跃数 `198/220/208`，没有触及 2048 活跃预算；这些数不是流式饱和曲线。
- **V3**：T0/T1 的红绿边缘图几乎相同，部分深度边缘与 RGB 纹理边缘本来就不同，不能把所有错色都解释为位姿错位。固定同权重置空记忆，所画查询 EPE：C5 `0.123→0.151 m`、BM1 `0.160→0.160 m`、C6 `0.127→0.187 m`。这是干预，不是另训的 No-Memory 基线。T1 尚未稳定改善 T0。
- **V4**：实际第 4 层、8 头平均、每条射线 Top-30 椭球已画出。8 头平均分布的归一化熵约 `0.9998`，接近均匀，热力高亮不能解释为已经学会强几何选择性。最终 O5/O6 残差门控约 `-0.0062/-0.00039`，尤其 O6 的输出作用仍弱。
- **V5**：已修正初版走廊框内无有效 GT 表面的问题，最终分别擦除 9/18/15 个高斯，框内有效像素 746/1439/1487。同权重零重训，点变化框内/框外均值约为 `2.14/2.18 mm`、`2.10/4.95 mm`、`2.52/3.30 mm`。变化很小，且没有集中在框内，不能宣称仅局部退化或局部因果已成立。

![真实训练曲线](/home/liwt/VGGT/results/smoke_seed17_v1/vis/training_curves.png)

## 下一步

继续正常架构调优，优先检查保留当前 patch 空间布局的自举位姿摘要、接近 180° 时的旋转损失梯度、O6 几何偏置的物理初始化/尺度与残差梯度比例，以及给 T1 明确的几何校准目标。目标是降低留出走廊重定位误差、形成实际几何选择性并让局部擦除呈现可解释的空间影响。先改完整系统，不开始组件消融，不据本轮结果关闭方法家族。实现与检验后再进入 master_plan 中的 Step 2 主可学习性训练；Step 3 的 Read-then-Write 与 V6 尚未执行。
