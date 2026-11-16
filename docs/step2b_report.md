# Step2b：18k等预算核心组件消融（进行中）

2026-11-17：三个消融已实现，正式18k结果尚未完成。原Full、No-Memory与Single-map产物只读，六方汇总输出至`results/step2b_matrix_seed17/`。此处仅记录实际已完成的实现与训练前验证。

## 固定协议与机制边界

- 三组均从官方VGGT全新初始化，原`configs/step2_splits.yaml`、seed17、168×224、AdamW参数组学习率/热身/余弦日程及18k预算不变。正式训练不会读取profile或Full训练参数初始化。
- 从Full正式18k日志重放相同scene/query/map顺序；从其初始Python/CUDA随机状态和已存随机分支恢复世界规范、读入扰动、配对擦除请求/半径。1500/3000/6000/12000/13000/18000的Python/CUDA状态均完全吻合，无需重训Full。增强仅为2.4MiB数据，保存于新协议目录`results/step2b_protocol/augmentation_replay.pt`。
- 配对擦除中心仍由各自当前注意力选择，无法在不同基元分组间复用高斯索引；当前criterion的低共视保护保留。查询仍只有RGB/K，评估G=I、固定45+79及2盲区，无后对齐。

| 变体 | 保留 | 明确移除 |
|---|---|---|
| Render-Only | 各向异性Writer/融合、T0/T1、O5注入/建图自监督/最终局部射线锚定 | O6模块为空、无调用或参数更新 |
| PE-Only | 各向异性Writer/融合、T0/T1、相机系O6高频PE与射线马氏先验 | O5模块为空、无光栅化/锚定/建图光栅损失；O6无渲染深度/opacity先验 |
| Isotropic-Point | 视觉潜特征、中心PE、单尺度点球几何、O5/O6 | Writer无三轴尺度/quat，融合与两级后验均投影为trace/3球；O5法向/O6法向余弦为零、无重根特征分解 |

PE-Only最终相机点严格为raw DPT点，移除锚定专属的额外相机/相对深度监督，保留原生几何与位姿/匹配准则。其余两组保留原v5全部损失。三组EPE均完整报告，不用Full的崩溃EPE阈值提前裁掉弱消融；非有限与活动模块持续断梯度检查保留。

## 已完成验证

训练前`REQUIRE_GSPLAT=1`全部严格测试127项通过；补充六方深度统计后当前全部138项通过。覆盖关闭分支禁止调用、PE无O5的criterion/反传、球Writer/融合/后验严格标量单位阵、无eigh/法向泄露、有限梯度、默认Full兼容、数据重放、调度互斥及六方分桶统计完整性。

三个真实VGGT各12步可丢弃profile已完成，未用于正式初始化；各自覆盖6个真实训练序列，活动Writer、Reader、接收/下游块、DPT、T0/T1均有有限非零梯度，关闭分支明确记为disabled。第12步活动读取通道门控已打开（mean_abs约0.00011–0.00012）。

| 变体 | 实际profile更新 | 峰值保留显存 GiB |
|---|---:|---:|
| Render-Only | 12 | 4.598 |
| PE-Only | 12 | 4.596 |
| Isotropic-Point | 12 | 4.625 |

## 调度与存储

独立变体仅在不同达标空闲卡并行；启动前至少两次利用率/显存检查，最多三组，单进程cap约7.7GiB，2GiB额外预留。只有一张达标卡时顺序运行，不终止其他进程。唯一协调器锁防止重复启动。

根盘仅余约3.8GiB，旧产物全部保留。每500步原子替换各自latest完整状态，18k保留hardlink；共享写锁避免三个tmp同时占盘。完整评测仍保存0/1500/3000/6000/12000/18000；仅减少冗余optimizer里程碑副本，不改变训练或评估。

正式目录为`results/step2_ablation_render_only_seed17/`、`results/step2_ablation_pe_only_seed17/`、`results/step2_ablation_isotropic_point_seed17/`。训练完成后补入六方指标、robotics分桶、曲线、实际V1/V2观察和最终测试/进程检查。Step3尚未启动。

## 资源等待与正式启动

2026-11-16T17:23:46+00:00：正式协调器已启动；连续资源检查仍无满足显存预留且重复低负载的GPU，因此三组正式18k尚未开始。协调器在`results/step2b_scheduler_console.log`持续记录资源并自动等待，条件满足后分配独立卡，最多三组并行。没有终止或信号任何其他进程。六方正式结果和图像审阅尚未完成，不将profile表述为18k结果。

随后自动调度在UTC 17:24 / 17:42 / 17:49分别从官方权重启动Render-Only（GPU6）、PE-Only（GPU1）、Isotropic-Point（GPU3），三组现已并行；每组启动前均再次通过127项严格测试。Render-Only已完成500步检查点保存，峰值保留显存约4.60GiB；该步快速评测中T1仍有退化，保持既定等预算训练，不据此改动消融模型或损失。

## 已实际审阅的补充图像

在独立空闲GPU4上完成Full18k同权重、首建图帧且无融合的Single-map V2，诊断进程已正常退出。四张图位于`results/step2b_matrix_seed17/single_map_frame_vis/`，已逐张打开：C5/C6/C11可见单视锥内稀疏薄椭球及近景局部外壳；robotics远距表面的面元明显更大，近景放大图可见水平与竖直薄面元。RGB表示最薄轴绝对分量，透明度来自alpha；后续与六方V1/V2并排比较。

旧三方分桶缓存缺少世界坐标L1等部分统计，已在另一张实时空闲GPU0上用各自18k权重重新前向固定11个robotics查询，补齐18项分桶指标，进程正常退出。新增缓存只写`results/step2b_matrix_seed17/diagnostics/`，不覆盖旧结果；GT仅在forward之后用于有效像素、近中远分桶及误差计算。六方正式模式要求所有方法的分桶字段齐全且有效像素数相同，当前合并状态仍为incomplete，等待三组消融完成。
