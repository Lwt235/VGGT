# Step 2 18k 最终三方定量补充

正式来源：results/step2_v5_full_seed17/final_comparison/full_vs_nomemory.json；三份逐查询源文件均明确 step=18000，Full/No-Memory/Single-map 的 126 个 scene/query/map-order/split 身份逐项一致。没有使用中期 No-Memory 数据。仅做现有产物的 CPU 分析，无 GPU 前向、模型变更或后对齐。

Full 和独立 No-Memory 各自从官方 VGGT 初始化训练 18000 步；Single-map 是 Full18k 同权重、仅首建图视图且不做空间融合的控制，没有独立训练 18k。Full 的同权重 empty 是干预，不是独立 No-Memory。

固定 benchmark_observable 为 cross-scene 45（C11 34、robotics 11）和 heldout 79；两个零共视查询单列。C11/robotics 是 cross-scene 的子组，不与 cross-scene 重复求总数；原始 3σ 学习后 support 不用于删帧。每格均为查询等权的“均值 / 中位数 / P90”，P90 线性插值；世界指标无任何输出后对齐。

## 四组最终世界几何

| 分组（n） | 方法 | World EPE m | World L1 m |
|---|---|---:|---:|
| cross_scene_val（45） | Full | 0.2964/0.1854/0.5436 | 0.1384/0.0882/0.2551 |
| cross_scene_val（45） | No-Memory | 0.8998/0.6033/1.8512 | 0.4263/0.2880/0.8707 |
| cross_scene_val（45） | Single-map | 0.3217/0.2201/0.5791 | 0.1506/0.1036/0.2744 |
| same_scene_heldout（79） | Full | 0.2352/0.1157/0.3359 | 0.1090/0.0540/0.1584 |
| same_scene_heldout（79） | No-Memory | 0.6573/0.5116/1.3033 | 0.3186/0.2449/0.5979 |
| same_scene_heldout（79） | Single-map | 0.2634/0.1231/0.3720 | 0.1229/0.0592/0.1766 |
| C11（34） | Full | 0.2070/0.1675/0.3373 | 0.0977/0.0803/0.1606 |
| C11（34） | No-Memory | 0.5676/0.5596/0.7672 | 0.2751/0.2691/0.3661 |
| C11（34） | Single-map | 0.2109/0.1660/0.3275 | 0.0989/0.0773/0.1556 |
| robotics（11） | Full | 0.5727/0.4364/0.9070 | 0.2642/0.1969/0.4417 |
| robotics（11） | No-Memory | 1.9267/1.7320/2.7217 | 0.8936/0.8518/1.3532 |
| robotics（11） | Single-map | 0.6644/0.5320/1.1979 | 0.3103/0.2528/0.5498 |

World L1 是有效像素上逐坐标绝对差的均值；World EPE 是逐像素三维欧氏误差的均值，二者不能互换。

| 分组（n） | 方法 | T0 旋转 ° | T1 旋转 ° | T0 平移 m | T1 平移 m |
|---|---|---:|---:|---:|---:|
| cross_scene_val（45） | Full | 2.9334/1.8376/3.6579 | 3.2542/2.4048/4.3595 | 0.1949/0.1273/0.3035 | 0.2084/0.1596/0.2941 |
| cross_scene_val（45） | No-Memory | 11.8486/8.7591/21.2498 | 11.8486/8.7591/21.2498 | 0.2677/0.1502/0.4263 | 0.2677/0.1502/0.4263 |
| cross_scene_val（45） | Single-map | 2.7025/1.8185/4.3473 | 2.8238/1.9127/3.9685 | 0.1974/0.1422/0.2835 | 0.1968/0.1331/0.3026 |
| same_scene_heldout（79） | Full | 5.8103/1.7260/5.2115 | 5.7263/1.8380/4.4262 | 0.3262/0.0701/0.3071 | 0.3218/0.0734/0.2540 |
| same_scene_heldout（79） | No-Memory | 13.6762/10.1307/24.4386 | 13.6762/10.1307/24.4386 | 0.2941/0.1742/0.5517 | 0.2941/0.1742/0.5517 |
| same_scene_heldout（79） | Single-map | 7.4855/1.7901/5.8214 | 7.3641/1.7569/4.6991 | 0.3597/0.0804/0.4629 | 0.3594/0.0791/0.4113 |
| C11（34） | Full | 2.2799/1.9043/3.6656 | 2.5771/2.6317/4.4375 | 0.1507/0.1213/0.2949 | 0.1592/0.1242/0.2886 |
| C11（34） | No-Memory | 11.3965/10.8499/21.4333 | 11.3965/10.8499/21.4333 | 0.2214/0.1524/0.4276 | 0.2214/0.1524/0.4276 |
| C11（34） | Single-map | 2.1772/1.8257/3.6601 | 2.2366/1.9003/3.9436 | 0.1363/0.1033/0.2172 | 0.1346/0.1106/0.2295 |
| robotics（11） | Full | 4.9533/1.5952/2.7444 | 5.3471/2.0045/3.2088 | 0.3313/0.1891/0.4216 | 0.3604/0.1967/0.3924 |
| robotics（11） | No-Memory | 13.2459/6.6978/16.1208 | 13.2459/6.6978/16.1208 | 0.4108/0.1502/0.4148 | 0.4108/0.1502/0.4148 |
| robotics（11） | Single-map | 4.3262/1.7900/4.6356 | 4.6389/2.0940/4.6704 | 0.3863/0.2581/0.5734 | 0.3888/0.2501/0.5300 |

Full 的逐查询 T0→T1 正负向计数、分方向幅度、V5 有符号及绝对变化另见 docs/step2_full18k_diagnostics.md。No-Memory 没有 T1 精化，T0=T1。

## 成对世界 EPE：同一查询上有多少真正改善

Δ=Full−对照，负值代表 Full 较好。下表胜/负/平直接比较已存 EPE；相对下降只按组内均值计算，不代表每个查询改善。

| 分组 | 对照 | Full 胜/负/平 | 成对 Δ m（均值/中位/P90） | Full 相对均值下降 |
|---|---|---:|---:|---:|
| cross_scene_val | No-Memory | 45/0/0 | -0.6034/-0.4246/-0.1735 | 67.06% |
| cross_scene_val | Single-map | 27/18/0 | -0.0253/-0.0118/0.0251 | 7.87% |
| same_scene_heldout | No-Memory | 77/2/0 | -0.4220/-0.4006/-0.1392 | 64.21% |
| same_scene_heldout | Single-map | 55/24/0 | -0.0281/-0.0087/0.0177 | 10.68% |
| C11 | No-Memory | 34/0/0 | -0.3606/-0.3608/-0.1174 | 63.53% |
| C11 | Single-map | 18/16/0 | -0.0038/-0.0046/0.0251 | 1.82% |
| robotics | No-Memory | 11/0/0 | -1.3540/-1.2300/-0.9655 | 70.27% |
| robotics | Single-map | 9/2/0 | -0.0917/-0.0989/0.0110 | 13.80% |

四组各方法最差的 3 个查询（按最终世界 EPE 排序），保留原始长尾：

| 分组 | 方法 | 场景/查询帧 | World EPE m | T1 旋转 ° | T1 平移 m |
|---|---|---|---:|---:|---:|
| cross_scene_val | Full | harvard_robotics_lab/46 | 1.0990 | 2.877 | 0.1684 |
| cross_scene_val | Full | harvard_robotics_lab/53 | 0.9070 | 1.807 | 0.1140 |
| cross_scene_val | Full | harvard_robotics_lab/362 | 0.8931 | 38.042 | 1.8810 |
| cross_scene_val | No-Memory | harvard_robotics_lab/46 | 3.5230 | 6.698 | 0.1640 |
| cross_scene_val | No-Memory | harvard_robotics_lab/53 | 2.7217 | 11.633 | 0.1312 |
| cross_scene_val | No-Memory | harvard_robotics_lab/91 | 2.1239 | 5.929 | 0.0372 |
| cross_scene_val | Single-map | harvard_robotics_lab/53 | 1.3247 | 1.540 | 0.2083 |
| cross_scene_val | Single-map | harvard_robotics_lab/46 | 1.1979 | 2.778 | 0.1910 |
| cross_scene_val | Single-map | harvard_robotics_lab/362 | 0.7224 | 27.842 | 1.7362 |
| same_scene_heldout | Full | brown_cogsci_1/332 | 3.1118 | 80.108 | 5.7734 |
| same_scene_heldout | Full | harvard_c8/859 | 2.6509 | 156.177 | 7.6407 |
| same_scene_heldout | Full | harvard_c5/1216 | 1.3681 | 53.911 | 3.7625 |
| same_scene_heldout | No-Memory | brown_cogsci_1/332 | 2.9679 | 59.037 | 0.7953 |
| same_scene_heldout | No-Memory | harvard_c5/1216 | 2.4790 | 89.732 | 5.0220 |
| same_scene_heldout | No-Memory | harvard_c8/859 | 2.0999 | 33.893 | 0.2429 |
| same_scene_heldout | Single-map | brown_cogsci_1/332 | 3.1075 | 78.405 | 6.0580 |
| same_scene_heldout | Single-map | harvard_c5/1216 | 2.7667 | 169.302 | 5.0289 |
| same_scene_heldout | Single-map | harvard_c8/859 | 2.6508 | 156.160 | 7.6408 |
| C11 | Full | harvard_c11/781 | 0.5836 | 2.701 | 0.4506 |
| C11 | Full | harvard_c11/765 | 0.4673 | 0.957 | 0.0591 |
| C11 | Full | harvard_c11/414 | 0.3528 | 5.370 | 0.2886 |
| C11 | No-Memory | harvard_c11/781 | 1.4146 | 20.148 | 0.2837 |
| C11 | No-Memory | harvard_c11/173 | 0.8156 | 24.738 | 0.1654 |
| C11 | No-Memory | harvard_c11/717 | 0.8033 | 20.026 | 0.2308 |
| C11 | Single-map | harvard_c11/781 | 0.5720 | 3.536 | 0.4507 |
| C11 | Single-map | harvard_c11/765 | 0.4878 | 0.985 | 0.0757 |
| C11 | Single-map | harvard_c11/414 | 0.3562 | 4.653 | 0.2408 |
| robotics | Full | harvard_robotics_lab/46 | 1.0990 | 2.877 | 0.1684 |
| robotics | Full | harvard_robotics_lab/53 | 0.9070 | 1.807 | 0.1140 |
| robotics | Full | harvard_robotics_lab/362 | 0.8931 | 38.042 | 1.8810 |
| robotics | No-Memory | harvard_robotics_lab/46 | 3.5230 | 6.698 | 0.1640 |
| robotics | No-Memory | harvard_robotics_lab/53 | 2.7217 | 11.633 | 0.1312 |
| robotics | No-Memory | harvard_robotics_lab/91 | 2.1239 | 5.929 | 0.0372 |
| robotics | Single-map | harvard_robotics_lab/53 | 1.3247 | 1.540 | 0.2083 |
| robotics | Single-map | harvard_robotics_lab/46 | 1.1979 | 2.778 | 0.1910 |
| robotics | Single-map | harvard_robotics_lab/362 | 0.7224 | 27.842 | 1.7362 |

Full 相比 Single-map 最负向的 5 个有共视查询（世界 EPE 增量从大到小）：

| 场景/查询帧 | Full EPE m | Single-map EPE m | Full−Single-map m |
|---|---:|---:|---:|
| harvard_robotics_lab/362 | 0.8931 | 0.7224 | 0.1707 |
| harvard_c6/282 | 0.3341 | 0.2567 | 0.0775 |
| harvard_c11/366 | 0.3336 | 0.2763 | 0.0573 |
| mit_w59_conference/272 | 0.3428 | 0.2894 | 0.0535 |
| mit_lab_hj/1192 | 0.1835 | 0.1466 | 0.0370 |

## 相机几何与两个盲区

camera_epe_m 是锚定前 DPT 原生相机点的误差；Full/Single-map 的 anchored_camera_epe_m 是最终锚定相机点误差。Full/Single-map 的该 DPT 分支可受记忆注入影响，不能将其称为注入前几何。两者同时列出，防止将世界规范失败与相机系几何质量混为一谈。No-Memory 没有锚定分支，以其原生相机几何为最终相机几何。

| 分组 | 方法 | 原生相机 EPE m | 最终相机 EPE m |
|---|---|---:|---:|
| cross_scene_val | Full | 0.4096/0.2684/0.8980 | 0.2966/0.2071/0.6324 |
| cross_scene_val | No-Memory | 0.6732/0.3452/1.5264 | 0.6732/0.3452/1.5264 |
| cross_scene_val | Single-map | 0.4207/0.2348/0.9788 | 0.3394/0.2251/0.7673 |
| same_scene_heldout | Full | 0.1864/0.1235/0.3093 | 0.1739/0.1168/0.2922 |
| same_scene_heldout | No-Memory | 0.2804/0.2559/0.4346 | 0.2804/0.2559/0.4346 |
| same_scene_heldout | Single-map | 0.1751/0.1352/0.3291 | 0.1593/0.1193/0.3089 |
| C11 | Full | 0.2327/0.1980/0.3519 | 0.1984/0.1741/0.3176 |
| C11 | No-Memory | 0.3311/0.3245/0.4687 | 0.3311/0.3245/0.4687 |
| C11 | Single-map | 0.2258/0.1899/0.3394 | 0.2002/0.1609/0.2996 |
| robotics | Full | 0.9563/0.8674/1.6029 | 0.6000/0.5391/0.9209 |
| robotics | No-Memory | 1.7305/1.4705/2.6138 | 1.7305/1.4705/2.6138 |
| robotics | Single-map | 1.0233/0.9709/1.8194 | 0.7696/0.7091/1.2813 |

| 盲区场景/帧 | 方法 | World EPE m | World L1 m | T1旋转 ° | T1平移 m | 原生相机 EPE m | 最终相机 EPE m |
|---|---|---:|---:|---:|---:|---:|---:|
| brown_bm_1/3330 | Full | 4.7652 | 2.4960 | 89.114 | 7.5730 | 0.8244 | 0.8485 |
| brown_bm_1/3330 | No-Memory | 5.6768 | 2.9754 | 49.790 | 7.2223 | 0.7251 | 0.7251 |
| brown_bm_1/3330 | Single-map | 4.5965 | 2.4269 | 80.825 | 6.9667 | 0.8420 | 0.7908 |
| home_at/13507 | Full | 7.3570 | 3.2399 | 24.014 | 8.3423 | 0.9238 | 1.0021 |
| home_at/13507 | No-Memory | 9.2326 | 3.9386 | 28.375 | 9.9361 | 0.7543 | 0.7543 |
| home_at/13507 | Single-map | 7.1472 | 3.2332 | 30.461 | 8.4104 | 1.1570 | 1.2572 |

两个盲区没有进入 45+79 汇总。Full 的低 3σ 支持与 world_supervision_enabled=0 是 criterion 保护，不能当成推理重定位成功；盲区能输出相机几何也不等价于确定相对首建图相机的世界位姿。

## 两条正式训练日志的一次序列核验

仅检查两个正式 metrics.jsonl，不混入 Full 的中断备份日志；没有重跑预测。

| 日志 | 总记录数 | 唯一步数 | 范围 | 1..18000 各一次 |
|---|---:|---:|---|---|
| Full | 18000 | 18000 | 1..18000 | 通过 |
| No-Memory | 18000 | 18000 | 1..18000 | 通过 |

逐步检查 step、scene、query_frame_id、mapping_frame_ids（含顺序），不一致步数：0。两条 18000 步 episode 序列一致。

各自首 100 步=1..100、末 100 步=17901..18000；每格为均值 / 中位数 / P90。Full 与 No-Memory 的 loss 包含不同的监督项和权重，绝对 loss 不用于横向优劣判断；仅观察各自训练变化。

| 方法 | 区间 | 总 loss | 训练世界 EPE m |
|---|---|---:|---:|
| Full | 首100 | 27.9515/22.2706/44.9324 | 0.8242/0.7350/1.2705 |
| Full | 末100 | 7.4291/6.3985/11.5498 | 0.1362/0.1143/0.2544 |
| No-Memory | 首100 | 2.2641/2.0337/3.5905 | 1.9353/1.7176/3.3549 |
| No-Memory | 末100 | 1.5038/1.0963/3.2302 | 1.0150/0.8225/1.8975 |

## 规范可辨识限制与仍需解决的问题

No-Memory.forward 只接收 RGB/K，训练/评估的标签世界系由首建图相机定义。模型没有该参考图像或其位姿输入，因此仅靠当前查询图像，通常不能唯一推断相对于未观测参考相机的 SE(3)；固定 G=I 固定的是监督规范，没有将参考信息注入查询。该比较量化的是“拥有公共历史参考信息的完整系统”相对“无参考输入单帧模型”的总体收益，不能仅凭它分离 Gaussian 融合、O5/O6 各组件的收益。

正式 18000 步日志中共有 331 个不同 scene/query 组合；其中 331 个组合重复出现且对应超过一个首建图帧 ID。首建图帧身份变化的最多数量为 28。相同 RGB/K 可对应不同参考世界系，这为上述条件不可辨识限制提供了 episode 级证据；这里只核验身份，没有用 GT 位姿重算差异。

Single-map 有参考视图，因而比无参考的 No-Memory 更直接检验多帧记忆对单视图地图的增量，但它仍是 Full 权重下的输入控制，并非同预算单独优化的架构基线。其胜负必须按同查询分布报告，不能因为 Full 远胜 No-Memory 就忽略 Full 逊于 Single-map 的查询。

仍需解决的定量问题很具体：Full 在 C11 的平均世界 EPE 仅比 Single-map 低 3.8 mm（1.82%），中位数 0.1675 vs 0.1660 m、P90 0.3373 vs 0.3275 m 反而更高；Full 的 C11 最终位姿均值也较差（2.5771°/0.1592 m vs 2.2366°/0.1346 m）。Full 在 robotics/362 的世界 EPE 比 Single-map 高 0.1707 m，且仍有 38.042°/1.8810 m 的位姿失败。heldout 上 Full 的平均平移误差 0.3218 m 也高于 No-Memory 的 0.2941 m，尽管中位数和 P90 更优；均值受到原始大姿态尾部拉高，不能只报告有利的中位数。

单 seed17 的这些均值和成对查询统计不构成跨 seed 稳健性结论。Full 的尾部姿态失败、T1 并非稳定改善、V5 的净负例及盲区世界重定位失败均仍存在；不能将 LM 内部能量下降解释为 GT 位姿必改善。实际 V1–V5 和远距图像审阅由主任务独立记录，本文件只给定量证据。
