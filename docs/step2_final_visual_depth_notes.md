# Step2 最终18k：三方图像审阅与深度分桶

2026-11-17，CPU只读审阅完成。Full 与 No-Memory 的完整评测均明确为18000步、126查询；Single-map 使用相同 Full18k 权重，仅首建图帧、无融合、无重训。此处读取现成评测和图片，没有启动GPU、运行额外前向、修改脚本/训练或主报告。

结论：Full相对No-Memory明显改善世界布局、远距深度及四场景参考墙位置；Full相对同权重Single-map主要在中远距和总体世界EPE受益，但近距像素EPE、个别墙厚和位姿并非全部占优。Full自身仍有远墙收缩与参考墙中心偏移，不能把“墙更薄”表述成“世界位置准确”。

## 实际打开的最终图片

以下PNG均通过完整性检查，并实际打开审阅。四张V1均为三行 Full / No-Memory / Single-map，三列为世界点云、固定规范鸟瞰、固定参考墙截面。

- [C5 三方V1](../results/step2_v5_full_seed17/final_comparison/harvard_c5__hv_c5_1_V1_full_vs_nomemory.png)
- [C6 三方V1](../results/step2_v5_full_seed17/final_comparison/harvard_c6__hv_c6_1_V1_full_vs_nomemory.png)
- [C11 三方V1](../results/step2_v5_full_seed17/final_comparison/harvard_c11__hv_c11_2_V1_full_vs_nomemory.png)
- [Robotics 三方V1](../results/step2_v5_full_seed17/final_comparison/harvard_robotics_lab__hv_s1_1_V1_full_vs_nomemory.png)
- [Robotics 三方相机深度散点](../results/step2_v5_full_seed17/final_comparison/robotics_depth_full_vs_nomemory.png)
- [Full / No-Memory 完整验证曲线](../results/step2_v5_full_seed17/final_comparison/full_vs_nomemory_curves.png)

数值来源：[三方深度分桶](../results/step2_v5_full_seed17/final_comparison/robotics_depth_bins.json)、[三方汇总](../results/step2_v5_full_seed17/final_comparison/full_vs_nomemory.json)、两运行各自的 `evaluation_18000_full.json` 与 [Single-map126查询记录](../results/step2_v5_full_seed17/final_comparison/single_map_frame_same_weights.json)。Single-map缓存版本为2，绘图冷灰参考为相同的完整历史RGB-D，只有模型读取限制为首帧。

## 同查询、同参考核查

三方126条 `(scene, query_frame_id, mapping_frame_ids, split)` 按顺序完全一致，均为18000步；两条盲区查询分别保留在三方原始JSON和 `blind_queries.json` 中。四个V1的查询、建图顺序一致，`mapping_wall_p90_p10_m`、参考ROI、参考平面中心/法向、墙评测像素数逐项核对一致。未做任何预测后对齐或去均值。

| 场景 | 查询 | 相同建图帧顺序 | 相同参考ROI xyxy | 截面像素数 |
|---|---:|---|---|---:|
| C5 | 71 | 66, 86, 91 | 0, 0, 56, 42 | 884 |
| C6 | 120 | 105, 90, 134 | 56, 84, 112, 126 | 7831 |
| C11 | 46 | 31, 62, 91 | 0, 0, 56, 42 | 3358 |
| Robotics | 46 | 31, 68, 98 | 168, 42, 224, 84 | 7073 |

## 四场景实际观察

以下宽度是JSON中的固定平面残差p90−p10，单位m；不是中心偏移。中心位置仅从现成直方图估读，未保存逐像素残差，因此不冒称精确均值或中位数。三行图各自保留原坐标轴范围，视觉宽窄应结合轴刻度和数值判断。

| 场景 | 冷灰参考宽度 | Full | No-Memory | Single-map |
|---|---:|---:|---:|---:|
| C5 | 0.1277 | 0.3270 | 0.3803 | 0.3189 |
| C6 | 0.1949 | 0.2077 | 0.3673 | 0.2246 |
| C11 | 0.0663 | 0.2290 | 0.5733 | 0.2884 |
| Robotics | 0.0874 | 0.2022 | 0.3516 | 0.1834 |

**C5，q71。** 三方均可辨房间/桌椅轮廓；Full与Single-map的鸟瞰整体更贴近历史布局，仍有近处弧形表面变形。No-Memory的参考墙截面主体明显移到负侧、约−0.5m量级；Full和Single-map主体更接近零，但仍较参考宽且有尾部。Single-map宽度略小于Full，单查询世界EPE也略低（0.1572对0.1592m）；此例不能作为Full每项都胜出的证据。No-Memory该查询世界EPE为0.2936m。

**C6，q120。** Full和Single-map保留的整体外轮廓与历史参考更接近，No-Memory在远端与边界收缩/偏移更明显。Full的截面主体位于零附近，宽度接近灰色参考但仍有次峰/尾部；Single-map稍宽且正侧分量更明显。No-Memory虽有近零峰，负侧宽尾也很明显，不能以峰接近零代替整面准确。三方单查询世界EPE为0.1499 / 0.4808 / 0.1620m。

**C11，q46。** Full与Single-map能保留更接近历史的房间尺度；No-Memory鸟瞰发生明显收缩和偏移，远边没有覆盖到参考位置。No-Memory墙截面大量分布在正侧、约+0.1至+0.7m区间，既宽又偏；Full的主峰更靠近零且厚尾较弱，Single-map也近零但更宽。Full0.2290m仍显著厚于参考0.0663m，不能表述成墙已恢复真值厚度。三方单查询世界EPE为0.1655 / 0.7051 / 0.1815m。

**Robotics，q46。** 三方差异最明显：参考鸟瞰远墙延伸约9–10m，Full与Single-map大致只到约8m，No-Memory则大幅缩到约4m量级。Full和Single-map的参考墙截面都明显落在零的负侧、约−0.2m量级；No-Memory主体约−0.9m，另有尾部。Single-map墙宽0.1834m比Full0.2022m更薄，但二者中心均错位；Full世界EPE1.0990m仍优于Single-map1.1979m，No-Memory为3.5230m。这个例子直接说明薄不能等同于准，Full也没有消除远墙收缩。

## Robotics：11个固定查询的像素分桶

按GT相机z分为[0.1,3)、[3,5)、[5,10)m，各档三方有效像素数完全相同，为115697、148413、79567。下表均为像素加权值，顺序始终为Full / No-Memory / Single-map，不等同于11个查询等权均值。偏差为最终相机z减GT相机z。

| 深度档 | 世界EPE / m | 世界EPE p90 / m | 最终深度L1 / m | 最终深度偏差 / m |
|---|---|---|---|---|
| 0.1–3m | 0.3430 / 1.0632 / 0.3416 | 0.7812 / 1.6423 / 0.6597 | 0.3444 / 0.7627 / 0.4532 | −0.3371 / −0.7626 / −0.4515 |
| 3–5m | 0.4217 / 1.6431 / 0.4807 | 0.7304 / 2.2887 / 0.8208 | 0.4221 / 1.3777 / 0.5349 | −0.3728 / −1.3777 / −0.4843 |
| 5–10m | 1.2138 / 3.7744 / 1.5080 | 1.9686 / 5.5465 / 2.5098 | 1.1641 / 3.4145 / 1.4979 | −1.1599 / −3.4145 / −1.4954 |

| 深度档 | 原生相机深度L1：Full / No-Memory / Single-map，m |
|---|---|
| 0.1–3m | 0.4816 / 0.7627 / 0.5308 |
| 3–5m | 0.6871 / 1.3777 / 0.7174 |
| 5–10m | 1.9471 / 3.4145 / 2.0978 |

Full的三档世界EPE比No-Memory分别低约67.7%、74.3%、67.8%；相对Single-map，中/远距低约12.3%/19.5%，近距反而略高约0.4%，近距p90也更高。Full的最终深度L1三档均低于Single-map，但不能由相机深度项单独推出每档世界EPE占优。

深度散点与分桶一致：No-Memory的native与final相同，GT8–10m大量预测为约3–4.5m；Full和Single-map的O5锚定将远距预测推向对角线，Full更接近，但GT最远段仍主要低于对角线。Full远距最终深度L1仍1.1641m，偏差−1.1599m，说明主要仍为系统性偏浅。不能将相对No-Memory的大幅改善称为远距重建已经准确。

## 完整验证曲线及其限制

实际审阅的曲线覆盖0、1500、3000、6000、12000、18000步，只画Full与No-Memory两条独立训练的跨场景/场景内曲线；Single-map没有独立训练曲线。

| 查询口径 | 模型 | 世界EPE@3k / 6k / 12k / 18k，m |
|---|---|---|
| 跨场景共视45查询 | Full | 0.3534 / 0.2904 / 0.2984 / 0.2964 |
| 跨场景共视45查询 | No-Memory | 0.8077 / 0.8526 / 0.9220 / 0.8998 |
| 场景内留出共视79查询 | Full | 0.3023 / 0.2855 / 0.2399 / 0.2352 |
| 场景内留出共视79查询 | No-Memory | 0.5955 / 0.6271 / 0.6705 / 0.6573 |

Full早期迅速降低世界误差，跨场景在6k之后约0.29–0.30m附近平台化；场景内留出继续改善。No-Memory在3k之后世界EPE没有持续下降，最终仍显著高于Full；不能把18k预算本身当作收敛充分的证据。Full旋转误差曲线也整体较低，但位姿指标并非全部占优：18k场景内留出的平均平移误差Full0.3218m、No-Memory0.2941m；跨场景则为0.2084/0.2677m。

最终查询等权世界EPE（Full / No-Memory / Single-map）：跨场景45查询0.2964 / 0.8998 / 0.3217m；场景内留出79查询0.2352 / 0.6573 / 0.2634m；Robotics11查询0.5727 / 1.9267 / 0.6644m。这与上述像素分桶属于不同聚合口径，均按原固定划分报告。

## 保留的问题

- Full相对No-Memory和Single-map有明确的总体及中远距世界误差收益，但绝对远距偏浅、远墙收缩和参考墙中心偏移仍存在。
- 细分近距、位姿和单例墙厚存在混合结果；参考墙更薄并不足以证明位置更准或整体布局更准。
- Full跨场景曲线已经平台化；本次只完成既定18k结果审阅，不据此追加训练、删帧、改变划分或启动组件消融。

此前同一Full运行3k→18k的独立诊断仍见 [Full18k深度说明](step2_full18k_depth_notes.md)，其中远距相对3k未改善及T1能量下降未保证GT位姿改善的结论不被本次三方优势替代。
