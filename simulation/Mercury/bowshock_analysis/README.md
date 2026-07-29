# Mercury bow-shock analysis

在 MATLAB R2025b 中将此目录设为当前目录，然后运行：

```matlab
mercury_bowshock_app
```

应用默认读取相邻的 `../201312_01s`，不会修改源 MAT 文件。

- 紫色虚线：自动预选（Winslow 平均弓激波面与轨迹的交点）
- 绿色实线：已确认
- 红色点线：已拒绝
- 橙色粗线：当前选择
- 灰色点划线及缺口：周期性仪器/标定型异常被排除；源 MAT 原值仍保留
- `Add / Move`：先选择已有点可移动；未选择时可点击图面新增点
- `Save`：保存 `bowshock_crossings.mat/.csv`
- `Final fit / Export`：只使用 `Confirmed` 点；同一入境/出境轨道段仅保留
  径向距离最大的最外侧点，且至少需要 3 个保留点

## 数据质量规则

- 以 `|B| > 1000 nT` 作为周期性异常触发条件，并向前、向后各扩展
  120 秒；相邻触发段合并。
- 2013 年 12 月共识别 4 个周周期异常区间、160 个触发样本；
  加保护窗后共排除 1,283 个样本。
- 被排除样本仅从图形显示、自动候选评分和最终拟合中舍去，不删除或改写
  原始 MAT 数据。
- 2013-12-27 的 ICME 是科学事件，未触发上述规则，完整保留。
- 详细区间见 `bowshock_exclusions.csv`，完整复核见
  `DATA_QUALITY_REVIEW.md`。

工具栏中的缩放、平移和数据提示可直接用于各磁场面板。初次批量生成的
`page17_bowshock_fit.png` 明确标为 `ProvisionalAuto`；人工复核后点击
`Final fit / Export` 才会生成只含确认点的最终拟合。

运行自动回归检查：

```matlab
test_mercury_bowshock
```
