# Terminology

One row per term: English form, Chinese form used in the Chinese guides, and the
document that owns the definition. Read the owning document before using a term;
do not restate the definition in another file.

Vocabulary here is descriptive. Behavioural rules live in
[ARCHITECTURE.md](../ARCHITECTURE.md); document structure in
[writing-standard](writing-standard.md).

## 1. Time and observation identity

| English | 中文 | Defined in |
| --- | --- | --- |
| observed UTC | 观测 UTC | [time_matching](time_matching.md) |
| time scale (UTC / TAI) | 时标（UTC / TAI） | [time_matching](time_matching.md) |
| layer (stream) | 层（数据流） | [time_matching](time_matching.md) |
| root master layer | 根主层 | [time_matching](time_matching.md) |
| master timeline | 主时间轴 | [time_matching](time_matching.md) |
| coverage | 覆盖 | [time_matching](time_matching.md) |
| effective matching clock | 有效匹配时钟 | [time_matching](time_matching.md) |
| coverage-forced edge | 覆盖强制边 | [time_matching](time_matching.md) |
| tolerance as a quality flag | 容差即质量标记 | [time_matching](time_matching.md) |
| validity mask | 有效掩膜 | [foundations](foundations.md) |

## 2. Data, units and coordinates

| English | 中文 | Defined in |
| --- | --- | --- |
| helioprojective WCS (HPLN/HPLT) | 日面投影坐标（HPLN/HPLT） | [foundations](foundations.md) |
| native pixel grid | 原始像素网格 | [radio_comparison](radio_comparison.md) |
| count rate vs DN | 计数率与 DN | [foundations](foundations.md) |
| SFU (solar flux unit) | 太阳流量单位 | [radio_comparison](radio_comparison.md) |
| circular polarization RR / LL | 圆偏振 RR / LL | [radio_comparison](radio_comparison.md) |
| display normalization | 显示归一化 | [aia_radio_composite](aia_radio_composite.md) |
| apparent solar radius | 视太阳半径 | [foundations](foundations.md) |

## 3. Fitting and uncertainty

| English | 中文 | Defined in |
| --- | --- | --- |
| elliptical Gaussian model | 椭圆高斯模型 | [radio_comparison](radio_comparison.md) |
| centroid | 质心 | [radio_comparison](radio_comparison.md) |
| parameter covariance | 参数协方差 | [radio_comparison](radio_comparison.md) |
| observed apparent source size | 观测视源尺寸 | [radio_comparison](radio_comparison.md) |
| intrinsic source size | 固有源尺寸 | [radio_comparison](radio_comparison.md) |
| beam deconvolution | 波束反卷积 | [radio_comparison](radio_comparison.md) |
| review flag | 复核标记 | [radio_comparison](radio_comparison.md) |

## 4. Models

| English | 中文 | Defined in |
| --- | --- | --- |
| PFSS frequency band | PFSS 频带 | [magnetic_models](magnetic_models.md) |
| global radial boundary | 全球径向边界 | [magnetic_models](magnetic_models.md) |
| SHARP CEA map | SHARP CEA 图 | [magnetic_models](magnetic_models.md) |
| local NLFFF grid | 局部 NLFFF 网格 | [magnetic_models](magnetic_models.md) |
| source-surface height | 源表面高度 | [magnetic_models](magnetic_models.md) |

## 5. Runtime and provenance

| English | 中文 | Defined in |
| --- | --- | --- |
| runtime root | 运行根目录 | [ARCHITECTURE](../ARCHITECTURE.md) |
| allowed roots | 允许根目录 | [Apps guide](../Apps/README.md) |
| project document (`.spapp.json`) | 项目文档 | [Apps guide](../Apps/README.md) |
| workflow document (`.spflow.json`) | 流程文档 | [Apps guide](../Apps/README.md) |
| evidence class `[observed]` / `[implemented]` / `[inferred]` | 证据分级 `[观测]` / `[实现]` / `[推断]` | [writing-standard](writing-standard.md) |
| migration record | 迁移记录 | [migration](migration.md) |

## 6. Writing rules

- Introduce a term in its owning document; when the document is Chinese, give the
  English form in parentheses at first use.
- Do not invent a second English phrasing for one concept.
- A new term needs a row here in the same change that introduces it.
