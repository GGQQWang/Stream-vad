# Stream-VAD Stage-1 实验手册

## 0. 路径清单（服务器）

| 用途 | 路径 |
|---|---|
| 基础模型 | `/data3/wgq/models/Qwen2-VL-7B-Instruct` |
| 训练标注 | `/data3/wgq/data/HIVAU-70k/raw_annotations/ucf_database_train.json` |
| 训练视频 | `/data1/wjq/data/UCF_Crime/training/videos` |
| 异常视频集（视频级标签来源） | `/data1/wjq/data/UCF_Crime/Anomaly-Videos-ALL` |
| 特征缓存 | `/data3/wgq/data/ucf_original_all_visual_cache_100352`（1900 视频，2.0G） |
| 测试标注 | `/data3/wgq/data/ucf_original_cache_manifests/ucf_original_test_cache_manifest.json` |
| 测试视频 | `/data1/wjq/data/UCF_Crime/testing/videos` |
| 测试 GT（帧级 .txt） | `/data1/wjq/data/UCF_Crime/testing/gt_labels` |
| 输出根目录 | `/data3/wgq/outputs/` |

## 1. 训练

### 1.1 首次训练（从零）

```bash
nohup python -u pipeline_stage1.py \
  --model-path /data3/wgq/models/Qwen2-VL-7B-Instruct \
  --train-json /data3/wgq/data/HIVAU-70k/raw_annotations/ucf_database_train.json \
  --video-root /data1/wjq/data/UCF_Crime/training/videos \
  --anomaly-video-root /data1/wjq/data/UCF_Crime/Anomaly-Videos-ALL \
  --feature-cache-root /data3/wgq/data/ucf_original_all_visual_cache_100352 \
  --min-pixels 100352 --max-pixels 100352 \
  --max-windows 8 \                                  # 实验变量：8 或 32
  --log-dir /data3/wgq/outputs/stage1_<实验名> \
  --epochs 1 \
  --device cuda:<N> \
  > /data3/wgq/outputs/train_<实验名>.log 2>&1 &
```

**启动后必须确认**（标签修复生效的标志）：

```bash
grep "skipped_template_broken" /data3/wgq/outputs/train_<实验名>.log
# 预期: skipped_template_broken=45
```

### 1.2 续跑（resume）

`--epochs` 传**全程总 epoch 数**；resume 从 epoch0 续跑第 2 个 epoch 时传 `--epochs 2`。

```bash
nohup python -u pipeline_stage1.py \
  <同首次训练的参数，一字不差> \
  --epochs 2 \
  --resume /data3/wgq/outputs/stage1_<实验名>/epoch0 \
  --device cuda:<N> \
  > /data3/wgq/outputs/train_<实验名>_epoch2.log 2>&1 &
```

启动日志应打印：`Resumed from epoch 0: ... training epochs 1..1`

### 1.3 进度查看

```bash
tail -f /data3/wgq/outputs/train_<实验名>.log     # 实时日志，Ctrl+C 退出查看
ps aux | grep pipeline_stage1                      # 进程存活
nvidia-smi                                          # GPU 占用
kill -9 <PID>                                       # 终止（换 PID）
```

## 2. 推理测评

### 2.1 带特征缓存（快，~2 小时）

```bash
nohup python -u infer_stage1_ucf.py \
  --model-path /data3/wgq/models/Qwen2-VL-7B-Instruct \
  --stage1-dir /data3/wgq/outputs/stage1_<实验名>/epoch<e> \
  --test-manifest /data3/wgq/data/ucf_original_cache_manifests/ucf_original_test_cache_manifest.json \
  --video-root /data1/wjq/data/UCF_Crime/testing/videos \
  --feature-cache-root /data3/wgq/data/ucf_original_all_visual_cache_100352 \
  --gt-root /data1/wjq/data/UCF_Crime/testing/gt_labels \
  --output-dir /data3/wgq/outputs/infer_<实验名>_epoch<e> \
  --device cuda:<N> \
  > /data3/wgq/outputs/infer_<实验名>_epoch<e>.log 2>&1 &
```

### 2.2 不带缓存（测真实 RTF，含解码+ViT）

去掉 `--feature-cache-root` 一行即可。先单视频快测：

```bash
python -u infer_stage1_ucf.py \
  <同上，去掉 --feature-cache-root> \
  --video-id Abuse028_x264 \
  --output-dir /data3/wgq/outputs/infer_nocache_single \
  --device cuda:<N>
```

### 2.3 结果解读

```bash
cat /data3/wgq/outputs/infer_<实验名>_epoch<e>/metrics.json
```

| 字段 | 含义 |
|---|---|
| `global_standard_auc` | 主表数字（窗口分数铺到窗口内帧，学术通行协议） |
| `global_causal_auc` | 严格因果协议（分数滞后一个窗口 ~1.6s） |
| `rtf_processing` | 模型处理 RTF，**< 1 即流式成立**（带缓存时 ~0.2） |
| `rtf_full` | 含数据加载的 RTF |
| `avg_window_ms` | 单窗口平均处理耗时 |

## 3. 实验框架

### 3.1 已完成 / 进行中

| 实验 | 配置 | 回答的问题 |
|---|---|---|
| baseline（旧） | 脏标签 + mw=8，1 epoch | 84.25 standard AUC（作废，标签脏） |
| clean-mw8 | 干净标签 + mw=8 | 标签修复单独贡献多少 |
| clean-mw32 | 干净标签 + mw=32 | SSM 遗忘梯度不足（惯性）假设 |
| nocache-RTF | 不带缓存推理 | 真实端到端 RTF |

### 3.2 待做

| 实验 | 目的 |
|---|---|
| 裸 VLM baseline（Qwen2-VL-7B 直接打分，同协议） | 坐实 "+X 点" 增益 |
| fps 扫描（2/4/10） | 与 ESOM 的 2fps/4fps 对齐 |
| 消融：无 SSM | "时序建模必要" |
| 消融：λ_sum=0 | "摘要对齐有用" |
| 消融：α=0 | "门控融合必要" |
| 因果 vs 双向（去 chunk detach） | "因果约束不掉点" |
| 世界模型 loss（IBQ 未来预测） | 第三支柱，未实现 |

### 3.3 对比表目标（UCF-Crime，在线严格因果）

| 方法 | AUC | 来源 |
|---|---|---|
| VERA (CVPR'25) | 75.27 | ESOM Table II |
| PANDA (NeurIPS'25) | 82.57 | ESOM Table II |
| MemoVAD (IJCAI'26) | 84.40 | 需确认出处 |
| MoniTor (NeurIPS'25) | 84.69 | 需确认出处 |
| ESOM (arXiv) | 86.18@4fps / 82.13@2fps | ESOM Table II/IV |
| Qwen2-VL-7B 裸跑 | 待测 | 必须补 |
| **Ours** | **84.25（旧）→ 待更新** | — |

## 4. 标签修复说明（2026-08 commit）

- **问题**：HIVAU 异常视频的 `events` 混有正常事件；原代码把异常视频所有 events 全标异常。
- **修复**：逐 event 解析 `events_summary_split[i]["judgement"]`（100% 覆盖实测措辞），否定判定的事件剔除。
- **兜底**：异常视频过滤后零阳性事件 → 整视频剔除（`skipped_template_broken=45`）。
- **效果**：异常事件 1354 个参与训练，1425 个正常事件排除；训练视频 1493 → 1448。

## 5. 坑位备忘

1. **训练参数必须全一致**：resume 时 `--min-pixels/--max-pixels` 等与首训不一致会被 config check 拒绝；换标签定义后禁止中途 resume（必须重训）。
2. **推理脚本 MAX_WINDOWS 硬编码 8**：mw=32 的 checkpoint 直接推理会 config mismatch，需先改推理脚本从 checkpoint 读配置。
3. **旧 checkpoint 作废**：标签修复前的所有 checkpoint（含 84.25 那个）训练标签是脏的，只能当历史参考。
4. **GPU 分配**：训练和推理尽量错开卡（cuda:0 / cuda:1）。
5. **`skipped_template_broken=45` 是正常输出**，不是报错。
