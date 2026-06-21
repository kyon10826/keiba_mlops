# keiba_mlops

JRA 中央競馬の予測 ML パイプライン。
**LightGBM ×2 ヘッド (複勝 + 穴馬) + Optuna 最適化 + Isotonic キャリブレーション + Harville モデル**で勝率/連対率を推定し、**フラクショナル・ケリー基準 + 1 レース予算上限**で実額 (円) の賭け方を提案する。

```
record CSV  ──▶  特徴量パイプライン ──▶  LightGBM 複勝モデル ─┐
              │                                              ├─▶ Harville (P(1着,2着,3着))
              └─▶  Time_Series_Odds ──▶  LightGBM 穴馬モデル ─┘
                                                              │
                                                  ┌───────────┴───────────┐
                                                  ▼                       ▼
                                            ケリー + 円表記           EV ≥ 1.0 フィルタ
                                            (per_race_cap)          (品質ガード + 控除考慮)
                                                  │                       │
                                                  └─▶ recommender ◀──────┘
                                                          │
                                                          ▼
                                                  単勝/複勝/三連複/三連単/穴馬の推奨
```

---

## 1. 概要

| 項目 | 内容 |
|---|---|
| **対象** | JRA 中央競馬 (10 競馬場・芝/ダート) |
| **予測単位** | 馬 × レース |
| **基本ターゲット** | 複勝 (`rank ∈ {1,2,3}`) の二値分類 |
| **追加ターゲット** | 穴馬 (`rank == 1 AND pop ≥ 5`) の二値分類 |
| **モデル** | LightGBM × 2 ヘッド (複勝 + 穴馬) |
| **HPO** | Optuna (Brier 最小化 / Average Precision 最大化) |
| **キャリブレーション** | Isotonic Regression (ホールドアウト) |
| **多券種推論** | Harville モデル (1 着→2 着→3 着の条件付き確率) |
| **賭け金サイジング** | Kelly + per_race_cap (円ベース) / Tier (固定額) |
| **馬券種** | 単勝・複勝・三連複・三連単・穴馬 (単勝/複勝) |
| **データ最古** | 1986 年 (record_data CSV) / 2003 年 (JD 時系列オッズ) |

---

## 2. ディレクトリ構造

```
keiba_mlops/
├── README.md                    # ← この文書
├── requirements.txt             # Python 依存
├── .gitignore                   # data/, models*/, cache/ を除外
│
├── config/                      # YAML 設定
│   ├── default.yaml             # 本番設定 (1986-2024 学習)
│   ├── smoke_anaba.yaml         # 動作確認用 (2022-2023 学習)
│   └── backtest_nov2025.yaml    # バックテスト用
│
├── src/
│   ├── data/                    # データ読み込み & スキーマ
│   │   ├── schema.py            # CSV 47 列の定義 + 場/競走種別マッピング
│   │   ├── loader.py            # 年別 CSV 結合 + train/valid/test 分割
│   │   ├── time_series_odds.py  # 時系列オッズ JD CSV パーサ + 馬×レース集約 + 年単位キャッシュ
│   │   ├── data/                # ★ 生 CSV (record_data_YYYY.csv) — gitignore
│   │   └── Time_Series_Odds/    # ★ JD/JR/JW CSV — gitignore (~16GB)
│   │
│   ├── features/                # 特徴量エンジニアリング
│   │   ├── race.py              # コース・距離・馬場・天候・クラス (8 列)
│   │   ├── horse.py             # 馬のローリング成績 + コールドスタート補完 (11 列)
│   │   ├── jockey.py            # 騎手のベイズ統計 (LCB95 など 5 列)
│   │   ├── sire.py              # 父馬のベイズ統計 + 芝/ダート別 (5 列)
│   │   ├── relative.py          # レース内 z-score / odds_rank (3 列)
│   │   ├── pipeline.py          # FeaturePipeline オーケストレータ (FEATURE_COLUMNS = 35 列)
│   │   └── odds_timeseries.py   # 時系列オッズ特徴量 (TS_ODDS_FEATURE_COLUMNS = 20 列)
│   │
│   ├── model/                   # モデル学習・推論・校正
│   │   ├── trainer.py           # 複勝モデル (LightGBM + Optuna, Brier 最適化)
│   │   ├── anaba_trainer.py     # 穴馬モデル (LightGBM + Optuna, AP 最適化, scale_pos_weight)
│   │   ├── calibrator.py        # Isotonic Regression キャリブレータ
│   │   ├── ensemble.py          # コールドスタート対策のオッズ-モデル確率ブレンド
│   │   └── evaluator.py         # AUC/Brier/LogLoss/閾値別 hit-rate + 可視化
│   │
│   ├── strategy/                # 推奨ロジック・サイジング・シミュレーション
│   │   ├── harville.py          # 1着→2着→3着の Harville 条件付き確率
│   │   ├── kelly.py             # ケリー基準 + per_race_cap + 確率比按分
│   │   ├── selector.py          # 閾値方式 / EV 方式の選定ロジック
│   │   ├── recommender.py       # 単勝/複勝/三連複/三連単/穴馬の統合推奨 + EV フィルタ
│   │   ├── algorithm.py         # 総合評価アルゴリズム (能力評価 + 割引 + ボーナス)
│   │   └── simulator.py         # バックテスト・シミュレータ
│   │
│   ├── scraper/                 # netkeiba スクレイピング
│   │   ├── race_card.py         # 出馬表 + 血統情報
│   │   ├── odds.py              # リアルタイム単複/三連複/三連単オッズ
│   │   └── results.py           # 過去レース結果 (バックテスト用)
│   │
│   └── api/                     # 当日実行用ラッパ
│       ├── client.py            # スクレイパー集約クライアント
│       └── runner.py            # 当日全レース推論ループ
│
├── scripts/                     # エントリーポイント (CLI)
│   ├── train.py                 # 学習 (複勝 + 穴馬 を順次)
│   ├── predict.py               # 当日推論 + 推奨
│   ├── backtest.py              # 過去データでの ROI 検証
│   └── scrape_results.py        # 結果データ収集
│
├── models_smoke/                # ★ スモーク学習成果物 (gitignore)
│   ├── lgbm_model.txt           # 複勝モデル (LightGBM 文字列形式)
│   ├── anaba_model.txt          # 穴馬モデル
│   ├── pipeline.pkl             # FeaturePipeline (騎手/父統計量含む)
│   ├── calibrator.pkl           # Isotonic Regression
│   ├── anaba_meta.pkl           # 穴馬モデルのメタ (feature_columns 58 列)
│   ├── jockey_stats.csv         # 騎手の rolling 統計量
│   ├── sire_stats.csv           # 父馬の rolling 統計量
│   ├── calibration_curve.png    # 校正前後の信頼度プロット
│   └── feature_importance.png   # gain importance Top
│
├── models_newfeat/              # ★ 本番学習成果物 (gitignore)
│   └── (同上、穴馬モデル未学習)
│
└── cache/                       # ★ Time_Series_Odds の年単位 parquet (gitignore)
    └── ts_odds/
        ├── jd_2022_agg.parquet
        ├── jd_2023_agg.parquet
        └── jd_2024_agg.parquet
```

★ = git 管理外 (`.gitignore`)。

---

## 3. データ仕様

### 3.1 メインデータ (`record_data_YYYY.csv`)

[src/data/data/](src/data/data/) に年単位で配置。**Shift-JIS / カラム名なし / 47 列固定**。  
完全な定義は [src/data/schema.py](src/data/schema.py:6-54) (`COLUMN_NAMES`)。

| カラム | 型 | 内容 |
|---|---|---|
| `race_id` | int64 (18桁) | レース ID + horse_num の連結 (`YYYYMMDDLLTTDDRRHH`) |
| `year/month/day` | int | 開催日 |
| `place` | str | 競馬場 (10 場) |
| `times/daily/race_num` | int | 開催回・日・R |
| `horse` | str | 馬名 |
| `jockey_id` | int | 騎手コード |
| `horse_N` | int | 出走頭数 |
| `waku_num/horse_num` | int | 枠番/馬番 |
| `track_code/state/weather` | int/str | コース/馬場/天候 |
| `class_code/age_code` | int | クラス・年齢限定 |
| `sex/age/basis_weight` | str/num | 性別/年齢/斤量 |
| `weight/inc_dec` | num | 馬体重/増減 |
| `win_odds/pop` | num | **確定オッズ・人気** |
| `rank` | int | **着順 (ターゲット)** |
| `time_diff/time` | str/num | 着差/走破タイム |
| `corner1-4_rank` | int | 通過順 |
| `last_3F_time/rank/Ave_3F/PCI` | num | 上がり 3F 関連 |
| `leg` | str | 脚質 |
| `prize` | num | 賞金 |
| `error_code` | int | 異常 (1=取消, 3=除外 など) |
| `father/mother/id` | str/int | 血統情報 |

### 3.2 時系列オッズ (`Time_Series_Odds/JD*.CSV`)

[src/data/Time_Series_Odds/](src/data/Time_Series_Odds/) に **1 レース 1 ファイル**で配置。**~22 万ファイル / ~16GB**。

#### ファイル名規則
`JD[place 2][year 2][times×10+daily][race_num].CSV` (例: `JD05241101.CSV` = 東京 2024年 1回1日 1R)

#### CSV 構造 (Shift-JIS, 1 ヘッダ行 + N スナップショット行)
```
レースID,区分,月日時分,頭数,単勝票数,複勝票数,1単,1複Lo,1複Hi,...,N単,N複Lo,N複Hi
2024072001010101,1,07181500,16,1234,2345,3.5,1.2,1.4,...
2024072001010101,1,07181508,16,1245,2367,3.4,1.2,1.4,...
2024072001010101,3,07200905,16,1567,2890,3.2,1.1,1.3,...
2024072001010101,4,07201105,16,1789,3123,3.1,1.1,1.3,...
```

- **区分** = フェーズ (1=前日, 3=当日朝, 4=直前) の大分類
- **月日時分** = MMDDHHMM 形式の取得時刻
- **N 単 / N 複Lo / N 複Hi** = 馬番 N の単勝オッズ / 複勝下限 / 複勝上限
- 1 レースあたり典型 50-170 スナップショット
- レース ID は **16 桁** (record_data の 18 桁から末尾 `horse_num` 2 桁を除いたもの)

### 3.3 race_id マッピング

```python
# src/data/time_series_odds.py:27-31
def race_id_main_to_ts(main_race_id: int | str) -> int:
    """main 形式 (18 桁) → ts 形式 (16 桁) に変換。"""
    s = str(int(main_race_id))
    return int(s[:16]) if len(s) >= 16 else int(s)
```

---

## 4. 特徴量パイプライン

[src/features/pipeline.py](src/features/pipeline.py) が全体をオーケストレーション。`FeaturePipeline.fit()` で騎手/父の Beta-Binomial 統計量を計算し、`transform()` で 35 列を生成する。

### 4.1 レース特徴量 (8 列)
[src/features/race.py](src/features/race.py)

| 列 | 計算 |
|---|---|
| `place_encoded` | 競馬場の Label encoding |
| `track_type` | `track_code // 10 % 10` (1=芝, 2=ダート, 3=障害) |
| `dist` | 距離 (m) パススルー |
| `dist_category` | S(<1400)/M(<1800)/I(<2200)/L(<2800)/E(≥2800) |
| `condition_encoded` | 良/稍重/重/不良 |
| `weather_encoded` | 晴/曇/小雨/雨/小雪/雪 |
| `class_grade` | クラスコードから 0-5 のグレード |
| `field_size` | 出走頭数 |

### 4.2 馬のローリング特徴量 (11 列)
[src/features/horse.py](src/features/horse.py) — **リーク対策で全て `shift(1)` 適用**

| 列 | 計算 |
|---|---|
| `rank_last` | 前走の着順 |
| `rank_rolling_3 / 5` | 直近 3/5 走の着順平均 |
| `show_rate_last_5` | 直近 5 走の複勝率 |
| `last_3f_rolling_3` | 直近 3 走の上がり 3F 平均 |
| `time_diff_rolling_3` | 直近 3 走のタイム差平均 |
| `weight_horse / weight_change` | 馬体重 / 増減 |
| `race_span_days` | 前走からの経過日数 |
| `prize_cumsum` | 累積賞金 (shift cumsum) |
| `label_momentum` | 直近 2 走の着順差 |

#### コールドスタート対応
[src/features/horse.py:30-78](src/features/horse.py#L30-L78) `compute_cold_start_defaults()` で、**初出走馬向けに「父馬グループの平均値」を補完値として算出**。フォールバックは全体平均 (`_GLOBAL_`)。

### 4.3 騎手特徴量 (5 列)
[src/features/jockey.py](src/features/jockey.py) — Beta-Binomial 事前分布で勝率/複勝率を平滑化

| 列 | 計算 |
|---|---|
| `jockey_encoded` | Label encoding |
| `jockey_show_rate / win_rate` | 累積ベイズ事後平均 (`α=2, β=5` 事前) |
| `jockey_race_count` | 累積出走数 |
| **`jockey_lcb95`** | 複勝率の **95% 信頼区間下限** (= 真の腕の保守的推定) |

### 4.4 父馬 (種牡馬) 特徴量 (5 列)
[src/features/sire.py](src/features/sire.py)

| 列 | 計算 |
|---|---|
| `father_encoded` | Label encoding |
| `sire_show_rate / lcb95` | 騎手と同じく Beta-Binomial |
| `sire_show_rate_turf / dirt` | 芝/ダート別の複勝率 |

### 4.5 レース内相対特徴量 (3 列)
[src/features/relative.py](src/features/relative.py)

| 列 | 計算 |
|---|---|
| `odds_rank` | レース内の単勝オッズ順位 |
| `weight_zscore` | レース内の馬体重 z-score |
| `age_relative` | レース内の年齢中央値からの差 |

### 4.6 そのまま通す列 (3 列)
`horse_num`, `waku_num`, `age`

### 4.7 時系列オッズ特徴量 (穴馬モデル専用, 20 列)
[src/features/odds_timeseries.py](src/features/odds_timeseries.py:16-26) — JD CSV から `aggregate_jd_per_horse()` で集約

| 列 | 内容 |
|---|---|
| `ts_n_snapshots` | スナップショット数 |
| `ts_win_first/last/min/max/std` | 単勝オッズの統計 |
| `ts_win_drop_pct` | `(first - last) / first` (正 = 人気上昇) |
| `ts_win_late_drop_pct` | **後半 25% での単勝オッズ低下率** ← 「賢い金」シグナル |
| `ts_show_lo_last / hi_last` | 直前の複勝オッズ |
| `ts_show_first_mid / last_mid` | 複勝中値 |
| `ts_show_drop_pct` | 複勝オッズ変動率 |
| `ts_implied_prob_first/last` | `1 / オッズ` (粗インプライド確率) |
| `ts_implied_prob_last_norm` | レース内合計 1.0 に正規化 (overround 補正) |
| `ts_pop_rank_change` | 序盤→直前の人気順位の変化 |
| `ts_log_win/show_votes_last` | `log1p` した票数 |
| **`ts_anomaly_score`** | `late_drop_pct × 0.6 + pop_rank_change × 0.4` の合成 |

`has_ts_odds` フラグ (0/1) で「TS データが存在しない場合」を識別。

---

## 5. モデル

### 5.1 複勝モデル (メイン)
[src/model/trainer.py](src/model/trainer.py)

| 項目 | 設定 |
|---|---|
| アルゴリズム | LightGBM (gbdt / dart) |
| 目的 | 二値分類 (binary logloss) |
| 最適化 | **Brier Score 最小化** (Optuna) |
| 特徴量 | 35 列 (FEATURE_COLUMNS) |
| カテゴリ列 | `place_encoded, track_type, dist_category, condition_encoded, weather_encoded, father_encoded, jockey_encoded` |
| Early stopping | 100 rounds (dart 除く) |
| 探索空間 | learning_rate (log), num_leaves, max_depth, subsample, colsample_bytree, reg_alpha/lambda (log), min_child_samples |
| 最終学習 | train + valid 結合データで best_iter まで再学習 |

### 5.2 穴馬モデル
[src/model/anaba_trainer.py](src/model/anaba_trainer.py)

| 項目 | 設定 |
|---|---|
| ターゲット | `rank == 1 AND pop >= 5` (デフォルト) |
| 陽性率 | ~1.5-1.8% (希少陽性) |
| クラス不均衡対応 | **`scale_pos_weight = neg / pos`** |
| 最適化 | **Average Precision 最大化** (-AP を最小化) |
| 特徴量 | **58 列** = 基礎能力 35 + `pop` + `win_odds` + TS_ODDS 20 + `has_ts_odds` |
| 評価指標 | AP, AUC, hit_at_k_positives |

#### スモーク学習実績 (2022-2023 train / 2024 test)
- **Test AUC 0.852 / AP 0.067** (baseline 0.015 の 4.6x)
- **Top 20 中 3 件 = 15.0% 的中** (lift 10.3x)

### 5.3 キャリブレーション
[src/model/calibrator.py](src/model/calibrator.py) — **Isotonic Regression** (`sklearn.isotonic.IsotonicRegression`)

- ホールドアウト分割: 検証データの **時系列順 前半 50% で fit / 後半 50% で評価**
- 学習データで校正するリークを回避
- 単調性保証で「過信」を補正

### 5.4 オッズアンサンブル (オプション)
[src/model/ensemble.py](src/model/ensemble.py) — コールドスタート馬対策

- 既知馬: `0.7 × モデル + 0.3 × オッズインプライド`
- 未知馬: `0.3 × モデル + 0.7 × オッズインプライド`

---

## 6. 推論パイプライン

### 6.1 単勝確率の Harville 推定
[src/strategy/harville.py](src/strategy/harville.py)

複勝モデルが直接出すのは「P(3 着以内)」。これを単勝・三連系に変換する:

1. **複勝 → 単勝**: `win_probability_from_show()`
   - オッズあり: `1 / win_odds` の正規化 (overround 補正)
   - オッズなし: `pred_prob ^ k` で先鋭化 → 正規化
2. **三連単**: Harville モデル `P(i=1, j=2, k=3) = p_i × p_j/(1-p_i) × p_k/(1-p_i-p_j)`
3. **三連複**: 三連単 6 通りの和

### 6.2 賭け金サイジング
[src/strategy/kelly.py](src/strategy/kelly.py)

#### フラクショナル・ケリー
```python
kf = (prob × b - q) / b      # b = odds - 1, q = 1 - prob
bet = bankroll × kf × fraction
bet = min(bet, bankroll × max_bet_fraction)
bet = min(bet, per_bet_cap)
bet = floor(bet / 100) × 100  # JRA 最低単位 100 円
```

| パラメータ | デフォルト |
|---|---|
| `kelly_fraction` | 0.25 (クォーターケリー) |
| `max_bet_fraction` | 0.05 (= バンクロールの 5%) |
| `min_bet` | 100 円 |
| `per_race_cap` | 10,000 円 (1 レース合計上限) |

#### `size_bets_per_race()` 
[src/strategy/kelly.py:130-159](src/strategy/kelly.py#L130-L159) — オッズの有無で振り分け

| 状態 | 処理 |
|---|---|
| **全頭オッズあり** | 各馬 Kelly 計算 → 合計が `per_race_cap` 超なら比例縮小 |
| **オッズなし** | `allocate_by_probability()`: 確率の **二乗** で重みつき按分 (高確率馬を厚く) |

### 6.3 Tier 方式 (固定額)
[src/strategy/kelly.py:90-122](src/strategy/kelly.py#L90-L122)

| 予測確率 | 賭け金 |
|---|---|
| ≥ 0.5 (`tier_high_threshold`) | **1,500 円** |
| ≥ 0.4 | 500 円 |
| ≥ 0.3 | 200 円 |
| < 0.3 | 0 円 |

オッズ無しでも動く。`bet_sizing: "tier"` で有効化。

### 6.4 推奨統合
[src/strategy/recommender.py](src/strategy/recommender.py) — `generate_full_recommendation()` がすべての券種を一括出力

| 関数 | 出力 |
|---|---|
| `recommend_show()` | 複勝 (pred_prob + show_odds_avg + EV + bet_amount) |
| `recommend_win()` | 単勝 (win_prob + win_odds + EV + bet_amount) |
| `recommend_trio()` | 三連複ボックス (Harville prob + odds + EV) |
| `recommend_trifecta()` | 三連単 (Harville prob + odds + EV) |

### 6.5 EV ≥ 1.0 フィルタ + 品質ガード
`--ev-only` フラグで起動。**3 段階の品質ガード**で「数値アーティファクト」を除外:

1. **最小的中確率** (`min_prob`): 三連複 0.5% / 三連単 0.2% 未満は外挿誤差として除外
2. **最大オッズ** (`max_odds`): 三連複 200 倍 / 三連単 1,000 倍超は理論オッズ領域として除外
3. **JRA 控除考慮 EV** (`--apply-takeout`): `EV × (1 − 控除率) ≥ 1.0` で判定
   - 三連複: 25.0% → EV ≥ **1.33** 必要
   - 三連単: 27.5% → EV ≥ **1.38** 必要

#### 効果 (2026 ダービー)
| モード | 三連複 | 三連単 | 合計 |
|---|---|---|---|
| ガード無し | 36 | 550 | **586** |
| ガードのみ | 18 | 51 | 69 |
| ガード + 控除考慮 | 0 | 2 | **2** |

### 6.6 穴馬出力
[scripts/predict.py:_print_anaba_recs](scripts/predict.py#L505-L535) — 別枠で表示

- `anaba_prob ≥ score_threshold (0.15)` の馬を **anaba_prob 降順**で最大 5 件
- 該当無しなら最高スコア馬を「参考」として表示

---

## 7. CLI コマンド

### 7.1 学習
```bash
# 本番設定で全モデル学習 (複勝 + 穴馬, Optuna 10 trials × 各モデル)
python3 scripts/train.py --config config/default.yaml

# スモーク (2022-2023, Optuna 2 trials, ~10 分)
python3 scripts/train.py --config config/smoke_anaba.yaml
```

### 7.2 推論
```bash
# 単一レース (12 桁 netkeiba race_id)
python3 scripts/predict.py --config config/smoke_anaba.yaml \
  --race_id 202605021211 --bankroll 100000

# 日付一括 (YYYYMMDD のその日の全レース)
python3 scripts/predict.py --config config/smoke_anaba.yaml \
  --date 20260531 --bankroll 100000

# EV ≥ 1.0 の組み合わせのみ全件 + 品質ガード + JRA 控除考慮 (推奨)
python3 scripts/predict.py --config config/smoke_anaba.yaml \
  --race_id 202605021211 --bankroll 100000 \
  --ev-only --top-n-combo 12 --apply-takeout

# オッズ取得スキップ (高速、ただし anaba/EV は機能限定)
python3 scripts/predict.py --config config/smoke_anaba.yaml \
  --race_id 202605021211 --no-odds
```

### 7.3 主な CLI フラグ

| フラグ | 役割 |
|---|---|
| `--config <yaml>` | 設定ファイル |
| `--race_id <12桁>` | 単一レース予測 |
| `--date <YYYYMMDD>` | 日付一括 |
| `--bankroll <円>` | バンクロール上書き |
| `--no-odds` | オッズ取得スキップ |
| `--algorithm` | 総合評価アルゴリズム併用 |
| `--ev-only` | EV ≥ 閾値の組み合わせ全件出力 |
| `--min-ev <float>` | EV 閾値 (デフォルト 1.0) |
| `--top-n-combo <int>` | 三連複/三連単の検査対象 (デフォルト 10) |
| `--min-prob-trio <float>` | 三連複最小的中確率 (デフォルト 0.005) |
| `--min-prob-trifecta <float>` | 三連単最小的中確率 (デフォルト 0.002) |
| `--max-odds-trio <float>` | 三連複最大オッズ (デフォルト 200) |
| `--max-odds-trifecta <float>` | 三連単最大オッズ (デフォルト 1000) |
| `--apply-takeout` | JRA 控除率考慮 EV |
| `--no-quality-guard` | 品質ガードを全部 OFF |

---

## 8. 設定ファイル (`config/default.yaml`)

```yaml
data:
  dir: "./data"                    # record CSV 配置先
  train_years: [2015, ..., 2024]
  valid_year: 2024
  valid_split_month: 7             # 1-6 月 valid, 7-12 月 test

# 時系列オッズ (穴馬モデルのみ利用)
odds_timeseries:
  enabled: true
  dir: "./src/data/Time_Series_Odds"
  cache_dir: "./cache/ts_odds"     # 年単位 parquet キャッシュ

# 穴馬予測ヘッド
anaba:
  enabled: true
  min_pop: 5                       # 何番人気以下を穴馬とみなすか
  use_ts_odds: true                # 時系列オッズ特徴量を使うか
  optuna_n_trials: 15
  score_threshold: 0.15            # 穴馬候補と見なす確率

model:
  dir: "./models"
  type: "lightgbm"
  objective: "binary"
  optuna:
    n_trials: 10
    timeout: 600
  search_space:
    learning_rate: [0.005, 0.1]    # log-uniform
    num_leaves: [20, 300]
    max_depth: [3, 12]
    # ... reg_alpha/lambda log-uniform, etc.

calibration:
  method: "isotonic"
  holdout_fraction: 0.5

bayesian:                          # 騎手/父の Beta-Binomial 事前
  alpha_prior: 2
  beta_prior: 5

strategy:
  initial_bankroll: 1000000
  per_race_cap: 10000              # 1 レース上限 (円)
  kelly_fraction: 0.25             # クォーターケリー
  max_bet_fraction: 0.05
  bet_sizing: "kelly"              # or "tier"
  tier_low/mid/high_amount: 200/500/1500    # 円
  selection_method: "threshold"    # or "ev"
  prob_threshold: 0.3
```

---

## 9. 使用技術

| カテゴリ | ライブラリ | バージョン要件 | 用途 |
|---|---|---|---|
| **ML** | LightGBM | ≥ 4.0 | 二値分類モデル (複勝 + 穴馬) |
| **HPO** | Optuna | ≥ 3.0 | TPE サンプラーによる HPO |
| **校正** | scikit-learn (Isotonic) | ≥ 1.3 | キャリブレーション |
| **数値** | NumPy / SciPy | ≥ 1.24 / ≥ 1.10 | 確率計算・統計 |
| **DataFrame** | pandas | ≥ 2.0 | 全データ操作 |
| **可視化** | matplotlib | ≥ 3.7 | 校正曲線 / 特徴量重要度 |
| **シリアライズ** | joblib / pickle / parquet | ≥ 1.3 | モデル/キャッシュ保存 |
| **スクレイピング** | requests + BeautifulSoup4 + lxml | ≥ 2.31 / ≥ 4.12 | netkeiba HTML パース |
| **進捗** | tqdm | ≥ 4.65 | バッチ処理進捗表示 |
| **解釈** | SHAP | ≥ 0.42 | 特徴量重要度 (オプション) |
| **設定** | PyYAML | ≥ 6.0 | YAML 設定読み込み |

### 主要なアルゴリズム/手法

| 手法 | 由来 | 使用箇所 |
|---|---|---|
| **LightGBM** | Microsoft Research (2017) | 複勝/穴馬モデル |
| **TPE (Tree-structured Parzen Estimator)** | Bergstra et al. (2011) | Optuna での HPO |
| **Isotonic Regression** | Barlow & Brunk (1972) | 確率キャリブレーション |
| **Beta-Binomial 共役事前** | Jeffreys (1946) | 騎手/父の率の平滑化 |
| **LCB95 (Lower Confidence Bound)** | Bandit literature | 騎手/父の保守的能力推定 |
| **Brier Score** | Brier (1950) | 二値分類の校正評価 |
| **Average Precision** | Mean of precision at recall | 不均衡データの順序評価 |
| **Harville モデル** | Harville (1973) | 三連単/三連複の条件付き確率 |
| **Kelly 基準** | Kelly (1956) | 賭け金最適化 |
| **フラクショナル・ケリー** | Thorp (1969) | ケリー過剰賭けの緩和 |

---

## 10. データフロー (詳細図)

```
record_data_YYYY.csv (Shift-JIS, 47列)
        │
        ▼ load_year_csv() + filter_errors()
   train_df / valid_df / test_df
        │
        ▼ pd.concat (rolling 特徴量を全期間で正確算出するため)
   all_data
        │
        ▼ pipeline.transform()
        │   ├─ add_race_features (8)        ── place/track/dist/condition/weather/class/field
        │   ├─ add_horse_features (11)      ── shift(1) rolling + cold-start defaults
        │   ├─ add_jockey_features (5)      ── Beta-Binomial 統計量 join
        │   ├─ add_sire_features (5)        ── 同上
        │   └─ add_relative_features (3)    ── レース内 z-score
        ▼
   all_transformed (35 列)
        │
        ├─ iloc[:n_train] → train_x   ────────┐
        ├─ iloc[n_train:n_train+n_valid] → valid_x  ─┐
        └─ iloc[n_train+n_valid:] → test_x    │      │
                                              │      │
                                              ▼      ▼
                                     ┌────────────────────────┐
                                     │ ① 複勝モデル学習       │
                                     │   train_model()        │
                                     │   ↓ Optuna 10 trials   │
                                     │   ↓ Brier 最小化        │
                                     │   ↓ early_stopping=100 │
                                     │   ↓ train + valid 結合 │
                                     │     で再学習            │
                                     │   → lgbm_model.txt     │
                                     └────────────────────────┘
                                              │
                                              ▼
                                     ┌────────────────────────┐
                                     │ ② キャリブレーション   │
                                     │   calibrate_model()    │
                                     │   ↓ valid 前半で Isotonic fit
                                     │   ↓ 後半で評価         │
                                     │   → calibrator.pkl     │
                                     └────────────────────────┘
                                              │
                                              ▼
                                     ┌────────────────────────┐
                                     │ ③ 穴馬モデル学習       │
                                     │   train_anaba_head()   │
                                     │   ↓ TS odds parquet    │
                                     │     load (cache hit時 1秒)
                                     │   ↓ merge_ts_odds_features
                                     │   ↓ +pop +win_odds     │
                                     │   ↓ scale_pos_weight   │
                                     │   ↓ Optuna -AP 最適化   │
                                     │   → anaba_model.txt    │
                                     │   → anaba_meta.pkl (feature_columns 58)
                                     └────────────────────────┘

【推論時 (scripts/predict.py)】

netkeiba スクレイピング (race_card + odds)
        │
        ▼
race_card DataFrame (47 列の一部 + horse 名 + 血統)
        │
        ▼ hist_df と結合 (rolling 特徴量計算用)
   combined (history + 当日)
        │
        ▼ pipeline.transform()
   all_transformed
        │
        ▼ _is_predict マーカーで当日行のみ抽出
   race_feat (35 列)
        │
        ├─ 複勝モデル推論 → pred_prob (calibrated)
        ├─ 穴馬モデル推論 (TS は当日なしで 0 埋め) → anaba_prob
        │
        ▼
   ┌──────────────────────────────────────┐
   │ generate_full_recommendation()       │
   │  ├─ recommend_show                   │
   │  ├─ recommend_win                    │
   │  ├─ recommend_trio (Harville)        │
   │  ├─ recommend_trifecta (Harville)    │
   │  └─ EV フィルタ + 品質ガード         │
   │     ├─ min_prob                      │
   │     ├─ max_odds                      │
   │     └─ × (1 - takeout_rate)          │
   └──────────────────────────────────────┘
        │
        ▼
   _print_show / win / anaba / trio / trifecta
   (円表記 + バンクロール残高比率 + EV-only サマリ)
```

---

## 11. 学習〜推論の所要時間 (実測)

| ステップ | 時間 |
|---|---|
| **record_data CSV 読み込み** (2022-2024) | ~5 秒 |
| **特徴量パイプライン** (concat → transform) | ~30 秒 |
| **複勝モデル Optuna 2 trials** (smoke) | ~30 秒 |
| **時系列オッズ初回パース** (3 年, 10k CSV) | **~10 分** |
| **時系列オッズ 2 回目 (parquet キャッシュ)** | ~1 秒 |
| **穴馬モデル Optuna 2 trials** | ~30 秒 |
| **netkeiba 出馬表スクレイピング** (18 頭) | ~30 秒 (リクエスト間隔 1.5 秒) |
| **netkeiba オッズスクレイピング** | ~10 秒 |
| **モデルロード + 推論 + 推奨** | ~5 秒 |

**初回 smoke 学習合計: ~15 分**、2 回目以降は ~3 分

---

## 12. 設計判断 (なぜそうしたか)

### 12.1 「複勝」を主ターゲットにした理由
- 勝率 ~7% (1/18 頭で当たり) vs 複勝率 ~17% (3/18 頭で当たり)
- **データ量が 2.3 倍**多いので統計的に安定
- 単勝確率は Harville で複勝確率から導出可能 (誤差は小さい)

### 12.2 穴馬を別ヘッドにした理由
- 複勝予測モデルは陽性率 ~21% でクラスバランスが良いが、「**人気 5 番手以下で 1 着**」は陽性率 **1.5%** と希少
- 同じモデルで学習すると人気馬への過剰適合が起きる
- 別モデル + `scale_pos_weight` で「人気薄の中での 1 着確率」を学習させる

### 12.3 時系列オッズを穴馬モデルにだけ入れた理由
- メモリ ([feedback_odds_independence](.claude/projects/.../memory/feedback_odds_independence.md)): 「複勝モデルは市場非依存で組む」原則
- オッズ変動 = 「賢い金」シグナル → **穴馬検出にだけ適している**情報
- 複勝モデルに混ぜると過剰にオッズ追従するモデルになる

### 12.4 Kelly + per_race_cap + tier の 3 並列
- **Kelly**: 数学的最適 (EV>1 で正のサイズ、EV<1 で 0)
- **per_race_cap**: 「1 レースで 5 万円超賭けたくない」現実制約
- **tier**: オッズが取れない (例: 1 週間先) ケースでも動く保険

### 12.5 EV ≥ 1.0 + 品質ガード
- 純粋な EV 計算は **「確率 0.001% × オッズ 130,000 倍 = EV 1.3」のような数値アーティファクト**を量産する
- min_prob で「外挿誤差レベル」を切り、max_odds で「理論オッズ領域」を切り、takeout で「市場効率の歪み」を吸収

---

## 13. 既知の制限事項

| 制限 | 影響 | 緩和策 |
|---|---|---|
| **時系列オッズが当日レースに無い** | 推論時 `has_ts_odds=0` → 穴馬モデルの TS 列が全て 0 で予測精度低下 | 当日 TS odds スクレイパーを追加 (未実装) |
| **smoke モデルは 2 年学習** | Test AUC 0.82 程度。本番品質ではない | `config/default.yaml` で 1986-2024 全期間 + Optuna 10+ trials で再学習 |
| **18 頭立て G1 は学習データに少ない** | 大頭数レースの確率推定が外挿になり、Harville 確率が偏る | 大頭数のみで再 fine-tune (未実装) |
| **train/valid/test の iloc 分割が pipeline.transform 後のソート順に依存** | `add_horse_features` 内で `sort_values(["id", "race_id"])` が走り、iloc[:n_train] が「2022-2023 のみ」を保証しない可能性 | 明示的に `_is_train`/`_is_test` マーカーで抽出する (predict.py は対応済み、train.py は未対応の既存実装) |
| **JRA-VAN リアルタイム連携なし** | netkeiba スクレイピング依存 (サイト変更でブレる) | JRA-VAN API 契約 + 別アダプタ実装 (未実装) |
| **馬連・馬単・ワイド・枠連未対応** | recommend は単複/三連複/三連単のみ | recommender に追加実装 (未実装) |

---

## 14. 拡張ロードマップ

| 優先度 | 内容 |
|---|---|
| ★★★ | 本番モデルで穴馬ヘッドを再学習 (1986-2024, Optuna 20 trials) |
| ★★★ | 当日 TS odds スクレイパー実装 → 推論時に anaba がフル活用 |
| ★★ | 馬連・馬単・ワイド・枠連の recommend 関数追加 |
| ★★ | Harville → **Henery モデル** (べき乗補正) へ拡張 |
| ★★ | バックテスト (`scripts/backtest.py`) を anaba 込みで動かす |
| ★ | 複数モデルアンサンブル (本番モデル + smoke モデルのスタッキング) |
| ★ | SHAP で穴馬モデルの説明可視化 |
| ★ | レース直前の TS データだけで「動いた馬」を検出するリアルタイム alert |

---

## 15. ライセンス・免責

- **本ソフトウェアは予測ツールであり、勝敗を保証するものではない**
- 賭博は計画的に。自己責任で。
- JRA・netkeiba 公式とは無関係の個人プロジェクト
- スクレイピング時はサイトの利用規約と robots.txt を尊重し、リクエスト間隔を空けること

---

## 付録 A: 主要関数の入出力一覧

| 関数 | 入力 | 出力 | 場所 |
|---|---|---|---|
| `load_all_data(cfg)` | config | `(train_df, valid_df, test_df)` | [src/data/loader.py:84](src/data/loader.py#L84) |
| `FeaturePipeline.fit_transform(df)` | DataFrame | 35 列の transformed DF | [src/features/pipeline.py:108](src/features/pipeline.py#L108) |
| `train_model(train_x, train_y, valid_x, valid_y, cfg, cols)` | matrices | `(lgb.Booster, optuna.Study)` | [src/model/trainer.py:114](src/model/trainer.py#L114) |
| `train_anaba_model(...)` | 同上 | 同上 | [src/model/anaba_trainer.py:69](src/model/anaba_trainer.py#L69) |
| `calibrate_model(model, valid_x, valid_y)` | model + valid | `(HoldoutCalibrator, eval_x, eval_y)` | [src/model/calibrator.py:52](src/model/calibrator.py#L52) |
| `parse_jd_csv(path)` | JD CSV パス | long形式 DataFrame | [src/data/time_series_odds.py:39](src/data/time_series_odds.py#L39) |
| `aggregate_jd_per_horse(long_df)` | long DataFrame | 馬×レース集約 (20 列) | [src/data/time_series_odds.py:88](src/data/time_series_odds.py#L88) |
| `merge_ts_odds_features(main_df, ts)` | main + TS | TS 列追加版 | [src/features/odds_timeseries.py:30](src/features/odds_timeseries.py#L30) |
| `size_bets_per_race(probs, odds, ...)` | 確率/オッズ配列 | 賭け金配列 (円) | [src/strategy/kelly.py:130](src/strategy/kelly.py#L130) |
| `generate_full_recommendation(race_feat, ...)` | race_feat | `dict[券種, DataFrame]` | [src/strategy/recommender.py:454](src/strategy/recommender.py#L454) |

## 付録 B: 馬券種ごとの計算式

| 券種 | 確率の計算 | EV の計算 |
|---|---|---|
| **複勝** | `pred_prob` (キャリブレーション済) | `pred_prob × (show_odds_min + show_odds_max) / 2` |
| **単勝** | `win_probability_from_show(pred_prob, win_odds)` | `win_prob × win_odds` |
| **三連複** | `Σ Harville(perm)` (6 通り合計) | `trio_prob × 三連複オッズ` |
| **三連単** | `Harville(i, j, k) = p_i × p_j/(1-p_i) × p_k/(1-p_i-p_j)` | `harville_prob × 三連単オッズ` |

### Harville 確率の擬似コード
```python
def harville_probability(probs, order):
    """order = (i, j, k) で「i が 1 着, j が 2 着, k が 3 着」の確率"""
    p = probs.copy()
    p1 = p[order[0]]
    p2 = p[order[1]] / (1.0 - p1)
    p3 = p[order[2]] / (1.0 - p1 - p[order[1]])
    return p1 * p2 * p3
```
