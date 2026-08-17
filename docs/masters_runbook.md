# AI 競馬予想マスターズ 2026 運用手順書 (Runbook)

## 大会スケジュール

| 日程 | 内容 |
|---|---|
| 8/15(土), 8/16(日), 8/22(土), 8/23(日) | **試験運用日** (API 疎通確認、投票がプラットフォームに反映されるか必ず確認) |
| 8/29, 8/30, 9/5, 9/6, 9/12, 9/13, 9/19, 9/20, 9/21 | **コンペ本番** (9 日間) |

## コンペ制約 (順位付け対象になる条件)

- **96 レース以上**に投票
- **累計 50 万 pt 以上**を投票
- 初期ポイント 100 万 pt、払戻は実確定オッズで計算
- 投票締切: **発走 3 分前** (netkeiba 投票 API)

---

## 0. 事前準備 (一度だけ)

```bash
# 認証情報を環境変数に設定 (~/.zshrc に書くと楽)
export NETKEIBA_LOGIN_ID="あなたのログインID"
export NETKEIBA_PASSWORD="あなたのパスワード"
```

モデルが `models_masters/` に揃っていることを確認:

```bash
ls models_masters/
# 必要: win_model.txt / win_meta.pkl / win_calibrator.pkl / pipeline.pkl
```

無ければ学習を実行 (約 1-2 時間):

```bash
python3 scripts/train.py --config config/masters_2026.yaml
```

## 0.5 試験運用日の前に: オフライン通しリハーサル (API 不要)

大会 API は試験運用日まで叩けないが、リプレイモードで当日ループの全経路
(出走表 → 特徴量構築 → 予測 → レース毎オッズ → 投票判定 → ログ/状態管理) を
過去日の record_data で検証できる:

```bash
python3 scripts/run_live.py --config config/masters_2026.yaml --date 20250831 --replay
```

- dry-run + skip-wait を強制。実 API・実投票は一切行わない
- 確定オッズを 5 分前オッズの代用にする (実運用より情報が新しい点だけ差分)
- 当日以降の履歴行はローリング特徴量から除外 (結果リークなし)
- 終了時に実結果で損益を集計 (的中数 / ROI / bet_kind 別内訳)
- ログは `logs/masters_live_bets_YYYYMMDD_replay.csv` に分離される

ユニットテスト (投票判定・ID 変換・bet_data 形式):

```bash
python3 -m unittest discover -s tests
```

## 0.7 実 API のスキーマと補完 (2026-08-17 実データで確認済み)

当日出馬表 API の実レスポンスは学習データ (record_data 47 列) と大きく異なる
**14 列** (`place, race_num, horse_num, dist, horse, sex, age, jockey(名前), loaf_weight,
father, mother, id, waku_num, race_id`)。`normalize_runtable()` と `enrich_from_netkeiba()`
が以下のように埋める:

| 学習時の列 | 当日の入手元 | 備考 |
|---|---|---|
| class_code / track_code / year / month / day | timetable API を (place, race_num) で結合 | |
| jockey_id | **netkeiba 出馬表** (レースごと 1 リクエスト、JRA 騎手コード) | API は騎手名しか返さない。`data/jockey_master.csv` に名前→ID をキャッシュ (netkeiba 不通時のフォールバック) |
| weight / inc_dec (馬体重) | 朝: 履歴の前走馬体重で補完 (増減 0) → 発走 4分10秒前: netkeiba から実測値で更新して再予測 | 0 埋めは学習分布外になるため厳禁 |
| state / weather | netkeiba 出馬表 (判明していれば)、無ければ 良 / 晴 | |
| basis_weight | loaf_weight を改名 | |
| horse_N / times / daily | 頭数集計 / race_id から導出 | |
| 結果系 (rank, pop, prize …) | 0 | 全ローリング特徴量が shift(1) なので当該行の値は混入しない |

- 朝の netkeiba 補完は 36 レースで約 1 分。8/9 分の実データで騎手 ID 解決 97 名、
  全 36 レースの判定・ログ出力まで通ることを `--dry-run --skip-wait` で確認済み。
- 大会 API は過去日も返すので、任意の過去開催日で `--dry-run --skip-wait` を回せば
  実 API 経路のリハーサルができる (投票は行わない)。

### 投票 API の仕様メモ (2026 参加マニュアル)

- 投票レスポンスは `data.success_count / error_count / list_error` と
  トップレベルの `remaining_money`。`error_count>0` や `success_count==0` はクライアントが失敗扱いにする
- **投票確認 (GET /bet) は投票直後だと未反映になりうる (非同期。1 分程度あける)**。
  run_live は投票直後には確認せず、次レースの処理時に 60 秒以上経過分をまとめて確認し、
  日次終了時に残り全件を確認する (`verified` 列に反映)
- 締切内なら再投票可 (前の投票は上書き)。ランキングは 手持ち pt > レース的中率 > ◎勝率

## 1. 試験運用日 (8/15, 16, 22, 23) の手順

### 朝 9:00 過ぎ: API 疎通確認

```bash
python3 scripts/check_masters_api.py --date 20260815 --check-vote
```

確認ポイント:
- [1] Racecards が取得でき timetable / runtable が非空
- [2] Odds の構造 (odds_type=1 が単勝、comb が馬番ゼロ埋め 2 桁)
- [3] 投票 API ログイン成功
- [4] GET /bet の投票内容確認が通る

### 最初のレース前: ドライラン

```bash
python3 scripts/run_live.py --config config/masters_2026.yaml --date 20260815 --dry-run
```

- 投票 API は叩かず、判定 (main/sub、馬番、金額) だけログに出す
- `logs/masters_live_bets_20260815.csv` に判定が記録される

### 問題なければ: 少額の本番投票テスト

```bash
python3 scripts/run_live.py --config config/masters_2026.yaml --date 20260815
```

**投票後は必ずプラットフォーム (netkeiba マスターズのマイページ) で反映を目視確認する。**
資料の注意書き: 「API のステータスが正常でも、プラットフォームへの反映がされていない場合、その投票は無効」

反映確認は 2 段構え:
1. **自動 (投票直後)**: run_live.py は投票のたびに GET /bet で反映を確認し、
   ログ CSV の `verified` 列に記録する。`verified=False` の行はログに ★ 警告が出る。
2. **自動 (日次の締め)**: 全レース終了後に照合スクリプトを実行する:

```bash
python3 scripts/verify_bets.py --date 20260815
```

   ログ CSV と GET /bet の登録内容を全件突き合わせ、OK / MISSING / MISMATCH を表示。
   OK 以外が 1 件でもあれば終了コード 1。MISSING はマイページで目視確認のうえ
   運営 Slack に報告する。

※ dry-run の投票ログと累積状態は `_dryrun` サフィックスの別ファイルに書かれるため、
   本番の制約トラッキング (`logs/masters_live_state.json`) を汚染しない。
※ 再実行時は投票済みレース (state の races_bet に記録) を自動スキップするので、
   クラッシュしてもそのまま同じコマンドを再実行すればよい (二重投票しない)。

## 2. コンペ本番日の手順

```bash
# 朝 9:00 過ぎに起動して放置 (発走時刻に合わせて自動投票)
python3 scripts/run_live.py --config config/masters_2026.yaml --date 20260829
```

- 出走表取得 → 特徴量構築 → 全レース予測 (~10-20 分) → 各レース発走 4分10秒前にオッズ取得 → 投票
- 進捗は標準出力と `logs/masters_live_bets_YYYYMMDD.csv` で確認
- 累積の制約充足状況は `logs/masters_live_state.json` に永続化 (日をまたいで積算)

### 途中でクラッシュした場合

そのまま再実行すれば OK (過ぎたレースはスキップされる):

```bash
python3 scripts/run_live.py --config config/masters_2026.yaml --date 20260829
```

## 3. 投票戦略

### 検証結果 (2026-08): ev_market は基準不合格 → favorite_concentration を継続採用

市場残差モデル (win_market, オッズ特徴量入り) を v1 (spw あり)・v2 (spw なし) の
2 版で学習し、事前登録基準「EV≥1.05 バケットの実現 ROI が 2024/2025 両窓でプラス」
を検証した結果:

| 版 | 校正後 Brier (test) | 市場ベースライン | EV≥1.05 実現 ROI (2024 窓 / 2025 窓) |
|---|---|---|---|
| v1 (spw) | 0.05811 | 0.05771 | -42.5% / -60.1% ✗ |
| v2 (spw なし) | 0.05790 | 0.05771 | +120.6% / **-83.0%** ✗ |

- v2 全期間の EV≥1.05 は +8.7% に見えるが、平均オッズ 44-47 倍の大穴的中数本に
  依存しており (odds≤15 に絞ると -26.6%)、推定標準誤差 ±10% でゼロと区別不能。
  両窓の乖離 (+121% vs -83%) は宝くじ分散の典型。
- 結論: **現特徴量 (過去成績・騎手・血統) では市場に対する残差エッジは検出できない**。
  市場は調教・馬体・当日情報を織り込んでおり、それを持たないモデルの乖離は
  情報不足由来。実マネーでエッジを得るには調教データ等の新情報源が必要。
- run_live.py の ev_market モードは実装済みのまま温存 (mode 切替だけで有効化可)。
  新データ源を足して再検証で合格したら切り替える。

### ev_market モード (温存中の実装) の仕組み

実マネー運用を見据え、投票判定は 2 層構造:

```
レース毎 (発走 4分10秒前):
  5 分前オッズ取得 → 全馬の市場特徴量を計算 → win_market モデルで予測・校正
      ↓
  EV = 校正後確率 × オッズ が最大の馬について
      EV ≥ 1.05 かつ オッズ ≤ 30 ?
      ├─ YES → Kelly サイズで投票 (bet_kind="ev")   ← 利益を狙う層
      └─ NO  → favorite_concentration へフォールバック ← 制約充足の層
                 (top-1 オッズ ≤1.5 なら大口 / それ以外 100pt)
```

- **ev 層**: 市場残差モデルが「市場より高い勝率」を主張する馬にだけ賭ける。
  市場確率 × オッズ ≈ 0.8 (控除後) なので、EV ≥ 1.05 はモデルが市場に対して
  +25% 以上のエッジを主張する場合にのみ発生する。
- **フォールバック層**: EV 機会がないレースでも 96 レース × 50 万 pt 制約を
  必ず満たす (favorite_concentration の損失最小化ロジックをそのまま継承)。
- **採否のゲート**: ev 層を有効にする前に `scripts/analyze_market_edge.py` で
  「EV ≥ 1.05 バケットの実現 ROI が 2024/2025 両窓でプラス」を確認すること。
  確認できない場合は `live.mode: favorite_concentration` に戻す。

### フォールバック層の詳細 (favorite_concentration)

### 戦略の根拠 (バックテスト検証済み、行順バグ修正後の正データ)

1. **EV フィルタ方式は機能しない**: 384 設定のグリッドサーチで、2024/2025 どちらの窓でも
   ベースラインを上回る「モデル確率 × オッズ ≥ 閾値」設定は**ゼロ**。
   オッズなしモデルが市場と強く乖離した馬は、市場の方が正しい (逆選択)。
2. **全オッズ帯でプラス ROI なし**。最小損失は**オッズ 1.0-1.5 の圧倒的人気帯 (-13.6%)**
   (favorite-longshot バイアス)。それ以外の帯は -20〜-44%。
3. よって最適戦略 = **「制約で強制される投票を、最小損失の形で消化する」**:
   - 全レース top-1 に **100 pt** (96 レース制約を総額 ~3 万 pt でクリア)
   - top-1 の 5 分前オッズが **1.5 以下**のレースだけ大口 (~2.5 万 pt/回、動的計算)
   - 大口額は「最終日に累計 51 万 pt に着地する」よう残り日数から毎朝自動計算

### 期待成績

- 期待最終ポイント: **~925,000 pt** (最低額分 -1 万 + 大口 48 万 × -13.6%)
- 窓ごとのブレ: ±8 万 pt 程度
- 参考: サンプルプログラム的な素朴な戦略 (全レースに残高/オッズ比例investment) は
  控除率 -20〜-30% をフル投票額で被るため、多くの参加者は 700-850k 着地と予想

### パラメータ (`config/masters_2026.yaml` の `live:`)

| パラメータ | 既定値 | 意味 |
|---|---|---|
| `mode` | favorite_concentration | 戦略モード ("hybrid" = 旧 EV 方式、非推奨) |
| `min_bet_per_race` | 100 | 全レースへの最低額 |
| `fav_max_odds` | 1.5 | 大口対象のオッズ上限 |
| `target_total_wagered` | 510000 | 期間累計の目標投票額 |
| `big_amount_min/max` | 5000 / 60000 | 大口額のクランプ |

### 終盤の手動調整

- 9/19 時点で累計投票が 40 万 pt 未満なら `fav_max_odds: 2.5` に広げる
  (大口機会が増える。2.0-2.5 帯の損失は -27% 程度に悪化するが制約未達よりマシ)
- 逆に達成済みなら自動で大口が 5,000 pt に落ちる (追加損失を抑制)

## 4. トラブルシューティング

| 症状 | 対処 |
|---|---|
| データ API がタイムアウト | リトライは自動 (3 回)。継続失敗なら運営 Slack に報告 |
| オッズが空で返る | 発走 4分30秒前より前に叩いている。`odds_fetch_before_sec` (既定 250 秒) を確認 |
| 投票 API 401/403 | トークン期限切れ (5 分)。クライアントはレース毎に再ログインする設計なので、認証情報を確認 |
| 投票がプラットフォームに反映されない | **その投票は無効扱い**。試験運用日に必ず反映確認。運営 Slack に報告 |
| バンクロール枯渇が近い | `main_bet_amount` / `sub_bet_amount` を下げる。制約 (50 万 pt) は死守 |

## 5. 関連ファイル

| ファイル | 役割 |
|---|---|
| `scripts/check_masters_api.py` | API 疎通確認 |
| `scripts/run_live.py` | 当日実行ループ (自動投票 + 投票直後の反映確認) |
| `scripts/verify_bets.py` | 日次の投票照合 (ログ CSV vs GET /bet 全件突き合わせ) |
| `tests/` | 投票判定・ID 変換・bet_data 形式のユニットテスト (`python3 -m unittest discover -s tests`) |
| `scripts/train.py` + `config/masters_2026.yaml` | 本番モデル学習 |
| `scripts/backtest_masters.py` + `config/backtest_masters_prod*.yaml` | バックテスト |
| `src/api/masters_client.py` | データ API / 投票 API クライアント |
| `src/strategy/masters_simulator.py` | コンペルール準拠シミュレータ |
