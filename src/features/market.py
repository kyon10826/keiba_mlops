"""市場 (オッズ) 由来の特徴量。市場残差モデル (market head) 専用。

================================================================================
【レビューガイド】なぜオッズを特徴量に入れるのか — 方針転換の経緯
================================================================================
当初はコンペ差別化のため「モデル入力にオッズを使わない」方針だったが、
バックテストで以下が実証されたため実マネー前提の設計に転換した:

  - オッズなしモデルは全オッズ帯で控除率 (20%) を超えられない
  - モデルが市場と強く乖離した馬は市場の方が正しい (逆選択)
    → 市場は調教・馬体・関係者情報を織り込んでおり、これを無視する
      モデルの「乖離」はほぼ情報不足由来のノイズだった

実マネーで通用する競馬 AI の定石は「市場を土台に、市場が取りこぼす歪みだけを
取る」市場残差アプローチ:
  - 市場確率 (オッズの逆数を overround 補正したもの) を特徴量に入れる
  - モデルは市場確率をベースラインとして再現しつつ、基礎能力特徴量が
    市場と食い違う局面だけ確率を上下に修正することを学習する
  - 賭けるのは「校正済み予測確率 × オッズ ≥ 閾値」の馬だけ
    (市場確率 × オッズ ≈ 1 − 控除率 ≈ 0.8 なので、EV ≥ 1 は
     モデルが市場に対して本物のエッジを持つ場合にしか発生しない)

【train-serving skew の注意】
学習時の win_odds は「確定オッズ」、本番の判定は「5 分前オッズ」。
人気馬 (本戦略の主戦場) では両者の差は小さいが、人気薄では 10-20% ずれる。
将来的には Time_Series_Odds の最終スナップショット (ts_win_last) への
置き換えでスキューを解消できる (cache/ts_odds に 2022-2024 の集約済みあり)。
================================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 市場残差モデルに追加する特徴量列
MARKET_FEATURE_COLUMNS = [
    "log_win_odds",     # log(オッズ)。オッズは裾が重いので対数化して分割を安定させる
    "market_prob",      # overround 補正後の市場確率 (レース内で合計 1.0 に正規化)
    "market_odds_rank", # レース内のオッズ順位 (1 = 1 番人気)
]

_RACE_KEY = ["year", "month", "day", "place", "race_num"]


def add_market_features(df: pd.DataFrame, odds_col: str = "win_odds") -> pd.DataFrame:
    """オッズ列から市場特徴量 3 列を計算して付与する。

    取消馬 (odds<=0/NaN) は market_prob=0, log_win_odds=0, rank=0 になる。
    学習側では error_code フィルタで除外されるため実害なし。

    Args:
        df: win_odds と _RACE_KEY 列を持つ DataFrame。
        odds_col: オッズ列名 (学習時は確定オッズ、本番は 5 分前オッズを入れる)。
    """
    out = df.copy()
    odds = pd.to_numeric(out[odds_col], errors="coerce")

    # log オッズ (odds<1 は理論上存在しないので 1.0 でクリップ → log=0)
    out["log_win_odds"] = np.log(odds.clip(lower=1.0)).fillna(0.0)

    # 市場確率: 1/odds をレース内合計で正規化 (overround = 控除率分の上乗せを除去)
    raw_p = 1.0 / odds.where(odds > 0)
    grp = [out[k] for k in _RACE_KEY]
    sum_p = raw_p.groupby(grp).transform("sum")
    out["market_prob"] = (raw_p / sum_p.where(sum_p > 0)).fillna(0.0)

    # レース内オッズ順位 (低いオッズ = 人気 = 小さい順位)
    out["market_odds_rank"] = odds.groupby(grp).rank(method="min").fillna(0.0)

    return out
