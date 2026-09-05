#!/usr/bin/env python3
"""AI 競馬予想マスターズ 2026 当日実行ループ。

使い方:
    # ドライラン (投票 API を叩かず判定だけ表示。試験運用日の最初はこれで確認)
    python scripts/run_live.py --config config/masters_2026.yaml --date 20260815 --dry-run

    # 本番 (要 NETKEIBA_LOGIN_ID / NETKEIBA_PASSWORD 環境変数)
    export NETKEIBA_LOGIN_ID="..."
    export NETKEIBA_PASSWORD="..."
    python scripts/run_live.py --config config/masters_2026.yaml --date 20260829

流れ:
    1. 朝: 当日データ API から出走表 (Racecards) を取得
    2. 履歴データと結合して特徴量パイプライン適用 → 単勝モデルで全レース予測
       (履歴との結合が必要な理由: 「直近 3 走平均」等のローリング特徴量は
        当日の出走表だけでは計算できないため。_is_predict マーカーで当日行を追跡)
    3. タイムテーブル順に各レースの発走 4分10秒前まで待機
    4. 5分前オッズを取得し、戦略モードに従って投票額を決定
    5. 投票 API で単勝 1 点投票 (締切は発走 3 分前)
    6. logs/ に投票ログ CSV と累積状態 JSON を保存 (クラッシュ後の再実行に対応)

================================================================================
【レビューガイド】採用戦略 favorite_concentration の理屈
================================================================================
バックテスト (行順バグ修正後の正データ、4,173 レース) で判明した事実:
  1. モデル top-1 の的中率は 23.6% と高い (ランキング能力は本物) が、
     どのオッズ帯でもフラット投票の ROI はマイナス = 控除率 (20%) を超えられない。
  2. EV フィルタ (モデル確率 × オッズ ≥ 閾値) は 384 設定全てで機能しない。
     オッズなしモデルが市場と強く乖離した馬は市場が正しい (逆選択)。
  3. 最も損失が小さい投票先は「top-1 かつオッズ 1.0-1.5 の圧倒的人気馬」で
     ROI -13.6% (favorite-longshot バイアス: 人気馬は過小評価されやすく、
     人気薄は過大評価されやすいという競馬市場の実証的な歪み)。

したがって最適戦略は「勝ちに行く」ではなく「制約で強制される投票の損失最小化」:
  - 全レース top-1 に 100 pt → 96 レース制約を総額 ~3 万 pt でクリア
  - top-1 のオッズ ≤ 1.5 のときだけ大口 → 強制される 50 万 pt を最小損失帯に配置
  - 大口額は「最終日に累計 51 万 pt に着地」するよう残り日数から毎朝動的計算
    (compute_big_amount 参照。序盤に消化しすぎず、終盤に未達にもしない)
期待最終ポイント ~92.5 万 pt。素朴な戦略の参加者 (控除率をフル投票額で被る) は
70-85 万 pt 着地が見込まれるため、相対順位で上位を狙う設計。
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from src.data.loader import load_all_data, load_config
from src.data.schema import COLUMN_NAMES
from src.features.pipeline import (
    FeaturePipeline,
    FEATURE_COLUMNS,
    assert_no_market_info,
)
from src.features.market import add_market_features
from src.model.calibrator import HoldoutCalibrator
from src.model.win_trainer import load_win_meta, load_win_model
from src.strategy.kelly import compute_bet_amount
from src.api.masters_client import (
    MastersDataClient,
    MastersVoteClient,
    normalize_runtable,
    race_id_to_odds_id,
    race_id_to_vote_id,
)
from src.strategy.multi_bet import select_multi_bets
from src.scraper.race_card import scrape_shutuba_light

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_live")

# 投票戦略のデフォルト (config の live: セクションで上書き可)
#
# mode:
#   "favorite_concentration" (推奨): バックテストで実証された損失最小化戦略。
#       - 全レースの top-1 に最低額 (min_bet_per_race) を投票 → 96 レース制約を最安でクリア
#       - top-1 のオッズが fav_max_odds 以下 (圧倒的人気馬) のときだけ大口投票
#         → favorite-longshot バイアスにより期待損失が最小の帯 (実測 -13.6%)
#       - 大口額は「累計 50 万 pt 制約に着地する」よう残り日数から動的計算
#   "hybrid": 旧 EV フィルタ方式 (バックテストで頑健な優位性なしと判明。参考実装)
DEFAULT_LIVE_STRATEGY = {
    "mode": "favorite_concentration",
    # --- ev_market (市場残差モデルの EV 投票。実マネー志向) 用 ---
    # レース毎に 5 分前オッズから市場特徴量を計算し win_market モデルで予測、
    # 校正後確率 × オッズ ≥ ev_min_ev の馬に Kelly サイズで投票する。
    # EV 条件を満たさないレースは favorite_concentration ロジックへフォールバック
    # するため、96 レース × 50 万 pt 制約は常に守られる。
    "ev_min_ev": 1.05,           # EV 閾値 (analyze_market_edge.py で検証した値を設定)
    "ev_max_odds": 30.0,         # EV 投票のオッズ上限 (超高オッズの確率推定は不安定)
    "ev_kelly_fraction": 0.25,   # クォーターケリー
    "ev_max_bet_fraction": 0.02, # 1 ベット = バンクロールの 2% まで
    "ev_max_bet": 20000,         # 1 ベット上限 (pt)
    # --- favorite_concentration 用 ---
    "min_bet_per_race": 100,        # 全レースに置く最低額 (レース数制約用)
    "fav_max_odds": 1.5,            # 大口対象: top-1 の 5 分前オッズがこの値以下
    "fav_fallback_odds": 2.5,       # 終盤に投票額が不足しそうな場合の拡張帯
    "target_total_wagered": 510000, # 期間累計の目標投票額 (50 万制約 + バッファ)
    "big_amount_min": 5000,         # 大口の下限
    "big_amount_max": 60000,        # 大口の上限
    "races_per_day_est": 36,        # 1 日の想定レース数
    "fav_rate_est": 0.06,           # top-1 がオッズ≤1.5 になるレースの想定割合
    "remaining_days_default": 9,    # コンペ残り日数 (state で自動更新されるが初期値)
    # 大口 chunking (仕様バグ回避): 単発で N,NNN pt を送ると status=OK だが差引されない
    # 事象 (8/30 24,700pt, 9/5 31,100pt) に対応。fav_big をチャンクに分けて送る。
    # 100pt は必ず成立するので、chunk_size を小さくすれば確実に累計投票が積める。
    "fav_big_chunk_size": 1000,     # 1 チャンクあたりの金額 (0 = 分割せず単発送信)
    "fav_big_chunk_max_time_sec": 100,  # 全チャンク送信にかける最大秒数 (締切前予算)
    # --- 穴狙いサブベット (毎レース、10 倍以上で model 確率が閾値以上なら 100pt) ---
    # 目的: 収支の上振れ余地を作る。1 レース +100pt、~30% で発火 = 1 日 ~30 * 100 = 3,000pt。
    # 9 日で ~27,000pt を使うが、期待値は「モデル確率が校正されている」前提で±0 前後。
    # 大外れでも 100pt × 発火数なので 50 万 pt 制約への影響は微小。
    # バックテスト (4,173 レース) 結果:
    #   (min_odds=10, max_odds=50, min_prob=0.05) → 発火率 94%、的中率 3.4%、
    #   ROI -11.4% ≒ 9 日で ~30k pt 使って期待損 ~3.5k pt (的中時に +3000pt 級)
    #   ※ min_prob を上げると ROI 悪化 (0.10 で -35.9%): モデルの穴推しは市場より弱い
    #   → デフォルトは「上振れ余地を残しつつ損失を最小化」の設定にした
    "anaba_enabled": True,
    "anaba_min_odds": 10.0,         # 10 倍以上 (ユーザー指定)
    "anaba_max_odds": 50.0,         # 50 倍まで許容 (バックテストで最良帯)
    "anaba_min_prob": 0.05,         # 校正後勝率 5% 以上 (緩め: 高くすると ROI 悪化)
    "anaba_amount": 100,            # 固定 100pt (100 pt 単位の最小)
    "anaba_max_per_race": 1,        # 1 レース最大何頭に賭けるか
    # --- 多重ベット (三連複・三連単・馬単・馬連) — Harville で EV 計算 ---
    # 期待値目標: 高分散でも上振れの余地 (三連単 100-500 倍配当) を狙う設計。
    # 平均 ROI は控除率のためマイナス想定だが、1 hit で +30k-500k の上振れが可能。
    # 目安: 1 レース 6 点×300pt = 1,800pt、9 日 20 レース/日 = 32.4 万 pt / 9 日投入。
    # 三連複 hit rate ~5%, 三連単 ~1%, 期待損失 -8〜-15% (単勝 fav_big より悪い)。
    "multi_enabled": True,
    "multi_bet_types": ["trio", "trifecta"],  # 券種を絞ると分散抑制
    "multi_top_n_trio": 4,                    # 上位 4 頭で組合せ (4C3=4 通り)
    "multi_top_n_trifecta": 4,                # 上位 4 頭で順列 (4P3=24 通り)
    "multi_top_n_exacta": 5,
    "multi_top_n_quinella": 5,
    "multi_min_ev": 1.30,                     # 三連複 takeout=0.25 → 1/(1-0.25)=1.33 が損益分岐
    "multi_min_prob_trio": 0.005,
    "multi_min_prob_trifecta": 0.002,
    "multi_min_prob_exacta": 0.01,
    "multi_min_prob_quinella": 0.02,
    "multi_max_odds": 500.0,
    "multi_amount_per_bet": 300,
    "multi_max_bets_per_race": 6,
    "multi_max_total_per_race": 3000,
    "multi_use_padded_bet_id": False,          # NG が返る場合 True に切替
    # --- hybrid (旧方式) 用 ---
    "win_prob_min": 0.10,
    "min_ev": 1.05,
    "min_odds": 5.0,
    "max_odds": 15.0,
    "main_bet_amount": 3000,
    "sub_bet_amount": 2000,
    # --- 共通 ---
    "min_bet_unit": 100,
    "odds_fetch_before_sec": 250,   # 発走何秒前にオッズ取得するか (4分10秒)
    "bet_deadline_sec": 180,        # 発走何秒前までに投票を終えるか (3分 = API 締切)
}

# コンペ本番の開催日 (残り日数の自動計算に使用)
COMPETITION_DAYS = [
    "20260829", "20260830", "20260905", "20260906",
    "20260912", "20260913", "20260919", "20260920", "20260921",
]


def compute_big_amount(strat: dict, state: dict, date_str: str) -> int:
    """favorite_concentration の大口投票額を残り日数から動的に計算する。

    目標: コンペ最終日までに total_wagered が target_total_wagered に着地する。
    大口額 = (残り必要額 − 最低額投票の残り見込み) / 残りの大口機会見込み
    """
    remaining_days = [d for d in COMPETITION_DAYS if d >= date_str]
    n_days = max(len(remaining_days), 1) if date_str in COMPETITION_DAYS or any(
        d >= date_str for d in COMPETITION_DAYS
    ) else int(strat["remaining_days_default"])

    races_left = n_days * float(strat["races_per_day_est"])
    fav_left = max(races_left * float(strat["fav_rate_est"]), 1.0)

    need = float(strat["target_total_wagered"]) - float(state.get("total_wagered", 0))
    need -= races_left * float(strat["min_bet_per_race"])  # 最低額投票で消化される分
    if need <= 0:
        return int(strat["big_amount_min"])

    unit = int(strat["min_bet_unit"])
    amt = need / fav_left
    amt = max(float(strat["big_amount_min"]), min(float(strat["big_amount_max"]), amt))
    return int(amt // unit) * unit


JOCKEY_CACHE_PATH = "data/jockey_master.csv"
NETKEIBA_INTERVAL_SEC = 1.5


def enrich_from_netkeiba(runtable: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
    """大会 API に無い列 (騎手 ID・馬体重・天候・馬場) を netkeiba の出馬表で補う。

    大会 API の出馬表は騎手を「名前 (4 文字)」でしか返さないが、学習済みモデルの
    騎手特徴量は JRA 騎手コード (jockey_id) で結合している。netkeiba の出馬表は
    同じ JRA コードをリンクに持つので、レースごとに 1 リクエストで
    (馬番 → jockey_id) を引き、(place, race_num, horse_num) で結合する。

    フォールバック順: netkeiba (当日・正確) → 騎手名キャッシュ CSV → 0 (コールドスタート)。
    netkeiba が落ちていても投票ループ自体は止めない。
    """
    out = runtable.copy()
    name_cache: dict[str, int] = {}
    if use_cache and os.path.exists(JOCKEY_CACHE_PATH):
        try:
            cache_df = pd.read_csv(JOCKEY_CACHE_PATH)
            name_cache = dict(zip(cache_df["jockey_name_api"], cache_df["jockey_id"]))
        except Exception as e:  # noqa: BLE001
            logger.warning("騎手キャッシュ読込失敗 (%s)", e)

    race_ids = out["race_id"].astype("int64").unique()
    logger.info("netkeiba から騎手 ID / 馬体重を補完中 (%d レース、約 %d 秒)...",
                len(race_ids), int(len(race_ids) * NETKEIBA_INTERVAL_SEC))
    n_ok = 0
    for rid in race_ids:
        nk_id = race_id_to_vote_id(rid)  # 投票用 12 桁 = netkeiba の race_id
        try:
            res = scrape_shutuba_light(nk_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("netkeiba 出馬表取得失敗 race=%s (%s)", nk_id, e)
            res = None
        time.sleep(NETKEIBA_INTERVAL_SEC)
        if res is None:
            continue
        n_ok += 1
        mask = out["race_id"].astype("int64") == rid
        h = res["horses"].set_index("horse_num")
        idx = out.loc[mask, "horse_num"].astype(int)
        out.loc[mask, "jockey_id"] = idx.map(h["jockey_id"]).fillna(0).astype(int).values
        if h["weight"].notna().any():
            out.loc[mask, "weight"] = idx.map(h["weight"]).values
            out.loc[mask, "inc_dec"] = idx.map(h["inc_dec"]).values
        if res.get("state"):
            out.loc[mask, "state"] = res["state"]
        if res.get("weather"):
            out.loc[mask, "weather"] = res["weather"]
        # 騎手名 (API 表記) → ID をキャッシュに蓄積
        if "jockey" in out.columns:
            pairs = out.loc[mask & (out["jockey_id"] > 0), ["jockey", "jockey_id"]]
            for name, jid in pairs.itertuples(index=False):
                name_cache[str(name)] = int(jid)

    # netkeiba で引けなかった馬はキャッシュで補完
    if "jockey" in out.columns and name_cache:
        miss = out["jockey_id"].fillna(0).astype(int) == 0
        out.loc[miss, "jockey_id"] = (
            out.loc[miss, "jockey"].astype(str).map(name_cache).fillna(0).astype(int).values
        )
    resolved = int((out["jockey_id"].fillna(0).astype(int) > 0).sum())
    logger.info("騎手 ID 解決: %d/%d 頭 (netkeiba %d/%d レース成功)",
                resolved, len(out), n_ok, len(race_ids))

    if use_cache and name_cache:
        os.makedirs(os.path.dirname(JOCKEY_CACHE_PATH), exist_ok=True)
        pd.DataFrame(sorted(name_cache.items()), columns=["jockey_name_api", "jockey_id"]) \
            .to_csv(JOCKEY_CACHE_PATH, index=False)
    return out


def impute_weight_from_history(runtable: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """馬体重が未公開の馬に、履歴の直近馬体重を補完する (増減は 0)。

    学習時の weight_horse は当日の実測馬体重だが、朝の出馬表 API には無い。
    0 で埋めると学習分布外 (最軽量馬 380kg 未満) になり予測が歪むため、
    馬体重は安定している (前走比 ±10kg 程度) ことを利用して前走値で代用する。
    初出走馬など履歴が無い場合はレース内平均、それも無ければ 470kg。
    """
    out = runtable.copy()
    if "weight" not in out.columns:
        out["weight"] = float("nan")
    need = out["weight"].isna()
    if not need.any():
        return out
    h = hist[pd.to_numeric(hist["weight"], errors="coerce").fillna(0) > 0]
    last_w = (
        h.sort_values("race_id").groupby("id")["weight"].last()
    )
    out.loc[need, "weight"] = out.loc[need, "id"].map(last_w).values
    still = out["weight"].isna()
    if still.any():
        race_mean = out.groupby(["place", "race_num"])["weight"].transform("mean")
        out.loc[still, "weight"] = race_mean[still]
    out["weight"] = out["weight"].fillna(470.0)
    out["inc_dec"] = pd.to_numeric(out["inc_dec"], errors="coerce").fillna(0.0)
    logger.info("馬体重: 前走値で補完 %d 頭 / 実測 %d 頭",
                int(need.sum()), int((~need).sum()))
    return out


def refresh_live_weight(
    race_rows: pd.DataFrame, netkeiba_race_id: str,
) -> tuple[pd.DataFrame, bool]:
    """発走直前に公開された実測馬体重で weight 系 3 特徴量を差し替える。

    weight_horse / weight_change / weight_zscore は当該レース内だけで
    完結する特徴量なので、他の 43 列を再計算せずに更新できる。
    取得失敗・未公開なら朝の補完値のまま返す。

    Returns:
        (rows, changed) — changed=True のとき呼び出し側は再予測する
    """
    try:
        res = scrape_shutuba_light(netkeiba_race_id)
    except Exception as e:  # noqa: BLE001
        logger.warning("直前馬体重の取得失敗 (%s)。朝の補完値を使用", e)
        return race_rows, False
    if res is None or res["horses"]["weight"].isna().all():
        return race_rows, False
    rows = race_rows.copy()
    h = res["horses"].set_index("horse_num")
    idx = rows["horse_num"].astype(int)
    w = idx.map(h["weight"])
    d = idx.map(h["inc_dec"])
    ok = w.notna().values
    if not ok.any():
        return race_rows, False
    rows.loc[ok, "weight_horse"] = w[ok].values
    rows.loc[ok, "weight_change"] = d[ok].fillna(0).values
    std = rows["weight_horse"].std()
    std = std if std and std > 0 else 1.0
    rows["weight_zscore"] = (rows["weight_horse"] - rows["weight_horse"].mean()) / std
    logger.info("直前馬体重を反映: %d/%d 頭", int(ok.sum()), len(rows))
    return rows, True


def build_day_features(
    cfg: dict,
    runtable: pd.DataFrame,
    model_dir: str,
    date_str: str,
    hist: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """当日出走馬の特徴量を履歴データと結合して構築する。

    Args:
        date_str: 対象日 "YYYYMMDD"。この日以降の履歴行はローリング特徴量の
            計算から除外する (リプレイ時に当日の結果が「過去」として混入する
            リークの防止。実運用日は当日行が履歴に存在しないため無操作)
        hist: 読み込み済み履歴 (リプレイで二重ロードを避ける用)。None なら読み込む
    """
    for col in COLUMN_NAMES:
        if col not in runtable.columns:
            runtable[col] = 0

    if hist is None:
        logger.info("Loading history data for rolling features...")
        train_df, valid_df, test_df = load_all_data(cfg)
        hist = pd.concat([train_df, valid_df, test_df], axis=0).reset_index(drop=True)

    from src.api.replay_client import date_str_of
    hist = hist[date_str_of(hist) < date_str].reset_index(drop=True)
    logger.info("History rows (< %s): %d", date_str, len(hist))

    pipeline = FeaturePipeline.load(os.path.join(model_dir, "pipeline.pkl"), cfg)

    hist["_is_predict"] = False
    runtable = runtable.copy()
    runtable["_is_predict"] = True
    combined = pd.concat([hist, runtable], axis=0).reset_index(drop=True)

    transformed = pipeline.transform(combined)
    day_feat = transformed[transformed["_is_predict"] == True].reset_index(drop=True)  # noqa: E712
    day_feat.drop(columns=["_is_predict"], inplace=True, errors="ignore")
    return day_feat


def load_market_bundle(model_dir: str):
    """市場残差ヘッド (モデル + キャリブレータ + 特徴量リスト) をロードする。

    存在しなければ None を返し、呼び出し側は favorite_concentration に
    フォールバックする。
    """
    model_path = os.path.join(model_dir, "win_market_model.txt")
    meta_path = os.path.join(model_dir, "win_market_meta.pkl")
    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        return None
    model = load_win_model(model_path)
    meta = load_win_meta(meta_path)
    cal_path = os.path.join(model_dir, "win_market_calibrator.pkl")
    calibrator = HoldoutCalibrator.load(cal_path) if os.path.exists(cal_path) else None
    logger.info("Market-residual head loaded (%d features)", len(meta["feature_columns"]))
    return {"model": model, "calibrator": calibrator, "feature_columns": meta["feature_columns"]}


def decide_bet_ev_market(
    race_rows: pd.DataFrame,
    win_odds_df: pd.DataFrame,
    strat: dict,
    market_bundle: dict | None,
    bankroll: float,
    big_amount: int,
) -> dict | None:
    """ev_market モードの投票判定。

    流れ:
      1. 5 分前オッズを全馬に結合 → 市場特徴量 3 列を計算
      2. 市場残差モデルで全馬の勝率を予測 → Isotonic 校正
      3. EV = 校正後確率 × オッズ が最大の馬を選び、
         EV ≥ ev_min_ev なら Kelly サイズで投票 (bet_kind="ev")
      4. 条件を満たさなければ favorite_concentration ロジックにフォールバック
         (bet_kind="fav_big" または "min") — 制約充足を常に保証する
    """
    if market_bundle is None or win_odds_df.empty:
        return decide_bet(race_rows, win_odds_df, strat, big_amount=big_amount)

    rows = race_rows.copy()
    rows["comb"] = rows["horse_num"].astype(int).astype(str).str.zfill(2)
    merged = rows.merge(win_odds_df, on="comb", how="inner")
    if merged.empty:
        return decide_bet(race_rows, win_odds_df, strat, big_amount=big_amount)

    # 学習時の win_odds (確定オッズ) の位置に 5 分前オッズを差し込む
    merged["win_odds"] = pd.to_numeric(merged["odds"], errors="coerce").fillna(0)
    merged = merged[merged["win_odds"] > 0]
    if merged.empty:
        return decide_bet(race_rows, win_odds_df, strat, big_amount=big_amount)

    merged = add_market_features(merged)
    feat_cols = market_bundle["feature_columns"]
    for c in feat_cols:
        if c not in merged.columns:
            merged[c] = 0.0
    raw = market_bundle["model"].predict(merged[feat_cols])
    p_cal = market_bundle["calibrator"].predict(raw) if market_bundle["calibrator"] else raw
    merged["p_cal"] = p_cal
    merged["ev"] = merged["p_cal"] * merged["win_odds"]

    best = merged.sort_values("ev", ascending=False).iloc[0]
    if (
        float(best["ev"]) >= float(strat["ev_min_ev"])
        and float(best["win_odds"]) <= float(strat["ev_max_odds"])
    ):
        amount = compute_bet_amount(
            prob=float(best["p_cal"]),
            odds=float(best["win_odds"]),
            bankroll=bankroll,
            fraction=float(strat["ev_kelly_fraction"]),
            max_bet_fraction=float(strat["ev_max_bet_fraction"]),
            min_bet=int(strat["min_bet_unit"]),
            per_bet_cap=float(strat["ev_max_bet"]),
        )
        if amount > 0:
            return {
                "horse_num": int(best["horse_num"]), "amount": int(amount),
                "bet_kind": "ev", "pred_prob": float(best["p_cal"]),
                "odds": float(best["win_odds"]), "ev": float(best["ev"]),
            }

    # EV 条件を満たさない → 損失最小化ロジックへ (制約充足はここが担保)
    return decide_bet(race_rows, win_odds_df, strat, big_amount=big_amount)


_WIN_BUNDLE_CACHE: dict[str, dict] = {}


def _load_win_bundle(model_dir: str) -> dict:
    """単勝モデル・メタ・キャリブレータを一度だけ読み込んでキャッシュする
    (レース毎の再予測で毎回ディスクから読まないため)。"""
    if model_dir not in _WIN_BUNDLE_CACHE:
        win_meta = load_win_meta(os.path.join(model_dir, "win_meta.pkl"))
        assert_no_market_info(win_meta["feature_columns"])
        cal_path = os.path.join(model_dir, "win_calibrator.pkl")
        _WIN_BUNDLE_CACHE[model_dir] = {
            "model": load_win_model(os.path.join(model_dir, "win_model.txt")),
            "feature_columns": win_meta["feature_columns"],
            "calibrator": HoldoutCalibrator.load(cal_path) if os.path.exists(cal_path) else None,
        }
        if _WIN_BUNDLE_CACHE[model_dir]["calibrator"] is None:
            logger.info("No win calibrator found; using raw probs")
    return _WIN_BUNDLE_CACHE[model_dir]


def predict_win_probs(day_feat: pd.DataFrame, model_dir: str, verbose: bool = True) -> pd.DataFrame:
    """単勝モデル (+ キャリブレータがあれば校正) で win_pred_prob を付与する。"""
    b = _load_win_bundle(model_dir)
    day_feat = day_feat.copy()
    for c in b["feature_columns"]:
        if c not in day_feat.columns:
            day_feat[c] = 0.0
    raw = b["model"].predict(day_feat[b["feature_columns"]])
    day_feat["win_pred_prob_raw"] = raw
    day_feat["win_pred_prob"] = b["calibrator"].predict(raw) if b["calibrator"] else raw
    if verbose:
        logger.info("Win probs (calibrated=%s) range: %.4f - %.4f",
                    b["calibrator"] is not None,
                    day_feat["win_pred_prob"].min(), day_feat["win_pred_prob"].max())
    return day_feat


def decide_anaba_bet(
    race_rows: pd.DataFrame,
    win_odds_df: pd.DataFrame,
    strat: dict,
    exclude_horse_num: int | None = None,
) -> dict | None:
    """穴狙いサブベット: 10 倍以上のオッズ帯で校正後勝率が閾値以上の馬に固定 100pt。

    favorite_concentration の primary bet に上乗せする独立ベット。
    - オッズ >= anaba_min_odds かつ <= anaba_max_odds
    - 校正後勝率 >= anaba_min_prob (10 倍 × 10% = EV 1.0 が損益分岐)
    - primary で既に投票した馬 (exclude_horse_num) は同じ bet_id 二重送信を避けて除外
    - 該当馬の中で EV (= prob × odds) が最大の馬 1 頭を選ぶ

    Returns:
        {"horse_num", "amount", "bet_kind"="anaba", "pred_prob", "odds", "ev"} または None
    """
    if not strat.get("anaba_enabled", False):
        return None
    if race_rows.empty or win_odds_df.empty:
        return None

    rows = race_rows.copy()
    rows["comb"] = rows["horse_num"].astype(int).astype(str).str.zfill(2)
    merged = rows.merge(win_odds_df, on="comb", how="inner")
    if merged.empty:
        return None
    merged["odds"] = pd.to_numeric(merged["odds"], errors="coerce")
    merged = merged.dropna(subset=["odds"])

    lo = float(strat["anaba_min_odds"])
    hi = float(strat["anaba_max_odds"])
    minp = float(strat["anaba_min_prob"])
    cand = merged[
        (merged["odds"] >= lo)
        & (merged["odds"] <= hi)
        & (merged["win_pred_prob"] >= minp)
    ]
    if exclude_horse_num is not None:
        cand = cand[cand["horse_num"].astype(int) != int(exclude_horse_num)]
    if cand.empty:
        return None

    cand = cand.copy()
    cand["ev"] = cand["win_pred_prob"] * cand["odds"]
    best = cand.sort_values("ev", ascending=False).iloc[0]

    return {
        "horse_num": int(best["horse_num"]),
        "amount": int(strat["anaba_amount"]),
        "bet_kind": "anaba",
        "pred_prob": float(best["win_pred_prob"]),
        "odds": float(best["odds"]),
        "ev": float(best["ev"]),
    }


def decide_bet(
    race_rows: pd.DataFrame,
    win_odds_df: pd.DataFrame,
    strat: dict,
    big_amount: int = 0,
) -> dict | None:
    """1 レース分の投票判定 (mode により分岐)。

    Returns:
        {"horse_num", "amount", "bet_kind", "pred_prob", "odds", "ev"} または None (見送り)
    """
    if race_rows.empty:
        return None

    rows = race_rows.sort_values("win_pred_prob", ascending=False)
    top = rows.iloc[0]
    prob = float(top["win_pred_prob"])
    horse_num = int(top["horse_num"])

    odds = None
    if not win_odds_df.empty:
        comb = str(horse_num).zfill(2)
        m = win_odds_df[win_odds_df["comb"] == comb]
        if not m.empty:
            odds = float(m["odds"].iloc[0])

    min_unit = int(strat["min_bet_unit"])
    mode = strat.get("mode", "favorite_concentration")

    if mode == "favorite_concentration":
        # 大口判定: top-1 のオッズが fav_max_odds 以下 (圧倒的人気馬 = 期待損失最小帯)
        if odds is not None and 0 < odds <= float(strat["fav_max_odds"]) and big_amount > 0:
            amount = int(big_amount // min_unit) * min_unit
            return {
                "horse_num": horse_num, "amount": amount, "bet_kind": "fav_big",
                "pred_prob": prob, "odds": odds, "ev": prob * odds,
            }
        # それ以外は最低額 (レース数制約用)
        amount = int(int(strat["min_bet_per_race"]) // min_unit) * min_unit
        amount = max(amount, min_unit)
        return {
            "horse_num": horse_num, "amount": amount, "bet_kind": "min",
            "pred_prob": prob, "odds": odds if odds is not None else float("nan"),
            "ev": (prob * odds) if odds else float("nan"),
        }

    # --- mode == "hybrid" (旧 EV フィルタ方式、参考) ---
    if odds is not None and odds > 0 and prob >= strat["win_prob_min"]:
        ev = prob * odds
        if strat["min_odds"] <= odds <= strat["max_odds"] and ev >= strat["min_ev"]:
            amount = int(strat["main_bet_amount"] // min_unit) * min_unit
            return {
                "horse_num": horse_num, "amount": amount, "bet_kind": "main",
                "pred_prob": prob, "odds": odds, "ev": ev,
            }
    amount = int(strat["sub_bet_amount"] // min_unit) * min_unit
    return {
        "horse_num": horse_num, "amount": amount, "bet_kind": "sub",
        "pred_prob": prob, "odds": odds if odds is not None else float("nan"),
        "ev": (prob * odds) if odds else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description="AI 競馬予想マスターズ 当日実行ループ")
    parser.add_argument("--config", default="config/masters_2026.yaml")
    parser.add_argument("--date", default=None, help="対象日 YYYYMMDD (省略時は今日)")
    parser.add_argument("--dry-run", action="store_true", help="投票 API を叩かない")
    parser.add_argument("--skip-wait", action="store_true",
                        help="発走時刻を待たず全レース即判定 (動作確認用。オッズは現時点の値)")
    parser.add_argument("--replay", action="store_true",
                        help="過去日を record_data からオフライン再生 (API 不要。"
                             "dry-run + skip-wait を強制し、終了時に実結果で損益を集計)")
    args = parser.parse_args()

    if args.replay:
        args.dry_run = True
        args.skip_wait = True

    date_str = args.date or datetime.now().strftime("%Y%m%d")
    if not (len(date_str) == 8 and date_str.isdigit()):
        logger.error("日付は YYYYMMDD の 8 桁で指定してください: '%s' (%d 桁)",
                     date_str, len(date_str))
        return
    try:
        datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        logger.error("存在しない日付です: %s", date_str)
        return
    cfg = load_config(args.config)
    model_dir = cfg["model"]["dir"]
    strat = {**DEFAULT_LIVE_STRATEGY, **cfg.get("live", {})}

    os.makedirs("logs", exist_ok=True)
    # dry-run / replay は本番の累積状態 (制約トラッキング) を汚染しないよう別ファイルに分離
    suffix = "_replay" if args.replay else ("_dryrun" if args.dry_run else "")
    bets_log_path = f"logs/masters_live_bets_{date_str}{suffix}.csv"
    state_path = f"logs/masters_live_state{suffix}.json"

    # 累積状態 (期間全体の制約トラッキング)。リプレイは毎回ゼロから開始
    # (前回リプレイの races_bet が残っていると二重投票ガードで全レーススキップになる)
    state = {"races_bet": [], "total_wagered": 0, "remaining_points": None}
    if not args.replay and os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
    logger.info("累積状態: %d レース投票済 / 累計 %s pt",
                len(state["races_bet"]), f"{state['total_wagered']:,}")

    # 投票クライアント
    login_id = os.environ.get("NETKEIBA_LOGIN_ID", "")
    password = os.environ.get("NETKEIBA_PASSWORD", "")
    if not args.dry_run and (not login_id or not password):
        logger.error("NETKEIBA_LOGIN_ID / NETKEIBA_PASSWORD が未設定です。"
                     "--dry-run で動作確認するか、環境変数を設定してください。")
        return
    vote_client = MastersVoteClient(login_id, password, dry_run=args.dry_run)

    hist_all = None
    if args.replay:
        from src.api.replay_client import ReplayDataClient
        logger.info("[REPLAY] Loading history data...")
        train_df, valid_df, test_df = load_all_data(cfg)
        hist_all = pd.concat([train_df, valid_df, test_df], axis=0).reset_index(drop=True)
        data_client = ReplayDataClient(hist_all, date_str)
    else:
        data_client = MastersDataClient()

    # Step 1: 出走表取得
    logger.info("Fetching racecards for %s ...", date_str)
    timetable, runtable = data_client.get_racecards(date_str)
    logger.info("timetable: %d races / runtable: %d horses", len(timetable), len(runtable))
    if timetable.empty or runtable.empty:
        logger.error("出走表が空です。開催日か API 状態を確認してください。")
        return

    # Step 1.5: API スキーマ (14 列) → record_data スキーマ (47 列)
    # 大会 API に無い列は netkeiba (騎手 ID・馬体重・馬場) と履歴 (前走馬体重) で補う。
    # リプレイ (record_data 由来) では normalize は無変更、補完もスキップされる。
    runtable = normalize_runtable(runtable, timetable)
    if not args.replay:
        runtable = enrich_from_netkeiba(runtable)
    if hist_all is None:
        train_df, valid_df, test_df = load_all_data(cfg)
        hist_all = pd.concat([train_df, valid_df, test_df], axis=0).reset_index(drop=True)
    runtable = impute_weight_from_history(runtable, hist_all)

    # Step 2: 特徴量 + 予測
    # 静的特徴量 (46 列) は朝に一度だけ構築。市場特徴量はレース毎に
    # 5 分前オッズ取得後、decide_bet_ev_market 内で都度計算する 2 段構成。
    day_feat = build_day_features(cfg, runtable, model_dir, date_str, hist=hist_all)
    day_feat = predict_win_probs(day_feat, model_dir)
    logger.info("Predicted %d horses", len(day_feat))

    # ev_market モードなら市場残差ヘッドをロード (無ければ自動フォールバック)
    market_bundle = None
    if strat.get("mode") == "ev_market":
        market_bundle = load_market_bundle(model_dir)
        if market_bundle is None:
            logger.warning("win_market モデルが見つからないため favorite_concentration で動作します")

    # race_id 変換列
    day_feat["race_id_odds"] = day_feat["race_id"].apply(race_id_to_odds_id)
    day_feat["race_id_vote"] = day_feat["race_id"].apply(race_id_to_vote_id)

    # Step 3-5: タイムテーブル順に投票ループ
    timetable = timetable.sort_values("start_time").reset_index(drop=True)
    bet_records: list[dict] = []
    # 反映確認待ちの投票 [(record_index, race_id_vote, horse_num, amount, placed_ts)]
    # 仕様上「投票直後は非同期処理のため未反映になりうる (1 分程度あける)」ので、
    # 投票直後ではなく次レース処理時と日次終了時にまとめて確認する
    pending_verify: list[tuple[int, str, int, int, float]] = []

    def flush_verifications(min_age_sec: float = 60.0, force: bool = False) -> None:
        """placed から min_age_sec 以上経った投票の反映を GET /bet でまとめて確認し、
        bet_records の verified を更新してログ CSV を書き直す。"""
        if args.dry_run or not pending_verify:
            return
        now_ts = time.time()
        due = [pv for pv in pending_verify if force or now_ts - pv[4] >= min_age_sec]
        if not due:
            return
        res = vote_client.verify_many([(rid, hn, amt) for _, rid, hn, amt, _ in due])
        for idx, rid, hn, amt, _ in due:
            v = res.get(str(rid))
            bet_records[idx]["verified"] = v
            if v is False:
                logger.error(
                    "★投票がプラットフォーム未反映/不一致 (race=%s 馬番%d %dpt)。"
                    "netkeiba マイページで目視確認し、締切前なら再投票を検討", rid, hn, amt,
                )
            elif v is True:
                logger.info("反映確認 OK: race=%s 馬番%d %dpt", rid, hn, amt)
        done_idx = {pv[0] for pv in due}
        pending_verify[:] = [pv for pv in pending_verify if pv[0] not in done_idx]
        pd.DataFrame(bet_records).to_csv(bets_log_path, index=False)

    # favorite_concentration の大口額を残り日数から動的計算 (朝に 1 回)
    big_amount = compute_big_amount(strat, state, date_str)
    if strat.get("mode", "favorite_concentration") == "favorite_concentration":
        fav_cap = float(strat["fav_max_odds"])
        chunk = int(strat.get("fav_big_chunk_size", 0))
        if fav_cap <= 0:
            big_desc = "大口 [無効化]"
        elif chunk > 0:
            n_chunks = max(1, big_amount // chunk)
            big_desc = f"大口 {big_amount:,} pt (chunk {chunk} pt × {n_chunks} 回、オッズ≤{fav_cap:.1f})"
        else:
            big_desc = f"大口 {big_amount:,} pt (単発、オッズ≤{fav_cap:.1f})"
        logger.info(
            "戦略: favorite_concentration | 最低額 %d pt/レース + %s (目標累計 %s pt, 現在 %s pt)",
            int(strat["min_bet_per_race"]), big_desc,
            f"{int(strat['target_total_wagered']):,}", f"{state['total_wagered']:,}",
        )

    for _, race in timetable.iterrows():
        place = race["place"]
        race_num = int(race["race_num"])
        start_time = str(race["start_time"])

        race_rows = day_feat[
            (day_feat["place"] == place) & (day_feat["race_num"] == race_num)
        ]
        if race_rows.empty:
            logger.warning("%s %dR: runtable に該当なし。スキップ", place, race_num)
            continue

        race_id_odds = str(race_rows["race_id_odds"].iloc[0])
        race_id_vote = str(race_rows["race_id_vote"].iloc[0])

        # 二重投票ガード: クラッシュ後の再実行で投票済みレースに再投票しない
        # (state は投票のたびに書き出しているので、ここに載っていれば投票は完了済み)
        if race_id_vote in state["races_bet"]:
            logger.info("%s %dR: 投票済み (再実行スキップ)", place, race_num)
            continue

        # 発走時刻まで待機
        target = datetime.strptime(start_time, "%H:%M").replace(
            year=datetime.now().year, month=datetime.now().month, day=datetime.now().day,
        )
        if not args.skip_wait:
            wait_sec = (target - datetime.now()).total_seconds() - strat["odds_fetch_before_sec"]
            if wait_sec < -strat["odds_fetch_before_sec"] + strat["bet_deadline_sec"]:
                logger.info("%s %dR (%s): 締切超過のためスキップ", place, race_num, start_time)
                continue
            if wait_sec > 0:
                logger.info("%s %dR (%s): オッズ取得まで %d 秒待機",
                            place, race_num, start_time, int(wait_sec))
                time.sleep(wait_sec)

        # 前レースまでの投票の反映確認 (60 秒以上経過分)
        flush_verifications()

        # 発走直前に公開される実測馬体重で weight 系特徴量を更新して再予測
        # (朝は前走値で補完しているため。失敗時は朝の値のまま)
        if not args.replay:
            race_rows, changed = refresh_live_weight(race_rows, race_id_vote)
            if changed:
                race_rows = predict_win_probs(race_rows, model_dir, verbose=False)

        # 5 分前オッズ取得 (失敗しても sub 投票は続行)
        try:
            win_odds_df = data_client.get_win_odds(race_id_odds)
        except Exception as e:
            logger.warning("%s %dR: オッズ取得失敗 (%s)。sub 投票にフォールバック", place, race_num, e)
            win_odds_df = pd.DataFrame(columns=["comb", "odds"])

        # 投票判定 (ev_market モードは市場残差モデルで EV 判定 → 不成立なら
        # favorite_concentration ロジックへ内部フォールバック)
        current_bankroll = float(state.get("remaining_points") or 1_000_000)
        if strat.get("mode") == "ev_market":
            decision = decide_bet_ev_market(
                race_rows, win_odds_df, strat, market_bundle,
                bankroll=current_bankroll, big_amount=big_amount,
            )
        else:
            decision = decide_bet(race_rows, win_odds_df, strat, big_amount=big_amount)
        if decision is None:
            logger.info("%s %dR: 見送り", place, race_num)
            continue

        # 投票実行 (締切 = 発走 bet_deadline_sec 前。それまでの間はリトライ可)
        deadline_ts = target.timestamp() - float(strat["bet_deadline_sec"])
        # fav_big の場合、単発大口は API バグで無効化される事象があるため
        # fav_big_chunk_size に従って複数回に分けて送る。100pt は必ず成立する仕様。
        chunk_size = int(strat.get("fav_big_chunk_size", 0))
        is_fav_big = decision.get("bet_kind") == "fav_big"
        if is_fav_big and chunk_size > 0 and decision["amount"] > chunk_size:
            total = int(decision["amount"])
            unit = int(strat["min_bet_unit"])
            chunk = (chunk_size // unit) * unit
            n_full = total // chunk
            rem = total - n_full * chunk
            chunks = [chunk] * n_full + ([rem] if rem >= unit else [])
            logger.info(
                "%s %dR: [fav_big] 馬番%d を %d pt × %d 回に分割して送信 (合計 %d pt)",
                place, race_num, decision["horse_num"], chunk, len(chunks), total,
            )
            actual_amount = 0
            fail_count = 0
            budget = float(strat.get("fav_big_chunk_max_time_sec", 100))
            t0 = time.time()
            first_result = None
            for i, amt in enumerate(chunks):
                if time.time() - t0 > budget:
                    logger.warning("chunk 時間予算超過、%d/%d で打切", i, len(chunks))
                    break
                if deadline_ts and not args.skip_wait and time.time() >= deadline_ts:
                    logger.warning("chunk 締切超過、%d/%d で打切", i, len(chunks))
                    break
                try:
                    r = vote_client.place_win_bet(
                        race_id_vote, decision["horse_num"], amt,
                        deadline_ts=None if args.skip_wait else deadline_ts,
                    )
                    actual_amount += amt
                    if first_result is None:
                        first_result = r
                    else:
                        first_result = r  # 最後の残高を採用
                except Exception as e:
                    fail_count += 1
                    logger.warning("chunk %d/%d 失敗: %s", i+1, len(chunks), e)
                    if fail_count >= 3:
                        logger.error("chunk 連続失敗 3 回で打切")
                        break
            if first_result is None:
                logger.error("%s %dR: fav_big 全 chunk 失敗", place, race_num)
                continue
            # decision の amount を実際に成立した合計に更新
            decision["amount"] = actual_amount
            result = first_result
        else:
            try:
                result = vote_client.place_win_bet(
                    race_id_vote, decision["horse_num"], decision["amount"],
                    deadline_ts=None if args.skip_wait else deadline_ts,
                )
            except Exception as e:
                logger.error("%s %dR: 投票失敗 (%s)", place, race_num, e)
                continue

        if result.remaining_points is not None:
            state["remaining_points"] = result.remaining_points
        if race_id_vote not in state["races_bet"]:
            state["races_bet"].append(race_id_vote)
        state["total_wagered"] += decision["amount"]

        record = {
            "date": date_str, "place": place, "race_num": race_num,
            "start_time": start_time, "race_id_vote": race_id_vote,
            **decision,
            "remaining_points": state["remaining_points"],
            "verified": result.verified,
            "dry_run": args.dry_run,
        }
        bet_records.append(record)
        if not args.dry_run:
            pending_verify.append(
                (len(bet_records) - 1, race_id_vote, int(decision["horse_num"]),
                 int(decision["amount"]), time.time()))
        logger.info(
            "%s %dR: [%s] 馬番%d に %d pt (prob=%.3f odds=%s ev=%s) 残=%s",
            place, race_num, decision["bet_kind"], decision["horse_num"],
            decision["amount"], decision["pred_prob"],
            f"{decision['odds']:.1f}" if pd.notna(decision["odds"]) else "?",
            f"{decision['ev']:.2f}" if pd.notna(decision["ev"]) else "?",
            state["remaining_points"],
        )

        # 都度ログを書き出し (途中クラッシュ対策)
        pd.DataFrame(bet_records).to_csv(bets_log_path, index=False)
        with open(state_path, "w") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

        # --- 穴狙いサブベット (100pt 独立) ---
        anaba = decide_anaba_bet(
            race_rows, win_odds_df, strat,
            exclude_horse_num=decision["horse_num"],
        )
        if anaba is not None:
            try:
                result2 = vote_client.place_win_bet(
                    race_id_vote, anaba["horse_num"], anaba["amount"],
                    deadline_ts=None if args.skip_wait else deadline_ts,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("%s %dR: 穴狙い投票失敗 (%s)", place, race_num, e)
                result2 = None
            if result2 is not None:
                if result2.remaining_points is not None:
                    state["remaining_points"] = result2.remaining_points
                state["total_wagered"] += anaba["amount"]
                # 同じ race_id_vote は races_bet に追加しない (レース数は増えない)
                record2 = {
                    "date": date_str, "place": place, "race_num": race_num,
                    "start_time": start_time, "race_id_vote": race_id_vote,
                    **anaba,
                    "remaining_points": state["remaining_points"],
                    "verified": None,
                    "dry_run": args.dry_run,
                }
                bet_records.append(record2)
                if not args.dry_run:
                    pending_verify.append(
                        (len(bet_records) - 1, race_id_vote, int(anaba["horse_num"]),
                         int(anaba["amount"]), time.time()))
                logger.info(
                    "%s %dR: [anaba] 馬番%d に %d pt (prob=%.3f odds=%.1f ev=%.2f) 残=%s",
                    place, race_num, anaba["horse_num"], anaba["amount"],
                    anaba["pred_prob"], anaba["odds"], anaba["ev"],
                    state["remaining_points"],
                )
                pd.DataFrame(bet_records).to_csv(bets_log_path, index=False)
                with open(state_path, "w") as f:
                    json.dump(state, f, ensure_ascii=False, indent=2)

        # --- 多重ベット (三連複・三連単・馬単・馬連) ---
        if strat.get("multi_enabled", False) and not args.replay:
            try:
                all_odds = data_client.get_all_odds(race_id_odds)
            except Exception as e:  # noqa: BLE001
                logger.warning("%s %dR: 全券種オッズ取得失敗 (%s)", place, race_num, e)
                all_odds = pd.DataFrame(columns=["comb", "odds_type", "odds"])
            if not all_odds.empty:
                sorted_rows = race_rows.sort_values("horse_num").reset_index(drop=True)
                horse_nums_arr = sorted_rows["horse_num"].astype(int).to_numpy()
                win_probs_arr = sorted_rows["win_pred_prob"].to_numpy()
                multi_cands = select_multi_bets(
                    horse_nums_arr, win_probs_arr, all_odds, strat,
                )
                if multi_cands:
                    # split=True: 1 買い目ずつ独立に送信 (bet_id 形式不明でも
                    # 通ったものだけ集計、失敗したものはスキップ)
                    try:
                        results3 = vote_client.place_multi_bet(
                            race_id_vote, multi_cands,
                            deadline_ts=None if args.skip_wait else deadline_ts,
                            use_padded_bet_id=bool(strat.get("multi_use_padded_bet_id", False)),
                            split=True,
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning("%s %dR: 多重ベット送信失敗 (%s)", place, race_num, e)
                        results3 = []
                    n_ok = 0; n_fail = 0
                    for cand, res in zip(multi_cands, results3 or []):
                        if res is None:
                            n_fail += 1
                            logger.warning(
                                "%s %dR: [%s] 買い目 %s @%.1f (%d pt) → 拒否",
                                place, race_num, cand.bet_type,
                                "-".join(str(h) for h in cand.horses), cand.odds, cand.amount,
                            )
                            continue
                        n_ok += 1
                        if res.remaining_points is not None:
                            state["remaining_points"] = res.remaining_points
                        state["total_wagered"] += int(cand.amount)
                        rec = {
                            "date": date_str, "place": place, "race_num": race_num,
                            "start_time": start_time, "race_id_vote": race_id_vote,
                            "horse_num": cand.horses[0],
                            "amount": int(cand.amount),
                            "bet_kind": cand.bet_type,
                            "pred_prob": cand.joint_prob,
                            "odds": cand.odds,
                            "ev": cand.ev,
                            "remaining_points": state["remaining_points"],
                            "verified": None,
                            "dry_run": args.dry_run,
                            "bet_id": cand.bet_id,
                            "horses": "-".join(str(h) for h in cand.horses),
                        }
                        bet_records.append(rec)
                        if not args.dry_run:
                            pending_verify.append(
                                (len(bet_records) - 1, race_id_vote, int(cand.horses[0]),
                                 int(cand.amount), time.time()))
                    if n_ok + n_fail > 0:
                        logger.info(
                            "%s %dR: [multi] 成立 %d / 拒否 %d (合計 %d pt) 残=%s",
                            place, race_num, n_ok, n_fail,
                            sum(int(c.amount) for c, r in zip(multi_cands, results3 or []) if r is not None),
                            state["remaining_points"],
                        )
                        pd.DataFrame(bet_records).to_csv(bets_log_path, index=False)
                        with open(state_path, "w") as f:
                            json.dump(state, f, ensure_ascii=False, indent=2)

        # --- balance-check: API remaining_money から total_wagered を実残ベースで補正 ---
        # fav_big 24,700 が API 応答で成立したのに実際は差引されていなかった事象
        # (8/30 新潟 11R) を検知するため、初期額 100 万 - API 残 を真実として上書き。
        # これにより「引かれていない投票」を total_wagered に加算する事故を防ぐ。
        INITIAL_BANKROLL = 1_000_000
        if state.get("remaining_points") is not None:
            actual = INITIAL_BANKROLL - int(state["remaining_points"])
            if abs(actual - state["total_wagered"]) >= 200:
                logger.warning(
                    "★残高不整合: total_wagered=%d, 実引落=%d (差 %+d pt)。"
                    "投票が受理されたが差引されていない可能性。マイページで要確認",
                    state["total_wagered"], actual,
                    actual - state["total_wagered"],
                )
                state["total_wagered"] = actual  # 実残ベースに補正
                with open(state_path, "w") as f:
                    json.dump(state, f, ensure_ascii=False, indent=2)

    # 残りの反映確認 (最後の投票から 60 秒以上あけてから全件)
    if pending_verify and not args.dry_run:
        wait = max(0.0, 60.0 - (time.time() - pending_verify[-1][4]))
        if wait > 0 and not args.skip_wait:
            logger.info("最終の反映確認まで %d 秒待機", int(wait))
            time.sleep(wait)
        flush_verifications(force=True)
        n_bad = sum(1 for r in bet_records if r.get("verified") is False)
        n_unk = sum(1 for r in bet_records if r.get("verified") is None)
        logger.info("反映確認: OK %d / 未反映・不一致 %d / 判定不能 %d",
                    len(bet_records) - n_bad - n_unk, n_bad, n_unk)

    # 日次サマリ
    logger.info("=" * 60)
    logger.info("本日の投票: %d 件 / 累積投票レース: %d / 累計投票額: %s pt",
                len(bet_records), len(state["races_bet"]), f"{state['total_wagered']:,}")
    logger.info("コンペ制約: %s 96レース / %s 50万pt",
                "✓" if len(state["races_bet"]) >= 96 else f"{len(state['races_bet'])}/96",
                "✓" if state["total_wagered"] >= 500_000 else f"{state['total_wagered']:,}/500,000")
    if bet_records:
        logger.info("投票ログ: %s", bets_log_path)

    # リプレイなら実結果 (rank / 確定オッズ) で損益を集計
    if args.replay:
        from src.api.replay_client import summarize_replay
        summarize_replay(bet_records, data_client.day_df)


if __name__ == "__main__":
    main()
