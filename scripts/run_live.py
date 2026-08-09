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
    race_id_to_odds_id,
    race_id_to_vote_id,
)

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


def predict_win_probs(day_feat: pd.DataFrame, model_dir: str) -> pd.DataFrame:
    """単勝モデル (+ キャリブレータがあれば校正) で win_pred_prob を付与する。"""
    win_model = load_win_model(os.path.join(model_dir, "win_model.txt"))
    win_meta = load_win_meta(os.path.join(model_dir, "win_meta.pkl"))
    feature_columns = win_meta["feature_columns"]
    assert_no_market_info(feature_columns)

    for c in feature_columns:
        if c not in day_feat.columns:
            day_feat[c] = 0.0
    raw = win_model.predict(day_feat[feature_columns])
    day_feat["win_pred_prob_raw"] = raw

    cal_path = os.path.join(model_dir, "win_calibrator.pkl")
    if os.path.exists(cal_path):
        calibrator = HoldoutCalibrator.load(cal_path)
        day_feat["win_pred_prob"] = calibrator.predict(raw)
        logger.info("Win calibrator applied (prob range: %.4f - %.4f)",
                    day_feat["win_pred_prob"].min(), day_feat["win_pred_prob"].max())
    else:
        day_feat["win_pred_prob"] = raw
        logger.info("No win calibrator found; using raw probs")
    return day_feat


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

    # favorite_concentration の大口額を残り日数から動的計算 (朝に 1 回)
    big_amount = compute_big_amount(strat, state, date_str)
    if strat.get("mode", "favorite_concentration") == "favorite_concentration":
        logger.info(
            "戦略: favorite_concentration | 最低額 %d pt/レース + オッズ≤%.1f の大口 %d pt "
            "(目標累計 %s pt, 現在 %s pt)",
            int(strat["min_bet_per_race"]), float(strat["fav_max_odds"]), big_amount,
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
        try:
            result = vote_client.place_win_bet(
                race_id_vote, decision["horse_num"], decision["amount"],
                deadline_ts=None if args.skip_wait else deadline_ts,
            )
        except Exception as e:
            logger.error("%s %dR: 投票失敗 (%s)", place, race_num, e)
            continue

        if result.verified is False:
            logger.error(
                "%s %dR: ★投票がプラットフォーム未反映の疑い (race=%s)。"
                "netkeiba マイページで目視確認してください", place, race_num, race_id_vote,
            )

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
