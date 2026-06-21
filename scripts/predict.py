#!/usr/bin/env python3
"""レース当日の推論を行う予測エントリーポイント。

使い方:
    python scripts/predict.py --config config/default.yaml --date 20250905
    python scripts/predict.py --config config/default.yaml --race_id 202509050101
    python scripts/predict.py --config config/default.yaml --input race_card.csv
    python scripts/predict.py --config config/default.yaml --race_id 202509050101 --no-odds
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from src.data.loader import load_config, load_train_data, load_valid_data
from src.data.schema import COLUMN_NAMES
from src.features.pipeline import FeaturePipeline, FEATURE_COLUMNS
from src.features.odds_timeseries import (
    EXTRA_ANABA_BASE_COLUMNS,
    TS_ODDS_FEATURE_COLUMNS,
    load_ts_odds_features,
    merge_ts_odds_features,
)
from src.model.trainer import load_model
from src.model.calibrator import HoldoutCalibrator
from src.model.anaba_trainer import load_anaba_meta, load_anaba_model
from src.strategy.selector import select_bets_for_all_races, select_bets_dispatch
from src.strategy.kelly import (
    compute_bet_amount, compute_bet_amount_dispatch, size_bets_per_race,
)
from src.strategy.recommender import generate_full_recommendation
from src.strategy.algorithm import run_full_evaluation, print_evaluation
from src.scraper.race_card import scrape_race_card, scrape_today_races
from src.scraper.odds import scrape_odds, scrape_trio_odds, scrape_trifecta_odds

try:
    from src.model.ensemble import (
        compute_implied_probability, detect_unknown_horses, adaptive_blend,
        get_final_probability,
    )
    _HAS_ENSEMBLE = True
except ImportError:
    _HAS_ENSEMBLE = False


PLACE_CODE_MAP = {
    # JRA (中央競馬)
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
    # NAR (地方競馬)
    "30": "門別", "31": "北見", "32": "岩見沢", "33": "帯広", "34": "旭川",
    "35": "盛岡", "36": "水沢",
    "42": "浦和", "43": "船橋", "44": "大井", "45": "川崎",
    "46": "金沢", "47": "笠松", "48": "名古屋",
    "50": "園田", "51": "姫路",
    "54": "高知",
    "55": "佐賀",
}


def _display_race_info(race_id: str, race_card: pd.DataFrame) -> None:
    """レース情報と出走馬一覧をユーザー確認用に表示する。"""
    # race_cardのカラムを直接利用（scrape_race_cardで既に埋まっている）
    race_name = str(race_card["race_name"].iloc[0]) if "race_name" in race_card.columns and len(race_card) > 0 else ""
    race_name = race_name or race_id

    # race_idの5〜6桁目から開催場コードを取得（0-indexedで4:6）
    place_code = race_id[4:6] if len(race_id) >= 6 else ""
    place_name = PLACE_CODE_MAP.get(place_code, place_code)

    # race_cardから馬場種別と距離を取得
    dist = int(race_card["dist"].iloc[0]) if "dist" in race_card.columns and len(race_card) > 0 else 0
    track_code = int(race_card["track_code"].iloc[0]) if "track_code" in race_card.columns and len(race_card) > 0 else 0
    track_type = "芝" if track_code // 10 == 1 else "ダ" if track_code // 10 == 2 else "不明"
    dist_str = f"{track_type}{dist}m" if dist else "不明"

    num_horses = len(race_card)

    print("\n" + "=" * 50)
    print(f"  レース情報: {race_name}")
    print(f"  場所: {place_name}  距離: {dist_str}  出走頭数: {num_horses}頭")
    print("=" * 50)
    print(f"  {'馬番':>4s}  {'馬名':<14s}  {'騎手ID':>8s}  {'父馬'}")
    print("  " + "-" * 50)

    for _, row in race_card.iterrows():
        horse_num = int(row.get("horse_num", 0))
        horse_name = row.get("horse", "")
        jockey_id = row.get("jockey_id", "")
        jockey_str = str(int(jockey_id)) if jockey_id else "---"
        sire = row.get("father", "") or "---"

        print(f"  {horse_num:>4d}  {horse_name:<14s}  {jockey_str:>8s}  {sire}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Predict and recommend bets")
    parser.add_argument("--config", default="config/default.yaml", help="Config file path")
    parser.add_argument("--input", default=None, help="Race card CSV (legacy mode)")
    parser.add_argument("--race_id", default=None, help="Specific race ID (e.g. 202509050101)")
    parser.add_argument("--date", default=None, help="Target date YYYYMMDD (all races)")
    parser.add_argument("--bankroll", type=float, default=None, help="Current bankroll")
    parser.add_argument("--no-odds", action="store_true",
                        help="Skip odds fetching (threshold mode)")
    parser.add_argument("--algorithm", action="store_true",
                        help="Use comprehensive evaluation algorithm (総合評価アルゴリズム)")
    parser.add_argument("--ev-only", action="store_true",
                        help="EV >= min_ev の組み合わせのみ全件出力 (品質ガード付き)")
    parser.add_argument("--min-ev", type=float, default=None,
                        help="--ev-only 時の EV 閾値 (デフォルト 1.0)")
    parser.add_argument("--top-n-combo", type=int, default=None,
                        help="--ev-only 時に三連複/三連単で何頭から組み合わせを作るか (デフォルト 10)")
    parser.add_argument("--min-prob-trio", type=float, default=0.005,
                        help="--ev-only 時の三連複の最小的中確率 (デフォルト 0.5%)")
    parser.add_argument("--min-prob-trifecta", type=float, default=0.002,
                        help="--ev-only 時の三連単の最小的中確率 (デフォルト 0.2%)")
    parser.add_argument("--max-odds-trio", type=float, default=200.0,
                        help="--ev-only 時の三連複の最大オッズ (デフォルト 200倍)")
    parser.add_argument("--max-odds-trifecta", type=float, default=1000.0,
                        help="--ev-only 時の三連単の最大オッズ (デフォルト 1000倍)")
    parser.add_argument("--apply-takeout", action="store_true",
                        help="JRA 控除率を考慮した実勢 EV でフィルタ (三連複 -25%, 三連単 -27.5%)")
    parser.add_argument("--no-quality-guard", action="store_true",
                        help="品質ガード (min-prob/max-odds) を全部無効化 (前と同じ生の EV 全件)")
    args = parser.parse_args()

    if not args.input and not args.race_id and not args.date:
        parser.error("One of --input, --race_id, or --date is required")

    cfg = load_config(args.config)
    strat = cfg["strategy"]
    selection_method = strat.get("selection_method", "threshold")
    bet_sizing = strat.get("bet_sizing", "tier")

    # 競合チェック: --no-odds と ev 方式の組み合わせ
    if args.no_odds and selection_method == "ev":
        print("WARNING: --no-odds specified but selection_method='ev' requires odds. "
              "Falling back to selection_method='threshold'.")
        selection_method = "threshold"
        bet_sizing = "tier"

    model_dir = cfg["model"]["dir"]

    # モデル関連の資産を読み込む
    print("Loading model...")
    model = load_model(os.path.join(model_dir, "lgbm_model.txt"))
    calibrator = HoldoutCalibrator.load(os.path.join(model_dir, "calibrator.pkl"))
    pipeline = FeaturePipeline.load(os.path.join(model_dir, "pipeline.pkl"), cfg)

    # ローリング特徴量用に過去データを読み込む
    print("Loading historical data...")
    train_df = load_train_data(cfg)
    valid_df = load_valid_data(cfg)
    hist_df = pd.concat([train_df, valid_df], axis=0).reset_index(drop=True)

    # 出馬表データを構築
    trio_odds_df = None
    trifecta_odds_df = None
    if args.input:
        print(f"Loading race card from {args.input}...")
        race_card = pd.read_csv(args.input, encoding="utf-8", low_memory=False)
    elif args.race_id:
        print(f"Scraping race card for {args.race_id}...")
        race_card = scrape_race_card(args.race_id)
        if race_card is None or race_card.empty:
            print("Failed to scrape race card.")
            return

        # ユーザー確認用にレース情報を表示
        _display_race_info(args.race_id, race_card)

        # オッズ（単勝・複勝）を取得。--no-odds指定時はスキップ
        if args.no_odds:
            print("Skipping odds fetch (--no-odds).")
        else:
            print("Fetching odds...")
            odds_df = scrape_odds(args.race_id)
            if odds_df is not None and not odds_df.empty:
                # 単勝オッズをマージ
                if "win_odds" not in race_card.columns:
                    race_card = race_card.merge(
                        odds_df[["horse_num", "win_odds"]],
                        on="horse_num", how="left",
                    )
                else:
                    race_card = race_card.merge(
                        odds_df[["horse_num", "win_odds"]].rename(
                            columns={"win_odds": "win_odds_rt"}
                        ),
                        on="horse_num", how="left",
                    )
                    race_card["win_odds"] = race_card["win_odds_rt"].fillna(
                        race_card["win_odds"]
                    )
                    race_card.drop(columns=["win_odds_rt"], inplace=True)
                # 複勝オッズをマージ
                show_cols = ["horse_num", "show_odds_min", "show_odds_max"]
                if all(c in odds_df.columns for c in show_cols):
                    race_card = race_card.merge(
                        odds_df[show_cols], on="horse_num", how="left",
                    )

                # 三連複・三連単オッズを取得
                print("Fetching trio/trifecta odds...")
                time.sleep(2)
                try:
                    trio_odds_df = scrape_trio_odds(args.race_id)
                    if trio_odds_df.empty:
                        trio_odds_df = None
                except Exception as e:
                    print(f"  Trio odds fetch failed: {e}")
                time.sleep(2)
                try:
                    trifecta_odds_df = scrape_trifecta_odds(args.race_id)
                    if trifecta_odds_df.empty:
                        trifecta_odds_df = None
                except Exception as e:
                    print(f"  Trifecta odds fetch failed: {e}")
    else:
        print(f"Scraping race schedule for {args.date}...")
        races = scrape_today_races(args.date)
        if not races:
            print("No races found for this date.")
            return
        frames = []
        for race_info in races:
            rid = race_info["race_id"]
            print(f"  Scraping {rid}...")
            df = scrape_race_card(rid)
            if df is not None and not df.empty:
                frames.append(df)
            time.sleep(1)
        if not frames:
            print("No race data scraped.")
            return
        race_card = pd.concat(frames, axis=0).reset_index(drop=True)

    # 必須カラムが存在することを保証
    for col in COLUMN_NAMES:
        if col not in race_card.columns:
            race_card[col] = 0

    # ローリング特徴量計算のため過去データと結合
    # パイプラインによる並び替え後も予測対象行を特定できるようマーク
    hist_df["_is_predict"] = False
    race_card["_is_predict"] = True
    all_data = pd.concat([hist_df, race_card], axis=0).reset_index(drop=True)
    all_transformed = pipeline.transform(all_data)

    # マーカーを使って出馬表の行だけを抽出（パイプラインが並び替えを行うため）
    race_feat = all_transformed[all_transformed["_is_predict"] == True].reset_index(drop=True)  # noqa: E712
    all_transformed.drop(columns=["_is_predict"], inplace=True, errors="ignore")
    race_feat.drop(columns=["_is_predict"], inplace=True, errors="ignore")
    hist_df.drop(columns=["_is_predict"], inplace=True, errors="ignore")
    available_features = [c for c in FEATURE_COLUMNS if c in race_feat.columns]

    # 予測
    raw_probs = model.predict(race_feat[available_features])
    race_feat["pred_prob"] = calibrator.predict(raw_probs)

    # 穴馬ヘッドの推論 (有効化されているとき)
    anaba_model_path = os.path.join(model_dir, "anaba_model.txt")
    anaba_meta_path = os.path.join(model_dir, "anaba_meta.pkl")
    if (
        cfg.get("anaba", {}).get("enabled", False)
        and os.path.exists(anaba_model_path)
        and os.path.exists(anaba_meta_path)
    ):
        print("Predicting anaba (穴馬) head...")
        try:
            anaba_model = load_anaba_model(anaba_model_path)
            anaba_meta = load_anaba_meta(anaba_meta_path)
            anaba_feature_columns = anaba_meta["feature_columns"]
            use_ts_odds = anaba_meta.get("use_ts_odds", False)

            anaba_race_feat = race_feat.copy()
            if use_ts_odds:
                # 当日レースは Time_Series_Odds に未収録 → has_ts_odds=0 フォールバック
                ts_features = pd.DataFrame()
                anaba_race_feat = merge_ts_odds_features(anaba_race_feat, ts_features)
            else:
                for col in TS_ODDS_FEATURE_COLUMNS:
                    if col not in anaba_race_feat.columns:
                        anaba_race_feat[col] = 0.0
                if "has_ts_odds" not in anaba_race_feat.columns:
                    anaba_race_feat["has_ts_odds"] = 0

            for col in EXTRA_ANABA_BASE_COLUMNS:
                if col not in anaba_race_feat.columns:
                    anaba_race_feat[col] = 0.0
                else:
                    anaba_race_feat[col] = pd.to_numeric(
                        anaba_race_feat[col], errors="coerce"
                    ).fillna(0.0)
            for col in anaba_feature_columns:
                if col not in anaba_race_feat.columns:
                    anaba_race_feat[col] = 0.0
            race_feat["anaba_prob"] = anaba_model.predict(
                anaba_race_feat[anaba_feature_columns]
            )
        except Exception as e:
            print(f"  WARNING: anaba prediction failed ({e}); skipping.")
            race_feat["anaba_prob"] = 0.0
    else:
        race_feat["anaba_prob"] = 0.0

    # 確率アンサンブル（コールドスタート対策）
    if _HAS_ENSEMBLE:
        race_key = ["year", "month", "day", "place", "race_num"]
        available_keys = [k for k in race_key if k in race_feat.columns]

        if selection_method == "threshold":
            print("Applying threshold ensemble...")
            if available_keys:
                for _, grp in race_feat.groupby(available_keys):
                    idx = grp.index
                    race_feat.loc[idx, "pred_prob"] = get_final_probability(
                        grp["pred_prob"], grp, method="threshold",
                    )
            else:
                race_feat["pred_prob"] = get_final_probability(
                    race_feat["pred_prob"], race_feat, method="threshold",
                )
        elif selection_method == "ev" and "win_odds" in race_feat.columns and (race_feat["win_odds"] > 0).any():
            print("Applying odds-implied ensemble (legacy)...")
            if available_keys:
                for _, grp in race_feat.groupby(available_keys):
                    idx = grp.index
                    race_feat.loc[idx, "pred_prob"] = get_final_probability(
                        grp["pred_prob"], grp, method="ev",
                        win_odds=grp["win_odds"],
                    )
            else:
                race_feat["pred_prob"] = get_final_probability(
                    race_feat["pred_prob"], race_feat, method="ev",
                    win_odds=race_feat["win_odds"],
                )

    # ================================================================
    # アルゴリズムモード: 総合評価
    # ================================================================
    use_algorithm = args.algorithm or strat.get("algorithm", {}).get("enabled", False)
    if use_algorithm:
        print("\n=== 総合評価アルゴリズム実行中 ===")
        race_key = ["year", "month", "day", "place", "race_num"]
        available_keys = [k for k in race_key if k in race_feat.columns]

        if available_keys:
            for group_key, grp in race_feat.groupby(available_keys):
                evaluation = run_full_evaluation(grp, hist_df=hist_df, cfg=cfg)
                print_evaluation(evaluation)
        else:
            evaluation = run_full_evaluation(race_feat, hist_df=hist_df, cfg=cfg)
            print_evaluation(evaluation)

        print("\n=== 従来方式の参考出力 ===")

    # 馬券対象を選定
    bet_df = select_bets_dispatch(
        race_feat,
        method=selection_method,
        top_n_popularity=strat.get("top_n_popularity", 3),
        min_expected_value=strat.get("min_expected_value", 1.0),
        prob_threshold=strat.get("prob_threshold", 0.3),
        max_popularity=strat.get("max_popularity", 3),
    )

    if selection_method == "threshold":
        method_label = "Threshold"
    else:
        method_label = "Legacy EV"

    if bet_df.empty:
        print(f"\nNo bets recommended ({method_label}).")
    else:
        bankroll_val = args.bankroll or strat["initial_bankroll"]
        per_race_cap = float(strat.get("per_race_cap", 10_000))

        if bet_sizing == "tier":
            bet_df["bet_amount"] = bet_df["pred_prob"].apply(
                lambda p: compute_bet_amount_dispatch(
                    p, method="tier",
                    tier_low_threshold=strat.get("tier_low_threshold", 0.3),
                    tier_mid_threshold=strat.get("tier_mid_threshold", 0.4),
                    tier_high_threshold=strat.get("tier_high_threshold", 0.5),
                    tier_low_amount=strat.get("tier_low_amount", 200),
                    tier_mid_amount=strat.get("tier_mid_amount", 500),
                    tier_high_amount=strat.get("tier_high_amount", 1500),
                ),
            )
        else:
            # Kelly + per_race_cap (オッズ無しは確率按分にフォールバック)
            if "win_odds" in bet_df.columns:
                bet_df["show_odds_est"] = bet_df["win_odds"] / 3.0
                odds_vals = bet_df["show_odds_est"].tolist()
            else:
                bet_df["show_odds_est"] = float("nan")
                odds_vals = None
            bet_df["bet_amount"] = size_bets_per_race(
                probs=bet_df["pred_prob"].tolist(),
                odds=odds_vals,
                bankroll=bankroll_val,
                per_race_cap=per_race_cap,
                fraction=strat.get("kelly_fraction", 0.25),
                max_bet_fraction=strat.get("max_bet_fraction", 0.05),
                min_bet=strat.get("min_bet", 100),
                min_prob=strat.get("prob_threshold", 0.30),
            )

        if "win_odds" in bet_df.columns:
            bet_df["show_odds_est"] = bet_df.get("show_odds_est", bet_df["win_odds"] / 3.0)
            bet_df["expected_value"] = (bet_df["pred_prob"] * bet_df["show_odds_est"]).round(2)

        display_cols = ["place", "race_num", "horse", "horse_num", "pred_prob",
                        "win_odds", "expected_value", "jockey_lcb95", "sire_lcb95",
                        "bet_amount"]
        display_cols = [c for c in display_cols if c in bet_df.columns]

        print("\n" + "=" * 80)
        print(f"RECOMMENDED HORSES ({method_label} / sizing={bet_sizing})")
        print("=" * 80)
        print(bet_df[display_cols].to_string(index=False))
        total_bet = float(bet_df["bet_amount"].sum()) if "bet_amount" in bet_df.columns else 0.0
        n_betting = int((bet_df["bet_amount"] > 0).sum()) if "bet_amount" in bet_df.columns else 0
        print(f"\n選定: {len(bet_df)} 頭 / 実際に賭ける: {n_betting} 頭")
        print(f"  バンクロール:        {bankroll_val:>10,.0f} 円")
        print(f"  1レース予算上限:     {per_race_cap:>10,.0f} 円")
        print(f"  この推奨の合計:      {total_bet:>10,.0f} 円 ({total_bet/max(bankroll_val,1):.2%} of bankroll)")
        if bet_sizing == "kelly" and total_bet == 0.0 and n_betting == 0:
            print("  → Kelly判定: 推定 EV < 1.0 → 賭けない方が良い (この方式では掛金0)。")
            print("    『複勝推奨 (Show)』『単勝推奨 (Win)』のティア方式の額も参考にしてください。")
        elif "expected_value" in bet_df.columns and not bet_df["expected_value"].isna().all() and total_bet > 0:
            roi = (bet_df["bet_amount"] * bet_df["expected_value"]).sum() - total_bet
            sign = "+" if roi >= 0 else ""
            print(f"  期待損益 (Σ bet·EV − Σ bet): {sign}{roi:>10,.0f} 円")

    # ================================================================
    # 複数券種のまとめて推奨
    # ================================================================
    bankroll = args.bankroll or strat["initial_bankroll"]
    print("\n\n" + "=" * 60)
    print("  ALL TICKET RECOMMENDATIONS")
    print("=" * 60)

    common_tier_kwargs = {
        "tier_low_threshold": strat.get("tier_low_threshold", 0.3),
        "tier_mid_threshold": strat.get("tier_mid_threshold", 0.4),
        "tier_high_threshold": strat.get("tier_high_threshold", 0.5),
        "tier_low_amount": strat.get("tier_low_amount", 200),
        "tier_mid_amount": strat.get("tier_mid_amount", 500),
        "tier_high_amount": strat.get("tier_high_amount", 1500),
    }
    ev_filter = args.ev_only
    min_ev_show = args.min_ev if args.min_ev is not None else 1.0
    top_n_for_combo = args.top_n_combo if args.top_n_combo is not None else (
        10 if ev_filter else strat.get("trio_top_n", 5)
    )
    quality_guard = ev_filter and not args.no_quality_guard
    # JRA 控除率: 三連複 25%, 三連単 27.5% (どちらも対象なら apply-takeout 必要)
    takeout_trio = 0.25 if (ev_filter and args.apply_takeout) else 0.0
    takeout_trifecta = 0.275 if (ev_filter and args.apply_takeout) else 0.0
    guard_kwargs = {}
    if quality_guard:
        guard_kwargs = {
            "min_prob_trio": args.min_prob_trio,
            "min_prob_trifecta": args.min_prob_trifecta,
            "max_odds_trio": args.max_odds_trio,
            "max_odds_trifecta": args.max_odds_trifecta,
        }
    try:
        recs = generate_full_recommendation(
            race_feat,
            min_ev=min_ev_show if ev_filter else strat.get("show_min_ev", 1.0),
            bankroll=bankroll,
            kelly_frac=strat.get("kelly_fraction", 0.25),
            top_n=top_n_for_combo,
            trio_odds_df=trio_odds_df,
            trifecta_odds_df=trifecta_odds_df,
            method=selection_method,
            prob_threshold=strat.get("prob_threshold", 0.3),
            ev_filter=ev_filter,
            takeout_trio=takeout_trio,
            takeout_trifecta=takeout_trifecta,
            **guard_kwargs,
            **common_tier_kwargs,
        )
    except TypeError:
        recs = generate_full_recommendation(
            race_feat,
            min_ev=min_ev_show if ev_filter else strat.get("show_min_ev", 1.0),
            bankroll=bankroll,
            kelly_frac=strat.get("kelly_fraction", 0.25),
            top_n=top_n_for_combo,
            trio_odds_df=trio_odds_df,
            trifecta_odds_df=trifecta_odds_df,
            **common_tier_kwargs,
        )

    if ev_filter:
        print(f"\n*** EV-ONLY MODE: EV ≥ {min_ev_show:.2f} の組み合わせを全件表示 (top_n_combo={top_n_for_combo}) ***")
        if quality_guard:
            print(f"    品質ガード: 三連複[prob≥{args.min_prob_trio:.1%} & odds≤{args.max_odds_trio:.0f}]  "
                  f"三連単[prob≥{args.min_prob_trifecta:.1%} & odds≤{args.max_odds_trifecta:.0f}]")
        else:
            print("    品質ガード: なし (--no-quality-guard)")
        if args.apply_takeout:
            print(f"    JRA 控除考慮 EV: 三連複×0.75, 三連単×0.725 後に閾値判定")

    # --- 複勝 ---
    _print_show_recs(recs["show"], strat, ev_only=ev_filter)

    # --- 単勝 ---
    _print_win_recs(recs["win"], race_feat, strat, ev_only=ev_filter)

    # --- 穴馬候補 ---
    _print_anaba_recs(race_feat, cfg)

    # --- 三連複 ---
    _print_trio_recs(recs["trio"], strat, ev_only=ev_filter)

    # --- 三連単 ---
    _print_trifecta_recs(recs["trifecta"], strat, ev_only=ev_filter)

    # --- EV プラスの合計 ---
    if ev_filter:
        _print_ev_summary(recs, min_ev_show)


def _print_anaba_recs(race_feat: pd.DataFrame, cfg: dict) -> None:
    """穴馬ヘッドの推奨を表示する。"""
    anaba_cfg = cfg.get("anaba", {})
    if not anaba_cfg.get("enabled", False):
        return
    if "anaba_prob" not in race_feat.columns:
        return
    if float(race_feat["anaba_prob"].max() or 0) <= 0:
        return

    threshold = float(anaba_cfg.get("score_threshold", 0.15))
    min_pop = int(anaba_cfg.get("min_pop", 5))

    print("\n" + "=" * 50)
    print(f"  穴馬候補 (Anaba: ≥{min_pop}番人気で1着想定 / score≥{threshold:.2f})")
    print("=" * 50)

    df = race_feat.copy()
    df["anaba_prob"] = pd.to_numeric(df["anaba_prob"], errors="coerce").fillna(0.0)
    cands = df[df["anaba_prob"] >= threshold].sort_values("anaba_prob", ascending=False)
    if cands.empty:
        top1 = df.sort_values("anaba_prob", ascending=False).head(1)
        if not top1.empty:
            print("  参考: 該当馬なし、最高 anaba_score:")
            for _, row in top1.iterrows():
                _print_anaba_row(row)
        else:
            print("  該当馬なし")
        return
    for _, row in cands.head(5).iterrows():
        _print_anaba_row(row)


def _print_anaba_row(row) -> None:
    name = row.get("horse", "")
    horse_num = int(row.get("horse_num", 0))
    pop = row.get("pop", "")
    pop_str = f"{int(pop)}人気" if pd.notna(pop) and pop != "" else "---"
    win_odds = row.get("win_odds", 0)
    win_str = f"{float(win_odds):.1f}" if pd.notna(win_odds) and float(win_odds) > 0 else "---"
    score = float(row.get("anaba_prob", 0.0))
    main_p = float(row.get("pred_prob", 0.0))
    print(
        f"  {horse_num:>2}番 {name:<10s}  "
        f"穴馬score {score:.3f}  複勝予測 {main_p:.1%}  "
        f"単勝オッズ {win_str:>6s}  {pop_str:>4s}"
    )


def _print_show_recs(show_df: pd.DataFrame, strat: dict, ev_only: bool = False) -> None:
    """複勝の推奨結果を表示する。"""
    print("\n" + "=" * 50)
    print("  複勝推奨 (Show)" + ("  [EV-only]" if ev_only else ""))
    print("=" * 50)
    if show_df.empty:
        print(f"  EV >= {strat.get('show_min_ev', 1.0)} の該当馬なし")
        return
    for _, row in show_df.iterrows():
        name = row.get("horse", "")
        odds_str = f"{row['show_odds_avg']:.1f}" if pd.notna(row.get("show_odds_avg")) else "---"
        ev_str = f"{row['ev']:.2f}" if pd.notna(row.get("ev")) else "---"
        bet_str = f"{row['bet_amount']:,.0f}円" if row.get("bet_amount", 0) > 0 else "---"
        print(f"  {int(row['horse_num']):>2}番 {name:<10s}  "
              f"予測複勝率 {row['pred_prob']:.1%}  "
              f"オッズ(平均) {odds_str:>6s}  "
              f"EV {ev_str:>6s}  "
              f"推奨額 {bet_str:>10s}")
    if "bet_amount" in show_df.columns:
        sub = float(show_df["bet_amount"].fillna(0).sum())
        if sub > 0:
            print(f"  ─ 複勝 小計: {sub:,.0f} 円 ({len(show_df)}頭)")


def _print_win_recs(win_df: pd.DataFrame, race_feat: pd.DataFrame, strat: dict, ev_only: bool = False) -> None:
    """単勝の推奨結果を表示する。"""
    print("\n" + "=" * 50)
    print("  単勝推奨 (Win)" + ("  [EV-only]" if ev_only else ""))
    print("=" * 50)
    if win_df.empty:
        print(f"  EV >= {strat.get('win_min_ev', 1.2)} の該当馬なし（参考: 上位5頭）")
        # 参考として予測確率上位5頭を表示（取消馬は除外）
        valid_horses = race_feat[race_feat.get("win_odds", pd.Series(dtype=float)) > 0] if "win_odds" in race_feat.columns else race_feat
        ref = valid_horses.sort_values("pred_prob", ascending=False).head(5)
        for _, row in ref.iterrows():
            name = row.get("horse", "")
            win_odds = row.get("win_odds", 0)
            odds_str = f"{win_odds:.1f}" if win_odds > 0 else "---"
            print(f"  {int(row['horse_num']):>2}番 {name:<10s}  "
                  f"予測複勝率 {row['pred_prob']:.1%}  "
                  f"単勝オッズ {odds_str:>6s}")
        return
    for _, row in win_df.iterrows():
        name = row.get("horse", "")
        ev_str = f"{row['ev']:.2f}" if pd.notna(row.get("ev")) else "---"
        bet_str = f"{row['bet_amount']:,.0f}円" if row.get("bet_amount", 0) > 0 else "---"
        print(f"  {int(row['horse_num']):>2}番 {name:<10s}  "
              f"予測勝率 {row['win_prob']:.1%}  "
              f"オッズ {row['win_odds']:.1f}  "
              f"EV {ev_str:>6s}  "
              f"推奨額 {bet_str:>10s}")
    if "bet_amount" in win_df.columns:
        sub = float(win_df["bet_amount"].fillna(0).sum())
        if sub > 0:
            print(f"  ─ 単勝 小計: {sub:,.0f} 円 ({len(win_df)}頭)")


def _print_trio_recs(trio_df: pd.DataFrame, strat: dict, ev_only: bool = False) -> None:
    """三連複の推奨結果を表示する。"""
    max_display = len(trio_df) if ev_only else strat.get("trio_max_display", 5)
    print("\n" + "=" * 60)
    print("  三連複推奨 (Trio) - ボックス買い" + ("  [EV-only 全件]" if ev_only else ""))
    print("=" * 60)
    if trio_df.empty:
        print("  推奨なし")
        return
    print(f"  {'組み合わせ':<14s}  {'的中確率':>8s}  {'オッズ':>8s}  {'EV':>8s}")
    print("  " + "-" * 50)
    for i, row in trio_df.head(max_display).iterrows():
        combo = f"{int(row['horse1'])}-{int(row['horse2'])}-{int(row['horse3'])}"
        prob_str = f"{row['trio_prob']:.1%}"
        odds_str = f"{row['odds']:.1f}" if pd.notna(row.get("odds")) else "---"
        ev_str = f"{row['ev']:.2f}" if pd.notna(row.get("ev")) else "---"
        print(f"  {combo:<14s}  {prob_str:>8s}  {odds_str:>8s}  {ev_str:>8s}")
    if ev_only:
        print(f"  ─ 三連複 EV≥{1.0:.1f} の件数: {len(trio_df)}")


def _print_trifecta_recs(trifecta_df: pd.DataFrame, strat: dict, ev_only: bool = False) -> None:
    """三連単の推奨結果を表示する。"""
    max_display = len(trifecta_df) if ev_only else strat.get("trifecta_max_display", 10)
    print("\n" + "=" * 60)
    print("  三連単推奨 (Trifecta) - Harville Model" + ("  [EV-only 全件]" if ev_only else ""))
    print("=" * 60)
    if trifecta_df.empty:
        print("  推奨なし")
        return
    print(f"  {'1着→2着→3着':<14s}  {'的中確率':>8s}  {'オッズ':>8s}  {'EV':>8s}")
    print("  " + "-" * 50)
    for i, row in trifecta_df.head(max_display).iterrows():
        combo = f"{int(row['horse1'])}→{int(row['horse2'])}→{int(row['horse3'])}"
        prob_str = f"{row['harville_prob']:.1%}"
        odds_str = f"{row['odds']:.1f}" if pd.notna(row.get("odds")) else "---"
        ev_str = f"{row['ev']:.2f}" if pd.notna(row.get("ev")) else "---"
        print(f"  {combo:<14s}  {prob_str:>8s}  {odds_str:>8s}  {ev_str:>8s}")
    if ev_only:
        print(f"  ─ 三連単 EV≥{1.0:.1f} の件数: {len(trifecta_df)}")


def _print_ev_summary(recs: dict, min_ev: float) -> None:
    """EV-only モードでの全券種サマリ。"""
    print("\n" + "=" * 60)
    print(f"  EV ≥ {min_ev:.2f} サマリ")
    print("=" * 60)
    n_show = len(recs.get("show", []))
    n_win = len(recs.get("win", []))
    n_trio = len(recs.get("trio", []))
    n_tri = len(recs.get("trifecta", []))
    total = n_show + n_win + n_trio + n_tri
    print(f"  複勝:   {n_show:>4d} 件")
    print(f"  単勝:   {n_win:>4d} 件")
    print(f"  三連複: {n_trio:>4d} 件")
    print(f"  三連単: {n_tri:>4d} 件")
    print(f"  ─ 合計: {total:>4d} 件")
    if total == 0:
        print("  ※ EV プラスの組み合わせ無し: このレースは見送り推奨")


if __name__ == "__main__":
    main()
