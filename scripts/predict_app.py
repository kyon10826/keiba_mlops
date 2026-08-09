#!/usr/bin/env python3
"""競馬予想アプリ — コンペ用の検証済みモデルで通常のレースを予想表示する。

run_live.py (大会 API + 自動投票) とは独立した**閲覧専用**ツール。
netkeiba から出馬表・オッズをスクレイピングし、オッズ不使用の
単勝/複勝ヘッド (models_masters、行順バグ修正後に再学習した最新モデル) で
校正済み確率を計算して表形式で表示する。投票は一切行わない。

旧 scripts/predict.py との違い:
  - predict.py は旧系統 (./models、オッズ特徴量+穴馬ヘッド) 用で、
    現在 ./models が存在しないため動かない
  - 本スクリプトは models_masters をそのまま使うので再学習不要

使い方:
    # 特定レース (netkeiba の 12 桁 race_id)
    python3 scripts/predict_app.py --race_id 202605020811

    # 指定日の全レース (省略時は今日)
    python3 scripts/predict_app.py --date 20260808

    # オッズ取得を省略して確率のみ (スクレイピングが最小になる)
    python3 scripts/predict_app.py --race_id 202605020811 --no-odds

注意 (バックテストで確定している事実):
    モデルの順位付け能力は本物 (top-1 的中率 23.6% ≒ 平均の 3.5 倍) だが、
    どのオッズ帯でもフラット投票の ROI は控除率 (20%) を超えられない。
    表示される EV は「市場との乖離の参考値」であり、EV > 1 に従って
    買えば勝てるという意味ではない。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

# リポジトリルートと scripts/ を絶対パスで通す。
# 注意: `from scripts.run_live import ...` 形式は使わない —
# Anaconda の site-packages に同名の `scripts` パッケージが存在する環境では
# そちらが優先されて ModuleNotFoundError になるため、run_live を
# トップレベルモジュールとして import する。
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS_DIR)
for _p in (_ROOT, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

from src.data.loader import load_config
from src.features.pipeline import FEATURE_COLUMNS
from src.model.calibrator import HoldoutCalibrator
from src.model.trainer import load_model
from src.scraper.race_card import scrape_race_card, scrape_today_races
from src.scraper.odds import scrape_odds

# run_live の特徴量構築・単勝予測をそのまま再利用する
from run_live import build_day_features, predict_win_probs

MARKS = ["◎", "○", "▲", "△", "△"]
SCRAPE_INTERVAL = 2.0  # netkeiba への連続リクエスト間隔 (秒)


def validate_date(date_str: str) -> str:
    """YYYYMMDD 形式かを検証する (桁数ミスを早期に検出)。"""
    if not (len(date_str) == 8 and date_str.isdigit()):
        raise SystemExit(
            f"日付は YYYYMMDD の 8 桁で指定してください: '{date_str}' ({len(date_str)} 桁)")
    try:
        datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        raise SystemExit(f"存在しない日付です: {date_str}")
    return date_str


def fetch_race_cards(args) -> list[pd.DataFrame]:
    """--race_id / --date から出馬表を収集する。"""
    if args.race_id:
        card = scrape_race_card(args.race_id)
        if card is None or card.empty:
            raise SystemExit(f"出馬表の取得に失敗しました: {args.race_id}")
        return [card]

    date = args.date or datetime.now().strftime("%Y%m%d")
    races = scrape_today_races(date)
    if not races:
        raise SystemExit(
            f"{date} のレース一覧が取得できません。\n"
            "  - 中央競馬の開催日 (土日祝) か確認してください\n"
            "  - 未来の日付の場合、netkeiba の出馬表公開は開催 2 日前頃です")
    print(f"{date}: {len(races)} レースの出馬表を取得します (数分かかります)...")
    cards = []
    for i, r in enumerate(races):
        print(f"  [{i + 1}/{len(races)}] {r['place']} {r['race_num']}R {r['race_name']}")
        card = scrape_race_card(r["race_id"])
        if card is not None and not card.empty:
            cards.append(card)
        time.sleep(SCRAPE_INTERVAL)
    if not cards:
        raise SystemExit("出馬表を 1 件も取得できませんでした")
    return cards


def predict_show_probs(day_feat: pd.DataFrame, model_dir: str) -> pd.DataFrame:
    """複勝ヘッド (lgbm_model.txt + calibrator.pkl) で show_pred_prob を付与する。"""
    model = load_model(os.path.join(model_dir, "lgbm_model.txt"))
    for c in FEATURE_COLUMNS:
        if c not in day_feat.columns:
            day_feat[c] = 0.0
    raw = model.predict(day_feat[FEATURE_COLUMNS])
    cal_path = os.path.join(model_dir, "calibrator.pkl")
    if os.path.exists(cal_path):
        raw = HoldoutCalibrator.load(cal_path).predict(raw)
    day_feat["show_pred_prob"] = raw
    return day_feat


def display_race(race_feat: pd.DataFrame, odds_df: pd.DataFrame | None) -> None:
    """1 レース分の予想テーブルを表示する。"""
    rid = str(int(race_feat["race_id"].iloc[0]))
    place = race_feat["place"].iloc[0]
    race_num = int(race_feat["race_num"].iloc[0])
    race_name = ""
    if "race_name" in race_feat.columns:
        race_name = str(race_feat["race_name"].iloc[0] or "")

    rows = race_feat.sort_values("win_pred_prob", ascending=False).reset_index(drop=True)

    if odds_df is not None and not odds_df.empty:
        rows = rows.merge(
            odds_df[["horse_num", "win_odds", "popularity"]].rename(
                columns={"win_odds": "odds_now"}),
            on="horse_num", how="left",
        )
    else:
        rows["odds_now"] = float("nan")
        rows["popularity"] = float("nan")

    print("\n" + "=" * 74)
    print(f"  {place} {race_num}R {race_name}  (race_id={rid}, {len(rows)} 頭)")
    print("=" * 74)
    print(f"  {'':<2}{'馬番':>4} {'馬名':<16}{'勝率':>7}{'複勝率':>8}{'オッズ':>8}{'人気':>5}{'EV':>7}")
    print("  " + "-" * 70)
    for i, row in rows.iterrows():
        mark = MARKS[i] if i < len(MARKS) else "  "
        odds = row.get("odds_now")
        pop = row.get("popularity")
        has_odds = pd.notna(odds) and odds > 0
        # 人気 9999 やオッズ欠損は出走取消・除外馬の可能性が高い
        scratched = (odds_df is not None and not odds_df.empty and not has_odds)
        odds_str = f"{odds:>8.1f}" if has_odds else f"{'---':>8}"
        pop_str = (f"{int(pop):>5}" if has_odds and pd.notna(pop) and 0 < pop < 999
                   else f"{'-':>5}")
        ev_str = f"{row['win_pred_prob'] * odds:>7.2f}" if has_odds else f"{'---':>7}"
        note = "  (取消?)" if scratched else ""
        print(
            f"  {mark:<2}{int(row['horse_num']):>4} {str(row['horse'])[:15]:<16}"
            f"{row['win_pred_prob'] * 100:>6.1f}%{row['show_pred_prob'] * 100:>7.1f}%"
            f"{odds_str}{pop_str}{ev_str}{note}"
        )


def main():
    parser = argparse.ArgumentParser(description="競馬予想アプリ (閲覧専用)")
    parser.add_argument("--config", default="config/masters_2026.yaml")
    parser.add_argument("--race_id", default=None, help="netkeiba の 12 桁 race_id")
    parser.add_argument("--date", default=None, help="対象日 YYYYMMDD (全レース)")
    parser.add_argument("--no-odds", action="store_true", help="オッズ取得を省略")
    args = parser.parse_args()

    if not args.race_id and not args.date:
        # 引数なしは今日の全レース
        args.date = datetime.now().strftime("%Y%m%d")
    if args.date:
        validate_date(args.date)
    if args.race_id and not (len(str(args.race_id)) == 12 and str(args.race_id).isdigit()):
        raise SystemExit(
            f"race_id は netkeiba の 12 桁で指定してください: '{args.race_id}'")

    cfg = load_config(args.config)
    model_dir = cfg["model"]["dir"]
    for f in ("win_model.txt", "lgbm_model.txt", "pipeline.pkl"):
        if not os.path.exists(os.path.join(model_dir, f)):
            raise SystemExit(
                f"{model_dir}/{f} がありません。"
                "python3 scripts/train.py --config config/masters_2026.yaml で学習してください")

    # 1. 出馬表収集
    cards = fetch_race_cards(args)
    runtable = pd.concat(cards, axis=0).reset_index(drop=True)
    print(f"\n出馬表: {len(cards)} レース {len(runtable)} 頭")

    # 2. 特徴量 + 予測 (履歴ロード込みで数分)
    today = args.date or datetime.now().strftime("%Y%m%d")
    day_feat = build_day_features(cfg, runtable, model_dir, date_str=today)
    day_feat = predict_win_probs(day_feat, model_dir)
    day_feat = predict_show_probs(day_feat, model_dir)

    # 3. レース毎に表示 (オッズはレース単位でスクレイピング)
    for rid, race_feat in day_feat.groupby("race_id", sort=True):
        odds_df = None
        if not args.no_odds:
            try:
                odds_df = scrape_odds(str(int(rid)).zfill(12))
                time.sleep(SCRAPE_INTERVAL)
            except Exception as e:
                print(f"  (オッズ取得失敗: {e})")
        display_race(race_feat.reset_index(drop=True), odds_df)

    print("\n" + "-" * 74)
    print("※ 勝率・複勝率はオッズ不使用モデルの校正済み確率。EV = 勝率×単勝オッズ。")
    print("※ バックテストでは全オッズ帯で控除率を超えられていない (EV は参考値)。")
    print("-" * 74)


if __name__ == "__main__":
    main()
