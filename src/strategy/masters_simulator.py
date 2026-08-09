"""AI 競馬予想マスターズ 2026 用のバックテストシミュレータ。

================================================================================
【レビューガイド】何をシミュレートしているか / 何を近似しているか
================================================================================
コンペルール準拠:
    - 初期ポイント: 100 万 pt
    - 最低 96 レース投票 かつ 累計 50 万 pt 以上投票 (未達 = 順位付け対象外)
    - 払戻は実確定オッズ (win_odds 列) で計算

シミュレーションの近似 (実運用との差分):
    1. 投票判定に使うオッズ = record_data の「確定オッズ」。
       実運用は「5 分前オッズ」なので、締切直前のオッズ変動分だけズレる。
       特に人気薄は 5 分前→確定で 10-20% 動くことがあり、EV 判定が楽観的に
       なりうる点は割り引いて見ること。
    2. 払戻確定のタイムラグ (次レースに間に合わない場合がある) は無視。
       バンクロールは投票時点で即時精算している。
    3. 自分の投票がオッズを動かす影響 (マーケットインパクト) は無視。
       個人の数千 pt では実害なし。

投票ロジック (hybrid モード、simulate_masters_backtest 内):
    レースごとに予測確率 top-1 を選び、
      main: 確率フロア + オッズ帯 + EV (prob×odds) の 3 条件を全て満たせば大口
      sub : main 不成立でも hybrid_enabled なら top-1 に少額 (制約充足用)
    ※ グリッドサーチの結果、main には頑健なエッジが無いと判明済み。
      本番 (run_live.py) は favorite_concentration 方式を採用しており、
      本シミュレータの hybrid は検証の経緯を残す参考実装の位置づけ。
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.data.schema import ERROR_CODES_EXCLUDE
from src.strategy.kelly import compute_bet_amount


@dataclass
class MastersBet:
    """1 レース分の投票記録。"""
    race_id: int
    date: pd.Timestamp
    place: str
    race_num: int
    bet_type: str          # "win" / "trio"
    horse_num: int
    pred_prob: float       # モデルの予測 (単勝1着確率)
    win_odds: float        # 実際の win_odds (払戻計算用)
    bet_amount: float      # 投票額 (pt)
    hit: int               # 的中 = 1, 不的中 = 0
    payout: float          # 払戻額
    bankroll_after: float


@dataclass
class MastersBacktestResult:
    """バックテスト結果サマリ。"""
    initial_bankroll: float
    final_bankroll: float
    n_races_available: int    # 期間中の全レース数
    n_races_bet: int          # 実際に投票したレース数
    n_hits: int
    total_wagered: float
    total_payout: float
    bets: pd.DataFrame        # 全投票の詳細
    daily_summary: pd.DataFrame
    bankroll_history: list[float]
    # コンペルール充足チェック
    min_races_bet: int
    min_total_wagered: float
    meets_race_constraint: bool
    meets_wagered_constraint: bool

    @property
    def hit_rate(self) -> float:
        return self.n_hits / self.n_races_bet if self.n_races_bet > 0 else 0.0

    @property
    def roi(self) -> float:
        return (self.total_payout - self.total_wagered) / self.total_wagered if self.total_wagered > 0 else 0.0

    @property
    def recovery_rate(self) -> float:
        return self.final_bankroll / self.initial_bankroll if self.initial_bankroll > 0 else 0.0


def simulate_masters_backtest(
    race_feat: pd.DataFrame,
    cfg: dict,
    prob_col: str = "win_pred_prob",
    win_odds_col: str = "win_odds",
) -> MastersBacktestResult:
    """AI 競馬予想マスターズ の単勝1点バックテストを実行する。

    Args:
        race_feat: 予測後の race_feat。以下の列が必要:
            - `race_id`, `year`, `month`, `day`, `place`, `race_num`, `horse_num`
            - `rank` (真の着順), `error_code`
            - `win_odds` (真の確定オッズ) — 払戻計算用に必須
            - prob_col (単勝1着確率の予測値)
        cfg: masters_rules + strategy + backtest を含む設定辞書
        prob_col: モデルの単勝予測確率の列名
        win_odds_col: 実際の win_odds の列名 (払戻計算に使う)

    Returns:
        MastersBacktestResult
    """
    rules = cfg["masters_rules"]
    strat = cfg["strategy"]

    initial_bankroll = float(rules["initial_bankroll"])
    min_races_bet = int(rules.get("min_races_bet", 96))
    min_total_wagered = float(rules.get("min_total_wagered", 500_000))
    min_bet_unit = int(rules.get("min_bet_unit", 100))

    bet_type = strat.get("bet_type", "win")
    win_prob_min = float(strat.get("win_prob_min", 0.15))
    win_bet_amount_default = float(strat.get("win_bet_amount", 5000))
    select_top_n = int(strat.get("select_top_n_per_race", 1))

    # Phase 4: EV フィルタ + Kelly サイジング
    sizing_method = strat.get("sizing_method", "flat")  # "flat" / "kelly"
    min_ev = float(strat.get("min_ev", 0.0))            # pred_prob*win_odds >= min_ev で投票
    max_odds = float(strat.get("max_odds", 1e9))        # 上限オッズ (数値アーティファクト回避)
    min_odds = float(strat.get("min_odds", 0.0))        # 下限オッズ (人気馬回避)
    kelly_fraction = float(strat.get("kelly_fraction", 0.25))
    max_bet_fraction = float(strat.get("max_bet_fraction", 0.05))
    per_bet_cap = float(strat.get("per_race_cap", 10000))
    prob_scaling = float(strat.get("prob_scaling", 1.0))  # 生確率を Kelly 用に校正 (実的中率へ引き上げ)

    # Phase 4b: ハイブリッド戦略 (メイン + サブベット)
    hybrid_enabled = bool(strat.get("hybrid_enabled", False))
    sub_bet_amount = float(strat.get("sub_bet_amount", 500))  # メイン通過ゼロのレースへのサブ投票額

    if bet_type != "win":
        raise NotImplementedError(f"bet_type={bet_type!r} は Phase 4c で拡張予定")

    df = race_feat.copy()
    # 日付列を組み立て (年/月/日は整数、year は 2 桁の場合 20xx として補完)
    df["year_full"] = df["year"].astype(int).apply(lambda y: 2000 + y if y < 100 else y)
    df["_date"] = pd.to_datetime(
        df[["year_full", "month", "day"]].rename(
            columns={"year_full": "year", "month": "month", "day": "day"}
        ),
        errors="coerce",
    )

    # バックテスト期間で絞り込む
    bt = cfg.get("backtest", {})
    if "start_date" in bt:
        df = df[df["_date"] >= pd.to_datetime(bt["start_date"])]
    if "end_date" in bt:
        df = df[df["_date"] <= pd.to_datetime(bt["end_date"])]

    df = df.sort_values(["_date", "place", "race_num", "horse_num"]).reset_index(drop=True)

    race_key = ["year", "month", "day", "place", "race_num"]
    n_races_available = df.groupby(race_key).ngroups

    bankroll = initial_bankroll
    bets: list[MastersBet] = []
    history: list[float] = [bankroll]

    for race_id_key, race_grp in df.groupby(race_key, sort=False):
        # 中止・除外馬を除く
        race_valid = race_grp[~race_grp["error_code"].isin(ERROR_CODES_EXCLUDE)].copy()
        if race_valid.empty:
            continue

        # 予測確率上位 select_top_n 頭を候補に
        race_valid_sorted = race_valid.sort_values(prob_col, ascending=False)
        candidates = race_valid_sorted.head(select_top_n)

        main_bet_placed = False  # ハイブリッドモード用: このレースでメインベットが入ったか

        for _, row in candidates.iterrows():
            prob = float(row[prob_col])
            if prob < win_prob_min:
                continue  # 生確率が閾値未満なら投票せず

            actual_odds = float(row.get(win_odds_col, 0) or 0)
            # オッズ範囲フィルタ (人気馬・理論値超高オッズを除外)
            if actual_odds <= 0 or actual_odds < min_odds or actual_odds > max_odds:
                continue

            # EV フィルタ: pred_prob (校正済み) × win_odds が閾値以上
            # 生確率は圧縮されているので prob_scaling で校正 (実的中率に近づける)
            calibrated_prob = min(prob * prob_scaling, 1.0)
            ev = calibrated_prob * actual_odds
            if ev < min_ev:
                continue

            # 投票額: flat / kelly
            if sizing_method == "kelly":
                bet_amount = compute_bet_amount(
                    prob=calibrated_prob,
                    odds=actual_odds,
                    bankroll=bankroll,
                    fraction=kelly_fraction,
                    max_bet_fraction=max_bet_fraction,
                    min_bet=min_bet_unit,
                    per_bet_cap=per_bet_cap,
                )
            else:
                bet_amount = win_bet_amount_default

            # min_bet 単位に丸め
            bet_amount = int(bet_amount // min_bet_unit) * min_bet_unit
            if bet_amount <= 0:
                continue
            # バンクロール切れは投票しない
            if bankroll < bet_amount:
                continue

            actual_rank = int(row.get("rank", 0) or 0)
            hit = 1 if actual_rank == 1 else 0
            payout = bet_amount * actual_odds if hit else 0.0

            bankroll = bankroll - bet_amount + payout
            main_bet_placed = True

            bets.append(MastersBet(
                race_id=int(row["race_id"]),
                date=row["_date"],
                place=str(row["place"]),
                race_num=int(row["race_num"]),
                bet_type="win",
                horse_num=int(row["horse_num"]),
                pred_prob=prob,
                win_odds=actual_odds,
                bet_amount=bet_amount,
                hit=hit,
                payout=payout,
                bankroll_after=bankroll,
            ))
            history.append(bankroll)

        # === Phase 4b: ハイブリッド戦略のサブベット ===
        # メインベットが入らなかったレースに、上位1頭に小額サブベット (制約充足用)
        if hybrid_enabled and not main_bet_placed:
            sub_row = race_valid_sorted.iloc[0]  # 予測確率最大の馬
            sub_odds = float(sub_row.get(win_odds_col, 0) or 0)
            if sub_odds > 0:  # 取消馬は除外
                sub_amount = int(sub_bet_amount // min_bet_unit) * min_bet_unit
                if sub_amount > 0 and bankroll >= sub_amount:
                    sub_rank = int(sub_row.get("rank", 0) or 0)
                    sub_hit = 1 if sub_rank == 1 else 0
                    sub_payout = sub_amount * sub_odds if sub_hit else 0.0
                    bankroll = bankroll - sub_amount + sub_payout

                    bets.append(MastersBet(
                        race_id=int(sub_row["race_id"]),
                        date=sub_row["_date"],
                        place=str(sub_row["place"]),
                        race_num=int(sub_row["race_num"]),
                        bet_type="win_sub",  # メインと区別
                        horse_num=int(sub_row["horse_num"]),
                        pred_prob=float(sub_row[prob_col]),
                        win_odds=sub_odds,
                        bet_amount=sub_amount,
                        hit=sub_hit,
                        payout=sub_payout,
                        bankroll_after=bankroll,
                    ))
                    history.append(bankroll)

    bets_df = pd.DataFrame([b.__dict__ for b in bets])

    if bets_df.empty:
        daily = pd.DataFrame(columns=["date", "n_bets", "n_hits", "wagered", "payout", "roi_daily"])
        n_races_bet = 0
        n_hits = 0
        total_wagered = 0.0
        total_payout = 0.0
    else:
        # 日次サマリ
        daily = bets_df.groupby("date").agg(
            n_bets=("bet_amount", "count"),
            n_hits=("hit", "sum"),
            wagered=("bet_amount", "sum"),
            payout=("payout", "sum"),
        ).reset_index()
        daily["roi_daily"] = (daily["payout"] - daily["wagered"]) / daily["wagered"]

        n_races_bet = int(bets_df["race_id"].nunique())  # 1レース = 複数投票してもレース数は1
        n_hits = int(bets_df["hit"].sum())
        total_wagered = float(bets_df["bet_amount"].sum())
        total_payout = float(bets_df["payout"].sum())

    return MastersBacktestResult(
        initial_bankroll=initial_bankroll,
        final_bankroll=bankroll,
        n_races_available=n_races_available,
        n_races_bet=n_races_bet,
        n_hits=n_hits,
        total_wagered=total_wagered,
        total_payout=total_payout,
        bets=bets_df,
        daily_summary=daily,
        bankroll_history=history,
        min_races_bet=min_races_bet,
        min_total_wagered=min_total_wagered,
        meets_race_constraint=(n_races_bet >= min_races_bet),
        meets_wagered_constraint=(total_wagered >= min_total_wagered),
    )


def print_masters_summary(result: MastersBacktestResult) -> None:
    """整形されたコンペ準拠のバックテストサマリを出力する。"""
    print("=" * 70)
    print("  AI 競馬予想マスターズ バックテスト結果")
    print("=" * 70)
    print(f"  期間内の全レース数:      {result.n_races_available:>10d}")
    print(f"  実際に投票したレース数:  {result.n_races_bet:>10d}")
    print(f"  的中数:                  {result.n_hits:>10d}")
    print(f"  的中率:                  {result.hit_rate*100:>9.2f}%")
    print(f"  総投票額:                {result.total_wagered:>10,.0f} pt")
    print(f"  総払戻額:                {result.total_payout:>10,.0f} pt")
    print(f"  ROI:                     {result.roi*100:>+9.2f}%")

    # ハイブリッド戦略のメイン/サブ内訳
    if not result.bets.empty and "bet_type" in result.bets.columns:
        by_type = result.bets.groupby("bet_type").agg(
            n_bets=("bet_amount", "count"),
            n_hits=("hit", "sum"),
            wagered=("bet_amount", "sum"),
            payout=("payout", "sum"),
        )
        if len(by_type) > 1:
            print()
            print("  --- 券種内訳 (main = EV通過 / sub = 制約充足用) ---")
            for bt, r in by_type.iterrows():
                roi_pct = (r["payout"] - r["wagered"]) / r["wagered"] * 100 if r["wagered"] > 0 else 0
                hit_pct = r["n_hits"] / r["n_bets"] * 100 if r["n_bets"] > 0 else 0
                print(f"  {bt:>10}: 投票 {int(r['n_bets']):>3} / 的中 {int(r['n_hits']):>3} "
                      f"({hit_pct:>5.1f}%) / 投資 {int(r['wagered']):>7,d} / "
                      f"払戻 {int(r['payout']):>7,d} / ROI {roi_pct:>+6.1f}%")
    print()
    print(f"  初期バンクロール:        {result.initial_bankroll:>10,.0f} pt")
    print(f"  最終バンクロール:        {result.final_bankroll:>10,.0f} pt")
    print(f"  回収率:                  {result.recovery_rate*100:>9.2f}%")
    print()
    print("  --- コンペルール充足チェック ---")
    ok_race = "✓" if result.meets_race_constraint else "✗"
    ok_wag = "✓" if result.meets_wagered_constraint else "✗"
    print(f"  {ok_race} 96 レース以上投票:      {result.n_races_bet}/{result.min_races_bet}")
    print(f"  {ok_wag} 累計 50万pt 以上投票:   {result.total_wagered:,.0f}/{result.min_total_wagered:,.0f}")
    if not (result.meets_race_constraint and result.meets_wagered_constraint):
        print("  ⚠️  制約未達 → 順位付け対象外になります")
    else:
        print("  ✅ 全制約を満たしています → 順位付け対象")
    print("=" * 70)


def print_daily_summary(result: MastersBacktestResult, max_rows: int = 30) -> None:
    """日次サマリを表示。"""
    daily = result.daily_summary
    if daily.empty:
        print("  (日次データなし)")
        return
    print("\n  --- 日次サマリ ---")
    print(f"  {'日付':<12} {'投票':>4} {'的中':>4} {'投票額':>10} {'払戻':>10} {'ROI':>7}")
    for _, row in daily.head(max_rows).iterrows():
        d = row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else "?"
        print(f"  {d:<12} {int(row['n_bets']):>4} {int(row['n_hits']):>4} "
              f"{row['wagered']:>10,.0f} {row['payout']:>10,.0f} "
              f"{row['roi_daily']*100:>+6.1f}%")


def print_top_bets(result: MastersBacktestResult, n: int = 10) -> None:
    """払戻額 Top N のヒット買い目を表示。"""
    if result.bets.empty:
        return
    top = result.bets[result.bets["hit"] == 1].nlargest(n, "payout")
    if top.empty:
        return
    print(f"\n  --- 払戻額 Top {n} ヒット ---")
    print(f"  {'日付':<12} {'場所':<4} {'R':>3} {'馬番':>4} {'オッズ':>7} {'投票':>7} {'払戻':>10}")
    for _, r in top.iterrows():
        d = r["date"].strftime("%Y-%m-%d") if pd.notna(r["date"]) else "?"
        print(f"  {d:<12} {str(r['place']):<4} {int(r['race_num']):>3} "
              f"{int(r['horse_num']):>4} {r['win_odds']:>7.1f} "
              f"{r['bet_amount']:>7,.0f} {r['payout']:>10,.0f}")
