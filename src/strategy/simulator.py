"""賭け戦略のためのバックテストシミュレーター。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import ERROR_CODES_EXCLUDE
from src.strategy.kelly import compute_bet_amount, compute_tier_bet_amount


def simulate_backtest(
    bet_df: pd.DataFrame,
    cfg: dict,
    prob_col: str = "pred_prob",
    odds_col: str = "win_odds",
) -> dict:
    """選定された賭け対象馬についてバックテストを実行する。

    Args:
        bet_df: 1レース1行の DataFrame。以下の列が必要:
            pred_prob, win_odds, rank, error_code
        cfg: 戦略パラメータを含む設定辞書
        prob_col: キャリブレーション済み確率の列名
        odds_col: オッズの列名

    Returns:
        シミュレーション結果とバンクロール履歴を含む辞書
    """
    strat = cfg["strategy"]
    bankroll = float(strat["initial_bankroll"])
    kelly_frac = strat["kelly_fraction"]
    max_bet_frac = strat["max_bet_fraction"]
    min_bet = strat["min_bet"]
    bet_sizing = strat.get("bet_sizing", "kelly")

    # ティアパラメータ (閾値方式用)
    tier_kwargs = {
        k: strat[k] for k in [
            "tier_low_threshold", "tier_mid_threshold", "tier_high_threshold",
            "tier_low_amount", "tier_mid_amount", "tier_high_amount",
        ] if k in strat
    }

    history = []
    bets = []

    for idx, row in bet_df.iterrows():
        # 中止・除外レースはスキップ
        if "error_code" in row and row["error_code"] in ERROR_CODES_EXCLUDE:
            continue

        prob = row[prob_col]
        # 複勝オッズを win_odds / 3 として推定
        show_odds_est = row[odds_col] / 3.0
        hit = (row["rank"] >= 1) and (row["rank"] <= 3)

        if bet_sizing == "tier":
            bet_amount = compute_tier_bet_amount(prob, **tier_kwargs)
        else:
            bet_amount = compute_bet_amount(
                prob, show_odds_est, bankroll,
                fraction=kelly_frac,
                max_bet_fraction=max_bet_frac,
                min_bet=min_bet,
            )

        if bet_amount <= 0:
            continue

        # バンクロールを更新
        if hit:
            payout = bet_amount * show_odds_est
        else:
            payout = 0.0

        bankroll = bankroll - bet_amount + payout

        bets.append({
            "bet_amount": bet_amount,
            "show_odds_est": show_odds_est,
            "hit": int(hit),
            "payout": payout,
            "bankroll_after": bankroll,
            "horse": row.get("horse", ""),
            "rank": row["rank"],
        })
        history.append(bankroll)

    bets_df = pd.DataFrame(bets) if bets else pd.DataFrame()
    initial = float(strat["initial_bankroll"])

    n_bets = len(bets_df)
    n_hits = int(bets_df["hit"].sum()) if n_bets > 0 else 0
    total_wagered = bets_df["bet_amount"].sum() if n_bets > 0 else 0
    total_payout = bets_df["payout"].sum() if n_bets > 0 else 0

    # 最大ドローダウン
    peak = initial
    max_dd = 0.0
    for b in history:
        if b > peak:
            peak = b
        dd = (peak - b) / peak
        if dd > max_dd:
            max_dd = dd

    results = {
        "initial_bankroll": initial,
        "final_bankroll": bankroll,
        "recovery_rate": bankroll / initial * 100 if initial > 0 else 0,
        "n_bets": n_bets,
        "n_hits": n_hits,
        "hit_rate": n_hits / n_bets * 100 if n_bets > 0 else 0,
        "total_wagered": total_wagered,
        "total_payout": total_payout,
        "roi": (total_payout - total_wagered) / total_wagered * 100 if total_wagered > 0 else 0,
        "max_drawdown": max_dd * 100,
        "bankroll_history": history,
        "bets": bets_df,
    }

    return results


def print_backtest_summary(results: dict) -> None:
    """整形されたバックテスト結果を出力する。"""
    print("=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Initial Bankroll:  {results['initial_bankroll']:>12,.0f} pts")
    print(f"  Final Bankroll:    {results['final_bankroll']:>12,.0f} pts")
    print(f"  Recovery Rate:     {results['recovery_rate']:>11.2f}%")
    print(f"  Total Bets:        {results['n_bets']:>12d}")
    print(f"  Hits:              {results['n_hits']:>12d}")
    print(f"  Hit Rate:          {results['hit_rate']:>11.2f}%")
    print(f"  Total Wagered:     {results['total_wagered']:>12,.0f} pts")
    print(f"  Total Payout:      {results['total_payout']:>12,.0f} pts")
    print(f"  ROI:               {results['roi']:>11.2f}%")
    print(f"  Max Drawdown:      {results['max_drawdown']:>11.2f}%")
    print("=" * 60)


def plot_bankroll_history(results: dict, save_path: str | None = None) -> None:
    """時系列のバンクロール推移をプロットする。"""
    import matplotlib.pyplot as plt

    history = results["bankroll_history"]
    initial = results["initial_bankroll"]

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(history, linewidth=1.5)
    ax.axhline(y=initial, color="r", linestyle="--", alpha=0.7, label="Initial")
    ax.set_xlabel("Race number")
    ax.set_ylabel("Bankroll (pts)")
    ax.set_title("Bankroll History")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
