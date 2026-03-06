"""
daily_rebalance.py — Entry-point for the GitHub Actions scheduled job.

Run: python daily_rebalance.py [--horizon week|day|month]
"""
import argparse
import sys
from portfolio import load_portfolio, rebalance, save_portfolio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', default='week', choices=['day', 'week', 'month'])
    args = parser.parse_args()

    print(f"Loading portfolio...")
    p = load_portfolio()

    initial = p['initial_balance']
    last_val = p['daily_values'][-1]['value'] if p['daily_values'] else initial
    print(f"  Cash: ${p['cash']:,.2f}  |  Positions: {len(p['positions'])}  |  Last value: ${last_val:,.2f}")

    print(f"Rebalancing (horizon={args.horizon})...")
    p, summary = rebalance(p, args.horizon)
    print(f"  {summary}")

    save_portfolio(p)
    print("Portfolio saved.")

    # Print current positions
    if p['positions']:
        print("\nOpen positions:")
        for sym, pos in p['positions'].items():
            print(f"  {sym}: {pos['shares']} shares @ ${pos['entry_price']:.2f}  "
                  f"(entered {pos['entry_date']}, pred={pos['entry_pred_pct']:+.1f}%, "
                  f"conf={pos['entry_conf']:.0%})")
    else:
        print("\nNo open positions.")

    return 0

if __name__ == '__main__':
    sys.exit(main())
