"""
daily_rebalance.py — Entry-point for the GitHub Actions scheduled job.

Run: python daily_rebalance.py [--horizon week|day|month]
Logs are written to rebalance.log (tee'd to stdout) for the webhook step.
"""
import argparse
import logging
import sys
from portfolio import load_portfolio, rebalance, save_portfolio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('rebalance.log', mode='w'),
    ],
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', default='week', choices=['day', 'week', 'month'])
    args = parser.parse_args()

    log.info('Loading portfolio...')
    p = load_portfolio()

    last_val = p['daily_values'][-1]['value'] if p['daily_values'] else p['initial_balance']
    log.info('Cash: $%.2f  |  Positions: %d  |  Last value: $%.2f',
             p['cash'], len(p['positions']), last_val)

    log.info('Rebalancing (horizon=%s)...', args.horizon)
    p, summary = rebalance(p, args.horizon)
    log.info(summary)

    save_portfolio(p)
    log.info('Portfolio saved.')

    if p['positions']:
        log.info('Open positions:')
        for sym, pos in p['positions'].items():
            log.info('  %s: %d shares @ $%.2f  (entered %s, pred=%+.1f%%, conf=%.0f%%)',
                     sym, pos['shares'], pos['entry_price'], pos['entry_date'],
                     pos['entry_pred_pct'], pos['entry_conf'] * 100)
    else:
        log.info('No open positions.')

    return 0


if __name__ == '__main__':
    sys.exit(main())
