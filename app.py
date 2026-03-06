from flask import Flask, jsonify, request, send_from_directory
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
import os
import json
import time
import urllib.request
from datetime import datetime, timedelta
from portfolio import load_portfolio, rebalance, save_portfolio, portfolio_value

# Optional: set GITHUB_REPO=owner/repo so /api/portfolio always reads the
# latest portfolio.json committed by GitHub Actions (no git pull needed).
GITHUB_REPO = os.environ.get('GITHUB_REPO', 'noaRoblesLevy/StockOracle')

def _fetch_portfolio_from_github() -> dict | None:
    """Download portfolio.json from GitHub raw and save locally if newer."""
    if not GITHUB_REPO:
        return None
    url = f'https://raw.githubusercontent.com/{GITHUB_REPO}/master/portfolio.json'
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            remote = json.loads(resp.read().decode())
        local = load_portfolio()
        # Use remote if it was updated more recently (or local has never been updated)
        remote_date = remote.get('last_updated') or ''
        local_date  = local.get('last_updated')  or ''
        if remote_date > local_date:
            save_portfolio(remote)
            return remote
        return local
    except Exception:
        return None

warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='static')

# ─── Popular stocks / ETFs for the searchable list ───────────────────────────
POPULAR_TICKERS = [
    # ── US Large Cap Tech ────────────────────────────────────────────────────
    {"symbol": "AAPL",  "name": "Apple Inc.",                        "type": "Stock", "sector": "Technology"},
    {"symbol": "MSFT",  "name": "Microsoft Corporation",             "type": "Stock", "sector": "Technology"},
    {"symbol": "GOOGL", "name": "Alphabet Inc. (Class A)",           "type": "Stock", "sector": "Technology"},
    {"symbol": "GOOG",  "name": "Alphabet Inc. (Class C)",           "type": "Stock", "sector": "Technology"},
    {"symbol": "AMZN",  "name": "Amazon.com Inc.",                   "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "NVDA",  "name": "NVIDIA Corporation",                "type": "Stock", "sector": "Technology"},
    {"symbol": "META",  "name": "Meta Platforms Inc.",               "type": "Stock", "sector": "Technology"},
    {"symbol": "TSLA",  "name": "Tesla Inc.",                        "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "AVGO",  "name": "Broadcom Inc.",                     "type": "Stock", "sector": "Technology"},
    {"symbol": "ORCL",  "name": "Oracle Corporation",                "type": "Stock", "sector": "Technology"},
    {"symbol": "CSCO",  "name": "Cisco Systems Inc.",                "type": "Stock", "sector": "Technology"},
    {"symbol": "ACN",   "name": "Accenture plc",                     "type": "Stock", "sector": "Technology"},
    {"symbol": "IBM",   "name": "IBM Corporation",                   "type": "Stock", "sector": "Technology"},
    {"symbol": "QCOM",  "name": "Qualcomm Inc.",                     "type": "Stock", "sector": "Technology"},
    {"symbol": "TXN",   "name": "Texas Instruments Inc.",            "type": "Stock", "sector": "Technology"},
    {"symbol": "INTC",  "name": "Intel Corporation",                 "type": "Stock", "sector": "Technology"},
    {"symbol": "AMD",   "name": "Advanced Micro Devices Inc.",       "type": "Stock", "sector": "Technology"},
    {"symbol": "ADBE",  "name": "Adobe Inc.",                        "type": "Stock", "sector": "Technology"},
    {"symbol": "CRM",   "name": "Salesforce Inc.",                   "type": "Stock", "sector": "Technology"},
    {"symbol": "NOW",   "name": "ServiceNow Inc.",                   "type": "Stock", "sector": "Technology"},
    {"symbol": "INTU",  "name": "Intuit Inc.",                       "type": "Stock", "sector": "Technology"},
    {"symbol": "PLTR",  "name": "Palantir Technologies Inc.",        "type": "Stock", "sector": "Technology"},
    {"symbol": "UBER",  "name": "Uber Technologies Inc.",            "type": "Stock", "sector": "Technology"},
    {"symbol": "PANW",  "name": "Palo Alto Networks Inc.",           "type": "Stock", "sector": "Technology"},
    {"symbol": "CRWD",  "name": "CrowdStrike Holdings Inc.",         "type": "Stock", "sector": "Technology"},
    {"symbol": "NET",   "name": "Cloudflare Inc.",                   "type": "Stock", "sector": "Technology"},
    {"symbol": "SNOW",  "name": "Snowflake Inc.",                    "type": "Stock", "sector": "Technology"},
    {"symbol": "DDOG",  "name": "Datadog Inc.",                      "type": "Stock", "sector": "Technology"},
    {"symbol": "ZS",    "name": "Zscaler Inc.",                      "type": "Stock", "sector": "Technology"},
    {"symbol": "OKTA",  "name": "Okta Inc.",                         "type": "Stock", "sector": "Technology"},
    {"symbol": "MDB",   "name": "MongoDB Inc.",                      "type": "Stock", "sector": "Technology"},
    {"symbol": "TEAM",  "name": "Atlassian Corporation",             "type": "Stock", "sector": "Technology"},
    {"symbol": "WDAY",  "name": "Workday Inc.",                      "type": "Stock", "sector": "Technology"},
    {"symbol": "FTNT",  "name": "Fortinet Inc.",                     "type": "Stock", "sector": "Technology"},
    {"symbol": "ZM",    "name": "Zoom Video Communications Inc.",    "type": "Stock", "sector": "Technology"},
    {"symbol": "DOCU",  "name": "DocuSign Inc.",                     "type": "Stock", "sector": "Technology"},
    {"symbol": "TWLO",  "name": "Twilio Inc.",                       "type": "Stock", "sector": "Technology"},
    {"symbol": "APP",   "name": "AppLovin Corporation",              "type": "Stock", "sector": "Technology"},
    {"symbol": "SMCI",  "name": "Super Micro Computer Inc.",         "type": "Stock", "sector": "Technology"},
    {"symbol": "ARM",   "name": "Arm Holdings plc",                  "type": "Stock", "sector": "Technology"},
    {"symbol": "DELL",  "name": "Dell Technologies Inc.",            "type": "Stock", "sector": "Technology"},
    {"symbol": "HPQ",   "name": "HP Inc.",                           "type": "Stock", "sector": "Technology"},
    {"symbol": "AMAT",  "name": "Applied Materials Inc.",            "type": "Stock", "sector": "Technology"},
    {"symbol": "LRCX",  "name": "Lam Research Corporation",         "type": "Stock", "sector": "Technology"},
    {"symbol": "KLAC",  "name": "KLA Corporation",                   "type": "Stock", "sector": "Technology"},
    {"symbol": "MRVL",  "name": "Marvell Technology Inc.",           "type": "Stock", "sector": "Technology"},
    {"symbol": "MU",    "name": "Micron Technology Inc.",            "type": "Stock", "sector": "Technology"},
    {"symbol": "WDC",   "name": "Western Digital Corporation",       "type": "Stock", "sector": "Technology"},
    {"symbol": "STX",   "name": "Seagate Technology Holdings",       "type": "Stock", "sector": "Technology"},
    # ── Financials ───────────────────────────────────────────────────────────
    {"symbol": "BRK-B", "name": "Berkshire Hathaway Inc.",           "type": "Stock", "sector": "Financials"},
    {"symbol": "JPM",   "name": "JPMorgan Chase & Co.",              "type": "Stock", "sector": "Financials"},
    {"symbol": "V",     "name": "Visa Inc.",                         "type": "Stock", "sector": "Financials"},
    {"symbol": "MA",    "name": "Mastercard Inc.",                   "type": "Stock", "sector": "Financials"},
    {"symbol": "BAC",   "name": "Bank of America Corp.",             "type": "Stock", "sector": "Financials"},
    {"symbol": "GS",    "name": "Goldman Sachs Group Inc.",          "type": "Stock", "sector": "Financials"},
    {"symbol": "MS",    "name": "Morgan Stanley",                    "type": "Stock", "sector": "Financials"},
    {"symbol": "C",     "name": "Citigroup Inc.",                    "type": "Stock", "sector": "Financials"},
    {"symbol": "WFC",   "name": "Wells Fargo & Company",             "type": "Stock", "sector": "Financials"},
    {"symbol": "BLK",   "name": "BlackRock Inc.",                    "type": "Stock", "sector": "Financials"},
    {"symbol": "SPGI",  "name": "S&P Global Inc.",                   "type": "Stock", "sector": "Financials"},
    {"symbol": "AXP",   "name": "American Express Company",          "type": "Stock", "sector": "Financials"},
    {"symbol": "COF",   "name": "Capital One Financial Corp.",       "type": "Stock", "sector": "Financials"},
    {"symbol": "USB",   "name": "U.S. Bancorp",                      "type": "Stock", "sector": "Financials"},
    {"symbol": "PNC",   "name": "PNC Financial Services Group",      "type": "Stock", "sector": "Financials"},
    {"symbol": "TFC",   "name": "Truist Financial Corporation",      "type": "Stock", "sector": "Financials"},
    {"symbol": "SCHW",  "name": "Charles Schwab Corporation",        "type": "Stock", "sector": "Financials"},
    {"symbol": "PYPL",  "name": "PayPal Holdings Inc.",              "type": "Stock", "sector": "Financials"},
    {"symbol": "SQ",    "name": "Block Inc.",                        "type": "Stock", "sector": "Financials"},
    {"symbol": "COIN",  "name": "Coinbase Global Inc.",              "type": "Stock", "sector": "Financials"},
    {"symbol": "HOOD",  "name": "Robinhood Markets Inc.",            "type": "Stock", "sector": "Financials"},
    {"symbol": "SOFI",  "name": "SoFi Technologies Inc.",            "type": "Stock", "sector": "Financials"},
    # ── Healthcare ───────────────────────────────────────────────────────────
    {"symbol": "JNJ",   "name": "Johnson & Johnson",                 "type": "Stock", "sector": "Healthcare"},
    {"symbol": "UNH",   "name": "UnitedHealth Group Inc.",           "type": "Stock", "sector": "Healthcare"},
    {"symbol": "ABBV",  "name": "AbbVie Inc.",                       "type": "Stock", "sector": "Healthcare"},
    {"symbol": "MRK",   "name": "Merck & Co. Inc.",                  "type": "Stock", "sector": "Healthcare"},
    {"symbol": "TMO",   "name": "Thermo Fisher Scientific Inc.",     "type": "Stock", "sector": "Healthcare"},
    {"symbol": "LLY",   "name": "Eli Lilly and Company",             "type": "Stock", "sector": "Healthcare"},
    {"symbol": "PFE",   "name": "Pfizer Inc.",                       "type": "Stock", "sector": "Healthcare"},
    {"symbol": "BMY",   "name": "Bristol-Myers Squibb Company",      "type": "Stock", "sector": "Healthcare"},
    {"symbol": "AMGN",  "name": "Amgen Inc.",                        "type": "Stock", "sector": "Healthcare"},
    {"symbol": "GILD",  "name": "Gilead Sciences Inc.",              "type": "Stock", "sector": "Healthcare"},
    {"symbol": "MRNA",  "name": "Moderna Inc.",                      "type": "Stock", "sector": "Healthcare"},
    {"symbol": "REGN",  "name": "Regeneron Pharmaceuticals Inc.",    "type": "Stock", "sector": "Healthcare"},
    {"symbol": "VRTX",  "name": "Vertex Pharmaceuticals Inc.",       "type": "Stock", "sector": "Healthcare"},
    {"symbol": "BIIB",  "name": "Biogen Inc.",                       "type": "Stock", "sector": "Healthcare"},
    {"symbol": "ISRG",  "name": "Intuitive Surgical Inc.",           "type": "Stock", "sector": "Healthcare"},
    {"symbol": "MDT",   "name": "Medtronic plc",                     "type": "Stock", "sector": "Healthcare"},
    {"symbol": "SYK",   "name": "Stryker Corporation",               "type": "Stock", "sector": "Healthcare"},
    {"symbol": "ABT",   "name": "Abbott Laboratories",               "type": "Stock", "sector": "Healthcare"},
    {"symbol": "CVS",   "name": "CVS Health Corporation",            "type": "Stock", "sector": "Healthcare"},
    {"symbol": "HCA",   "name": "HCA Healthcare Inc.",               "type": "Stock", "sector": "Healthcare"},
    # ── Consumer Discretionary ───────────────────────────────────────────────
    {"symbol": "HD",    "name": "The Home Depot Inc.",               "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "MCD",   "name": "McDonald's Corporation",            "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "ABNB",  "name": "Airbnb Inc.",                       "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "RIVN",  "name": "Rivian Automotive Inc.",            "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "LCID",  "name": "Lucid Group Inc.",                  "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "NKE",   "name": "Nike Inc.",                         "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "SBUX",  "name": "Starbucks Corporation",             "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "TGT",   "name": "Target Corporation",                "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "LOW",   "name": "Lowe's Companies Inc.",             "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "F",     "name": "Ford Motor Company",                "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "GM",    "name": "General Motors Company",            "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "TJX",   "name": "TJX Companies Inc.",               "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "LULU",  "name": "Lululemon Athletica Inc.",          "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "ROST",  "name": "Ross Stores Inc.",                  "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "RBLX",  "name": "Roblox Corporation",                "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "U",     "name": "Unity Software Inc.",               "type": "Stock", "sector": "Consumer Discretionary"},
    # ── Consumer Staples ─────────────────────────────────────────────────────
    {"symbol": "WMT",   "name": "Walmart Inc.",                      "type": "Stock", "sector": "Consumer Staples"},
    {"symbol": "PG",    "name": "Procter & Gamble Co.",              "type": "Stock", "sector": "Consumer Staples"},
    {"symbol": "KO",    "name": "The Coca-Cola Company",             "type": "Stock", "sector": "Consumer Staples"},
    {"symbol": "PEP",   "name": "PepsiCo Inc.",                      "type": "Stock", "sector": "Consumer Staples"},
    {"symbol": "COST",  "name": "Costco Wholesale Corporation",      "type": "Stock", "sector": "Consumer Staples"},
    {"symbol": "PM",    "name": "Philip Morris International Inc.",  "type": "Stock", "sector": "Consumer Staples"},
    {"symbol": "MO",    "name": "Altria Group Inc.",                 "type": "Stock", "sector": "Consumer Staples"},
    {"symbol": "CL",    "name": "Colgate-Palmolive Company",         "type": "Stock", "sector": "Consumer Staples"},
    {"symbol": "MDLZ",  "name": "Mondelez International Inc.",       "type": "Stock", "sector": "Consumer Staples"},
    {"symbol": "KHC",   "name": "The Kraft Heinz Company",           "type": "Stock", "sector": "Consumer Staples"},
    # ── Communication Services ───────────────────────────────────────────────
    {"symbol": "NFLX",  "name": "Netflix Inc.",                      "type": "Stock", "sector": "Communication Services"},
    {"symbol": "SNAP",  "name": "Snap Inc.",                         "type": "Stock", "sector": "Communication Services"},
    {"symbol": "DIS",   "name": "The Walt Disney Company",           "type": "Stock", "sector": "Communication Services"},
    {"symbol": "T",     "name": "AT&T Inc.",                         "type": "Stock", "sector": "Communication Services"},
    {"symbol": "VZ",    "name": "Verizon Communications Inc.",       "type": "Stock", "sector": "Communication Services"},
    {"symbol": "CMCSA", "name": "Comcast Corporation",               "type": "Stock", "sector": "Communication Services"},
    {"symbol": "TMUS",  "name": "T-Mobile US Inc.",                  "type": "Stock", "sector": "Communication Services"},
    {"symbol": "CHTR",  "name": "Charter Communications Inc.",       "type": "Stock", "sector": "Communication Services"},
    {"symbol": "WBD",   "name": "Warner Bros. Discovery Inc.",       "type": "Stock", "sector": "Communication Services"},
    {"symbol": "PARA",  "name": "Paramount Global",                  "type": "Stock", "sector": "Communication Services"},
    {"symbol": "LYFT",  "name": "Lyft Inc.",                         "type": "Stock", "sector": "Communication Services"},
    # ── Energy ───────────────────────────────────────────────────────────────
    {"symbol": "XOM",   "name": "Exxon Mobil Corporation",           "type": "Stock", "sector": "Energy"},
    {"symbol": "CVX",   "name": "Chevron Corporation",               "type": "Stock", "sector": "Energy"},
    {"symbol": "COP",   "name": "ConocoPhillips",                    "type": "Stock", "sector": "Energy"},
    {"symbol": "OXY",   "name": "Occidental Petroleum Corporation",  "type": "Stock", "sector": "Energy"},
    {"symbol": "SLB",   "name": "SLB (Schlumberger)",                "type": "Stock", "sector": "Energy"},
    {"symbol": "HAL",   "name": "Halliburton Company",               "type": "Stock", "sector": "Energy"},
    {"symbol": "EOG",   "name": "EOG Resources Inc.",                "type": "Stock", "sector": "Energy"},
    {"symbol": "MPC",   "name": "Marathon Petroleum Corporation",    "type": "Stock", "sector": "Energy"},
    {"symbol": "PSX",   "name": "Phillips 66",                       "type": "Stock", "sector": "Energy"},
    {"symbol": "VLO",   "name": "Valero Energy Corporation",         "type": "Stock", "sector": "Energy"},
    {"symbol": "DVN",   "name": "Devon Energy Corporation",          "type": "Stock", "sector": "Energy"},
    {"symbol": "FANG",  "name": "Diamondback Energy Inc.",           "type": "Stock", "sector": "Energy"},
    # ── Industrials ──────────────────────────────────────────────────────────
    {"symbol": "BA",    "name": "Boeing Company",                    "type": "Stock", "sector": "Industrials"},
    {"symbol": "CAT",   "name": "Caterpillar Inc.",                  "type": "Stock", "sector": "Industrials"},
    {"symbol": "GE",    "name": "GE Aerospace",                      "type": "Stock", "sector": "Industrials"},
    {"symbol": "HON",   "name": "Honeywell International Inc.",      "type": "Stock", "sector": "Industrials"},
    {"symbol": "LMT",   "name": "Lockheed Martin Corporation",       "type": "Stock", "sector": "Industrials"},
    {"symbol": "RTX",   "name": "RTX Corporation",                   "type": "Stock", "sector": "Industrials"},
    {"symbol": "UPS",   "name": "United Parcel Service Inc.",        "type": "Stock", "sector": "Industrials"},
    {"symbol": "FDX",   "name": "FedEx Corporation",                 "type": "Stock", "sector": "Industrials"},
    {"symbol": "MMM",   "name": "3M Company",                        "type": "Stock", "sector": "Industrials"},
    {"symbol": "DE",    "name": "Deere & Company",                   "type": "Stock", "sector": "Industrials"},
    {"symbol": "EMR",   "name": "Emerson Electric Co.",              "type": "Stock", "sector": "Industrials"},
    {"symbol": "GD",    "name": "General Dynamics Corporation",      "type": "Stock", "sector": "Industrials"},
    {"symbol": "NOC",   "name": "Northrop Grumman Corporation",      "type": "Stock", "sector": "Industrials"},
    {"symbol": "ITW",   "name": "Illinois Tool Works Inc.",          "type": "Stock", "sector": "Industrials"},
    {"symbol": "ETN",   "name": "Eaton Corporation plc",             "type": "Stock", "sector": "Industrials"},
    # ── Materials ────────────────────────────────────────────────────────────
    {"symbol": "FCX",   "name": "Freeport-McMoRan Inc.",             "type": "Stock", "sector": "Materials"},
    {"symbol": "NEM",   "name": "Newmont Corporation",               "type": "Stock", "sector": "Materials"},
    {"symbol": "LIN",   "name": "Linde plc",                         "type": "Stock", "sector": "Materials"},
    {"symbol": "APD",   "name": "Air Products and Chemicals Inc.",   "type": "Stock", "sector": "Materials"},
    {"symbol": "SHW",   "name": "Sherwin-Williams Company",          "type": "Stock", "sector": "Materials"},
    {"symbol": "DOW",   "name": "Dow Inc.",                          "type": "Stock", "sector": "Materials"},
    {"symbol": "DD",    "name": "DuPont de Nemours Inc.",            "type": "Stock", "sector": "Materials"},
    {"symbol": "AA",    "name": "Alcoa Corporation",                 "type": "Stock", "sector": "Materials"},
    # ── Utilities ────────────────────────────────────────────────────────────
    {"symbol": "NEE",   "name": "NextEra Energy Inc.",               "type": "Stock", "sector": "Utilities"},
    {"symbol": "DUK",   "name": "Duke Energy Corporation",           "type": "Stock", "sector": "Utilities"},
    {"symbol": "SO",    "name": "Southern Company",                  "type": "Stock", "sector": "Utilities"},
    {"symbol": "AEP",   "name": "American Electric Power Co. Inc.",  "type": "Stock", "sector": "Utilities"},
    {"symbol": "EXC",   "name": "Exelon Corporation",                "type": "Stock", "sector": "Utilities"},
    {"symbol": "D",     "name": "Dominion Energy Inc.",              "type": "Stock", "sector": "Utilities"},
    # ── Real Estate ──────────────────────────────────────────────────────────
    {"symbol": "AMT",   "name": "American Tower Corporation",        "type": "Stock", "sector": "Real Estate"},
    {"symbol": "PLD",   "name": "Prologis Inc.",                     "type": "Stock", "sector": "Real Estate"},
    {"symbol": "EQIX",  "name": "Equinix Inc.",                      "type": "Stock", "sector": "Real Estate"},
    {"symbol": "O",     "name": "Realty Income Corporation",         "type": "Stock", "sector": "Real Estate"},
    {"symbol": "SPG",   "name": "Simon Property Group Inc.",         "type": "Stock", "sector": "Real Estate"},
    {"symbol": "WELL",  "name": "Welltower Inc.",                    "type": "Stock", "sector": "Real Estate"},
    {"symbol": "PSA",   "name": "Public Storage",                    "type": "Stock", "sector": "Real Estate"},
    # ── International ────────────────────────────────────────────────────────
    {"symbol": "TSM",   "name": "Taiwan Semiconductor Mfg Co.",      "type": "Stock", "sector": "Technology"},
    {"symbol": "ASML",  "name": "ASML Holding N.V.",                 "type": "Stock", "sector": "Technology"},
    {"symbol": "NVO",   "name": "Novo Nordisk A/S",                  "type": "Stock", "sector": "Healthcare"},
    {"symbol": "SAP",   "name": "SAP SE",                            "type": "Stock", "sector": "Technology"},
    {"symbol": "SHOP",  "name": "Shopify Inc.",                      "type": "Stock", "sector": "Technology"},
    {"symbol": "BABA",  "name": "Alibaba Group Holding Ltd.",        "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "PDD",   "name": "PDD Holdings Inc.",                 "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "JD",    "name": "JD.com Inc.",                       "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "NIO",   "name": "NIO Inc.",                          "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "VALE",  "name": "Vale S.A.",                         "type": "Stock", "sector": "Materials"},
    {"symbol": "RIO",   "name": "Rio Tinto Group",                   "type": "Stock", "sector": "Materials"},
    {"symbol": "BHP",   "name": "BHP Group Limited",                 "type": "Stock", "sector": "Materials"},
    {"symbol": "TM",    "name": "Toyota Motor Corporation",          "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "SONY",  "name": "Sony Group Corporation",            "type": "Stock", "sector": "Consumer Discretionary"},
    # ── Broad Market ETFs ────────────────────────────────────────────────────
    {"symbol": "SPY",   "name": "SPDR S&P 500 ETF Trust",           "type": "ETF", "sector": "Broad Market"},
    {"symbol": "QQQ",   "name": "Invesco QQQ Trust",                 "type": "ETF", "sector": "Technology"},
    {"symbol": "IWM",   "name": "iShares Russell 2000 ETF",          "type": "ETF", "sector": "Small Cap"},
    {"symbol": "DIA",   "name": "SPDR Dow Jones Industrial Avg ETF", "type": "ETF", "sector": "Broad Market"},
    {"symbol": "VTI",   "name": "Vanguard Total Stock Market ETF",   "type": "ETF", "sector": "Broad Market"},
    {"symbol": "VOO",   "name": "Vanguard S&P 500 ETF",              "type": "ETF", "sector": "Broad Market"},
    {"symbol": "IVV",   "name": "iShares Core S&P 500 ETF",          "type": "ETF", "sector": "Broad Market"},
    {"symbol": "RSP",   "name": "Invesco S&P 500 Equal Weight ETF",  "type": "ETF", "sector": "Broad Market"},
    {"symbol": "MDY",   "name": "SPDR S&P MidCap 400 ETF",           "type": "ETF", "sector": "Mid Cap"},
    {"symbol": "IJH",   "name": "iShares Core S&P Mid-Cap ETF",      "type": "ETF", "sector": "Mid Cap"},
    {"symbol": "IJR",   "name": "iShares Core S&P Small-Cap ETF",    "type": "ETF", "sector": "Small Cap"},
    {"symbol": "ACWI",  "name": "iShares MSCI ACWI ETF",             "type": "ETF", "sector": "Global"},
    {"symbol": "VEU",   "name": "Vanguard FTSE All-World ex-US ETF", "type": "ETF", "sector": "International"},
    {"symbol": "SPDW",  "name": "SPDR Portfolio Developed World ETF", "type": "ETF", "sector": "International"},
    # ── Sector ETFs ──────────────────────────────────────────────────────────
    {"symbol": "XLK",   "name": "Technology Select Sector SPDR",     "type": "ETF", "sector": "Technology"},
    {"symbol": "XLF",   "name": "Financial Select Sector SPDR",      "type": "ETF", "sector": "Financials"},
    {"symbol": "XLE",   "name": "Energy Select Sector SPDR",         "type": "ETF", "sector": "Energy"},
    {"symbol": "XLV",   "name": "Health Care Select Sector SPDR",    "type": "ETF", "sector": "Healthcare"},
    {"symbol": "XLI",   "name": "Industrial Select Sector SPDR",     "type": "ETF", "sector": "Industrials"},
    {"symbol": "XLP",   "name": "Consumer Staples Select Sector SPDR","type": "ETF", "sector": "Consumer Staples"},
    {"symbol": "XLY",   "name": "Consumer Discretionary Select SPDR", "type": "ETF", "sector": "Consumer Discretionary"},
    {"symbol": "XLB",   "name": "Materials Select Sector SPDR",      "type": "ETF", "sector": "Materials"},
    {"symbol": "XLU",   "name": "Utilities Select Sector SPDR",      "type": "ETF", "sector": "Utilities"},
    {"symbol": "XLRE",  "name": "Real Estate Select Sector SPDR",    "type": "ETF", "sector": "Real Estate"},
    {"symbol": "XLC",   "name": "Communication Services Select SPDR", "type": "ETF", "sector": "Communication Services"},
    {"symbol": "VGT",   "name": "Vanguard Information Technology ETF","type": "ETF", "sector": "Technology"},
    {"symbol": "VNQ",   "name": "Vanguard Real Estate ETF",           "type": "ETF", "sector": "Real Estate"},
    {"symbol": "VFH",   "name": "Vanguard Financials ETF",            "type": "ETF", "sector": "Financials"},
    {"symbol": "VDE",   "name": "Vanguard Energy ETF",                "type": "ETF", "sector": "Energy"},
    {"symbol": "VCR",   "name": "Vanguard Consumer Discretionary ETF","type": "ETF", "sector": "Consumer Discretionary"},
    {"symbol": "VHT",   "name": "Vanguard Health Care ETF",           "type": "ETF", "sector": "Healthcare"},
    # ── Thematic ETFs ────────────────────────────────────────────────────────
    {"symbol": "SMH",   "name": "VanEck Semiconductor ETF",           "type": "ETF", "sector": "Technology"},
    {"symbol": "SOXX",  "name": "iShares Semiconductor ETF",          "type": "ETF", "sector": "Technology"},
    {"symbol": "XBI",   "name": "SPDR S&P Biotech ETF",               "type": "ETF", "sector": "Healthcare"},
    {"symbol": "IBB",   "name": "iShares Biotechnology ETF",          "type": "ETF", "sector": "Healthcare"},
    {"symbol": "KRE",   "name": "SPDR S&P Regional Banking ETF",      "type": "ETF", "sector": "Financials"},
    {"symbol": "XRT",   "name": "SPDR S&P Retail ETF",                "type": "ETF", "sector": "Consumer Discretionary"},
    {"symbol": "JETS",  "name": "US Global Jets ETF",                 "type": "ETF", "sector": "Industrials"},
    {"symbol": "ICLN",  "name": "iShares Global Clean Energy ETF",    "type": "ETF", "sector": "Energy"},
    {"symbol": "TAN",   "name": "Invesco Solar ETF",                  "type": "ETF", "sector": "Energy"},
    {"symbol": "ROBO",  "name": "Robo Global Robotics & Auto ETF",    "type": "ETF", "sector": "Technology"},
    {"symbol": "BOTZ",  "name": "Global X Robotics & AI ETF",         "type": "ETF", "sector": "Technology"},
    {"symbol": "HACK",  "name": "ETFMG Prime Cyber Security ETF",     "type": "ETF", "sector": "Technology"},
    {"symbol": "IBIT",  "name": "iShares Bitcoin Trust ETF",          "type": "ETF", "sector": "Crypto"},
    {"symbol": "BITO",  "name": "ProShares Bitcoin Strategy ETF",     "type": "ETF", "sector": "Crypto"},
    {"symbol": "KWEB",  "name": "KraneShares China Internet ETF",     "type": "ETF", "sector": "International"},
    {"symbol": "FXI",   "name": "iShares China Large-Cap ETF",        "type": "ETF", "sector": "International"},
    {"symbol": "EWJ",   "name": "iShares MSCI Japan ETF",             "type": "ETF", "sector": "International"},
    {"symbol": "EWZ",   "name": "iShares MSCI Brazil ETF",            "type": "ETF", "sector": "International"},
    {"symbol": "EWY",   "name": "iShares MSCI South Korea ETF",       "type": "ETF", "sector": "International"},
    {"symbol": "EWG",   "name": "iShares MSCI Germany ETF",           "type": "ETF", "sector": "International"},
    {"symbol": "EWU",   "name": "iShares MSCI United Kingdom ETF",    "type": "ETF", "sector": "International"},
    {"symbol": "EEM",   "name": "iShares MSCI Emerging Markets ETF",  "type": "ETF", "sector": "Emerging Markets"},
    {"symbol": "EFA",   "name": "iShares MSCI EAFE ETF",              "type": "ETF", "sector": "International"},
    {"symbol": "VEA",   "name": "Vanguard FTSE Developed Markets ETF","type": "ETF", "sector": "International"},
    {"symbol": "VWO",   "name": "Vanguard FTSE Emerging Markets ETF", "type": "ETF", "sector": "Emerging Markets"},
    # ── Factor / Dividend ETFs ───────────────────────────────────────────────
    {"symbol": "IWF",   "name": "iShares Russell 1000 Growth ETF",    "type": "ETF", "sector": "Growth"},
    {"symbol": "IWD",   "name": "iShares Russell 1000 Value ETF",     "type": "ETF", "sector": "Value"},
    {"symbol": "SCHD",  "name": "Schwab US Dividend Equity ETF",      "type": "ETF", "sector": "Dividend"},
    {"symbol": "VIG",   "name": "Vanguard Dividend Appreciation ETF", "type": "ETF", "sector": "Dividend"},
    {"symbol": "DVY",   "name": "iShares Select Dividend ETF",        "type": "ETF", "sector": "Dividend"},
    {"symbol": "DGRO",  "name": "iShares Core Dividend Growth ETF",   "type": "ETF", "sector": "Dividend"},
    {"symbol": "JEPI",  "name": "JPMorgan Equity Premium Income ETF", "type": "ETF", "sector": "Income"},
    {"symbol": "JEPQ",  "name": "JPMorgan Nasdaq Equity Premium ETF", "type": "ETF", "sector": "Income"},
    {"symbol": "QYLD",  "name": "Global X NASDAQ 100 Covered Call ETF","type": "ETF", "sector": "Income"},
    {"symbol": "XYLD",  "name": "Global X S&P 500 Covered Call ETF",  "type": "ETF", "sector": "Income"},
    {"symbol": "USMV",  "name": "iShares MSCI USA Min Vol Factor ETF","type": "ETF", "sector": "Low Volatility"},
    {"symbol": "QUAL",  "name": "iShares MSCI USA Quality Factor ETF","type": "ETF", "sector": "Quality"},
    {"symbol": "MTUM",  "name": "iShares MSCI USA Momentum Factor ETF","type": "ETF","sector": "Momentum"},
    {"symbol": "VLUE",  "name": "iShares MSCI USA Value Factor ETF",  "type": "ETF", "sector": "Value"},
    # ── Bond ETFs ────────────────────────────────────────────────────────────
    {"symbol": "TLT",   "name": "iShares 20+ Year Treasury Bond ETF", "type": "ETF", "sector": "Bonds"},
    {"symbol": "IEF",   "name": "iShares 7-10 Year Treasury Bond ETF","type": "ETF", "sector": "Bonds"},
    {"symbol": "SHY",   "name": "iShares 1-3 Year Treasury Bond ETF", "type": "ETF", "sector": "Bonds"},
    {"symbol": "AGG",   "name": "iShares Core US Aggregate Bond ETF", "type": "ETF", "sector": "Bonds"},
    {"symbol": "BND",   "name": "Vanguard Total Bond Market ETF",     "type": "ETF", "sector": "Bonds"},
    {"symbol": "LQD",   "name": "iShares IG Corporate Bond ETF",      "type": "ETF", "sector": "Bonds"},
    {"symbol": "HYG",   "name": "iShares High Yield Corporate Bond ETF","type":"ETF", "sector": "Bonds"},
    {"symbol": "EMB",   "name": "iShares JP Morgan EM Bond ETF",      "type": "ETF", "sector": "Bonds"},
    {"symbol": "BNDX",  "name": "Vanguard Total Intl Bond ETF",       "type": "ETF", "sector": "Bonds"},
    # ── Commodities & Alternatives ───────────────────────────────────────────
    {"symbol": "GLD",   "name": "SPDR Gold Shares",                   "type": "ETF", "sector": "Commodities"},
    {"symbol": "IAU",   "name": "iShares Gold Trust",                 "type": "ETF", "sector": "Commodities"},
    {"symbol": "SLV",   "name": "iShares Silver Trust",               "type": "ETF", "sector": "Commodities"},
    {"symbol": "USO",   "name": "United States Oil Fund",             "type": "ETF", "sector": "Commodities"},
    {"symbol": "UNG",   "name": "United States Natural Gas Fund",     "type": "ETF", "sector": "Commodities"},
    {"symbol": "DBC",   "name": "Invesco DB Commodity Index ETF",     "type": "ETF", "sector": "Commodities"},
    # ── Leveraged / Inverse / Volatility ETFs ───────────────────────────────
    {"symbol": "ARKK",  "name": "ARK Innovation ETF",                 "type": "ETF", "sector": "Innovation"},
    {"symbol": "ARKG",  "name": "ARK Genomic Revolution ETF",         "type": "ETF", "sector": "Healthcare"},
    {"symbol": "ARKW",  "name": "ARK Next Generation Internet ETF",   "type": "ETF", "sector": "Technology"},
    {"symbol": "TQQQ",  "name": "ProShares UltraPro QQQ (3x)",        "type": "ETF", "sector": "Leveraged"},
    {"symbol": "SQQQ",  "name": "ProShares UltraPro Short QQQ (3x)", "type": "ETF", "sector": "Inverse"},
    {"symbol": "SPXL",  "name": "Direxion Daily S&P 500 Bull 3X",    "type": "ETF", "sector": "Leveraged"},
    {"symbol": "SPXS",  "name": "Direxion Daily S&P 500 Bear 3X",    "type": "ETF", "sector": "Inverse"},
    {"symbol": "UVXY",  "name": "ProShares Ultra VIX Short-Term ETF", "type": "ETF", "sector": "Volatility"},
    {"symbol": "VXX",   "name": "iPath S&P 500 VIX Short-Term ETN",  "type": "ETF", "sector": "Volatility"},
    {"symbol": "SVXY",  "name": "ProShares Short VIX Short-Term ETF", "type": "ETF", "sector": "Volatility"},

    # ── Broad Market ETFs (additional) ───────────────────────────────────────
    {"symbol": "SPLG",  "name": "SPDR Portfolio S&P 500 ETF",           "type": "ETF", "sector": "Broad Market"},
    {"symbol": "QQQM",  "name": "Invesco NASDAQ 100 ETF",               "type": "ETF", "sector": "Technology"},
    {"symbol": "VT",    "name": "Vanguard Total World Stock ETF",        "type": "ETF", "sector": "Global"},
    {"symbol": "VXUS",  "name": "Vanguard Total International Stock ETF","type": "ETF", "sector": "International"},
    {"symbol": "SCHB",  "name": "Schwab US Broad Market ETF",           "type": "ETF", "sector": "Broad Market"},
    {"symbol": "ITOT",  "name": "iShares Core S&P Total US Stock ETF",  "type": "ETF", "sector": "Broad Market"},
    {"symbol": "VV",    "name": "Vanguard Large-Cap ETF",                "type": "ETF", "sector": "Broad Market"},
    {"symbol": "VO",    "name": "Vanguard Mid-Cap ETF",                  "type": "ETF", "sector": "Mid Cap"},
    {"symbol": "VB",    "name": "Vanguard Small-Cap ETF",                "type": "ETF", "sector": "Small Cap"},
    {"symbol": "SCHA",  "name": "Schwab US Small-Cap ETF",               "type": "ETF", "sector": "Small Cap"},
    {"symbol": "SCHX",  "name": "Schwab US Large-Cap ETF",               "type": "ETF", "sector": "Broad Market"},
    {"symbol": "IWB",   "name": "iShares Russell 1000 ETF",              "type": "ETF", "sector": "Broad Market"},
    {"symbol": "IWR",   "name": "iShares Russell Mid-Cap ETF",           "type": "ETF", "sector": "Mid Cap"},
    {"symbol": "SPTM",  "name": "SPDR Portfolio S&P 1500 Composite ETF", "type": "ETF", "sector": "Broad Market"},
    {"symbol": "SCHK",  "name": "Schwab 1000 Index ETF",                 "type": "ETF", "sector": "Broad Market"},
    {"symbol": "IWC",   "name": "iShares Micro-Cap ETF",                 "type": "ETF", "sector": "Small Cap"},
    {"symbol": "VXF",   "name": "Vanguard Extended Market ETF",          "type": "ETF", "sector": "Broad Market"},

    # ── International ETFs (additional) ──────────────────────────────────────
    {"symbol": "MCHI",  "name": "iShares MSCI China ETF",               "type": "ETF", "sector": "International"},
    {"symbol": "INDA",  "name": "iShares MSCI India ETF",               "type": "ETF", "sector": "International"},
    {"symbol": "EWA",   "name": "iShares MSCI Australia ETF",           "type": "ETF", "sector": "International"},
    {"symbol": "EWC",   "name": "iShares MSCI Canada ETF",              "type": "ETF", "sector": "International"},
    {"symbol": "EWI",   "name": "iShares MSCI Italy ETF",               "type": "ETF", "sector": "International"},
    {"symbol": "EWP",   "name": "iShares MSCI Spain ETF",               "type": "ETF", "sector": "International"},
    {"symbol": "EWQ",   "name": "iShares MSCI France ETF",              "type": "ETF", "sector": "International"},
    {"symbol": "EWD",   "name": "iShares MSCI Sweden ETF",              "type": "ETF", "sector": "International"},
    {"symbol": "EWN",   "name": "iShares MSCI Netherlands ETF",         "type": "ETF", "sector": "International"},
    {"symbol": "EWL",   "name": "iShares MSCI Switzerland ETF",         "type": "ETF", "sector": "International"},
    {"symbol": "EWS",   "name": "iShares MSCI Singapore ETF",           "type": "ETF", "sector": "International"},
    {"symbol": "EWT",   "name": "iShares MSCI Taiwan ETF",              "type": "ETF", "sector": "International"},
    {"symbol": "EWH",   "name": "iShares MSCI Hong Kong ETF",           "type": "ETF", "sector": "International"},
    {"symbol": "EWM",   "name": "iShares MSCI Malaysia ETF",            "type": "ETF", "sector": "International"},
    {"symbol": "EWW",   "name": "iShares MSCI Mexico ETF",              "type": "ETF", "sector": "International"},
    {"symbol": "THD",   "name": "iShares MSCI Thailand ETF",            "type": "ETF", "sector": "International"},
    {"symbol": "EPOL",  "name": "iShares MSCI Poland ETF",              "type": "ETF", "sector": "International"},
    {"symbol": "EPHE",  "name": "iShares MSCI Philippines ETF",         "type": "ETF", "sector": "International"},
    {"symbol": "ECH",   "name": "iShares MSCI Chile ETF",               "type": "ETF", "sector": "International"},
    {"symbol": "EPU",   "name": "iShares MSCI Peru ETF",                "type": "ETF", "sector": "International"},
    {"symbol": "IEFA",  "name": "iShares Core MSCI EAFE ETF",           "type": "ETF", "sector": "International"},
    {"symbol": "IEMG",  "name": "iShares Core MSCI Emerging Markets ETF","type": "ETF","sector": "Emerging Markets"},
    {"symbol": "GXC",   "name": "SPDR S&P China ETF",                   "type": "ETF", "sector": "International"},
    {"symbol": "CQQQ",  "name": "Invesco China Technology ETF",         "type": "ETF", "sector": "International"},
    {"symbol": "ASHR",  "name": "Xtrackers CSI 300 China A-Shares ETF", "type": "ETF", "sector": "International"},
    {"symbol": "INDY",  "name": "iShares India 50 ETF",                 "type": "ETF", "sector": "International"},
    {"symbol": "PIN",   "name": "Invesco India ETF",                    "type": "ETF", "sector": "International"},
    {"symbol": "VPL",   "name": "Vanguard FTSE Pacific ETF",            "type": "ETF", "sector": "International"},
    {"symbol": "VGK",   "name": "Vanguard FTSE Europe ETF",             "type": "ETF", "sector": "International"},
    {"symbol": "ILF",   "name": "iShares Latin America 40 ETF",         "type": "ETF", "sector": "International"},
    {"symbol": "EWK",   "name": "iShares MSCI Belgium ETF",             "type": "ETF", "sector": "International"},
    {"symbol": "EWO",   "name": "iShares MSCI Austria ETF",             "type": "ETF", "sector": "International"},
    {"symbol": "HEWJ",  "name": "iShares Currency Hedged MSCI Japan ETF","type": "ETF","sector": "International"},
    {"symbol": "DBEF",  "name": "Xtrackers MSCI EAFE Hedged Equity ETF","type": "ETF", "sector": "International"},

    # ── Additional Sector ETFs ────────────────────────────────────────────────
    {"symbol": "VIS",   "name": "Vanguard Industrials ETF",             "type": "ETF", "sector": "Industrials"},
    {"symbol": "VAW",   "name": "Vanguard Materials ETF",               "type": "ETF", "sector": "Materials"},
    {"symbol": "VPU",   "name": "Vanguard Utilities ETF",               "type": "ETF", "sector": "Utilities"},
    {"symbol": "VOX",   "name": "Vanguard Communication Services ETF",  "type": "ETF", "sector": "Communication Services"},
    {"symbol": "SCHH",  "name": "Schwab US REIT ETF",                   "type": "ETF", "sector": "Real Estate"},
    {"symbol": "IYR",   "name": "iShares US Real Estate ETF",           "type": "ETF", "sector": "Real Estate"},
    {"symbol": "REM",   "name": "iShares Mortgage Real Estate ETF",     "type": "ETF", "sector": "Real Estate"},
    {"symbol": "FTEC",  "name": "Fidelity MSCI Information Technology ETF","type": "ETF","sector": "Technology"},
    {"symbol": "IGV",   "name": "iShares Expanded Tech-Software ETF",   "type": "ETF", "sector": "Technology"},
    {"symbol": "IYF",   "name": "iShares US Financials ETF",            "type": "ETF", "sector": "Financials"},
    {"symbol": "KBE",   "name": "SPDR S&P Bank ETF",                    "type": "ETF", "sector": "Financials"},
    {"symbol": "IAI",   "name": "iShares US Broker-Dealers ETF",        "type": "ETF", "sector": "Financials"},
    {"symbol": "OIH",   "name": "VanEck Oil Services ETF",              "type": "ETF", "sector": "Energy"},
    {"symbol": "FENY",  "name": "Fidelity MSCI Energy ETF",             "type": "ETF", "sector": "Energy"},
    {"symbol": "IYH",   "name": "iShares US Healthcare ETF",            "type": "ETF", "sector": "Healthcare"},
    {"symbol": "IYJ",   "name": "iShares US Industrials ETF",           "type": "ETF", "sector": "Industrials"},
    {"symbol": "IYC",   "name": "iShares US Consumer Discretionary ETF","type": "ETF", "sector": "Consumer Discretionary"},
    {"symbol": "IYK",   "name": "iShares US Consumer Staples ETF",      "type": "ETF", "sector": "Consumer Staples"},
    {"symbol": "IYE",   "name": "iShares US Energy ETF",                "type": "ETF", "sector": "Energy"},
    {"symbol": "IYM",   "name": "iShares US Basic Materials ETF",       "type": "ETF", "sector": "Materials"},
    {"symbol": "IYZ",   "name": "iShares US Telecommunications ETF",    "type": "ETF", "sector": "Communication Services"},

    # ── Thematic ETFs (additional) ────────────────────────────────────────────
    {"symbol": "SKYY",  "name": "First Trust Cloud Computing ETF",      "type": "ETF", "sector": "Technology"},
    {"symbol": "WCLD",  "name": "WisdomTree Cloud Computing ETF",       "type": "ETF", "sector": "Technology"},
    {"symbol": "BUG",   "name": "Global X Cybersecurity ETF",           "type": "ETF", "sector": "Technology"},
    {"symbol": "CIBR",  "name": "First Trust NASDAQ Cybersecurity ETF", "type": "ETF", "sector": "Technology"},
    {"symbol": "IRBO",  "name": "iShares Robotics & AI Multisector ETF","type": "ETF", "sector": "Technology"},
    {"symbol": "AIQ",   "name": "Global X Artificial Intelligence ETF", "type": "ETF", "sector": "Technology"},
    {"symbol": "CHAT",  "name": "Roundhill Generative AI & Tech ETF",   "type": "ETF", "sector": "Technology"},
    {"symbol": "IGPT",  "name": "Invesco AI and Next Gen Software ETF", "type": "ETF", "sector": "Technology"},
    {"symbol": "ESPO",  "name": "VanEck Video Gaming & eSports ETF",    "type": "ETF", "sector": "Technology"},
    {"symbol": "HERO",  "name": "Global X Video Games & Esports ETF",   "type": "ETF", "sector": "Technology"},
    {"symbol": "METV",  "name": "Roundhill Ball Metaverse ETF",         "type": "ETF", "sector": "Technology"},
    {"symbol": "BETZ",  "name": "Roundhill Sports Betting & iGaming ETF","type": "ETF","sector": "Consumer Discretionary"},
    {"symbol": "MOON",  "name": "Direxion Moonshot Innovators ETF",     "type": "ETF", "sector": "Innovation"},
    {"symbol": "UFO",   "name": "Procure Space ETF",                    "type": "ETF", "sector": "Innovation"},
    {"symbol": "ARKQ",  "name": "ARK Autonomous Tech & Robotics ETF",   "type": "ETF", "sector": "Innovation"},
    {"symbol": "ARKX",  "name": "ARK Space Exploration ETF",            "type": "ETF", "sector": "Innovation"},
    {"symbol": "ARKF",  "name": "ARK Fintech Innovation ETF",           "type": "ETF", "sector": "Financials"},
    {"symbol": "PRNT",  "name": "3D Printing ETF",                      "type": "ETF", "sector": "Innovation"},
    {"symbol": "DAPP",  "name": "VanEck Digital Transformation ETF",    "type": "ETF", "sector": "Crypto"},
    {"symbol": "BLOK",  "name": "Amplify Transformational Data Sharing ETF","type":"ETF","sector": "Crypto"},
    {"symbol": "BITB",  "name": "Bitwise Bitcoin ETF",                  "type": "ETF", "sector": "Crypto"},
    {"symbol": "FBTC",  "name": "Fidelity Wise Origin Bitcoin Fund",    "type": "ETF", "sector": "Crypto"},
    {"symbol": "ARKB",  "name": "ARK 21Shares Bitcoin ETF",             "type": "ETF", "sector": "Crypto"},
    {"symbol": "GBTC",  "name": "Grayscale Bitcoin Trust",              "type": "ETF", "sector": "Crypto"},
    {"symbol": "ETHE",  "name": "Grayscale Ethereum Trust",             "type": "ETF", "sector": "Crypto"},
    {"symbol": "FAN",   "name": "First Trust Global Wind Energy ETF",   "type": "ETF", "sector": "Energy"},
    {"symbol": "QCLN",  "name": "First Trust NASDAQ Clean Edge Green Energy ETF","type":"ETF","sector":"Energy"},
    {"symbol": "GRID",  "name": "First Trust NASDAQ Clean Edge Smart Grid ETF","type":"ETF","sector":"Utilities"},
    {"symbol": "NLR",   "name": "VanEck Uranium + Nuclear Energy ETF",  "type": "ETF", "sector": "Energy"},
    {"symbol": "URA",   "name": "Global X Uranium ETF",                 "type": "ETF", "sector": "Energy"},
    {"symbol": "LIT",   "name": "Global X Lithium & Battery Tech ETF",  "type": "ETF", "sector": "Materials"},
    {"symbol": "COPX",  "name": "Global X Copper Miners ETF",           "type": "ETF", "sector": "Materials"},
    {"symbol": "GDX",   "name": "VanEck Gold Miners ETF",               "type": "ETF", "sector": "Materials"},
    {"symbol": "GDXJ",  "name": "VanEck Junior Gold Miners ETF",        "type": "ETF", "sector": "Materials"},
    {"symbol": "SIL",   "name": "Global X Silver Miners ETF",           "type": "ETF", "sector": "Materials"},
    {"symbol": "GAMR",  "name": "Amplify Video Game Tech ETF",          "type": "ETF", "sector": "Technology"},
    {"symbol": "NERD",  "name": "Roundhill Video Games ETF",            "type": "ETF", "sector": "Technology"},
    {"symbol": "THNQ",  "name": "ROBO Global Artificial Intelligence ETF","type":"ETF","sector": "Technology"},

    # ── Dividend / Income ETFs (additional) ──────────────────────────────────
    {"symbol": "VYM",   "name": "Vanguard High Dividend Yield ETF",     "type": "ETF", "sector": "Dividend"},
    {"symbol": "HDV",   "name": "iShares Core High Dividend ETF",       "type": "ETF", "sector": "Dividend"},
    {"symbol": "NOBL",  "name": "ProShares S&P 500 Dividend Aristocrats ETF","type":"ETF","sector":"Dividend"},
    {"symbol": "SDY",   "name": "SPDR S&P Dividend ETF",                "type": "ETF", "sector": "Dividend"},
    {"symbol": "SPHD",  "name": "Invesco S&P 500 High Dividend Low Vol ETF","type":"ETF","sector":"Dividend"},
    {"symbol": "DGRW",  "name": "WisdomTree US Quality Dividend Growth ETF","type":"ETF","sector":"Dividend"},
    {"symbol": "COWZ",  "name": "Pacer US Cash Cows 100 ETF",           "type": "ETF", "sector": "Value"},
    # ── Factor / Style ETFs (additional) ─────────────────────────────────────
    {"symbol": "VUG",   "name": "Vanguard Growth ETF",                  "type": "ETF", "sector": "Growth"},
    {"symbol": "VTV",   "name": "Vanguard Value ETF",                   "type": "ETF", "sector": "Value"},
    {"symbol": "SPYG",  "name": "SPDR Portfolio S&P 500 Growth ETF",    "type": "ETF", "sector": "Growth"},
    {"symbol": "SPYV",  "name": "SPDR Portfolio S&P 500 Value ETF",     "type": "ETF", "sector": "Value"},
    {"symbol": "SCHG",  "name": "Schwab US Large-Cap Growth ETF",       "type": "ETF", "sector": "Growth"},
    {"symbol": "SCHV",  "name": "Schwab US Large-Cap Value ETF",        "type": "ETF", "sector": "Value"},
    {"symbol": "IWO",   "name": "iShares Russell 2000 Growth ETF",      "type": "ETF", "sector": "Growth"},
    {"symbol": "IWN",   "name": "iShares Russell 2000 Value ETF",       "type": "ETF", "sector": "Value"},

    # ── Fixed Income ETFs (additional) ───────────────────────────────────────
    {"symbol": "GOVT",  "name": "iShares US Treasury Bond ETF",         "type": "ETF", "sector": "Bonds"},
    {"symbol": "TIP",   "name": "iShares TIPS Bond ETF",                "type": "ETF", "sector": "Bonds"},
    {"symbol": "BIL",   "name": "SPDR Bloomberg 1-3 Month T-Bill ETF",  "type": "ETF", "sector": "Bonds"},
    {"symbol": "SGOV",  "name": "iShares 0-3 Month Treasury Bond ETF",  "type": "ETF", "sector": "Bonds"},
    {"symbol": "JNK",   "name": "SPDR Bloomberg High Yield Bond ETF",   "type": "ETF", "sector": "Bonds"},
    {"symbol": "VCIT",  "name": "Vanguard Intermediate-Term Corp Bond ETF","type":"ETF","sector": "Bonds"},
    {"symbol": "VCSH",  "name": "Vanguard Short-Term Corp Bond ETF",    "type": "ETF", "sector": "Bonds"},
    {"symbol": "VGIT",  "name": "Vanguard Intermediate-Term Treasury ETF","type":"ETF","sector": "Bonds"},
    {"symbol": "VGSH",  "name": "Vanguard Short-Term Treasury ETF",     "type": "ETF", "sector": "Bonds"},
    {"symbol": "VGLT",  "name": "Vanguard Long-Term Treasury ETF",      "type": "ETF", "sector": "Bonds"},
    {"symbol": "STIP",  "name": "iShares 0-5 Year TIPS Bond ETF",       "type": "ETF", "sector": "Bonds"},
    {"symbol": "TMF",   "name": "Direxion Daily 20+ Year Treasury Bull 3X","type":"ETF","sector":"Leveraged"},
    {"symbol": "TMV",   "name": "Direxion Daily 20+ Year Treasury Bear 3X","type":"ETF","sector":"Inverse"},
    {"symbol": "TBF",   "name": "ProShares Short 20+ Year Treasury ETF","type": "ETF", "sector": "Inverse"},
    {"symbol": "HYD",   "name": "VanEck High Yield Muni ETF",           "type": "ETF", "sector": "Bonds"},
    {"symbol": "MUB",   "name": "iShares National Muni Bond ETF",       "type": "ETF", "sector": "Bonds"},

    # ── Commodity ETFs (additional) ──────────────────────────────────────────
    {"symbol": "DBA",   "name": "Invesco DB Agriculture Fund",          "type": "ETF", "sector": "Commodities"},
    {"symbol": "PDBC",  "name": "Invesco Optimum Yield Diversified Commodity","type":"ETF","sector":"Commodities"},
    {"symbol": "GSG",   "name": "iShares S&P GSCI Commodity ETF",       "type": "ETF", "sector": "Commodities"},
    {"symbol": "PALL",  "name": "Aberdeen Physical Palladium Shares ETF","type": "ETF","sector": "Commodities"},
    {"symbol": "PPLT",  "name": "Aberdeen Physical Platinum Shares ETF","type": "ETF", "sector": "Commodities"},
    {"symbol": "WEAT",  "name": "Teucrium Wheat Fund",                  "type": "ETF", "sector": "Commodities"},
    {"symbol": "CORN",  "name": "Teucrium Corn Fund",                   "type": "ETF", "sector": "Commodities"},
    {"symbol": "SOYB",  "name": "Teucrium Soybean Fund",                "type": "ETF", "sector": "Commodities"},
    {"symbol": "KRBN",  "name": "KraneShares Global Carbon Strategy ETF","type": "ETF","sector": "Commodities"},
    {"symbol": "CPER",  "name": "United States Copper Index Fund",      "type": "ETF", "sector": "Commodities"},

    # ── Leveraged ETFs (additional) ───────────────────────────────────────────
    {"symbol": "UPRO",  "name": "ProShares UltraPro S&P 500 (3x)",      "type": "ETF", "sector": "Leveraged"},
    {"symbol": "SPXU",  "name": "ProShares UltraPro Short S&P 500 (3x)","type": "ETF", "sector": "Inverse"},
    {"symbol": "SSO",   "name": "ProShares Ultra S&P 500 (2x)",         "type": "ETF", "sector": "Leveraged"},
    {"symbol": "SDS",   "name": "ProShares UltraShort S&P 500 (2x)",    "type": "ETF", "sector": "Inverse"},
    {"symbol": "QLD",   "name": "ProShares Ultra QQQ (2x)",             "type": "ETF", "sector": "Leveraged"},
    {"symbol": "QID",   "name": "ProShares UltraShort QQQ (2x)",        "type": "ETF", "sector": "Inverse"},
    {"symbol": "TECL",  "name": "Direxion Daily Technology Bull (3x)",  "type": "ETF", "sector": "Leveraged"},
    {"symbol": "TECS",  "name": "Direxion Daily Technology Bear (3x)",  "type": "ETF", "sector": "Inverse"},
    {"symbol": "SOXL",  "name": "Direxion Daily Semiconductor Bull (3x)","type": "ETF","sector": "Leveraged"},
    {"symbol": "SOXS",  "name": "Direxion Daily Semiconductor Bear (3x)","type": "ETF","sector": "Inverse"},
    {"symbol": "LABU",  "name": "Direxion Daily S&P Biotech Bull (3x)", "type": "ETF", "sector": "Leveraged"},
    {"symbol": "LABD",  "name": "Direxion Daily S&P Biotech Bear (3x)", "type": "ETF", "sector": "Inverse"},
    {"symbol": "TNA",   "name": "Direxion Daily Small Cap Bull (3x)",   "type": "ETF", "sector": "Leveraged"},
    {"symbol": "TZA",   "name": "Direxion Daily Small Cap Bear (3x)",   "type": "ETF", "sector": "Inverse"},
    {"symbol": "FAS",   "name": "Direxion Daily Financial Bull (3x)",   "type": "ETF", "sector": "Leveraged"},
    {"symbol": "FAZ",   "name": "Direxion Daily Financial Bear (3x)",   "type": "ETF", "sector": "Inverse"},
    {"symbol": "ERX",   "name": "Direxion Daily Energy Bull (2x)",      "type": "ETF", "sector": "Leveraged"},
    {"symbol": "ERY",   "name": "Direxion Daily Energy Bear (2x)",      "type": "ETF", "sector": "Inverse"},
    {"symbol": "SH",    "name": "ProShares Short S&P 500",              "type": "ETF", "sector": "Inverse"},
    {"symbol": "PSQ",   "name": "ProShares Short QQQ",                  "type": "ETF", "sector": "Inverse"},
    {"symbol": "DOG",   "name": "ProShares Short Dow30",                "type": "ETF", "sector": "Inverse"},
    {"symbol": "RWM",   "name": "ProShares Short Russell2000",          "type": "ETF", "sector": "Inverse"},

    # ── Multi-Asset / Balanced ETFs ───────────────────────────────────────────
    {"symbol": "AOR",   "name": "iShares Core Growth Allocation ETF",   "type": "ETF", "sector": "Multi-Asset"},
    {"symbol": "AOM",   "name": "iShares Core Moderate Allocation ETF", "type": "ETF", "sector": "Multi-Asset"},
    {"symbol": "AOA",   "name": "iShares Core Aggressive Allocation ETF","type": "ETF","sector": "Multi-Asset"},
    {"symbol": "AOK",   "name": "iShares Core Conservative Allocation ETF","type":"ETF","sector":"Multi-Asset"},
    {"symbol": "MDIV",  "name": "Multi-Asset Diversified Income ETF",   "type": "ETF", "sector": "Multi-Asset"},

    # ── Additional Technology Stocks ──────────────────────────────────────────
    {"symbol": "ANET",  "name": "Arista Networks Inc.",                 "type": "Stock", "sector": "Technology"},
    {"symbol": "CDNS",  "name": "Cadence Design Systems Inc.",          "type": "Stock", "sector": "Technology"},
    {"symbol": "SNPS",  "name": "Synopsys Inc.",                        "type": "Stock", "sector": "Technology"},
    {"symbol": "ANSS",  "name": "ANSYS Inc.",                           "type": "Stock", "sector": "Technology"},
    {"symbol": "PTC",   "name": "PTC Inc.",                             "type": "Stock", "sector": "Technology"},
    {"symbol": "HUBS",  "name": "HubSpot Inc.",                         "type": "Stock", "sector": "Technology"},
    {"symbol": "ZI",    "name": "ZoomInfo Technologies Inc.",           "type": "Stock", "sector": "Technology"},
    {"symbol": "GTLB",  "name": "GitLab Inc.",                          "type": "Stock", "sector": "Technology"},
    {"symbol": "BILL",  "name": "Bill.com Holdings Inc.",               "type": "Stock", "sector": "Technology"},
    {"symbol": "PCTY",  "name": "Paylocity Holding Corporation",        "type": "Stock", "sector": "Technology"},
    {"symbol": "PAYC",  "name": "Paycom Software Inc.",                 "type": "Stock", "sector": "Technology"},
    {"symbol": "TOST",  "name": "Toast Inc.",                           "type": "Stock", "sector": "Technology"},
    {"symbol": "BRZE",  "name": "Braze Inc.",                           "type": "Stock", "sector": "Technology"},
    {"symbol": "IOT",   "name": "Samsara Inc.",                         "type": "Stock", "sector": "Technology"},
    {"symbol": "ASTS",  "name": "AST SpaceMobile Inc.",                 "type": "Stock", "sector": "Technology"},
    {"symbol": "CIEN",  "name": "Ciena Corporation",                    "type": "Stock", "sector": "Technology"},
    {"symbol": "KEYS",  "name": "Keysight Technologies Inc.",           "type": "Stock", "sector": "Technology"},
    {"symbol": "NTAP",  "name": "NetApp Inc.",                          "type": "Stock", "sector": "Technology"},
    {"symbol": "GDDY",  "name": "GoDaddy Inc.",                         "type": "Stock", "sector": "Technology"},
    {"symbol": "GLOB",  "name": "Globant S.A.",                         "type": "Stock", "sector": "Technology"},
    {"symbol": "EPAM",  "name": "EPAM Systems Inc.",                    "type": "Stock", "sector": "Technology"},
    {"symbol": "CTSH",  "name": "Cognizant Technology Solutions",       "type": "Stock", "sector": "Technology"},
    {"symbol": "WIT",   "name": "Wipro Limited",                        "type": "Stock", "sector": "Technology"},
    {"symbol": "INFY",  "name": "Infosys Limited",                      "type": "Stock", "sector": "Technology"},
    {"symbol": "HCL",   "name": "HCL Technologies (OTC)",               "type": "Stock", "sector": "Technology"},
    {"symbol": "GLBE",  "name": "Global-E Online Ltd.",                 "type": "Stock", "sector": "Technology"},
    {"symbol": "S",     "name": "SentinelOne Inc.",                     "type": "Stock", "sector": "Technology"},
    {"symbol": "VRNS",  "name": "Varonis Systems Inc.",                 "type": "Stock", "sector": "Technology"},
    {"symbol": "RPD",   "name": "Rapid7 Inc.",                          "type": "Stock", "sector": "Technology"},
    {"symbol": "TENB",  "name": "Tenable Holdings Inc.",                "type": "Stock", "sector": "Technology"},
    {"symbol": "QLYS",  "name": "Qualys Inc.",                          "type": "Stock", "sector": "Technology"},
    {"symbol": "CYBR",  "name": "CyberArk Software Ltd.",               "type": "Stock", "sector": "Technology"},
    {"symbol": "SAIL",  "name": "SailPoint Technologies",               "type": "Stock", "sector": "Technology"},
    {"symbol": "TTD",   "name": "The Trade Desk Inc.",                  "type": "Stock", "sector": "Technology"},
    {"symbol": "MGNI",  "name": "Magnite Inc.",                         "type": "Stock", "sector": "Technology"},
    {"symbol": "PUBM",  "name": "PubMatic Inc.",                        "type": "Stock", "sector": "Technology"},
    {"symbol": "DV",    "name": "DoubleVerify Holdings Inc.",            "type": "Stock", "sector": "Technology"},
    {"symbol": "APPS",  "name": "Digital Turbine Inc.",                 "type": "Stock", "sector": "Technology"},
    {"symbol": "SMAR",  "name": "Smartsheet Inc.",                      "type": "Stock", "sector": "Technology"},
    {"symbol": "CFLT",  "name": "Confluent Inc.",                       "type": "Stock", "sector": "Technology"},
    {"symbol": "ESTC",  "name": "Elastic N.V.",                         "type": "Stock", "sector": "Technology"},
    {"symbol": "SUMO",  "name": "Sumo Logic Inc.",                      "type": "Stock", "sector": "Technology"},
    {"symbol": "DT",    "name": "Dynatrace Inc.",                       "type": "Stock", "sector": "Technology"},
    {"symbol": "NTNX",  "name": "Nutanix Inc.",                         "type": "Stock", "sector": "Technology"},
    {"symbol": "PSTG",  "name": "Pure Storage Inc.",                    "type": "Stock", "sector": "Technology"},
    {"symbol": "CEVA",  "name": "CEVA Inc.",                            "type": "Stock", "sector": "Technology"},
    {"symbol": "WOLF",  "name": "Wolfspeed Inc.",                       "type": "Stock", "sector": "Technology"},
    {"symbol": "LSCC",  "name": "Lattice Semiconductor Corporation",    "type": "Stock", "sector": "Technology"},
    {"symbol": "FORM",  "name": "FormFactor Inc.",                      "type": "Stock", "sector": "Technology"},
    {"symbol": "ONTO",  "name": "Onto Innovation Inc.",                 "type": "Stock", "sector": "Technology"},
    {"symbol": "RMBS",  "name": "Rambus Inc.",                          "type": "Stock", "sector": "Technology"},
    {"symbol": "MPWR",  "name": "Monolithic Power Systems Inc.",        "type": "Stock", "sector": "Technology"},
    {"symbol": "ENTG",  "name": "Entegris Inc.",                        "type": "Stock", "sector": "Technology"},

    # ── Additional Healthcare / Biotech Stocks ────────────────────────────────
    {"symbol": "DXCM",  "name": "DexCom Inc.",                          "type": "Stock", "sector": "Healthcare"},
    {"symbol": "IDXX",  "name": "IDEXX Laboratories Inc.",              "type": "Stock", "sector": "Healthcare"},
    {"symbol": "BSX",   "name": "Boston Scientific Corporation",        "type": "Stock", "sector": "Healthcare"},
    {"symbol": "EW",    "name": "Edwards Lifesciences Corporation",     "type": "Stock", "sector": "Healthcare"},
    {"symbol": "ZBH",   "name": "Zimmer Biomet Holdings Inc.",          "type": "Stock", "sector": "Healthcare"},
    {"symbol": "BDX",   "name": "Becton Dickinson and Company",         "type": "Stock", "sector": "Healthcare"},
    {"symbol": "MCK",   "name": "McKesson Corporation",                 "type": "Stock", "sector": "Healthcare"},
    {"symbol": "CI",    "name": "The Cigna Group",                      "type": "Stock", "sector": "Healthcare"},
    {"symbol": "ELV",   "name": "Elevance Health Inc.",                 "type": "Stock", "sector": "Healthcare"},
    {"symbol": "CNC",   "name": "Centene Corporation",                  "type": "Stock", "sector": "Healthcare"},
    {"symbol": "HUM",   "name": "Humana Inc.",                          "type": "Stock", "sector": "Healthcare"},
    {"symbol": "MOH",   "name": "Molina Healthcare Inc.",               "type": "Stock", "sector": "Healthcare"},
    {"symbol": "GEHC",  "name": "GE HealthCare Technologies Inc.",      "type": "Stock", "sector": "Healthcare"},
    {"symbol": "IQV",   "name": "IQVIA Holdings Inc.",                  "type": "Stock", "sector": "Healthcare"},
    {"symbol": "DGX",   "name": "Quest Diagnostics Inc.",               "type": "Stock", "sector": "Healthcare"},
    {"symbol": "LH",    "name": "Labcorp",                              "type": "Stock", "sector": "Healthcare"},
    {"symbol": "HOLX",  "name": "Hologic Inc.",                         "type": "Stock", "sector": "Healthcare"},
    {"symbol": "MASI",  "name": "Masimo Corporation",                   "type": "Stock", "sector": "Healthcare"},
    {"symbol": "INSP",  "name": "Inspire Medical Systems Inc.",         "type": "Stock", "sector": "Healthcare"},
    {"symbol": "ALGN",  "name": "Align Technology Inc.",                "type": "Stock", "sector": "Healthcare"},
    {"symbol": "NVAX",  "name": "Novavax Inc.",                         "type": "Stock", "sector": "Healthcare"},
    {"symbol": "ARCT",  "name": "Arctus Therapeutics Holdings",         "type": "Stock", "sector": "Healthcare"},
    {"symbol": "RXRX",  "name": "Recursion Pharmaceuticals Inc.",       "type": "Stock", "sector": "Healthcare"},
    {"symbol": "TWST",  "name": "Twist Bioscience Corporation",         "type": "Stock", "sector": "Healthcare"},
    {"symbol": "BEAM",  "name": "Beam Therapeutics Inc.",               "type": "Stock", "sector": "Healthcare"},
    {"symbol": "EDIT",  "name": "Editas Medicine Inc.",                 "type": "Stock", "sector": "Healthcare"},
    {"symbol": "CRSP",  "name": "CRISPR Therapeutics AG",               "type": "Stock", "sector": "Healthcare"},
    {"symbol": "NTLA",  "name": "Intellia Therapeutics Inc.",           "type": "Stock", "sector": "Healthcare"},
    {"symbol": "BLUE",  "name": "bluebird bio Inc.",                    "type": "Stock", "sector": "Healthcare"},
    {"symbol": "SGEN",  "name": "Seagen Inc.",                          "type": "Stock", "sector": "Healthcare"},
    {"symbol": "ALNY",  "name": "Alnylam Pharmaceuticals Inc.",         "type": "Stock", "sector": "Healthcare"},
    {"symbol": "IONS",  "name": "Ionis Pharmaceuticals Inc.",           "type": "Stock", "sector": "Healthcare"},
    {"symbol": "EXAS",  "name": "Exact Sciences Corporation",           "type": "Stock", "sector": "Healthcare"},
    {"symbol": "NVCR",  "name": "NovoCure Limited",                     "type": "Stock", "sector": "Healthcare"},
    {"symbol": "AZN",   "name": "AstraZeneca plc",                      "type": "Stock", "sector": "Healthcare"},
    {"symbol": "NVS",   "name": "Novartis AG",                          "type": "Stock", "sector": "Healthcare"},
    {"symbol": "SNY",   "name": "Sanofi S.A.",                          "type": "Stock", "sector": "Healthcare"},
    {"symbol": "GSK",   "name": "GSK plc",                              "type": "Stock", "sector": "Healthcare"},
    {"symbol": "RHHBY", "name": "Roche Holding AG",                     "type": "Stock", "sector": "Healthcare"},

    # ── Additional Consumer Discretionary Stocks ──────────────────────────────
    {"symbol": "BKNG",  "name": "Booking Holdings Inc.",                "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "EXPE",  "name": "Expedia Group Inc.",                   "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "TRIP",  "name": "TripAdvisor Inc.",                     "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "MGM",   "name": "MGM Resorts International",            "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "WYNN",  "name": "Wynn Resorts Limited",                 "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "LVS",   "name": "Las Vegas Sands Corp.",                "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "CZR",   "name": "Caesars Entertainment Inc.",           "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "DKNG",  "name": "DraftKings Inc.",                      "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "PENN",  "name": "PENN Entertainment Inc.",              "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "CMG",   "name": "Chipotle Mexican Grill Inc.",          "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "YUM",   "name": "Yum! Brands Inc.",                     "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "DRI",   "name": "Darden Restaurants Inc.",              "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "WING",  "name": "Wingstop Inc.",                        "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "NCLH",  "name": "Norwegian Cruise Line Holdings Ltd.",  "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "CCL",   "name": "Carnival Corporation & plc",           "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "RCL",   "name": "Royal Caribbean Cruises Ltd.",         "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "ORLY",  "name": "O'Reilly Automotive Inc.",             "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "AZO",   "name": "AutoZone Inc.",                        "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "TSCO",  "name": "Tractor Supply Company",               "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "ULTA",  "name": "Ulta Beauty Inc.",                     "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "TPR",   "name": "Tapestry Inc.",                        "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "DECK",  "name": "Deckers Outdoor Corporation",          "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "HAS",   "name": "Hasbro Inc.",                          "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "MAT",   "name": "Mattel Inc.",                          "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "CPRI",  "name": "Capri Holdings Limited",               "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "RL",    "name": "Ralph Lauren Corporation",             "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "PVH",   "name": "PVH Corp.",                            "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "HBI",   "name": "Hanesbrands Inc.",                     "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "W",     "name": "Wayfair Inc.",                         "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "ETSY",  "name": "Etsy Inc.",                            "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "EBAY",  "name": "eBay Inc.",                            "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "DASH",  "name": "DoorDash Inc.",                        "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "CART",  "name": "Instacart (Maplebear Inc.)",           "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "CVNA",  "name": "Carvana Co.",                          "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "KMX",   "name": "CarMax Inc.",                          "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "AN",    "name": "AutoNation Inc.",                      "type": "Stock", "sector": "Consumer Discretionary"},

    # ── Additional Consumer Staples Stocks ────────────────────────────────────
    {"symbol": "GIS",   "name": "General Mills Inc.",                   "type": "Stock", "sector": "Consumer Staples"},
    {"symbol": "K",     "name": "Kellanova",                            "type": "Stock", "sector": "Consumer Staples"},
    {"symbol": "CPB",   "name": "Campbell Soup Company",                "type": "Stock", "sector": "Consumer Staples"},
    {"symbol": "HRL",   "name": "Hormel Foods Corporation",             "type": "Stock", "sector": "Consumer Staples"},
    {"symbol": "TSN",   "name": "Tyson Foods Inc.",                     "type": "Stock", "sector": "Consumer Staples"},
    {"symbol": "CAG",   "name": "Conagra Brands Inc.",                  "type": "Stock", "sector": "Consumer Staples"},
    {"symbol": "SJM",   "name": "The J.M. Smucker Company",             "type": "Stock", "sector": "Consumer Staples"},
    {"symbol": "MKC",   "name": "McCormick & Company Inc.",             "type": "Stock", "sector": "Consumer Staples"},
    {"symbol": "CHD",   "name": "Church & Dwight Co. Inc.",             "type": "Stock", "sector": "Consumer Staples"},
    {"symbol": "CLX",   "name": "The Clorox Company",                   "type": "Stock", "sector": "Consumer Staples"},
    {"symbol": "EL",    "name": "Estée Lauder Companies Inc.",          "type": "Stock", "sector": "Consumer Staples"},
    {"symbol": "COTY",  "name": "Coty Inc.",                            "type": "Stock", "sector": "Consumer Staples"},
    {"symbol": "KR",    "name": "The Kroger Co.",                       "type": "Stock", "sector": "Consumer Staples"},
    {"symbol": "SFM",   "name": "Sprouts Farmers Market Inc.",          "type": "Stock", "sector": "Consumer Staples"},
    {"symbol": "MNST",  "name": "Monster Beverage Corporation",         "type": "Stock", "sector": "Consumer Staples"},
    {"symbol": "BUD",   "name": "Anheuser-Busch InBev SA/NV",           "type": "Stock", "sector": "Consumer Staples"},
    {"symbol": "STZ",   "name": "Constellation Brands Inc.",            "type": "Stock", "sector": "Consumer Staples"},
    {"symbol": "DEO",   "name": "Diageo plc",                           "type": "Stock", "sector": "Consumer Staples"},
    {"symbol": "TAP",   "name": "Molson Coors Beverage Company",        "type": "Stock", "sector": "Consumer Staples"},

    # ── Additional Financials Stocks ──────────────────────────────────────────
    {"symbol": "CB",    "name": "Chubb Limited",                        "type": "Stock", "sector": "Financials"},
    {"symbol": "MET",   "name": "MetLife Inc.",                         "type": "Stock", "sector": "Financials"},
    {"symbol": "PRU",   "name": "Prudential Financial Inc.",            "type": "Stock", "sector": "Financials"},
    {"symbol": "AFL",   "name": "Aflac Incorporated",                   "type": "Stock", "sector": "Financials"},
    {"symbol": "TRV",   "name": "The Travelers Companies Inc.",         "type": "Stock", "sector": "Financials"},
    {"symbol": "AIG",   "name": "American International Group Inc.",    "type": "Stock", "sector": "Financials"},
    {"symbol": "ALL",   "name": "The Allstate Corporation",             "type": "Stock", "sector": "Financials"},
    {"symbol": "PGR",   "name": "Progressive Corporation",              "type": "Stock", "sector": "Financials"},
    {"symbol": "ICE",   "name": "Intercontinental Exchange Inc.",       "type": "Stock", "sector": "Financials"},
    {"symbol": "CME",   "name": "CME Group Inc.",                       "type": "Stock", "sector": "Financials"},
    {"symbol": "NDAQ",  "name": "Nasdaq Inc.",                          "type": "Stock", "sector": "Financials"},
    {"symbol": "MCO",   "name": "Moody's Corporation",                  "type": "Stock", "sector": "Financials"},
    {"symbol": "FI",    "name": "Fiserv Inc.",                          "type": "Stock", "sector": "Financials"},
    {"symbol": "FIS",   "name": "Fidelity National Information Services","type": "Stock","sector": "Financials"},
    {"symbol": "GPN",   "name": "Global Payments Inc.",                 "type": "Stock", "sector": "Financials"},
    {"symbol": "ALLY",  "name": "Ally Financial Inc.",                  "type": "Stock", "sector": "Financials"},
    {"symbol": "SYF",   "name": "Synchrony Financial",                  "type": "Stock", "sector": "Financials"},
    {"symbol": "DFS",   "name": "Discover Financial Services",          "type": "Stock", "sector": "Financials"},
    {"symbol": "TROW",  "name": "T. Rowe Price Group Inc.",             "type": "Stock", "sector": "Financials"},
    {"symbol": "IVZ",   "name": "Invesco Ltd.",                         "type": "Stock", "sector": "Financials"},
    {"symbol": "AMG",   "name": "Affiliated Managers Group Inc.",       "type": "Stock", "sector": "Financials"},
    {"symbol": "BAM",   "name": "Brookfield Asset Management Ltd.",     "type": "Stock", "sector": "Financials"},
    {"symbol": "BX",    "name": "Blackstone Inc.",                      "type": "Stock", "sector": "Financials"},
    {"symbol": "KKR",   "name": "KKR & Co. Inc.",                       "type": "Stock", "sector": "Financials"},
    {"symbol": "APO",   "name": "Apollo Global Management Inc.",        "type": "Stock", "sector": "Financials"},
    {"symbol": "CG",    "name": "The Carlyle Group Inc.",               "type": "Stock", "sector": "Financials"},
    {"symbol": "ARES",  "name": "Ares Management Corporation",          "type": "Stock", "sector": "Financials"},
    {"symbol": "OWL",   "name": "Blue Owl Capital Inc.",                "type": "Stock", "sector": "Financials"},
    {"symbol": "HDB",   "name": "HDFC Bank Limited",                    "type": "Stock", "sector": "Financials"},
    {"symbol": "IBN",   "name": "ICICI Bank Limited",                   "type": "Stock", "sector": "Financials"},

    # ── Additional Industrials Stocks ─────────────────────────────────────────
    {"symbol": "TDG",   "name": "TransDigm Group Incorporated",         "type": "Stock", "sector": "Industrials"},
    {"symbol": "AXON",  "name": "Axon Enterprise Inc.",                 "type": "Stock", "sector": "Industrials"},
    {"symbol": "PWR",   "name": "Quanta Services Inc.",                 "type": "Stock", "sector": "Industrials"},
    {"symbol": "GNRC",  "name": "Generac Holdings Inc.",                "type": "Stock", "sector": "Industrials"},
    {"symbol": "DAL",   "name": "Delta Air Lines Inc.",                 "type": "Stock", "sector": "Industrials"},
    {"symbol": "UAL",   "name": "United Airlines Holdings Inc.",        "type": "Stock", "sector": "Industrials"},
    {"symbol": "AAL",   "name": "American Airlines Group Inc.",         "type": "Stock", "sector": "Industrials"},
    {"symbol": "LUV",   "name": "Southwest Airlines Co.",               "type": "Stock", "sector": "Industrials"},
    {"symbol": "ALK",   "name": "Alaska Air Group Inc.",                "type": "Stock", "sector": "Industrials"},
    {"symbol": "UNP",   "name": "Union Pacific Corporation",            "type": "Stock", "sector": "Industrials"},
    {"symbol": "NSC",   "name": "Norfolk Southern Corporation",         "type": "Stock", "sector": "Industrials"},
    {"symbol": "CSX",   "name": "CSX Corporation",                      "type": "Stock", "sector": "Industrials"},
    {"symbol": "CP",    "name": "Canadian Pacific Kansas City Ltd.",    "type": "Stock", "sector": "Industrials"},
    {"symbol": "CNI",   "name": "Canadian National Railway Company",    "type": "Stock", "sector": "Industrials"},
    {"symbol": "ODFL",  "name": "Old Dominion Freight Line Inc.",       "type": "Stock", "sector": "Industrials"},
    {"symbol": "XPO",   "name": "XPO Inc.",                             "type": "Stock", "sector": "Industrials"},
    {"symbol": "CHRW",  "name": "C.H. Robinson Worldwide Inc.",         "type": "Stock", "sector": "Industrials"},
    {"symbol": "EXPD",  "name": "Expeditors International Inc.",        "type": "Stock", "sector": "Industrials"},
    {"symbol": "GXO",   "name": "GXO Logistics Inc.",                   "type": "Stock", "sector": "Industrials"},
    {"symbol": "ACHR",  "name": "Archer Aviation Inc.",                 "type": "Stock", "sector": "Industrials"},
    {"symbol": "JOBY",  "name": "Joby Aviation Inc.",                   "type": "Stock", "sector": "Industrials"},
    {"symbol": "LUNR",  "name": "Intuitive Machines Inc.",              "type": "Stock", "sector": "Industrials"},
    {"symbol": "RKLB",  "name": "Rocket Lab USA Inc.",                  "type": "Stock", "sector": "Industrials"},
    {"symbol": "SPIR",  "name": "Spire Global Inc.",                    "type": "Stock", "sector": "Industrials"},
    {"symbol": "PH",    "name": "Parker-Hannifin Corporation",          "type": "Stock", "sector": "Industrials"},
    {"symbol": "ROK",   "name": "Rockwell Automation Inc.",             "type": "Stock", "sector": "Industrials"},
    {"symbol": "AME",   "name": "AMETEK Inc.",                          "type": "Stock", "sector": "Industrials"},
    {"symbol": "ROP",   "name": "Roper Technologies Inc.",              "type": "Stock", "sector": "Industrials"},
    {"symbol": "VRSK",  "name": "Verisk Analytics Inc.",                "type": "Stock", "sector": "Industrials"},
    {"symbol": "CSGP",  "name": "CoStar Group Inc.",                    "type": "Stock", "sector": "Industrials"},
    {"symbol": "WAB",   "name": "Westinghouse Air Brake Technologies",  "type": "Stock", "sector": "Industrials"},

    # ── Additional Energy Stocks ──────────────────────────────────────────────
    {"symbol": "KMI",   "name": "Kinder Morgan Inc.",                   "type": "Stock", "sector": "Energy"},
    {"symbol": "WMB",   "name": "The Williams Companies Inc.",          "type": "Stock", "sector": "Energy"},
    {"symbol": "OKE",   "name": "ONEOK Inc.",                           "type": "Stock", "sector": "Energy"},
    {"symbol": "ET",    "name": "Energy Transfer LP",                   "type": "Stock", "sector": "Energy"},
    {"symbol": "EPD",   "name": "Enterprise Products Partners L.P.",    "type": "Stock", "sector": "Energy"},
    {"symbol": "LNG",   "name": "Cheniere Energy Inc.",                 "type": "Stock", "sector": "Energy"},
    {"symbol": "AR",    "name": "Antero Resources Corporation",         "type": "Stock", "sector": "Energy"},
    {"symbol": "EQT",   "name": "EQT Corporation",                      "type": "Stock", "sector": "Energy"},
    {"symbol": "RRC",   "name": "Range Resources Corporation",          "type": "Stock", "sector": "Energy"},
    {"symbol": "CTRA",  "name": "Coterra Energy Inc.",                  "type": "Stock", "sector": "Energy"},
    {"symbol": "MRO",   "name": "Marathon Oil Corporation",             "type": "Stock", "sector": "Energy"},
    {"symbol": "APA",   "name": "APA Corporation",                      "type": "Stock", "sector": "Energy"},
    {"symbol": "BP",    "name": "BP plc",                               "type": "Stock", "sector": "Energy"},
    {"symbol": "SHEL",  "name": "Shell plc",                            "type": "Stock", "sector": "Energy"},
    {"symbol": "TTE",   "name": "TotalEnergies SE",                     "type": "Stock", "sector": "Energy"},
    {"symbol": "ENPH",  "name": "Enphase Energy Inc.",                  "type": "Stock", "sector": "Energy"},
    {"symbol": "FSLR",  "name": "First Solar Inc.",                     "type": "Stock", "sector": "Energy"},
    {"symbol": "RUN",   "name": "Sunrun Inc.",                          "type": "Stock", "sector": "Energy"},
    {"symbol": "ARRY",  "name": "Array Technologies Inc.",              "type": "Stock", "sector": "Energy"},
    {"symbol": "BE",    "name": "Bloom Energy Corporation",             "type": "Stock", "sector": "Energy"},
    {"symbol": "PLUG",  "name": "Plug Power Inc.",                      "type": "Stock", "sector": "Energy"},
    {"symbol": "BLDP",  "name": "Ballard Power Systems Inc.",           "type": "Stock", "sector": "Energy"},

    # ── Additional Real Estate Stocks ─────────────────────────────────────────
    {"symbol": "CCI",   "name": "Crown Castle Inc.",                    "type": "Stock", "sector": "Real Estate"},
    {"symbol": "SBAC",  "name": "SBA Communications Corporation",       "type": "Stock", "sector": "Real Estate"},
    {"symbol": "DLR",   "name": "Digital Realty Trust Inc.",            "type": "Stock", "sector": "Real Estate"},
    {"symbol": "VICI",  "name": "VICI Properties Inc.",                 "type": "Stock", "sector": "Real Estate"},
    {"symbol": "AVB",   "name": "AvalonBay Communities Inc.",           "type": "Stock", "sector": "Real Estate"},
    {"symbol": "EQR",   "name": "Equity Residential",                   "type": "Stock", "sector": "Real Estate"},
    {"symbol": "MAA",   "name": "Mid-America Apartment Communities",    "type": "Stock", "sector": "Real Estate"},
    {"symbol": "ESS",   "name": "Essex Property Trust Inc.",            "type": "Stock", "sector": "Real Estate"},
    {"symbol": "ARE",   "name": "Alexandria Real Estate Equities Inc.", "type": "Stock", "sector": "Real Estate"},
    {"symbol": "GLPI",  "name": "Gaming and Leisure Properties Inc.",   "type": "Stock", "sector": "Real Estate"},
    {"symbol": "NNN",   "name": "NNN REIT Inc.",                        "type": "Stock", "sector": "Real Estate"},
    {"symbol": "BXP",   "name": "BXP Inc.",                             "type": "Stock", "sector": "Real Estate"},
    {"symbol": "KIM",   "name": "Kimco Realty Corporation",             "type": "Stock", "sector": "Real Estate"},
    {"symbol": "REG",   "name": "Regency Centers Corporation",          "type": "Stock", "sector": "Real Estate"},
    {"symbol": "HST",   "name": "Host Hotels & Resorts Inc.",           "type": "Stock", "sector": "Real Estate"},
    {"symbol": "SUI",   "name": "Sun Communities Inc.",                 "type": "Stock", "sector": "Real Estate"},
    {"symbol": "CPT",   "name": "Camden Property Trust",                "type": "Stock", "sector": "Real Estate"},
    {"symbol": "UDR",   "name": "UDR Inc.",                             "type": "Stock", "sector": "Real Estate"},

    # ── Additional Materials Stocks ───────────────────────────────────────────
    {"symbol": "NUE",   "name": "Nucor Corporation",                    "type": "Stock", "sector": "Materials"},
    {"symbol": "STLD",  "name": "Steel Dynamics Inc.",                  "type": "Stock", "sector": "Materials"},
    {"symbol": "CLF",   "name": "Cleveland-Cliffs Inc.",                "type": "Stock", "sector": "Materials"},
    {"symbol": "CF",    "name": "CF Industries Holdings Inc.",          "type": "Stock", "sector": "Materials"},
    {"symbol": "MOS",   "name": "The Mosaic Company",                   "type": "Stock", "sector": "Materials"},
    {"symbol": "ECL",   "name": "Ecolab Inc.",                          "type": "Stock", "sector": "Materials"},
    {"symbol": "PPG",   "name": "PPG Industries Inc.",                  "type": "Stock", "sector": "Materials"},
    {"symbol": "RPM",   "name": "RPM International Inc.",               "type": "Stock", "sector": "Materials"},
    {"symbol": "IFF",   "name": "International Flavors & Fragrances",   "type": "Stock", "sector": "Materials"},
    {"symbol": "CTVA",  "name": "Corteva Inc.",                         "type": "Stock", "sector": "Materials"},
    {"symbol": "FMC",   "name": "FMC Corporation",                      "type": "Stock", "sector": "Materials"},
    {"symbol": "LYB",   "name": "LyondellBasell Industries N.V.",       "type": "Stock", "sector": "Materials"},
    {"symbol": "CE",    "name": "Celanese Corporation",                 "type": "Stock", "sector": "Materials"},
    {"symbol": "EMN",   "name": "Eastman Chemical Company",             "type": "Stock", "sector": "Materials"},
    {"symbol": "HUN",   "name": "Huntsman Corporation",                 "type": "Stock", "sector": "Materials"},
    {"symbol": "PKG",   "name": "Packaging Corporation of America",     "type": "Stock", "sector": "Materials"},
    {"symbol": "IP",    "name": "International Paper Company",          "type": "Stock", "sector": "Materials"},
    {"symbol": "BALL",  "name": "Ball Corporation",                     "type": "Stock", "sector": "Materials"},
    {"symbol": "CCK",   "name": "Crown Holdings Inc.",                  "type": "Stock", "sector": "Materials"},

    # ── Additional Utilities Stocks ───────────────────────────────────────────
    {"symbol": "AWK",   "name": "American Water Works Company Inc.",    "type": "Stock", "sector": "Utilities"},
    {"symbol": "WEC",   "name": "WEC Energy Group Inc.",                "type": "Stock", "sector": "Utilities"},
    {"symbol": "ES",    "name": "Eversource Energy",                    "type": "Stock", "sector": "Utilities"},
    {"symbol": "ETR",   "name": "Entergy Corporation",                  "type": "Stock", "sector": "Utilities"},
    {"symbol": "PPL",   "name": "PPL Corporation",                      "type": "Stock", "sector": "Utilities"},
    {"symbol": "FE",    "name": "FirstEnergy Corp.",                    "type": "Stock", "sector": "Utilities"},
    {"symbol": "CMS",   "name": "CMS Energy Corporation",               "type": "Stock", "sector": "Utilities"},
    {"symbol": "SRE",   "name": "Sempra",                               "type": "Stock", "sector": "Utilities"},
    {"symbol": "PCG",   "name": "PG&E Corporation",                     "type": "Stock", "sector": "Utilities"},
    {"symbol": "EIX",   "name": "Edison International",                 "type": "Stock", "sector": "Utilities"},
    {"symbol": "NRG",   "name": "NRG Energy Inc.",                      "type": "Stock", "sector": "Utilities"},
    {"symbol": "VST",   "name": "Vistra Corp.",                         "type": "Stock", "sector": "Utilities"},
    {"symbol": "CEG",   "name": "Constellation Energy Corporation",     "type": "Stock", "sector": "Utilities"},
    {"symbol": "NI",    "name": "NiSource Inc.",                        "type": "Stock", "sector": "Utilities"},
    {"symbol": "LNT",   "name": "Alliant Energy Corporation",           "type": "Stock", "sector": "Utilities"},

    # ── Additional Communication Services Stocks ──────────────────────────────
    {"symbol": "SPOT",  "name": "Spotify Technology S.A.",              "type": "Stock", "sector": "Communication Services"},
    {"symbol": "MTCH",  "name": "Match Group Inc.",                     "type": "Stock", "sector": "Communication Services"},
    {"symbol": "PINS",  "name": "Pinterest Inc.",                       "type": "Stock", "sector": "Communication Services"},
    {"symbol": "RDDT",  "name": "Reddit Inc.",                          "type": "Stock", "sector": "Communication Services"},
    {"symbol": "IAC",   "name": "IAC Inc.",                             "type": "Stock", "sector": "Communication Services"},
    {"symbol": "TME",   "name": "Tencent Music Entertainment Group",    "type": "Stock", "sector": "Communication Services"},
    {"symbol": "NTES",  "name": "NetEase Inc.",                         "type": "Stock", "sector": "Communication Services"},
    {"symbol": "BILI",  "name": "Bilibili Inc.",                        "type": "Stock", "sector": "Communication Services"},
    {"symbol": "IQ",    "name": "iQIYI Inc.",                           "type": "Stock", "sector": "Communication Services"},

    # ── Additional International / ADR Stocks ────────────────────────────────
    {"symbol": "SE",    "name": "Sea Limited",                          "type": "Stock", "sector": "Technology"},
    {"symbol": "GRAB",  "name": "Grab Holdings Limited",               "type": "Stock", "sector": "Technology"},
    {"symbol": "MELI",  "name": "MercadoLibre Inc.",                    "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "NU",    "name": "Nu Holdings Ltd.",                     "type": "Stock", "sector": "Financials"},
    {"symbol": "STNE",  "name": "StoneCo Ltd.",                         "type": "Stock", "sector": "Financials"},
    {"symbol": "XPEV",  "name": "XPeng Inc.",                           "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "LI",    "name": "Li Auto Inc.",                         "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "HMC",   "name": "Honda Motor Co. Ltd.",                 "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "BIDU",  "name": "Baidu Inc.",                           "type": "Stock", "sector": "Technology"},
    {"symbol": "HSBC",  "name": "HSBC Holdings plc",                    "type": "Stock", "sector": "Financials"},
    {"symbol": "SAN",   "name": "Banco Santander S.A.",                 "type": "Stock", "sector": "Financials"},
    {"symbol": "BBVA",  "name": "Banco Bilbao Vizcaya Argentaria",      "type": "Stock", "sector": "Financials"},
    {"symbol": "ING",   "name": "ING Groep N.V.",                       "type": "Stock", "sector": "Financials"},
    {"symbol": "UBS",   "name": "UBS Group AG",                         "type": "Stock", "sector": "Financials"},
    {"symbol": "DB",    "name": "Deutsche Bank AG",                     "type": "Stock", "sector": "Financials"},
    {"symbol": "ABB",   "name": "ABB Ltd",                              "type": "Stock", "sector": "Industrials"},
    {"symbol": "BEKE",  "name": "KE Holdings Inc.",                     "type": "Stock", "sector": "Real Estate"},
    {"symbol": "EDU",   "name": "New Oriental Education & Technology",  "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "TAL",   "name": "TAL Education Group",                  "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "BN",    "name": "Brookfield Corporation",               "type": "Stock", "sector": "Financials"},

    # ── Crypto & Bitcoin-related Stocks ──────────────────────────────────────
    {"symbol": "MSTR",  "name": "MicroStrategy Incorporated",           "type": "Stock", "sector": "Crypto"},
    {"symbol": "MARA",  "name": "MARA Holdings Inc.",                   "type": "Stock", "sector": "Crypto"},
    {"symbol": "RIOT",  "name": "Riot Platforms Inc.",                  "type": "Stock", "sector": "Crypto"},
    {"symbol": "CLSK",  "name": "CleanSpark Inc.",                      "type": "Stock", "sector": "Crypto"},
    {"symbol": "HUT",   "name": "Hut 8 Corp.",                          "type": "Stock", "sector": "Crypto"},
    {"symbol": "BITF",  "name": "Bitfarms Ltd.",                        "type": "Stock", "sector": "Crypto"},
    {"symbol": "CIFR",  "name": "Cipher Mining Inc.",                   "type": "Stock", "sector": "Crypto"},
    {"symbol": "IREN",  "name": "Iris Energy Limited",                  "type": "Stock", "sector": "Crypto"},
    {"symbol": "CORZ",  "name": "Core Scientific Inc.",                 "type": "Stock", "sector": "Crypto"},
    {"symbol": "WULF",  "name": "TeraWulf Inc.",                        "type": "Stock", "sector": "Crypto"},

    # ── SPACs / Innovation / Misc ─────────────────────────────────────────────
    {"symbol": "RKT",   "name": "Rocket Companies Inc.",                "type": "Stock", "sector": "Financials"},
    {"symbol": "OPEN",  "name": "Opendoor Technologies Inc.",           "type": "Stock", "sector": "Real Estate"},
    {"symbol": "AFRM",  "name": "Affirm Holdings Inc.",                 "type": "Stock", "sector": "Financials"},
    {"symbol": "UPST",  "name": "Upstart Holdings Inc.",                "type": "Stock", "sector": "Financials"},
    {"symbol": "LMND",  "name": "Lemonade Inc.",                        "type": "Stock", "sector": "Financials"},
    {"symbol": "ROOT",  "name": "Root Inc.",                            "type": "Stock", "sector": "Financials"},
    {"symbol": "PTON",  "name": "Peloton Interactive Inc.",             "type": "Stock", "sector": "Consumer Discretionary"},
    {"symbol": "BYND",  "name": "Beyond Meat Inc.",                     "type": "Stock", "sector": "Consumer Staples"},
    {"symbol": "OATLY", "name": "Oatly Group AB",                       "type": "Stock", "sector": "Consumer Staples"},
    {"symbol": "DNA",   "name": "Ginkgo Bioworks Holdings Inc.",        "type": "Stock", "sector": "Healthcare"},
    {"symbol": "PACB",  "name": "Pacific Biosciences of California",    "type": "Stock", "sector": "Healthcare"},
    {"symbol": "ILMN",  "name": "Illumina Inc.",                        "type": "Stock", "sector": "Healthcare"},
    {"symbol": "VEEV",  "name": "Veeva Systems Inc.",                   "type": "Stock", "sector": "Healthcare"},
    {"symbol": "PODD",  "name": "Insulet Corporation",                  "type": "Stock", "sector": "Healthcare"},
    {"symbol": "SWAV",  "name": "ShockWave Medical Inc.",               "type": "Stock", "sector": "Healthcare"},
    {"symbol": "TMDX",  "name": "TransMedics Group Inc.",               "type": "Stock", "sector": "Healthcare"},
    {"symbol": "AXSM",  "name": "Axsome Therapeutics Inc.",             "type": "Stock", "sector": "Healthcare"},
    {"symbol": "INVA",  "name": "Innoviva Inc.",                        "type": "Stock", "sector": "Healthcare"},
]


def compute_technical_indicators(df):
    """Compute RSI, MACD, Bollinger Bands, EMA crossovers."""
    close = df['Close']

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['BB_Upper'] = sma20 + 2 * std20
    df['BB_Lower'] = sma20 - 2 * std20
    df['BB_Mid'] = sma20
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / sma20

    # EMA crossovers
    df['EMA_9'] = close.ewm(span=9, adjust=False).mean()
    df['EMA_21'] = close.ewm(span=21, adjust=False).mean()
    df['EMA_50'] = close.ewm(span=50, adjust=False).mean()
    df['EMA_200'] = close.ewm(span=200, adjust=False).mean()

    # Volume indicators
    df['Volume_SMA20'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA20'].replace(0, np.nan)

    # ATR
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - close.shift()).abs()
    low_close = (df['Low'] - close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    return df


def _shrink(correct: int, total: int, alpha: float = 8.0) -> float:
    """
    Bayesian shrinkage toward 50% (the random baseline).

    Formula: (correct + α) / (total + 2α)

    With small samples the estimate is pulled strongly toward 0.5.
    As total grows, the shrinkage effect diminishes.

    α = 8 examples:
      9/10  raw → 65%   (was 90%)
      5/10  raw → 50%   (stays at random)
     13/15  raw → 68%   (was 87%)
     10/15  raw → 58%
     20/20  raw → 77%   (cap near 80%)
    """
    if total == 0:
        return 0.5
    return round((correct + alpha) / (total + 2 * alpha), 4)


def _lr_predict_fast(train_vals, horizon_days):
    """Minimal LR predict used by backtester."""
    n = len(train_vals)
    X = np.arange(n, dtype=float).reshape(-1, 1)
    mn, mx = X.min(), X.max()
    Xs = (X - mn) / (mx - mn + 1e-9)
    lr = LinearRegression().fit(Xs, train_vals)
    fX = (np.arange(n, n + horizon_days, dtype=float).reshape(-1, 1) - mn) / (mx - mn + 1e-9)
    return float(lr.predict(fX)[-1])


def _ema_predict_fast(train_ser, horizon_days):
    """Minimal EMA momentum predict used by backtester."""
    es = float(train_ser.ewm(span=9,  adjust=False).mean().iloc[-1])
    el = float(train_ser.ewm(span=21, adjust=False).mean().iloc[-1])
    base = float(train_ser.iloc[-1])
    momentum = (es - el) / base
    current = base
    for i in range(horizon_days):
        current += momentum * base * 0.1 * np.exp(-i * 0.05)
    return current


def compute_backtest_accuracy(prices, horizon_days, n_windows=20, train_window=80):
    """
    Walk-forward backtesting over n_windows periods.
    Returns Bayesian-shrunk directional accuracy per model.
    Shrinkage pulls small-sample estimates toward 0.50 (random baseline).
    Also returns raw correct/total counts for transparency.
    """
    prices = prices.dropna()
    needed = train_window + n_windows + horizon_days
    if len(prices) < needed:
        return {k: {'acc': 0.5, 'correct': 0, 'total': 0}
                for k in ['linear_regression', 'arima', 'ema_crossover', 'holt_winters']}

    correct = {k: 0 for k in ['linear_regression', 'arima', 'ema_crossover', 'holt_winters']}
    total   = {k: 0 for k in ['linear_regression', 'arima', 'ema_crossover', 'holt_winters']}

    for i in range(n_windows):
        end_train = len(prices) - n_windows - horizon_days + i
        train  = prices.iloc[end_train - train_window:end_train]
        future = prices.iloc[end_train:end_train + horizon_days]
        if len(train) < 20 or len(future) < horizon_days:
            continue

        baseline   = float(train.iloc[-1])
        actual_up  = float(future.iloc[-1]) >= baseline

        # ── Linear Regression ────────────────────────────────────────────────
        try:
            pred = _lr_predict_fast(train.values, horizon_days)
            correct['linear_regression'] += int((pred >= baseline) == actual_up)
            total['linear_regression']   += 1
        except Exception:
            pass

        # ── ARIMA(1,1,0) ─────────────────────────────────────────────────────
        try:
            res  = ARIMA(train.values, order=(1, 1, 0)).fit()
            pred = float(res.forecast(horizon_days)[-1])
            correct['arima'] += int((pred >= baseline) == actual_up)
            total['arima']   += 1
        except Exception:
            pass

        # ── EMA Crossover ────────────────────────────────────────────────────
        try:
            pred = _ema_predict_fast(train, horizon_days)
            correct['ema_crossover'] += int((pred >= baseline) == actual_up)
            total['ema_crossover']   += 1
        except Exception:
            pass

        # ── Holt-Winters ─────────────────────────────────────────────────────
        try:
            hw   = ExponentialSmoothing(
                train.values, trend='add', initialization_method='estimated'
            ).fit(optimized=False, smoothing_level=0.2, smoothing_trend=0.1)
            pred = float(hw.forecast(horizon_days)[-1])
            correct['holt_winters'] += int((pred >= baseline) == actual_up)
            total['holt_winters']   += 1
        except Exception:
            pass

    return {
        k: {
            'acc':     _shrink(correct[k], total[k]),
            'correct': correct[k],
            'total':   total[k],
        }
        for k in correct
    }


def predict_linear_regression(prices, horizon_days):
    """Trend-based linear regression."""
    prices = prices.dropna()
    n = len(prices)
    X = np.arange(n).reshape(-1, 1)
    y = prices.values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(X_scaled, y)
    future_X_scaled = scaler.transform(np.arange(n, n + horizon_days).reshape(-1, 1))
    return model.predict(future_X_scaled)


def predict_arima(prices, horizon_days):
    """ARIMA(2,1,2) for mean-reverting statistical forecasting."""
    prices = prices.dropna()
    for order in [(2, 1, 2), (1, 1, 1), (1, 1, 0)]:
        try:
            result = ARIMA(prices.values, order=order).fit()
            forecast = result.forecast(steps=horizon_days)
            ci = result.get_forecast(steps=horizon_days).conf_int()
            if isinstance(ci, np.ndarray):
                ci_lower, ci_upper = ci[:, 0].tolist(), ci[:, 1].tolist()
            else:
                ci_lower, ci_upper = ci.iloc[:, 0].tolist(), ci.iloc[:, 1].tolist()
            return np.array(forecast), ci_lower, ci_upper
        except Exception:
            continue
    return None, [], []


def predict_ema_crossover(prices, horizon_days):
    """EMA-based momentum extrapolation."""
    prices = prices.dropna()
    ema_short = prices.ewm(span=9,  adjust=False).mean()
    ema_long  = prices.ewm(span=21, adjust=False).mean()
    last_price = float(prices.iloc[-1])
    momentum   = (float(ema_short.iloc[-1]) - float(ema_long.iloc[-1])) / last_price
    daily_drift = momentum * last_price * 0.1
    predictions, current = [], last_price
    for i in range(horizon_days):
        current += daily_drift * np.exp(-i * 0.05)
        predictions.append(current)
    signal = "BULLISH" if momentum > 0 else "BEARISH"
    return np.array(predictions), signal


def predict_holt_winters(prices, horizon_days):
    """Holt-Winters exponential smoothing."""
    prices = prices.dropna()
    try:
        result = ExponentialSmoothing(
            prices.values, trend='add', seasonal=None,
            initialization_method='estimated'
        ).fit(optimized=True)
        return np.array(result.forecast(horizon_days))
    except Exception:
        return None


def ensemble_predict(prices, horizon_days):
    """
    Run all 4 models, backtest their directional accuracy,
    use those real accuracies as weights, and return rich metadata.
    """
    # ── Backtest each model (out-of-sample directional accuracy) ─────────────
    bt_acc = compute_backtest_accuracy(prices, horizon_days)

    results = {}

    # 1. Linear Regression
    lr_pred = predict_linear_regression(prices, horizon_days)
    bt_lr   = bt_acc['linear_regression']
    results['linear_regression'] = {
        'predictions': lr_pred.tolist(),
        'confidence':  bt_lr['acc'],
        'bt_correct':  bt_lr['correct'],
        'bt_total':    bt_lr['total'],
        'label': 'Linear Regression',
    }

    # 2. ARIMA
    arima_pred, ci_lower, ci_upper = predict_arima(prices, horizon_days)
    bt_ar = bt_acc['arima']
    if arima_pred is not None:
        results['arima'] = {
            'predictions':    arima_pred.tolist(),
            'confidence':     bt_ar['acc'],
            'bt_correct':     bt_ar['correct'],
            'bt_total':       bt_ar['total'],
            'label':          'ARIMA',
            'conf_int_lower': ci_lower,
            'conf_int_upper': ci_upper,
        }

    # 3. EMA Crossover
    ema_pred, ema_signal = predict_ema_crossover(prices, horizon_days)
    bt_ema = bt_acc['ema_crossover']
    results['ema_crossover'] = {
        'predictions': ema_pred.tolist(),
        'confidence':  bt_ema['acc'],
        'bt_correct':  bt_ema['correct'],
        'bt_total':    bt_ema['total'],
        'label':       'EMA Crossover',
        'signal':      ema_signal,
    }

    # 4. Holt-Winters
    hw_pred = predict_holt_winters(prices, horizon_days)
    bt_hw   = bt_acc['holt_winters']
    if hw_pred is not None:
        results['holt_winters'] = {
            'predictions': hw_pred.tolist(),
            'confidence':  bt_hw['acc'],
            'bt_correct':  bt_hw['correct'],
            'bt_total':    bt_hw['total'],
            'label':       'Holt-Winters',
        }

    # ── Weighted ensemble (weights = shrunk backtest accuracy) ────────────────
    valid = [(v['predictions'], v['confidence']) for v in results.values() if v.get('predictions')]
    if valid:
        weights = np.array([c for _, c in valid])
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(valid)) / len(valid)
        preds_matrix  = np.array([p for p, _ in valid])
        ensemble_preds = np.average(preds_matrix, axis=0, weights=weights)

        current_price = float(prices.dropna().iloc[-1])
        directions    = [p[-1] >= current_price for p, _ in valid]
        agree_count   = max(sum(directions), len(directions) - sum(directions))
        agreement     = agree_count / len(directions)
        consensus     = 'UP' if sum(directions) >= len(directions) / 2 else 'DOWN'

        all_final = [p[-1] for p, _ in valid]
        pred_range = {
            'low':        round(float(min(all_final)), 2),
            'high':       round(float(max(all_final)), 2),
            'spread_pct': round((max(all_final) - min(all_final)) / current_price * 100, 2),
        }

        # Ensemble confidence: shrunk mean, small boost for model agreement
        base_conf     = float(np.mean([c for _, c in valid]))
        agree_boost   = (agreement - 0.5) * 0.08   # up to +4 pp when unanimous
        ensemble_conf = round(min(0.78, base_conf + agree_boost), 4)
        total_windows = sum(v.get('bt_total', 0) for v in results.values())

        results['ensemble'] = {
            'predictions':     ensemble_preds.tolist(),
            'confidence':      ensemble_conf,
            'label':           'Ensemble (Weighted)',
            'agreement':       round(agreement, 3),
            'consensus':       consensus,
            'pred_range':      pred_range,
            'backtest_windows': total_windows,
        }

    # Return flat accuracy dict for the /api/predict response
    bt_flat = {k: v['acc'] for k, v in bt_acc.items()}
    return results, bt_flat


def compute_stats(df, ticker_info):
    """Compute comprehensive statistics."""
    close = df['Close']
    returns = close.pct_change().dropna()

    # Basic stats
    current_price = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) > 1 else current_price
    change = current_price - prev_close
    change_pct = (change / prev_close) * 100

    # Volatility
    daily_vol = returns.std()
    annual_vol = daily_vol * np.sqrt(252) * 100

    # Returns
    ret_1w = ((close.iloc[-1] / close.iloc[-5]) - 1) * 100 if len(close) >= 5 else None
    ret_1m = ((close.iloc[-1] / close.iloc[-21]) - 1) * 100 if len(close) >= 21 else None
    ret_3m = ((close.iloc[-1] / close.iloc[-63]) - 1) * 100 if len(close) >= 63 else None
    ret_1y = ((close.iloc[-1] / close.iloc[-252]) - 1) * 100 if len(close) >= 252 else None

    # Risk metrics
    sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

    # Max drawdown
    rolling_max = close.cummax()
    drawdown = (close - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min()) * 100

    # 52-week high/low
    recent_year = close.tail(252)
    high_52w = float(recent_year.max())
    low_52w = float(recent_year.min())

    # Beta approximation (vs own historical returns)
    market_corr = None

    # Support & Resistance (recent pivots)
    recent = close.tail(50)
    support = float(recent.min())
    resistance = float(recent.max())

    # Trend strength
    ema50 = df['EMA_50'].iloc[-1] if 'EMA_50' in df.columns else close.rolling(50).mean().iloc[-1]
    ema200 = df['EMA_200'].iloc[-1] if 'EMA_200' in df.columns else close.rolling(200).mean().iloc[-1]
    golden_cross = bool(ema50 > ema200)

    return {
        'current_price': round(current_price, 2),
        'change': round(change, 2),
        'change_pct': round(change_pct, 2),
        'volume': int(df['Volume'].iloc[-1]),
        'avg_volume': int(df['Volume'].tail(20).mean()),
        'daily_volatility': round(daily_vol * 100, 3),
        'annual_volatility': round(annual_vol, 2),
        'sharpe_ratio': round(sharpe, 3),
        'max_drawdown': round(max_drawdown, 2),
        'high_52w': round(high_52w, 2),
        'low_52w': round(low_52w, 2),
        'ret_1w': round(ret_1w, 2) if ret_1w is not None else None,
        'ret_1m': round(ret_1m, 2) if ret_1m is not None else None,
        'ret_3m': round(ret_3m, 2) if ret_3m is not None else None,
        'ret_1y': round(ret_1y, 2) if ret_1y is not None else None,
        'support': round(support, 2),
        'resistance': round(resistance, 2),
        'rsi': round(float(df['RSI'].iloc[-1]), 2) if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]) else None,
        'macd': round(float(df['MACD'].iloc[-1]), 4) if 'MACD' in df.columns else None,
        'macd_signal': round(float(df['MACD_Signal'].iloc[-1]), 4) if 'MACD_Signal' in df.columns else None,
        'bb_upper': round(float(df['BB_Upper'].iloc[-1]), 2) if 'BB_Upper' in df.columns else None,
        'bb_lower': round(float(df['BB_Lower'].iloc[-1]), 2) if 'BB_Lower' in df.columns else None,
        'golden_cross': golden_cross,
        'ema_50': round(float(ema50), 2),
        'ema_200': round(float(ema200), 2),
        'market_cap': ticker_info.get('marketCap'),
        'pe_ratio': ticker_info.get('trailingPE'),
        'forward_pe': ticker_info.get('forwardPE'),
        'dividend_yield': ticker_info.get('dividendYield'),
        'beta': ticker_info.get('beta'),
    }


# ─── Rankings infrastructure ──────────────────────────────────────────────────

# 30 high-interest tickers screened on the home page
RANK_TICKERS = [
    'AAPL', 'MSFT', 'NVDA', 'META', 'GOOGL', 'AMZN', 'TSLA', 'AVGO',
    'AMD',  'PLTR', 'APP',  'CRWD', 'NET',   'DDOG', 'PANW', 'CRM',
    'NFLX', 'COIN', 'SMCI', 'ARM',
    'SPY',  'QQQ',  'SMH',  'ARKK',
    'JPM',  'GS',   'BAC',  'LLY',  'XOM',   'GLD',
]

_rankings_cache: dict = {}   # key = horizon_days → {data, ts}
RANK_CACHE_TTL = 900         # 15 minutes


def _fast_screen(closes: pd.Series, horizon: int):
    """
    Quickly predict direction + magnitude for a single ticker using
    LR trend extrapolation and EMA momentum — no ARIMA, very fast.
    Returns (pred_pct, signal_conf) where conf is a Bayesian-shrunk 12-window mini-backtest.
    """
    closes = closes.dropna()
    if len(closes) < 30:
        return None, None

    last = float(closes.iloc[-1])

    # ── LR predicted % change ────────────────────────────────────────────────
    lr_pred = _lr_predict_fast(closes.values, horizon)
    lr_pct  = (lr_pred - last) / last * 100

    # ── EMA momentum ─────────────────────────────────────────────────────────
    ema_pred = _ema_predict_fast(closes, horizon)
    ema_pct  = (ema_pred - last) / last * 100

    # Weighted average (LR trend carries more weight)
    agree   = (lr_pct > 0) == (ema_pct > 0)
    pred_pct = lr_pct * 0.6 + ema_pct * 0.4 if agree else lr_pct

    # ── 12-window mini-backtest (LR + EMA) with Bayesian shrinkage ──────────
    n_win, train_w = 12, 40
    needed = train_w + n_win + horizon
    correct = total = 0
    if len(closes) >= needed:
        for i in range(n_win):
            end    = len(closes) - n_win - horizon + i
            tr     = closes.iloc[end - train_w:end]
            actual_up = float(closes.iloc[end + horizon - 1]) >= float(tr.iloc[-1])
            base   = float(tr.iloc[-1])
            for pred_fn in (_lr_predict_fast, _ema_predict_fast):
                try:
                    arg = tr.values if pred_fn is _lr_predict_fast else tr
                    p   = pred_fn(arg, horizon)
                    correct += int((p >= base) == actual_up)
                    total   += 1
                except Exception:
                    pass
    conf = _shrink(correct, total)

    return round(pred_pct, 2), conf


# ─── API Routes ───────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/tickers')
def get_tickers():
    return jsonify(POPULAR_TICKERS)

@app.route('/api/search')
def search_ticker():
    q = request.args.get('q', '').upper()
    if not q:
        return jsonify([])
    results = [t for t in POPULAR_TICKERS if q in t['symbol'] or q in t['name'].upper()]
    # Also try live search
    try:
        ticker = yf.Ticker(q)
        info = ticker.fast_info
        if hasattr(info, 'last_price') and info.last_price:
            existing = any(t['symbol'] == q for t in results)
            if not existing:
                results.insert(0, {"symbol": q, "name": ticker.info.get('longName', q), "type": "Stock", "sector": "Unknown"})
    except Exception:
        pass
    return jsonify(results[:20])

@app.route('/api/rankings')
def get_rankings():
    horizon_map = {'day': 1, 'week': 5, 'month': 21}
    horizon_key = request.args.get('horizon', 'week')
    horizon     = horizon_map.get(horizon_key, 5)
    force       = request.args.get('refresh', 'false') == 'true'

    cached = _rankings_cache.get(horizon_key)
    if not force and cached and (time.time() - cached['ts']) < RANK_CACHE_TTL:
        return jsonify({**cached['data'], 'from_cache': True})

    try:
        # Batch download — much faster than individual calls
        raw = yf.download(
            RANK_TICKERS, period='3mo', interval='1d',
            auto_adjust=True, progress=False, group_by='ticker'
        )

        results = []
        for sym in RANK_TICKERS:
            try:
                # Extract close series for this symbol
                if isinstance(raw.columns, pd.MultiIndex):
                    closes = raw[sym]['Close'].dropna()
                else:
                    closes = raw['Close'].dropna()

                pred_pct, conf = _fast_screen(closes, horizon)
                if pred_pct is None:
                    continue

                last_price = float(closes.iloc[-1])

                # RSI for context
                delta = closes.diff()
                gain  = delta.clip(lower=0).rolling(14).mean()
                loss  = (-delta.clip(upper=0)).rolling(14).mean()
                rs    = gain.iloc[-1] / loss.iloc[-1] if float(loss.iloc[-1]) != 0 else 0
                rsi   = round(100 - (100 / (1 + rs)), 1)

                # 5-day % change (momentum context)
                mom_5d = round(((last_price / float(closes.iloc[-5])) - 1) * 100, 2) if len(closes) >= 5 else None

                meta = next((t for t in POPULAR_TICKERS if t['symbol'] == sym),
                            {'name': sym, 'sector': 'Unknown', 'type': 'Stock'})

                results.append({
                    'symbol':    sym,
                    'name':      meta['name'],
                    'sector':    meta['sector'],
                    'type':      meta['type'],
                    'price':     round(last_price, 2),
                    'pred_pct':  pred_pct,
                    'direction': 'UP' if pred_pct >= 0 else 'DOWN',
                    'confidence': conf,
                    'rsi':       float(rsi),
                    'mom_5d':    mom_5d,
                })
            except Exception:
                continue

        # Sort: gainers first (highest pred_pct at top)
        results.sort(key=lambda x: x['pred_pct'], reverse=True)
        # Add rank
        for i, r in enumerate(results):
            r['rank'] = i + 1

        data = {
            'rankings':     results,
            'horizon':      horizon_key,
            'horizon_days': horizon,
            'generated_at': datetime.now().isoformat(),
            'from_cache':   False,
        }
        _rankings_cache[horizon_key] = {'data': data, 'ts': time.time()}
        return jsonify(data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict')
def predict():
    symbol = request.args.get('symbol', '').upper()
    horizon = request.args.get('horizon', 'week')  # day | week | month

    if not symbol:
        return jsonify({'error': 'Symbol required'}), 400

    horizon_map = {'day': 1, 'week': 5, 'month': 21}
    horizon_days = horizon_map.get(horizon, 5)

    try:
        # Fetch data — use 2 years for good model training
        ticker = yf.Ticker(symbol)
        df = ticker.history(period='2y', interval='1d')

        if df.empty or len(df) < 30:
            return jsonify({'error': f'Insufficient data for {symbol}'}), 404

        # Flatten column names if MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = compute_technical_indicators(df)

        # Get ticker info
        try:
            info = ticker.info
        except Exception:
            info = {}

        stats = compute_stats(df, info)

        # Run predictions on closing prices
        close = df['Close']
        predictions, bt_acc = ensemble_predict(close, horizon_days)

        # Build forecast dates (trading days only)
        last_date = df.index[-1]
        forecast_dates = []
        d = last_date
        count = 0
        while count < horizon_days:
            d = d + timedelta(days=1)
            if d.weekday() < 5:  # Mon–Fri
                forecast_dates.append(d.strftime('%Y-%m-%d'))
                count += 1

        # Historical chart data (last 90 days)
        hist_df = df.tail(90).copy()
        first_close = float(hist_df['Close'].iloc[0])
        historical = {
            'dates': [d.strftime('%Y-%m-%d') for d in hist_df.index],
            'close': [round(float(v), 2) for v in hist_df['Close']],
            'open': [round(float(v), 2) for v in hist_df['Open']],
            'high': [round(float(v), 2) for v in hist_df['High']],
            'low': [round(float(v), 2) for v in hist_df['Low']],
            'volume': [int(v) for v in hist_df['Volume']],
            # Daily % change (close-to-close)
            'daily_pct': [
                round(float(v) * 100, 3) if not pd.isna(v) else None
                for v in hist_df['Close'].pct_change()
            ],
            # Cumulative % change from first day of the 90-day window
            'cum_pct': [
                round(((float(v) - first_close) / first_close) * 100, 3) if not pd.isna(v) else None
                for v in hist_df['Close']
            ],
            'bb_upper': [round(float(v), 2) if not pd.isna(v) else None for v in hist_df['BB_Upper']],
            'bb_lower': [round(float(v), 2) if not pd.isna(v) else None for v in hist_df['BB_Lower']],
            'bb_mid': [round(float(v), 2) if not pd.isna(v) else None for v in hist_df['BB_Mid']],
            'ema_9': [round(float(v), 2) if not pd.isna(v) else None for v in hist_df['EMA_9']],
            'ema_21': [round(float(v), 2) if not pd.isna(v) else None for v in hist_df['EMA_21']],
            'ema_50': [round(float(v), 2) if not pd.isna(v) else None for v in hist_df['EMA_50']],
            'rsi': [round(float(v), 2) if not pd.isna(v) else None for v in hist_df['RSI']],
            'macd': [round(float(v), 4) if not pd.isna(v) else None for v in hist_df['MACD']],
            'macd_signal': [round(float(v), 4) if not pd.isna(v) else None for v in hist_df['MACD_Signal']],
            'macd_hist': [round(float(v), 4) if not pd.isna(v) else None for v in hist_df['MACD_Hist']],
        }

        return jsonify({
            'symbol': symbol,
            'name': info.get('longName', symbol),
            'horizon': horizon,
            'horizon_days': horizon_days,
            'forecast_dates': forecast_dates,
            'predictions': predictions,
            'backtest': bt_acc,
            'stats': stats,
            'historical': historical,
            'generated_at': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/portfolio')
def get_portfolio():
    """Return current portfolio state enriched with live mark-to-market prices.
    Pulls the latest portfolio.json from GitHub first so trades made by the
    scheduled GitHub Actions job are visible without a git pull."""
    p = _fetch_portfolio_from_github() or load_portfolio()

    # Fetch 3-month history for open positions (needed for current-signal screen)
    price_map  = {}
    closes_map = {}   # sym → full close series for _fast_screen
    if p['positions']:
        syms = list(p['positions'].keys())
        try:
            raw = yf.download(syms, period='3mo', interval='1d',
                              auto_adjust=True, progress=False, group_by='ticker')
            for sym in syms:
                try:
                    closes = (raw[sym]['Close'] if isinstance(raw.columns, pd.MultiIndex)
                              else raw['Close']).dropna()
                    if not closes.empty:
                        price_map[sym]  = round(float(closes.iloc[-1]), 2)
                        closes_map[sym] = closes
                except Exception:
                    pass
        except Exception:
            pass

    # Build enriched positions list with live signal check
    positions_list = []
    for sym, pos in p['positions'].items():
        current = price_map.get(sym, pos['entry_price'])
        cost    = pos['shares'] * pos['entry_price']
        value   = pos['shares'] * current
        pnl     = value - cost
        pnl_pct = (pnl / cost * 100) if cost else 0

        # Re-run fast screen to get the signal as it stands RIGHT NOW
        cur_pred_pct, cur_conf = None, None
        if sym in closes_map:
            try:
                cur_pred_pct, cur_conf = _fast_screen(closes_map[sym], horizon=5)
            except Exception:
                pass

        # A signal is "stale" when the model has flipped to bearish (pred_pct < 0)
        # or confidence has dropped below our entry threshold
        signal_stale = (cur_pred_pct is not None and cur_pred_pct < 0)

        positions_list.append({
            'symbol':           sym,
            'shares':           pos['shares'],
            'entry_price':      pos['entry_price'],
            'entry_date':       pos['entry_date'],
            'entry_conf':       pos['entry_conf'],
            'entry_pred_pct':   pos['entry_pred_pct'],
            'current_price':    round(current, 2),
            'cost_basis':       round(cost, 2),
            'current_value':    round(value, 2),
            'pnl':              round(pnl, 2),
            'pnl_pct':          round(pnl_pct, 2),
            'current_pred_pct': round(cur_pred_pct, 2) if cur_pred_pct is not None else None,
            'current_conf':     round(cur_conf, 4)     if cur_conf     is not None else None,
            'signal_stale':     signal_stale,
        })

    total_val    = portfolio_value(p, price_map)
    positions_val = total_val - p['cash']
    total_pnl    = total_val - p['initial_balance']
    total_pnl_pct = total_pnl / p['initial_balance'] * 100

    return jsonify({
        'initial_balance': p['initial_balance'],
        'cash':            round(p['cash'], 2),
        'positions_value': round(positions_val, 2),
        'total_value':     round(total_val, 2),
        'total_pnl':       round(total_pnl, 2),
        'total_pnl_pct':   round(total_pnl_pct, 2),
        'positions':       positions_list,
        'trades':          list(reversed(p['trades']))[:50],   # last 50, newest first
        'daily_values':    p['daily_values'],
        'last_updated':    p.get('last_updated'),
        'created':         p.get('created'),
    })


@app.route('/api/portfolio/rebalance', methods=['POST'])
def trigger_rebalance():
    """Manually trigger a rebalance (ignores today-already-ran guard)."""
    horizon = request.json.get('horizon', 'week') if request.is_json else 'week'
    p = load_portfolio()
    # Allow re-run by clearing last_updated
    p['last_updated'] = None
    p, summary = rebalance(p, horizon)
    save_portfolio(p)
    return jsonify({'ok': True, 'summary': summary})


if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    app.run(debug=True, port=5000)
