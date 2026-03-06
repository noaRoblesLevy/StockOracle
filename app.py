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

    # Fetch live prices for open positions
    price_map = {}
    if p['positions']:
        syms = list(p['positions'].keys())
        try:
            raw = yf.download(syms, period='5d', interval='1d',
                              auto_adjust=True, progress=False, group_by='ticker')
            for sym in syms:
                try:
                    closes = (raw[sym]['Close'] if isinstance(raw.columns, pd.MultiIndex)
                              else raw['Close']).dropna()
                    if not closes.empty:
                        price_map[sym] = round(float(closes.iloc[-1]), 2)
                except Exception:
                    pass
        except Exception:
            pass

    # Build enriched positions list
    positions_list = []
    for sym, pos in p['positions'].items():
        current = price_map.get(sym, pos['entry_price'])
        cost    = pos['shares'] * pos['entry_price']
        value   = pos['shares'] * current
        pnl     = value - cost
        pnl_pct = (pnl / cost * 100) if cost else 0
        positions_list.append({
            'symbol':          sym,
            'shares':          pos['shares'],
            'entry_price':     pos['entry_price'],
            'entry_date':      pos['entry_date'],
            'entry_conf':      pos['entry_conf'],
            'entry_pred_pct':  pos['entry_pred_pct'],
            'current_price':   round(current, 2),
            'cost_basis':      round(cost, 2),
            'current_value':   round(value, 2),
            'pnl':             round(pnl, 2),
            'pnl_pct':         round(pnl_pct, 2),
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
