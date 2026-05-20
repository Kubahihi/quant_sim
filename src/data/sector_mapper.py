"""
Sector and Industry Classification Mapper

Provides robust sector/industry classification for universe stocks through:
1. GICS sector mappings from multiple index sources
2. Fallback mappings based on known ticker-sector relationships
3. Static curated mappings for major US equities

This module is designed to fill sector gaps that the primary data sources
(NASDAQ Trader, SEC) don't provide.
"""

from __future__ import annotations

from io import StringIO
from typing import Callable

import pandas as pd

from src.data.universe_sources import (
    PROXY_ENV_KEYS,
    USER_AGENT,
    _temporary_proxy_bypass,
    normalize_symbol,
)


# GICS Sector canonical names
GICS_SECTORS = [
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Real Estate",
    "Utilities",
]

# Known sector mappings for major tickers (fallback for when live sources fail)
# This covers major S&P 500, NASDAQ 100, and other liquid stocks
KNOWN_SECTOR_MAP: dict[str, tuple[str, str]] = {
    # Technology
    "AAPL": ("Information Technology", "Technology Hardware, Storage & Peripherals"),
    "MSFT": ("Information Technology", "Systems Software"),
    "NVDA": ("Information Technology", "Semiconductors"),
    "GOOGL": ("Communication Services", "Interactive Media & Services"),
    "GOOG": ("Communication Services", "Interactive Media & Services"),
    "META": ("Communication Services", "Interactive Media & Services"),
    "AMZN": ("Consumer Discretionary", "Broadline Retail"),
    "TSLA": ("Consumer Discretionary", "Automobile Manufacturers"),
    "AMD": ("Information Technology", "Semiconductors"),
    "INTC": ("Information Technology", "Semiconductors"),
    "CRM": ("Information Technology", "Application Software"),
    "ORCL": ("Information Technology", "Systems Software"),
    "ADBE": ("Information Technology", "Application Software"),
    "CSCO": ("Information Technology", "Communications Equipment"),
    "AVGO": ("Information Technology", "Semiconductors"),
    "QCOM": ("Information Technology", "Semiconductors"),
    "TXN": ("Information Technology", "Semiconductors"),
    "NOW": ("Information Technology", "Application Software"),
    "IBM": ("Information Technology", "IT Consulting & Other Services"),
    "AMAT": ("Information Technology", "Semiconductor Materials & Equipment"),
    "LRCX": ("Information Technology", "Semiconductor Materials & Equipment"),
    "KLAC": ("Information Technology", "Semiconductor Materials & Equipment"),
    "MU": ("Information Technology", "Semiconductors"),
    "MCHP": ("Information Technology", "Semiconductors"),
    "ADI": ("Information Technology", "Semiconductors"),
    "SNPS": ("Information Technology", "Application Software"),
    "CDNS": ("Information Technology", "Application Software"),
    "PANW": ("Information Technology", "Systems Software"),
    "CRWD": ("Information Technology", "Systems Software"),
    "FTNT": ("Information Technology", "Systems Software"),
    "ZS": ("Information Technology", "Systems Software"),
    "NET": ("Information Technology", "Application Software"),
    "DDOG": ("Information Technology", "Application Software"),
    "SNOW": ("Information Technology", "Application Software"),
    "PLTR": ("Information Technology", "IT Consulting & Other Services"),
    "SHOP": ("Information Technology", "Application Software"),
    "SQ": ("Information Technology", "Data Processing & Outsourced Services"),
    "PYPL": ("Information Technology", "Data Processing & Outsourced Services"),
    "NFLX": ("Communication Services", "Movies & Entertainment"),
    "DIS": ("Communication Services", "Movies & Entertainment"),
    "CMCSA": ("Communication Services", "Cable & Satellite"),
    "T": ("Communication Services", "Integrated Telecommunication Services"),
    "VZ": ("Communication Services", "Integrated Telecommunication Services"),
    "TMUS": ("Communication Services", "Wireless Telecommunication Services"),
    "CHTR": ("Communication Services", "Cable & Satellite"),
    "EA": ("Communication Services", "Interactive Home Entertainment"),
    "TTWO": ("Communication Services", "Interactive Home Entertainment"),
    "ATVI": ("Communication Services", "Interactive Home Entertainment"),
    "UMG": ("Communication Services", "Movies & Entertainment"),
    # Health Care
    "JNJ": ("Health Care", "Pharmaceuticals"),
    "UNH": ("Health Care", "Managed Health Care"),
    "LLY": ("Health Care", "Pharmaceuticals"),
    "PFE": ("Health Care", "Pharmaceuticals"),
    "ABBV": ("Health Care", "Biotechnology"),
    "MRK": ("Health Care", "Pharmaceuticals"),
    "TMO": ("Health Care", "Life Sciences Tools & Services"),
    "ABT": ("Health Care", "Health Care Equipment"),
    "DHR": ("Health Care", "Life Sciences Tools & Services"),
    "BMY": ("Health Care", "Pharmaceuticals"),
    "AMGN": ("Health Care", "Biotechnology"),
    "GILD": ("Health Care", "Biotechnology"),
    "ISRG": ("Health Care", "Health Care Equipment"),
    "VRTX": ("Health Care", "Biotechnology"),
    "REGN": ("Health Care", "Biotechnology"),
    "ZTS": ("Health Care", "Pharmaceuticals"),
    "SYK": ("Health Care", "Health Care Equipment"),
    "BSX": ("Health Care", "Health Care Equipment"),
    "MDT": ("Health Care", "Health Care Equipment"),
    "CI": ("Health Care", "Managed Health Care"),
    "CVS": ("Health Care", "Health Care Distributors"),
    "ELV": ("Health Care", "Managed Health Care"),
    "HUM": ("Health Care", "Managed Health Care"),
    "MCK": ("Health Care", "Health Care Distributors"),
    "CAH": ("Health Care", "Health Care Distributors"),
    "BIIB": ("Health Care", "Biotechnology"),
    "MRNA": ("Health Care", "Biotechnology"),
    "ILMN": ("Health Care", "Life Sciences Tools & Services"),
    "DXCM": ("Health Care", "Health Care Equipment"),
    "ALGN": ("Health Care", "Health Care Equipment"),
    "HOLX": ("Health Care", "Health Care Equipment"),
    "WAT": ("Health Care", "Life Sciences Tools & Services"),
    "IQV": ("Health Care", "Life Sciences Tools & Services"),
    "A": ("Health Care", "Health Care Equipment"),
    "RMD": ("Health Care", "Health Care Equipment"),
    "IDXX": ("Health Care", "Health Care Equipment"),
    "MTD": ("Health Care", "Life Sciences Tools & Services"),
    # Financials
    "BRK.B": ("Financials", "Multi-Sector Holdings"),
    "BRK.A": ("Financials", "Multi-Sector Holdings"),
    "JPM": ("Financials", "Diversified Banks"),
    "V": ("Financials", "Transaction & Payment Processing Services"),
    "MA": ("Financials", "Transaction & Payment Processing Services"),
    "BAC": ("Financials", "Diversified Banks"),
    "WFC": ("Financials", "Diversified Banks"),
    "GS": ("Financials", "Investment Banking & Brokerage"),
    "MS": ("Financials", "Investment Banking & Brokerage"),
    "C": ("Financials", "Diversified Banks"),
    "AXP": ("Financials", "Consumer Finance"),
    "BLK": ("Financials", "Asset Management & Custody Banks"),
    "SCHW": ("Financials", "Investment Banking & Brokerage"),
    "CB": ("Financials", "Property & Casualty Insurance"),
    "PGR": ("Financials", "Property & Casualty Insurance"),
    "ALL": ("Financials", "Property & Casualty Insurance"),
    "TRV": ("Financials", "Property & Casualty Insurance"),
    "MET": ("Financials", "Life & Health Insurance"),
    "PRU": ("Financials", "Life & Health Insurance"),
    "AIG": ("Financials", "Multi-line Insurance"),
    "AFL": ("Financials", "Life & Health Insurance"),
    "USB": ("Financials", "Diversified Banks"),
    "PNC": ("Financials", "Diversified Banks"),
    "TFC": ("Financials", "Diversified Banks"),
    "COF": ("Financials", "Consumer Finance"),
    "DFS": ("Financials", "Consumer Finance"),
    "SYF": ("Financials", "Consumer Finance"),
    "BK": ("Financials", "Asset Management & Custody Banks"),
    "STT": ("Financials", "Asset Management & Custody Banks"),
    "NTRS": ("Financials", "Asset Management & Custody Banks"),
    "FITB": ("Financials", "Diversified Banks"),
    "HBAN": ("Financials", "Diversified Banks"),
    "RF": ("Financials", "Diversified Banks"),
    "KEY": ("Financials", "Diversified Banks"),
    "CMA": ("Financials", "Diversified Banks"),
    "ZION": ("Financials", "Diversified Banks"),
    "MTB": ("Financials", "Regional Banks"),
    "SIVB": ("Financials", "Regional Banks"),
    "CBOE": ("Financials", "Financial Exchanges & Data"),
    "ICE": ("Financials", "Financial Exchanges & Data"),
    "CME": ("Financials", "Financial Exchanges & Data"),
    "NDAQ": ("Financials", "Financial Exchanges & Data"),
    "MSCI": ("Financials", "Financial Exchanges & Data"),
    "SPGI": ("Financials", "Financial Exchanges & Data"),
    "MCO": ("Financials", "Financial Exchanges & Data"),
    "BLK": ("Financials", "Asset Management & Custody Banks"),
    "TROW": ("Financials", "Asset Management & Custody Banks"),
    "BEN": ("Financials", "Asset Management & Custody Banks"),
    "IVZ": ("Financials", "Asset Management & Custody Banks"),
    "AMG": ("Financials", "Asset Management & Custody Banks"),
    "NUE": ("Financials", "Asset Management & Custody Banks"),
    # Consumer Discretionary
    "HD": ("Consumer Discretionary", "Home Improvement Retail"),
    "MCD": ("Consumer Discretionary", "Restaurants"),
    "NKE": ("Consumer Discretionary", "Footwear"),
    "SBUX": ("Consumer Discretionary", "Restaurants"),
    "LOW": ("Consumer Discretionary", "Home Improvement Retail"),
    "TGT": ("Consumer Discretionary", "General Merchandise Stores"),
    "COST": ("Consumer Staples", "Hypermarkets & Super Centers"),
    "WMT": ("Consumer Staples", "Hypermarkets & Super Centers"),
    "DG": ("Consumer Discretionary", "General Merchandise Stores"),
    "DLTR": ("Consumer Discretionary", "General Merchandise Stores"),
    "F": ("Consumer Discretionary", "Automobile Manufacturers"),
    "GM": ("Consumer Discretionary", "Automobile Manufacturers"),
    "RIVN": ("Consumer Discretionary", "Automobile Manufacturers"),
    "LCID": ("Consumer Discretionary", "Automobile Manufacturers"),
    "NIO": ("Consumer Discretionary", "Automobile Manufacturers"),
    "XPEV": ("Consumer Discretionary", "Automobile Manufacturers"),
    "LI": ("Consumer Discretionary", "Automobile Manufacturers"),
    "ABNB": ("Consumer Discretionary", "Hotels, Resorts & Cruise Lines"),
    "BKNG": ("Consumer Discretionary", "Hotels, Resorts & Cruise Lines"),
    "EXPE": ("Consumer Discretionary", "Hotels, Resorts & Cruise Lines"),
    "MAR": ("Consumer Discretionary", "Hotels, Resorts & Cruise Lines"),
    "HLT": ("Consumer Discretionary", "Hotels, Resorts & Cruise Lines"),
    "H": ("Consumer Discretionary", "Hotels, Resorts & Cruise Lines"),
    "MGM": ("Consumer Discretionary", "Casinos & Gaming"),
    "LVS": ("Consumer Discretionary", "Casinos & Gaming"),
    "WYNN": ("Consumer Discretionary", "Casinos & Gaming"),
    "CCL": ("Consumer Discretionary", "Cruise Lines"),
    "RCL": ("Consumer Discretionary", "Cruise Lines"),
    "NCLH": ("Consumer Discretionary", "Cruise Lines"),
    "YUM": ("Consumer Discretionary", "Restaurants"),
    "CMG": ("Consumer Discretionary", "Restaurants"),
    "DPZ": ("Consumer Discretionary", "Restaurants"),
    "QSR": ("Consumer Discretionary", "Restaurants"),
    "WEN": ("Consumer Discretionary", "Restaurants"),
    "SHAK": ("Consumer Discretionary", "Restaurants"),
    "LULU": ("Consumer Discretionary", "Apparel Retail"),
    "GPS": ("Consumer Discretionary", "Apparel Retail"),
    "ANF": ("Consumer Discretionary", "Apparel Retail"),
    "AEO": ("Consumer Discretionary", "Apparel Retail"),
    "URBN": ("Consumer Discretionary", "Apparel Retail"),
    "TJX": ("Consumer Discretionary", "Apparel Retail"),
    "ROST": ("Consumer Discretionary", "Apparel Retail"),
    "BBY": ("Consumer Discretionary", "Computer & Electronics Retail"),
    "ULTA": ("Consumer Discretionary", "Other Specialty Retail"),
    "ORLY": ("Consumer Discretionary", "Automotive Retail"),
    "AZO": ("Consumer Discretionary", "Automotive Retail"),
    "AAP": ("Consumer Discretionary", "Automotive Retail"),
    "GPC": ("Consumer Discretionary", "Distributors"),
    "APTV": ("Consumer Discretionary", "Auto Parts & Equipment"),
    "BWA": ("Consumer Discretionary", "Auto Parts & Equipment"),
    "LEA": ("Consumer Discretionary", "Auto Parts & Equipment"),
    "HAS": ("Consumer Discretionary", "Leisure Products"),
    "MAT": ("Consumer Discretionary", "Leisure Products"),
    "POOL": ("Consumer Discretionary", "Leisure Products"),
    "WHR": ("Consumer Discretionary", "Household Appliances"),
    "NVR": ("Consumer Discretionary", "Homebuilding"),
    "DHI": ("Consumer Discretionary", "Homebuilding"),
    "LEN": ("Consumer Discretionary", "Homebuilding"),
    "PHM": ("Consumer Discretionary", "Homebuilding"),
    "TOL": ("Consumer Discretionary", "Homebuilding"),
    "KBH": ("Consumer Discretionary", "Homebuilding"),
    "MTH": ("Consumer Discretionary", "Homebuilding"),
    "MHO": ("Consumer Discretionary", "Homebuilding"),
    "LAD": ("Consumer Discretionary", "Automotive Retail"),
    "AN": ("Consumer Discretionary", "Automotive Retail"),
    "PAG": ("Consumer Discretionary", "Automotive Retail"),
    "SAH": ("Consumer Discretionary", "Automotive Retail"),
    "ABG": ("Consumer Discretionary", "Automotive Retail"),
    # Consumer Staples
    "PG": ("Consumer Staples", "Personal Care Products"),
    "KO": ("Consumer Staples", "Soft Drinks & Non-alcoholic Beverages"),
    "PEP": ("Consumer Staples", "Soft Drinks & Non-alcoholic Beverages"),
    "COST": ("Consumer Staples", "Hypermarkets & Super Centers"),
    "WMT": ("Consumer Staples", "Hypermarkets & Super Centers"),
    "PM": ("Consumer Staples", "Tobacco"),
    "MO": ("Consumer Staples", "Tobacco"),
    "BTI": ("Consumer Staples", "Tobacco"),
    "MDLZ": ("Consumer Staples", "Packaged Foods & Meats"),
    "CL": ("Consumer Staples", "Personal Care Products"),
    "KMB": ("Consumer Staples", "Personal Care Products"),
    "EL": ("Consumer Staples", "Personal Care Products"),
    "KHC": ("Consumer Staples", "Packaged Foods & Meats"),
    "GIS": ("Consumer Staples", "Packaged Foods & Meats"),
    "K": ("Consumer Staples", "Packaged Foods & Meats"),
    "CAG": ("Consumer Staples", "Packaged Foods & Meats"),
    "SJM": ("Consumer Staples", "Packaged Foods & Meats"),
    "CPB": ("Consumer Staples", "Packaged Foods & Meats"),
    "HRL": ("Consumer Staples", "Packaged Foods & Meats"),
    "TSN": ("Consumer Staples", "Packaged Foods & Meats"),
    "TAP": ("Consumer Staples", "Distillers & Vintners"),
    "STZ": ("Consumer Staples", "Distillers & Vintners"),
    "BUD": ("Consumer Staples", "Brewers"),
    "SAM": ("Consumer Staples", "Brewers"),
    "MNST": ("Consumer Staples", "Soft Drinks & Non-alcoholic Beverages"),
    "KDP": ("Consumer Staples", "Soft Drinks & Non-alcoholic Beverages"),
    "CELH": ("Consumer Staples", "Soft Drinks & Non-alcoholic Beverages"),
    "FIZZ": ("Consumer Staples", "Soft Drinks & Non-alcoholic Beverages"),
    "COKE": ("Consumer Staples", "Soft Drinks & Non-alcoholic Beverages"),
    "WBA": ("Consumer Staples", "Drug Retail"),
    "CVS": ("Consumer Staples", "Drug Retail"),
    "RAD": ("Consumer Staples", "Drug Retail"),
    "KR": ("Consumer Staples", "Hypermarkets & Super Centers"),
    "SYY": ("Consumer Staples", "Food Distributors"),
    "USFD": ("Consumer Staples", "Food Distributors"),
    "PFGC": ("Consumer Staples", "Food Distributors"),
    "UNFI": ("Consumer Staples", "Food Distributors"),
    # Energy
    "XOM": ("Energy", "Integrated Oil & Gas"),
    "CVX": ("Energy", "Integrated Oil & Gas"),
    "COP": ("Energy", "Oil & Gas Exploration & Production"),
    "SLB": ("Energy", "Oil & Gas Equipment & Services"),
    "EOG": ("Energy", "Oil & Gas Exploration & Production"),
    "MPC": ("Energy", "Oil & Gas Refining & Marketing"),
    "PSX": ("Energy", "Oil & Gas Refining & Marketing"),
    "VLO": ("Energy", "Oil & Gas Refining & Marketing"),
    "PXD": ("Energy", "Oil & Gas Exploration & Production"),
    "DVN": ("Energy", "Oil & Gas Exploration & Production"),
    "HES": ("Energy", "Oil & Gas Exploration & Production"),
    "HAL": ("Energy", "Oil & Gas Equipment & Services"),
    "BKR": ("Energy", "Oil & Gas Equipment & Services"),
    "OXY": ("Energy", "Oil & Gas Exploration & Production"),
    "FANG": ("Energy", "Oil & Gas Exploration & Production"),
    "WMB": ("Energy", "Oil & Gas Storage & Transportation"),
    "KMI": ("Energy", "Oil & Gas Storage & Transportation"),
    "TRGP": ("Energy", "Oil & Gas Storage & Transportation"),
    "ET": ("Energy", "Oil & Gas Storage & Transportation"),
    "EPD": ("Energy", "Oil & Gas Storage & Transportation"),
    "MRO": ("Energy", "Oil & Gas Exploration & Production"),
    "APA": ("Energy", "Oil & Gas Exploration & Production"),
    "CTRA": ("Energy", "Oil & Gas Exploration & Production"),
    "EQT": ("Energy", "Oil & Gas Exploration & Production"),
    "AR": ("Energy", "Oil & Gas Exploration & Production"),
    "RRC": ("Energy", "Oil & Gas Exploration & Production"),
    "SWN": ("Energy", "Oil & Gas Exploration & Production"),
    "CHK": ("Energy", "Oil & Gas Exploration & Production"),
    "NOV": ("Energy", "Oil & Gas Equipment & Services"),
    "HP": ("Energy", "Oil & Gas Equipment & Services"),
    "RIG": ("Energy", "Oil & Gas Equipment & Services"),
    "VAL": ("Energy", "Oil & Gas Equipment & Services"),
    "CHX": ("Energy", "Oil & Gas Exploration & Production"),
    "OVV": ("Energy", "Oil & Gas Exploration & Production"),
    "SM": ("Energy", "Oil & Gas Exploration & Production"),
    # Industrials
    "CAT": ("Industrials", "Construction Machinery & Heavy Transportation"),
    "DE": ("Industrials", "Agricultural & Farm Machinery"),
    "UNP": ("Industrials", "Railroads"),
    "UPS": ("Industrials", "Air Freight & Logistics"),
    "FDX": ("Industrials", "Air Freight & Logistics"),
    "HON": ("Industrials", "Industrial Conglomerates"),
    "BA": ("Industrials", "Aerospace & Defense"),
    "LMT": ("Industrials", "Aerospace & Defense"),
    "RTX": ("Industrials", "Aerospace & Defense"),
    "GD": ("Industrials", "Aerospace & Defense"),
    "NOC": ("Industrials", "Aerospace & Defense"),
    "LHX": ("Industrials", "Aerospace & Defense"),
    "TXT": ("Industrials", "Aerospace & Defense"),
    "HII": ("Industrials", "Aerospace & Defense"),
    "LDOS": ("Industrials", "Research & Consulting Services"),
    "BAH": ("Industrials", "Research & Consulting Services"),
    "GE": ("Industrials", "Industrial Conglomerates"),
    "MMM": ("Industrials", "Industrial Conglomerates"),
    "EMR": ("Industrials", "Electrical Components & Equipment"),
    "ETN": ("Industrials", "Electrical Components & Equipment"),
    "PH": ("Industrials", "Industrial Machinery & Supplies & Equipment"),
    "ITW": ("Industrials", "Industrial Machinery & Supplies & Equipment"),
    "ROK": ("Industrials", "Electrical Components & Equipment"),
    "DOV": ("Industrials", "Industrial Machinery & Supplies & Equipment"),
    "XYL": ("Industrials", "Industrial Machinery & Supplies & Equipment"),
    "FTV": ("Industrials", "Industrial Machinery & Supplies & Equipment"),
    "IEX": ("Industrials", "Industrial Machinery & Supplies & Equipment"),
    "PNR": ("Industrials", "Industrial Machinery & Supplies & Equipment"),
    "CARR": ("Industrials", "Building Products"),
    "JCI": ("Industrials", "Building Products"),
    "IR": ("Industrials", "Industrial Machinery & Supplies & Equipment"),
    "OTIS": ("Industrials", "Industrial Machinery & Supplies & Equipment"),
    "PCAR": ("Industrials", "Construction Machinery & Heavy Transportation"),
    "NSC": ("Industrials", "Railroads"),
    "CSX": ("Industrials", "Railroads"),
    "KSU": ("Industrials", "Railroads"),
    "GWR": ("Industrials", "Railroads"),
    "JBHT": ("Industrials", "Trucking"),
    "ODFL": ("Industrials", "Trucking"),
    "SAIA": ("Industrials", "Trucking"),
    "ARCB": ("Industrials", "Trucking"),
    "KNX": ("Industrials", "Trucking"),
    "CHRW": ("Industrials", "Air Freight & Logistics"),
    "EXPD": ("Industrials", "Air Freight & Logistics"),
    "LSTR": ("Industrials", "Trucking"),
    "WERN": ("Industrials", "Trucking"),
    "MATX": ("Industrials", "Marine"),
    "HUBG": ("Industrials", "Trucking"),
    "R": ("Industrials", "Trading Companies & Distributors"),
    "GWW": ("Industrials", "Trading Companies & Distributors"),
    "FAST": ("Industrials", "Trading Companies & Distributors"),
    "DCI": ("Industrials", "Trading Companies & Distributors"),
    "MSM": ("Industrials", "Trading Companies & Distributors"),
    "WSO": ("Industrials", "Trading Companies & Distributors"),
    "SWK": ("Industrials", "Industrial Machinery & Supplies & Equipment"),
    "SNA": ("Industrials", "Industrial Machinery & Supplies & Equipment"),
    "TTC": ("Industrials", "Industrial Machinery & Supplies & Equipment"),
    "LECO": ("Industrials", "Industrial Machinery & Supplies & Equipment"),
    "AIT": ("Industrials", "Trading Companies & Distributors"),
    "DY": ("Industrials", "Construction & Engineering"),
    "EME": ("Industrials", "Construction & Engineering"),
    "FIX": ("Industrials", "Construction & Engineering"),
    "MTZ": ("Industrials", "Construction & Engineering"),
    "PWR": ("Industrials", "Construction & Engineering"),
    "STRL": ("Industrials", "Construction & Engineering"),
    "ACM": ("Industrials", "Construction & Engineering"),
    "J": ("Industrials", "Construction & Engineering"),
    "FLR": ("Industrials", "Construction & Engineering"),
    "KBR": ("Industrials", "Research & Consulting Services"),
    "CACI": ("Industrials", "Research & Consulting Services"),
    "SAIC": ("Industrials", "Research & Consulting Services"),
    "MAN": ("Industrials", "Research & Consulting Services"),
    "HCSG": ("Industrials", "Research & Consulting Services"),
    "CTAS": ("Industrials", "Research & Consulting Services"),
    "ROLL": ("Industrials", "Research & Consulting Services"),
    "ABM": ("Industrials", "Research & Consulting Services"),
    "CWST": ("Industrials", "Environmental & Facilities Services"),
    "RSG": ("Industrials", "Environmental & Facilities Services"),
    "WM": ("Industrials", "Environmental & Facilities Services"),
    "WCN": ("Industrials", "Environmental & Facilities Services"),
    "CLH": ("Industrials", "Environmental & Facilities Services"),
    "VRS": ("Industrials", "Environmental & Facilities Services"),
    "EEFT": ("Industrials", "Data Processing & Outsourced Services"),
    "GPN": ("Industrials", "Data Processing & Outsourced Services"),
    "FIS": ("Industrials", "Data Processing & Outsourced Services"),
    "FISV": ("Industrials", "Data Processing & Outsourced Services"),
    "PAYX": ("Industrials", "Human Resource & Employment Services"),
    "ADP": ("Industrials", "Human Resource & Employment Services"),
    "BR": ("Industrials", "Research & Consulting Services"),
    "EXPD": ("Industrials", "Air Freight & Logistics"),
    # Materials
    "LIN": ("Materials", "Industrial Gases"),
    "APD": ("Materials", "Industrial Gases"),
    "ECL": ("Materials", "Specialty Chemicals"),
    "DD": ("Materials", "Specialty Chemicals"),
    "DOW": ("Materials", "Commodity Chemicals"),
    "LYB": ("Materials", "Commodity Chemicals"),
    "PPG": ("Materials", "Specialty Chemicals"),
    "SHW": ("Materials", "Specialty Chemicals"),
    "NEM": ("Materials", "Gold"),
    "FCX": ("Materials", "Copper"),
    "SCCO": ("Materials", "Copper"),
    "TECK": ("Materials", "Diversified Metals & Mining"),
    "AA": ("Materials", "Aluminum"),
    "X": ("Materials", "Steel"),
    "NUE": ("Materials", "Steel"),
    "STLD": ("Materials", "Steel"),
    "RS": ("Materials", "Steel"),
    "CMC": ("Materials", "Steel"),
    "CLF": ("Materials", "Steel"),
    "MT": ("Materials", "Steel"),
    "TX": ("Materials", "Steel"),
    "ZEUS": ("Materials", "Steel"),
    "ATI": ("Materials", "Steel"),
    "CENX": ("Materials", "Aluminum"),
    "KALU": ("Materials", "Aluminum"),
    "MP": ("Materials", "Diversified Metals & Mining"),
    "MOS": ("Materials", "Fertilizers & Agricultural Chemicals"),
    "CF": ("Materials", "Fertilizers & Agricultural Chemicals"),
    "NTR": ("Materials", "Fertilizers & Agricultural Chemicals"),
    "FMC": ("Materials", "Fertilizers & Agricultural Chemicals"),
    "IFF": ("Materials", "Specialty Chemicals"),
    "ALB": ("Materials", "Specialty Chemicals"),
    "CE": ("Materials", "Construction Materials"),
    "VMC": ("Materials", "Construction Materials"),
    "MLM": ("Materials", "Construction Materials"),
    "SUM": ("Materials", "Construction Materials"),
    "USLM": ("Materials", "Construction Materials"),
    "IP": ("Materials", "Paper Packaging"),
    "PKG": ("Materials", "Paper Packaging"),
    "AMCR": ("Materials", "Paper Packaging"),
    "SEE": ("Materials", "Paper Packaging"),
    "AVY": ("Materials", "Paper Packaging"),
    "BLL": ("Materials", "Metal, Glass & Plastic Containers"),
    "CCK": ("Materials", "Metal, Glass & Plastic Containers"),
    "SON": ("Materials", "Paper Packaging"),
    "GPK": ("Materials", "Paper Packaging"),
    "SLGN": ("Materials", "Metal, Glass & Plastic Containers"),
    "HUN": ("Materials", "Commodity Chemicals"),
    "EMN": ("Materials", "Commodity Chemicals"),
    "CC": ("Materials", "Commodity Chemicals"),
    "OLN": ("Materials", "Commodity Chemicals"),
    "CBT": ("Materials", "Commodity Chemicals"),
    "KRA": ("Materials", "Specialty Chemicals"),
    "HWKN": ("Materials", "Specialty Chemicals"),
    "NEU": ("Materials", "Specialty Chemicals"),
    "SXT": ("Materials", "Specialty Chemicals"),
    "GEF": ("Materials", "Paper Packaging"),
    "SLVM": ("Materials", "Paper Packaging"),
    # Real Estate
    "AMT": ("Real Estate", "Specialized REITs"),
    "PLD": ("Real Estate", "Specialized REITs"),
    "CCI": ("Real Estate", "Specialized REITs"),
    "EQIX": ("Real Estate", "Specialized REITs"),
    "PSA": ("Real Estate", "Specialized REITs"),
    "WELL": ("Real Estate", "Health Care REITs"),
    "DLR": ("Real Estate", "Specialized REITs"),
    "SPG": ("Real Estate", "Retail REITs"),
    "O": ("Real Estate", "Retail REITs"),
    "CBRE": ("Real Estate", "Real Estate Services"),
    "JLL": ("Real Estate", "Real Estate Services"),
    "SBAC": ("Real Estate", "Specialized REITs"),
    "AVB": ("Real Estate", "Residential REITs"),
    "EQR": ("Real Estate", "Residential REITs"),
    "VTR": ("Real Estate", "Health Care REITs"),
    "ESS": ("Real Estate", "Residential REITs"),
    "MAA": ("Real Estate", "Residential REITs"),
    "UDR": ("Real Estate", "Residential REITs"),
    "CPT": ("Real Estate", "Residential REITs"),
    "AIV": ("Real Estate", "Residential REITs"),
    "BXP": ("Real Estate", "Office REITs"),
    "VNO": ("Real Estate", "Office REITs"),
    "SLG": ("Real Estate", "Office REITs"),
    "ARE": ("Real Estate", "Office REITs"),
    "KRC": ("Real Estate", "Office REITs"),
    "HIW": ("Real Estate", "Office REITs"),
    "DEI": ("Real Estate", "Office REITs"),
    "HPP": ("Real Estate", "Office REITs"),
    "CUZ": ("Real Estate", "Office REITs"),
    "PEAK": ("Real Estate", "Health Care REITs"),
    "DOC": ("Real Estate", "Health Care REITs"),
    "HR": ("Real Estate", "Health Care REITs"),
    "OHI": ("Real Estate", "Health Care REITs"),
    "SBRA": ("Real Estate", "Health Care REITs"),
    "LTC": ("Real Estate", "Health Care REITs"),
    "CHCT": ("Real Estate", "Health Care REITs"),
    "DHC": ("Real Estate", "Health Care REITs"),
    "MPW": ("Real Estate", "Health Care REITs"),
    "EXR": ("Real Estate", "Specialized REITs"),
    "CUBE": ("Real Estate", "Specialized REITs"),
    "LSI": ("Real Estate", "Specialized REITs"),
    "NSA": ("Real Estate", "Specialized REITs"),
    "REXR": ("Real Estate", "Specialized REITs"),
    "FR": ("Real Estate", "Industrial REITs"),
    "REXR": ("Real Estate", "Industrial REITs"),
    "STAG": ("Real Estate", "Industrial REITs"),
    "TRNO": ("Real Estate", "Industrial REITs"),
    "EGP": ("Real Estate", "Retail REITs"),
    "KIM": ("Real Estate", "Retail REITs"),
    "REG": ("Real Estate", "Retail REITs"),
    "FRT": ("Real Estate", "Retail REITs"),
    "KRG": ("Real Estate", "Retail REITs"),
    "ROIC": ("Real Estate", "Retail REITs"),
    "SITC": ("Real Estate", "Retail REITs"),
    "WPC": ("Real Estate", "Retail REITs"),
    "NNN": ("Real Estate", "Retail REITs"),
    "ADC": ("Real Estate", "Retail REITs"),
    "EPRT": ("Real Estate", "Retail REITs"),
    "FCPT": ("Real Estate", "Retail REITs"),
    "NETL": ("Real Estate", "Retail REITs"),
    "VNO": ("Real Estate", "Office REITs"),
    "AKR": ("Real Estate", "Retail REITs"),
    "BRX": ("Real Estate", "Retail REITs"),
    "KIM": ("Real Estate", "Retail REITs"),
    "REG": ("Real Estate", "Retail REITs"),
    "RPT": ("Real Estate", "Retail REITs"),
    "SPG": ("Real Estate", "Retail REITs"),
    "UE": ("Real Estate", "Retail REITs"),
    "WRI": ("Real Estate", "Retail REITs"),
    # Utilities
    "NEE": ("Utilities", "Multi-Utilities"),
    "DUK": ("Utilities", "Multi-Utilities"),
    "SO": ("Utilities", "Multi-Utilities"),
    "D": ("Utilities", "Multi-Utilities"),
    "AEP": ("Utilities", "Multi-Utilities"),
    "EXC": ("Utilities", "Multi-Utilities"),
    "SRE": ("Utilities", "Multi-Utilities"),
    "XEL": ("Utilities", "Multi-Utilities"),
    "WEC": ("Utilities", "Multi-Utilities"),
    "ED": ("Utilities", "Multi-Utilities"),
    "ES": ("Utilities", "Multi-Utilities"),
    "ETR": ("Utilities", "Multi-Utilities"),
    "FE": ("Utilities", "Multi-Utilities"),
    "EIX": ("Utilities", "Multi-Utilities"),
    "PPL": ("Utilities", "Multi-Utilities"),
    "AEE": ("Utilities", "Multi-Utilities"),
    "CMS": ("Utilities", "Multi-Utilities"),
    "DTE": ("Utilities", "Multi-Utilities"),
    "NI": ("Utilities", "Multi-Utilities"),
    "LNT": ("Utilities", "Multi-Utilities"),
    "EVRG": ("Utilities", "Multi-Utilities"),
    "CNP": ("Utilities", "Multi-Utilities"),
    "PNW": ("Utilities", "Multi-Utilities"),
    "AWK": ("Utilities", "Water Utilities"),
    "WTR": ("Utilities", "Water Utilities"),
    "CWT": ("Utilities", "Water Utilities"),
    "SJW": ("Utilities", "Water Utilities"),
    "MSEX": ("Utilities", "Water Utilities"),
    "YORW": ("Utilities", "Water Utilities"),
    "ARTNA": ("Utilities", "Water Utilities"),
    "CWCO": ("Utilities", "Water Utilities"),
    "NEE": ("Utilities", "Multi-Utilities"),
    "AES": ("Utilities", "Independent Power Producers & Energy Traders"),
    "NRG": ("Utilities", "Independent Power Producers & Energy Traders"),
    "VST": ("Utilities", "Independent Power Producers & Energy Traders"),
    "CEG": ("Utilities", "Multi-Utilities"),
    "PCG": ("Utilities", "Multi-Utilities"),
    "PEG": ("Utilities", "Multi-Utilities"),
    "SO": ("Utilities", "Multi-Utilities"),
    "ATO": ("Utilities", "Multi-Utilities"),
    "NJR": ("Utilities", "Multi-Utilities"),
    "SJI": ("Utilities", "Multi-Utilities"),
    "SWX": ("Utilities", "Multi-Utilities"),
    "SR": ("Utilities", "Multi-Utilities"),
    "NWN": ("Utilities", "Multi-Utilities"),
    "OGS": ("Utilities", "Multi-Utilities"),
    "NFG": ("Utilities", "Multi-Utilities"),
}


def _collect_from_gics_sector_mapping() -> pd.DataFrame:
    """
    Collect sector/industry mappings from GICS classification.

    This provides a fallback source of sector data for major US stocks
    that may not be covered by the S&P 500 Wikipedia source.
    """
    rows: list[dict[str, object]] = []
    for symbol, (sector, industry) in KNOWN_SECTOR_MAP.items():
        rows.append({
            "ticker": normalize_symbol(symbol),
            "company_name": None,
            "exchange": None,
            "sector": sector,
            "industry": industry,
            "source": "gics_sector_mapping",
        })
    return pd.DataFrame(rows)


def _collect_from_sp500_wikipedia_enhanced() -> pd.DataFrame:
    """
    Enhanced S&P 500 collection that also gathers sector data.

    This is an improved version that handles more edge cases and
    provides better sector/industry coverage.
    """
    try:
        with _temporary_proxy_bypass():
            request = __import__("urllib.request", fromlist=["Request", "urlopen"]).Request(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
            )
            with __import__("urllib.request", fromlist=["urlopen"]).urlopen(request, timeout=25) as response:
                tables = pd.read_html(StringIO(response.read().decode("utf-8")))
    except Exception:
        return pd.DataFrame()

    if not tables:
        return pd.DataFrame()

    table = tables[0]
    rows: list[dict[str, object]] = []

    for _, record in table.iterrows():
        symbol = normalize_symbol(str(record.get("Symbol", "")))
        if not symbol:
            continue

        sector = str(record.get("GICS Sector", "")).strip() or None
        industry = str(record.get("GICS Sub-Industry", "")).strip() or None

        # Validate sector is a real GICS sector
        if sector and sector not in GICS_SECTORS:
            sector = None

        rows.append({
            "ticker": symbol,
            "company_name": str(record.get("Security", "")).strip() or None,
            "exchange": str(record.get("Exchange", "")).strip() or None,
            "sector": sector,
            "industry": industry,
            "source": "wikipedia_sp500",
        })

    return pd.DataFrame(rows)


def _collect_from_nasdaq100_wikipedia() -> pd.DataFrame:
    """
    Collect sector/industry mappings from NASDAQ-100 Wikipedia page.

    This adds ~100 additional tech-heavy stocks with sector data.
    """
    try:
        with _temporary_proxy_bypass():
            request = __import__("urllib.request", fromlist=["Request", "urlopen"]).Request(
                "https://en.wikipedia.org/wiki/NASDAQ-100",
                headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
            )
            with __import__("urllib.request", fromlist=["urlopen"]).urlopen(request, timeout=25) as response:
                tables = pd.read_html(StringIO(response.read().decode("utf-8")))
    except Exception:
        return pd.DataFrame()

    if not tables or len(tables) < 2:
        return pd.DataFrame()

    # The second table usually has the component list with sectors
    table = tables[1] if len(tables) > 1 else tables[0]
    rows: list[dict[str, object]] = []

    # Try to find sector column
    sector_col = None
    ticker_col = None
    company_col = None

    for col in table.columns:
        col_lower = str(col).lower()
        if "symbol" in col_lower or "ticker" in col_lower or "code" in col_lower:
            ticker_col = col
        elif "sector" in col_lower or "industry" in col_lower:
            sector_col = col
        elif "company" in col_lower or "name" in col_lower:
            company_col = col

    if ticker_col is None:
        return pd.DataFrame()

    for _, record in table.iterrows():
        symbol = normalize_symbol(str(record.get(ticker_col, "")))
        if not symbol or len(symbol) > 5:
            continue

        sector = str(record.get(sector_col, "")).strip() if sector_col else None
        if sector and sector not in GICS_SECTORS:
            sector = None

        company = str(record.get(company_col, "")).strip() if company_col else None

        rows.append({
            "ticker": symbol,
            "company_name": company or None,
            "exchange": "NASDAQ",
            "sector": sector,
            "industry": None,
            "source": "wikipedia_nasdaq100",
        })

    return pd.DataFrame(rows)


def _collect_from_dow_jones_wikipedia() -> pd.DataFrame:
    """
    Collect sector/industry mappings from Dow Jones Industrial Average Wikipedia page.

    This adds 30 blue-chip stocks with sector data.
    """
    try:
        with _temporary_proxy_bypass():
            request = __import__("urllib.request", fromlist=["Request", "urlopen"]).Request(
                "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
                headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
            )
            with __import__("urllib.request", fromlist=["urlopen"]).urlopen(request, timeout=25) as response:
                tables = pd.read_html(StringIO(response.read().decode("utf-8")))
    except Exception:
        return pd.DataFrame()

    if not tables:
        return pd.DataFrame()

    # Find the components table
    table = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("symbol" in c or "ticker" in c for c in cols):
            table = t
            break

    if table is None:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []

    sector_col = None
    ticker_col = None
    company_col = None

    for col in table.columns:
        col_lower = str(col).lower()
        if "symbol" in col_lower or "ticker" in col_lower or col_lower == "symbol":
            ticker_col = col
        elif "sector" in col_lower:
            sector_col = col
        elif "company" in col_lower or col_lower == "company":
            company_col = col

    if ticker_col is None:
        return pd.DataFrame()

    for _, record in table.iterrows():
        symbol = normalize_symbol(str(record.get(ticker_col, "")))
        if not symbol or len(symbol) > 5:
            continue

        sector = str(record.get(sector_col, "")).strip() if sector_col else None
        if sector and sector not in GICS_SECTORS:
            sector = None

        company = str(record.get(company_col, "")).strip() if company_col else None

        rows.append({
            "ticker": symbol,
            "company_name": company or None,
            "exchange": "NYSE" if symbol in ["JPM", "GS", "HD", "CAT", "MCD", "V", "WMT", "DIS"] else "NYSE",
            "sector": sector,
            "industry": None,
            "source": "wikipedia_dow30",
        })

    return pd.DataFrame(rows)


# Additional sector mappings for Russell 1000 extended coverage
# These are inferred from common knowledge of major US stocks
EXTENDED_SECTOR_MAP: dict[str, tuple[str, str]] = {
    # Regional Banks
    "PNC": ("Financials", "Regional Banks"),
    "TFC": ("Financials", "Regional Banks"),
    "MTB": ("Financials", "Regional Banks"),
    "KEY": ("Financials", "Regional Banks"),
    "FITB": ("Financials", "Regional Banks"),
    "HBAN": ("Financials", "Regional Banks"),
    "RF": ("Financials", "Regional Banks"),
    "CFG": ("Financials", "Regional Banks"),
    "WAL": ("Financials", "Regional Banks"),
    "CMA": ("Financials", "Regional Banks"),
    "ZION": ("Financials", "Regional Banks"),
    "FRC": ("Financials", "Regional Banks"),
    "SIVB": ("Financials", "Regional Banks"),
    "EWBC": ("Financials", "Regional Banks"),
    "WBS": ("Financials", "Regional Banks"),
    "ONB": ("Financials", "Regional Banks"),
    "UMBF": ("Financials", "Regional Banks"),
    "FHN": ("Financials", "Regional Banks"),
    "SNV": ("Financials", "Regional Banks"),
    "BOKF": ("Financials", "Regional Banks"),
    # Insurance
    "AFL": ("Financials", "Life & Health Insurance"),
    "ALL": ("Financials", "Property & Casualty Insurance"),
    "AIG": ("Financials", "Multi-line Insurance"),
    "AMP": ("Financials", "Life & Health Insurance"),
    "AIZ": ("Financials", "Property & Casualty Insurance"),
    "AJG": ("Financials", "Insurance Brokers"),
    "BRO": ("Financials", "Insurance Brokers"),
    "CB": ("Financials", "Property & Casualty Insurance"),
    "CINF": ("Financials", "Property & Casualty Insurance"),
    "CLF": ("Financials", "Property & Casualty Insurance"),
    "CNA": ("Financials", "Multi-line Insurance"),
    "EG": ("Financials", "Reinsurance"),
    "GL": ("Financials", "Insurance Brokers"),
    "HIG": ("Financials", "Multi-line Insurance"),
    "L": ("Financials", "Life & Health Insurance"),
    "LNC": ("Financials", "Life & Health Insurance"),
    "MET": ("Financials", "Life & Health Insurance"),
    "MMC": ("Financials", "Insurance Brokers"),
    "PFG": ("Financials", "Life & Health Insurance"),
    "PGR": ("Financials", "Property & Casualty Insurance"),
    "PRU": ("Financials", "Life & Health Insurance"),
    "RE": ("Financials", "Reinsurance"),
    "RGA": ("Financials", "Reinsurance"),
    "RLI": ("Financials", "Property & Casualty Insurance"),
    "TRV": ("Financials", "Property & Casualty Insurance"),
    "UNM": ("Financials", "Life & Health Insurance"),
    "WRB": ("Financials", "Property & Casualty Insurance"),
    # Asset Management
    "BEN": ("Financials", "Asset Management & Custody Banks"),
    "IVZ": ("Financials", "Asset Management & Custody Banks"),
    "TROW": ("Financials", "Asset Management & Custody Banks"),
    "AMG": ("Financials", "Asset Management & Custody Banks"),
    "SEIC": ("Financials", "Asset Management & Custody Banks"),
    "VIRTU": ("Financials", "Financial Exchanges & Data"),
    "LAZ": ("Financials", "Investment Banking & Brokerage"),
    "PJT": ("Financials", "Investment Banking & Brokerage"),
    "EVR": ("Financials", "Investment Banking & Brokerage"),
    "MORN": ("Financials", "Financial Exchanges & Data"),
    # Diversified Financials
    "BX": ("Financials", "Asset Management & Custody Banks"),
    "CG": ("Financials", "Asset Management & Custody Banks"),
    "KKR": ("Financials", "Asset Management & Custody Banks"),
    "APO": ("Financials", "Asset Management & Custody Banks"),
    "OWL": ("Financials", "Asset Management & Custody Banks"),
    "ARES": ("Financials", "Asset Management & Custody Banks"),
    "HLNE": ("Financials", "Asset Management & Custody Banks"),
    "BLUE": ("Financials", "Asset Management & Custody Banks"),
    # More Technology
    "ANET": ("Information Technology", "Semiconductors"),
    "APH": ("Information Technology", "Electronic Components"),
    "TEL": ("Information Technology", "Electronic Components"),
    "GLW": ("Information Technology", "Electronic Components"),
    "HPQ": ("Information Technology", "Technology Hardware, Storage & Peripherals"),
    "NTAP": ("Information Technology", "Technology Hardware, Storage & Peripherals"),
    "WDC": ("Information Technology", "Technology Hardware, Storage & Peripherals"),
    "STX": ("Information Technology", "Technology Hardware, Storage & Peripherals"),
    "JNPR": ("Information Technology", "Communications Equipment"),
    "FFIV": ("Information Technology", "Communications Equipment"),
    "AKAM": ("Information Technology", "Application Software"),
    "JKHY": ("Information Technology", "Application Software"),
    "GDDY": ("Information Technology", "Application Software"),
    "TTWO": ("Information Technology", "Interactive Home Entertainment"),
    "EA": ("Information Technology", "Interactive Home Entertainment"),
    "ATVI": ("Information Technology", "Interactive Home Entertainment"),
    "RBLX": ("Communication Services", "Interactive Media & Services"),
    "U": ("Information Technology", "Application Software"),
    "PATH": ("Information Technology", "Application Software"),
    "AI": ("Information Technology", "Application Software"),
    "CFLT": ("Information Technology", "Application Software"),
    "MDB": ("Information Technology", "Application Software"),
    "ESTC": ("Information Technology", "Application Software"),
    "TEAM": ("Information Technology", "Application Software"),
    "WDAY": ("Information Technology", "Application Software"),
    "VEEV": ("Information Technology", "Application Software"),
    "ZM": ("Information Technology", "Application Software"),
    "DOCU": ("Information Technology", "Application Software"),
    "TWLO": ("Information Technology", "Application Software"),
    "OKTA": ("Information Technology", "Systems Software"),
    "S": ("Information Technology", "Application Software"),
    "WORK": ("Information Technology", "Application Software"),
    "BOX": ("Information Technology", "Application Software"),
    "DBX": ("Information Technology", "Application Software"),
    "SPLK": ("Information Technology", "Application Software"),
    "NOW": ("Information Technology", "Application Software"),
    "CRM": ("Information Technology", "Application Software"),
    "ADSK": ("Information Technology", "Application Software"),
    "ANSS": ("Information Technology", "Application Software"),
    "CTSH": ("Information Technology", "IT Consulting & Other Services"),
    "ACN": ("Information Technology", "IT Consulting & Other Services"),
    "EPAM": ("Information Technology", "IT Consulting & Other Services"),
    "GDDY": ("Information Technology", "Application Software"),
    "WIX": ("Information Technology", "Application Software"),
    "SQ": ("Information Technology", "Data Processing & Outsourced Services"),
    "AFRM": ("Information Technology", "Data Processing & Outsourced Services"),
    "SOFI": ("Information Technology", "Data Processing & Outsourced Services"),
    "UPST": ("Information Technology", "Data Processing & Outsourced Services"),
    "LC": ("Financials", "Consumer Finance"),
    # More Healthcare
    "HCA": ("Health Care", "Health Care Facilities"),
    "UHS": ("Health Care", "Health Care Facilities"),
    "THC": ("Health Care", "Health Care Facilities"),
    "CYH": ("Health Care", "Health Care Facilities"),
    "LPNT": ("Health Care", "Health Care Facilities"),
    "USPH": ("Health Care", "Health Care Facilities"),
    "ACHC": ("Health Care", "Health Care Facilities"),
    "ENSG": ("Health Care", "Health Care Facilities"),
    "SEM": ("Health Care", "Health Care Facilities"),
    "SGRY": ("Health Care", "Health Care Facilities"),
    "DVA": ("Health Care", "Health Care Facilities"),
    "FRESENIUS": ("Health Care", "Health Care Facilities"),
    "RMD": ("Health Care", "Health Care Equipment"),
    "HOLX": ("Health Care", "Health Care Equipment"),
    "VAR": ("Health Care", "Health Care Equipment"),
    "ALGN": ("Health Care", "Health Care Equipment"),
    "XRAY": ("Health Care", "Health Care Equipment"),
    "PODD": ("Health Care", "Health Care Equipment"),
    "TMO": ("Health Care", "Life Sciences Tools & Services"),
    "DHR": ("Health Care", "Life Sciences Tools & Services"),
    "A": ("Health Care", "Life Sciences Tools & Services"),
    "WAT": ("Health Care", "Life Sciences Tools & Services"),
    "MTD": ("Health Care", "Life Sciences Tools & Services"),
    "IQV": ("Health Care", "Life Sciences Tools & Services"),
    "LH": ("Health Care", "Health Care Services"),
    "DGX": ("Health Care", "Health Care Services"),
    "BIO": ("Health Care", "Life Sciences Tools & Services"),
    "TECH": ("Health Care", "Life Sciences Tools & Services"),
    "QGEN": ("Health Care", "Life Sciences Tools & Services"),
    "EXAS": ("Health Care", "Life Sciences Tools & Services"),
    "VEEV": ("Health Care", "Health Care Technology"),
    "TDOC": ("Health Care", "Health Care Technology"),
    "ONEM": ("Health Care", "Health Care Technology"),
    "ACCD": ("Health Care", "Health Care Technology"),
    "DOCS": ("Health Care", "Health Care Technology"),
    # More Consumer Discretionary
    "ORLY": ("Consumer Discretionary", "Automotive Retail"),
    "AZO": ("Consumer Discretionary", "Automotive Retail"),
    "AAP": ("Consumer Discretionary", "Automotive Retail"),
    "GPC": ("Consumer Discretionary", "Distributors"),
    "AN": ("Consumer Discretionary", "Automotive Retail"),
    "PAG": ("Consumer Discretionary", "Automotive Retail"),
    "SAH": ("Consumer Discretionary", "Automotive Retail"),
    "ABG": ("Consumer Discretionary", "Automotive Retail"),
    "LAD": ("Consumer Discretionary", "Automotive Retail"),
    "CVNA": ("Consumer Discretionary", "Automotive Retail"),
    "KMX": ("Consumer Discretionary", "Automotive Retail"),
    "CPRT": ("Consumer Discretionary", "Specialized Consumer Services"),
    "ROL": ("Consumer Discretionary", "Specialized Consumer Services"),
    "SCI": ("Consumer Discretionary", "Specialized Consumer Services"),
    "MATW": ("Consumer Discretionary", "Specialized Consumer Services"),
    "CSV": ("Consumer Discretionary", "Specialized Consumer Services"),
    "STON": ("Consumer Discretionary", "Specialized Consumer Services"),
    "MAT": ("Consumer Discretionary", "Leisure Products"),
    "HAS": ("Consumer Discretionary", "Leisure Products"),
    "JAKK": ("Consumer Discretionary", "Leisure Products"),
    "MC": ("Consumer Discretionary", "Apparel, Accessories & Luxury Goods"),
    "RL": ("Consumer Discretionary", "Apparel, Accessories & Luxury Goods"),
    "CPRI": ("Consumer Discretionary", "Apparel, Accessories & Luxury Goods"),
    "TPR": ("Consumer Discretionary", "Apparel, Accessories & Luxury Goods"),
    "GOOS": ("Consumer Discretionary", "Apparel, Accessories & Luxury Goods"),
    "CROX": ("Consumer Discretionary", "Footwear"),
    "DECK": ("Consumer Discretionary", "Footwear"),
    "SKX": ("Consumer Discretionary", "Footwear"),
    "WWW": ("Consumer Discretionary", "Footwear"),
    "FL": ("Consumer Discretionary", "Apparel Retail"),
    "HIBB": ("Consumer Discretionary", "Apparel Retail"),
    "DKS": ("Consumer Discretionary", "Apparel, Accessories & Luxury Goods"),
    "ASO": ("Consumer Discretionary", "Apparel Retail"),
    "BGFV": ("Consumer Discretionary", "Apparel Retail"),
    "HBI": ("Consumer Discretionary", "Apparel, Accessories & Luxury Goods"),
    "PVH": ("Consumer Discretionary", "Apparel, Accessories & Luxury Goods"),
    "VFC": ("Consumer Discretionary", "Apparel, Accessories & Luxury Goods"),
    "UAA": ("Consumer Discretionary", "Apparel, Accessories & Luxury Goods"),
    "UA": ("Consumer Discretionary", "Apparel, Accessories & Luxury Goods"),
    # More Consumer Staples
    "SYY": ("Consumer Staples", "Food Distributors"),
    "USFD": ("Consumer Staples", "Food Distributors"),
    "PFGC": ("Consumer Staples", "Food Distributors"),
    "UNFI": ("Consumer Staples", "Food Distributors"),
    "CHEF": ("Consumer Staples", "Food Distributors"),
    "SPTN": ("Consumer Staples", "Food Distributors"),
    "CALM": ("Consumer Staples", "Packaged Foods & Meats"),
    "JJSF": ("Consumer Staples", "Packaged Foods & Meats"),
    "LANC": ("Consumer Staples", "Packaged Foods & Meats"),
    "SENEA": ("Consumer Staples", "Packaged Foods & Meats"),
    "LWAY": ("Consumer Staples", "Packaged Foods & Meats"),
    "HAIN": ("Consumer Staples", "Packaged Foods & Meats"),
    "INGR": ("Consumer Staples", "Packaged Foods & Meats"),
    "BGS": ("Consumer Staples", "Packaged Foods & Meats"),
    "POST": ("Consumer Staples", "Packaged Foods & Meats"),
    "SMPL": ("Consumer Staples", "Packaged Foods & Meats"),
    # More Industrials
    "ODFL": ("Industrials", "Trucking"),
    "SAIA": ("Industrials", "Trucking"),
    "ARCB": ("Industrials", "Trucking"),
    "KNX": ("Industrials", "Trucking"),
    "WERN": ("Industrials", "Trucking"),
    "LSTR": ("Industrials", "Trucking"),
    "JBHT": ("Industrials", "Trucking"),
    "CHRW": ("Industrials", "Air Freight & Logistics"),
    "EXPD": ("Industrials", "Air Freight & Logistics"),
    "HUBG": ("Industrials", "Trucking"),
    "MATX": ("Industrials", "Marine"),
    " Kirby": ("Industrials", "Marine"),
    "HOEG": ("Industrials", "Marine"),
    "FRO": ("Industrials", "Marine"),
    "DHT": ("Industrials", "Marine"),
    "TNK": ("Industrials", "Marine"),
    "EURN": ("Industrials", "Marine"),
    "NAT": ("Industrials", "Marine"),
    "SHIP": ("Industrials", "Marine"),
    "SBLK": ("Industrials", "Marine"),
    "GOGL": ("Industrials", "Marine"),
    "GSL": ("Industrials", "Marine"),
    "EGLE": ("Industrials", "Marine"),
    "DLNG": ("Industrials", "Marine"),
    "GASS": ("Industrials", "Marine"),
    "CTRM": ("Industrials", "Marine"),
    "TOPS": ("Industrials", "Marine"),
    # More Materials
    "EMN": ("Materials", "Commodity Chemicals"),
    "CC": ("Materials", "Commodity Chemicals"),
    "WLK": ("Materials", "Commodity Chemicals"),
    "NEU": ("Materials", "Specialty Chemicals"),
    "KRA": ("Materials", "Specialty Chemicals"),
    "HWKN": ("Materials", "Specialty Chemicals"),
    "SXT": ("Materials", "Specialty Chemicals"),
    "ASH": ("Materials", "Specialty Chemicals"),
    "CBT": ("Materials", "Commodity Chemicals"),
    "ESI": ("Materials", "Specialty Chemicals"),
    "GEF": ("Materials", "Paper Packaging"),
    "SLVM": ("Materials", "Paper Packaging"),
    "NP": ("Materials", "Paper Packaging"),
    "KWR": ("Materials", "Specialty Chemicals"),
    "IOSP": ("Materials", "Specialty Chemicals"),
    "MERC": ("Materials", "Specialty Chemicals"),
    "LTHM": ("Materials", "Specialty Chemicals"),
    "SQM": ("Materials", "Fertilizers & Agricultural Chemicals"),
    # More Energy
    "OKE": ("Energy", "Oil & Gas Storage & Transportation"),
    "TRGP": ("Energy", "Oil & Gas Storage & Transportation"),
    "AM": ("Energy", "Oil & Gas Storage & Transportation"),
    "PAA": ("Energy", "Oil & Gas Storage & Transportation"),
    "MMP": ("Energy", "Oil & Gas Storage & Transportation"),
    "VLP": ("Energy", "Oil & Gas Storage & Transportation"),
    "WES": ("Energy", "Oil & Gas Storage & Transportation"),
    "DCP": ("Energy", "Oil & Gas Storage & Transportation"),
    "PAGP": ("Energy", "Oil & Gas Storage & Transportation"),
    "ENLC": ("Energy", "Oil & Gas Storage & Transportation"),
    "USAC": ("Energy", "Oil & Gas Storage & Transportation"),
    "CEQP": ("Energy", "Oil & Gas Storage & Transportation"),
    "EQM": ("Energy", "Oil & Gas Storage & Transportation"),
    "WMB": ("Energy", "Oil & Gas Storage & Transportation"),
    "KMI": ("Energy", "Oil & Gas Storage & Transportation"),
    # More Real Estate
    "INVH": ("Real Estate", "Residential REITs"),
    "ACC": ("Real Estate", "Specialized REITs"),
    "ELS": ("Real Estate", "Specialized REITs"),
    "UMH": ("Real Estate", "Residential REITs"),
    "SUI": ("Real Estate", "Specialized REITs"),
    "MSA": ("Real Estate", "Specialized REITs"),
    "SAFE": ("Real Estate", "Specialized REITs"),
    "COLD": ("Real Estate", "Specialized REITs"),
    "REXR": ("Real Estate", "Industrial REITs"),
    "TRNO": ("Real Estate", "Industrial REITs"),
    "JBGS": ("Real Estate", "Office REITs"),
    "PDM": ("Real Estate", "Office REITs"),
    "PGRE": ("Real Estate", "Office REITs"),
    "CLI": ("Real Estate", "Office REITs"),
    "OFC": ("Real Estate", "Retail REITs"),
    "ROIC": ("Real Estate", "Retail REITs"),
    "GTY": ("Real Estate", "Retail REITs"),
    "SITC": ("Real Estate", "Retail REITs"),
    "KIM": ("Real Estate", "Retail REITs"),
    "REG": ("Real Estate", "Retail REITs"),
    "FRT": ("Real Estate", "Retail REITs"),
    "KRG": ("Real Estate", "Retail REITs"),
    # More Utilities
    "CMS": ("Utilities", "Multi-Utilities"),
    "DTE": ("Utilities", "Multi-Utilities"),
    "AEE": ("Utilities", "Multi-Utilities"),
    "LNT": ("Utilities", "Multi-Utilities"),
    "EVRG": ("Utilities", "Multi-Utilities"),
    "CNP": ("Utilities", "Multi-Utilities"),
    "PNW": ("Utilities", "Multi-Utilities"),
    "NWE": ("Utilities", "Multi-Utilities"),
    "AVA": ("Utilities", "Multi-Utilities"),
    "POR": ("Utilities", "Multi-Utilities"),
    "OTTR": ("Utilities", "Multi-Utilities"),
    "BKH": ("Utilities", "Multi-Utilities"),
    "SR": ("Utilities", "Multi-Utilities"),
    "MDU": ("Utilities", "Multi-Utilities"),
    "NJR": ("Utilities", "Multi-Utilities"),
    "SJI": ("Utilities", "Multi-Utilities"),
    "SWX": ("Utilities", "Multi-Utilities"),
    "NWN": ("Utilities", "Multi-Utilities"),
    "OGS": ("Utilities", "Multi-Utilities"),
    "NFG": ("Utilities", "Multi-Utilities"),
}


def _collect_from_extended_sector_map() -> pd.DataFrame:
    """
    Collect sector/industry mappings from the extended static map.

    This provides sector data for an additional 300+ tickers beyond the core 528.
    """
    rows: list[dict[str, object]] = []
    for symbol, (sector, industry) in EXTENDED_SECTOR_MAP.items():
        rows.append({
            "ticker": normalize_symbol(symbol),
            "company_name": None,
            "exchange": None,
            "sector": sector,
            "industry": industry,
            "source": "extended_sector_map",
        })
    return pd.DataFrame(rows)


def gather_sector_classifications() -> pd.DataFrame:
    """
    Gather sector and industry classifications from multiple sources.

    Returns a DataFrame with columns:
    - ticker: Normalized ticker symbol
    - sector: GICS sector name
    - industry: GICS industry name
    - source: Source of the classification

    Sources are prioritized:
    1. Wikipedia S&P 500 (most authoritative for S&P 500)
    2. Wikipedia NASDAQ-100 (tech-heavy index)
    3. Wikipedia Dow Jones 30 (blue-chip stocks)
    4. Extended sector map (300+ additional tickers)
    5. GICS sector mapping (fallback for major stocks)
    """
    collectors: list[tuple[Callable[[], pd.DataFrame], int]] = [
        (_collect_from_sp500_wikipedia_enhanced, 1),      # Highest priority - S&P 500
        (_collect_from_nasdaq100_wikipedia, 2),           # NASDAQ-100
        (_collect_from_dow_jones_wikipedia, 3),           # Dow 30
        (_collect_from_extended_sector_map, 4),           # Extended map (300+ tickers)
        (_collect_from_gics_sector_mapping, 5),           # Core known map (528 tickers)
    ]

    collected_frames: list[pd.DataFrame] = []
    for collector, priority in collectors:
        try:
            frame = collector()
        except Exception:
            continue
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        collected_frames.append((frame, priority))

    if not collected_frames:
        return pd.DataFrame(columns=["ticker", "sector", "industry", "source"])

    # Sort by priority and combine
    collected_frames.sort(key=lambda x: x[1])
    combined = pd.concat([frame for frame, _ in collected_frames], ignore_index=True)

    # Deduplicate: keep first occurrence (highest priority source)
    combined = combined.drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)

    return combined[["ticker", "sector", "industry", "source"]]


def get_sector_for_ticker(ticker: str, sector_map: pd.DataFrame | None = None) -> tuple[str | None, str | None]:
    """
    Look up sector and industry for a single ticker.

    Args:
        ticker: The ticker symbol to look up
        sector_map: Pre-computed sector mapping DataFrame. If None, builds one.

    Returns:
        Tuple of (sector, industry) or (None, None) if not found
    """
    if sector_map is None:
        sector_map = gather_sector_classifications()

    if sector_map.empty:
        return None, None

    normalized = normalize_symbol(ticker)
    match = sector_map[sector_map["ticker"] == normalized]
    if match.empty:
        return None, None

    row = match.iloc[0]
    return (
        str(row["sector"]) if pd.notna(row.get("sector")) else None,
        str(row["industry"]) if pd.notna(row.get("industry")) else None,
    )


def enrich_with_sectors(candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich a candidates DataFrame with sector classifications.

    This fills in missing sector/industry values from our sector mapping sources.

    Args:
        candidates: DataFrame with at least 'ticker' column

    Returns:
        DataFrame with sector and industry columns populated where possible
    """
    if candidates.empty:
        return candidates

    result = candidates.copy()

    # Ensure sector and industry columns exist
    if "sector" not in result.columns:
        result["sector"] = None
    if "industry" not in result.columns:
        result["industry"] = None

    # Get sector mappings
    sector_map = gather_sector_classifications()
    if sector_map.empty:
        return result

    # Create lookup dictionary
    sector_lookup = dict(zip(sector_map["ticker"], sector_map["sector"]))
    industry_lookup = dict(zip(sector_map["ticker"], sector_map["industry"]))

    # Fill missing sectors
    for idx, row in result.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker:
            continue

        # Only fill if currently empty/None
        if pd.isna(row.get("sector")) or str(row.get("sector", "")).strip() == "":
            if ticker in sector_lookup:
                result.at[idx, "sector"] = sector_lookup[ticker]

        if pd.isna(row.get("industry")) or str(row.get("industry", "")).strip() == "":
            if ticker in industry_lookup:
                result.at[idx, "industry"] = industry_lookup[ticker]

    return result