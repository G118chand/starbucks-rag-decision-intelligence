"""Synthetic Starbucks operational data generator.

Produces six CSV files and three Markdown reports in the data/synthetic directory
for use by DataIngestionPipeline.  Run directly:

    python data/synthetic/generate_data.py
"""

import logging
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── output paths ─────────────────────────────────────────────────────────────
OUTPUT_DIR: Path = Path(__file__).parent
REPORTS_DIR: Path = OUTPUT_DIR / "reports"

# ── domain constants ─────────────────────────────────────────────────────────
REGIONS: List[str] = [
    "Northeast", "Southeast", "Midwest", "Southwest",
    "West Coast", "Pacific Northwest", "Mountain", "Mid-Atlantic",
]

STORE_FORMATS: List[str] = ["Drive-Thru", "Cafe", "Reserve", "Kiosk", "Drive-Thru+Cafe"]

PRODUCT_CATEGORIES: List[str] = [
    "Hot Beverages", "Cold Beverages", "Frappuccino", "Teavana",
    "Food", "Merchandise", "At-Home Coffee",
]

PRODUCTS: dict = {
    "Hot Beverages": [
        ("Pike Place Roast", 3.45, 5, 260),
        ("Caffe Latte", 4.95, 5, 190),
        ("Cappuccino", 4.95, 5, 120),
        ("Caffe Americano", 3.95, 5, 15),
        ("Flat White", 5.25, 5, 220),
        ("Espresso", 2.95, 5, 10),
        ("Caramel Macchiato", 5.75, 5, 250),
        ("Pumpkin Spice Latte", 5.95, 5, 380),
    ],
    "Cold Beverages": [
        ("Cold Brew", 4.45, 5, 5),
        ("Nitro Cold Brew", 5.25, 5, 5),
        ("Iced Caffe Latte", 4.95, 5, 130),
        ("Iced Matcha Latte", 5.25, 5, 200),
        ("Iced Brown Sugar Oat Shaken Espresso", 5.95, 5, 120),
        ("Iced Caramel Macchiato", 5.75, 5, 250),
    ],
    "Frappuccino": [
        ("Caramel Frappuccino", 5.95, 5, 380),
        ("Mocha Frappuccino", 5.95, 5, 370),
        ("Vanilla Bean Frappuccino", 5.25, 5, 340),
        ("Java Chip Frappuccino", 6.25, 5, 460),
        ("Matcha Frappuccino", 5.75, 5, 420),
    ],
    "Teavana": [
        ("Chai Tea Latte", 4.95, 5, 240),
        ("Green Tea Latte", 5.25, 5, 320),
        ("Peach Green Tea Lemonade", 4.75, 5, 140),
        ("Passion Tango Tea", 3.45, 5, 45),
    ],
    "Food": [
        ("Butter Croissant", 3.45, 5, 240),
        ("Blueberry Muffin", 3.25, 5, 380),
        ("Spinach Feta Wrap", 6.45, 5, 290),
        ("Impossible Breakfast Sandwich", 6.95, 5, 420),
        ("Sous Vide Egg Bites", 6.45, 5, 310),
        ("Avocado Spread", 1.50, 5, 90),
        ("Banana Bread", 3.25, 5, 420),
    ],
    "Merchandise": [
        ("Starbucks Tumbler 24oz", 24.95, 5, 0),
        ("Starbucks Cold Cup 24oz", 19.95, 5, 0),
        ("Holiday Ornament Set", 14.95, 5, 0),
        ("Reusable Hot Cup", 12.95, 5, 0),
    ],
    "At-Home Coffee": [
        ("Pike Place Ground Coffee 12oz", 10.95, 5, 0),
        ("Veranda Blend K-Cups 10ct", 12.95, 5, 0),
        ("Espresso Roast Whole Bean 12oz", 12.95, 5, 0),
        ("Blonde Roast Capsules 10ct", 11.95, 5, 0),
    ],
}

STORE_CITIES: dict = {
    "Northeast":        ["New York", "Boston", "Philadelphia", "Hartford", "Providence"],
    "Southeast":        ["Atlanta", "Miami", "Charlotte", "Nashville", "Orlando"],
    "Midwest":          ["Chicago", "Detroit", "Minneapolis", "Columbus", "Indianapolis"],
    "Southwest":        ["Phoenix", "Las Vegas", "Albuquerque", "Tucson", "El Paso"],
    "West Coast":       ["Los Angeles", "San Francisco", "San Diego", "Sacramento", "Long Beach"],
    "Pacific Northwest":["Seattle", "Portland", "Tacoma", "Bellevue", "Eugene"],
    "Mountain":         ["Denver", "Salt Lake City", "Boise", "Reno", "Colorado Springs"],
    "Mid-Atlantic":     ["Washington DC", "Baltimore", "Richmond", "Virginia Beach", "Wilmington"],
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_store_id(region: str, index: int) -> str:
    """Return a store ID like 'NE-0042'."""
    prefix = "".join(w[0] for w in region.split())
    return f"{prefix}-{index:04d}"


def _date_range(start: str, end: str, freq: str = "W") -> List[str]:
    """Return a list of date strings between start and end at the given frequency."""
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start=start, end=end, freq=freq)]


# ── CSV generators ────────────────────────────────────────────────────────────

def generate_sales_data(n_stores: int = 80) -> pd.DataFrame:
    """Generate daily store-level sales across all regions.

    Args:
        n_stores: Total stores to simulate.

    Returns:
        DataFrame with columns store_id, store_name, region, format, date,
        revenue, transactions, avg_ticket_usd, mobile_order_pct, drive_thru_pct.
    """
    dates = _date_range("2023-01-01", "2024-12-31", freq="W")
    rows = []

    for i in range(n_stores):
        region = REGIONS[i % len(REGIONS)]
        city = random.choice(STORE_CITIES[region])
        store_id = _make_store_id(region, i + 1)
        store_name = f"Starbucks {city} #{i + 1}"
        fmt = random.choice(STORE_FORMATS)

        # base weekly revenue varies by format
        base = {"Drive-Thru": 32000, "Cafe": 27000, "Reserve": 42000,
                "Kiosk": 12000, "Drive-Thru+Cafe": 38000}[fmt]

        for date in dates:
            month = int(date[5:7])
            # seasonal lift: Q4 holiday +15%, summer +8%
            seasonal = 1.15 if month == 12 else 1.08 if month in (6, 7, 8) else 1.0
            revenue = round(base * seasonal * np.random.normal(1.0, 0.07), 2)
            transactions = int(revenue / np.random.uniform(7.5, 9.5))
            avg_ticket = round(revenue / max(transactions, 1), 2)
            mobile_pct = round(np.random.uniform(0.22, 0.45), 3)
            dt_pct = round(np.random.uniform(0.30, 0.65), 3) if "Drive" in fmt else round(np.random.uniform(0.0, 0.10), 3)

            rows.append({
                "store_id": store_id,
                "store_name": store_name,
                "region": region,
                "format": fmt,
                "date": date,
                "revenue": revenue,
                "transactions": transactions,
                "avg_ticket_usd": avg_ticket,
                "mobile_order_pct": mobile_pct,
                "drive_thru_pct": dt_pct,
            })

    df = pd.DataFrame(rows)
    logger.info("Generated sales_data: %d rows", len(df))
    return df


def generate_store_performance(n_stores: int = 80) -> pd.DataFrame:
    """Generate monthly store KPI snapshots.

    Args:
        n_stores: Number of stores to simulate.

    Returns:
        DataFrame with operational health metrics per store per month.
    """
    dates = _date_range("2023-01-01", "2024-12-01", freq="MS")
    rows = []

    for i in range(n_stores):
        region = REGIONS[i % len(REGIONS)]
        city = random.choice(STORE_CITIES[region])
        store_id = _make_store_id(region, i + 1)
        store_name = f"Starbucks {city} #{i + 1}"
        fmt = random.choice(STORE_FORMATS)
        base_csat = np.random.uniform(3.8, 4.7)

        for date in dates:
            rows.append({
                "store_id": store_id,
                "store_name": store_name,
                "region": region,
                "format": fmt,
                "date": date,
                "customer_satisfaction": round(np.clip(base_csat + np.random.normal(0, 0.15), 1, 5), 2),
                "drive_thru_wait_sec": int(np.random.normal(210, 40)) if "Drive" in fmt else None,
                "mobile_order_pct": round(np.random.uniform(0.20, 0.48), 3),
                "loyalty_redemption_pct": round(np.random.uniform(0.35, 0.65), 3),
                "staff_hours": int(np.random.normal(480, 40)),
                "waste_pct": round(np.random.uniform(0.01, 0.06), 3),
                "upsell_success_pct": round(np.random.uniform(0.10, 0.35), 3),
            })

    df = pd.DataFrame(rows)
    logger.info("Generated store_performance: %d rows", len(df))
    return df


def generate_product_catalog() -> pd.DataFrame:
    """Generate a static product catalogue with pricing and nutrition data.

    Returns:
        DataFrame with all Starbucks menu items across categories.
    """
    rows = []
    pid = 1
    for category, items in PRODUCTS.items():
        for name, base_price, _size, calories in items:
            is_seasonal = any(kw in name for kw in ("Pumpkin", "Holiday", "Peppermint"))
            rows.append({
                "product_id": f"SKU-{pid:04d}",
                "product_name": name,
                "category": category,
                "price_usd": base_price,
                "calories": calories if calories > 0 else None,
                "is_seasonal": is_seasonal,
                "customizable": category in ("Hot Beverages", "Cold Beverages", "Frappuccino", "Teavana"),
                "available_sizes": "Tall/Grande/Venti" if category not in ("Merchandise", "At-Home Coffee") else "One Size",
            })
            pid += 1

    df = pd.DataFrame(rows)
    logger.info("Generated product_catalog: %d rows", len(df))
    return df


def generate_regional_summary() -> pd.DataFrame:
    """Generate quarterly regional performance aggregates.

    Returns:
        DataFrame with revenue, store count, satisfaction and market data by region/quarter.
    """
    quarters = [
        "2023-Q1", "2023-Q2", "2023-Q3", "2023-Q4",
        "2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4",
    ]
    rows = []
    region_bases: dict = {
        "West Coast":        {"revenue": 420, "stores": 210, "share": 14.2},
        "Northeast":         {"revenue": 390, "stores": 195, "share": 12.8},
        "Pacific Northwest": {"revenue": 280, "stores": 140, "share": 9.5},
        "Southeast":         {"revenue": 310, "stores": 155, "share": 10.4},
        "Midwest":           {"revenue": 260, "stores": 130, "share": 8.9},
        "Mid-Atlantic":      {"revenue": 295, "stores": 148, "share": 9.8},
        "Southwest":         {"revenue": 230, "stores": 115, "share": 7.7},
        "Mountain":          {"revenue": 185, "stores": 93,  "share": 6.2},
    }

    for quarter in quarters:
        year, q = quarter.split("-")
        qnum = int(q[1])
        # derive a date for the quarter start
        month_start = {1: "01", 2: "04", 3: "07", 4: "10"}[qnum]
        date = f"{year}-{month_start}-01"

        for region, base in region_bases.items():
            growth = 1.0 + (int(year) - 2023) * 0.08 + (qnum - 1) * 0.015
            rows.append({
                "region": region,
                "quarter": quarter,
                "date": date,
                "total_revenue_millions": round(base["revenue"] * growth * np.random.normal(1, 0.03), 1),
                "total_stores": int(base["stores"] * growth),
                "new_stores_opened": random.randint(1, 8),
                "avg_customer_satisfaction": round(np.random.uniform(4.0, 4.6), 2),
                "market_share_pct": round(base["share"] * np.random.normal(1, 0.02), 2),
                "yoy_revenue_growth_pct": round(np.random.uniform(4.5, 12.5), 2),
                "digital_order_pct": round(np.random.uniform(0.28, 0.52), 3),
            })

    df = pd.DataFrame(rows)
    logger.info("Generated regional_summary: %d rows", len(df))
    return df


def generate_inventory() -> pd.DataFrame:
    """Generate weekly inventory snapshots for key ingredients and merchandise.

    Returns:
        DataFrame with stock levels, reorder flags, and shrinkage by region.
    """
    items = [
        ("INV-001", "Espresso Beans - Dark Roast", "Ingredients"),
        ("INV-002", "Espresso Beans - Blonde Roast", "Ingredients"),
        ("INV-003", "Whole Milk - Gallons", "Dairy"),
        ("INV-004", "Oat Milk - Cartons", "Dairy Alternative"),
        ("INV-005", "Almond Milk - Cartons", "Dairy Alternative"),
        ("INV-006", "Coconut Milk - Cartons", "Dairy Alternative"),
        ("INV-007", "Caramel Sauce - 64oz", "Syrups"),
        ("INV-008", "Vanilla Syrup - 1L", "Syrups"),
        ("INV-009", "Hazelnut Syrup - 1L", "Syrups"),
        ("INV-010", "Pumpkin Spice Sauce - 64oz", "Syrups"),
        ("INV-011", "Whipped Cream Canisters", "Ingredients"),
        ("INV-012", "Cold Brew Concentrate - 1gal", "Ingredients"),
        ("INV-013", "Matcha Powder - 1kg", "Ingredients"),
        ("INV-014", "Croissants - Each", "Food"),
        ("INV-015", "Breakfast Sandwiches - Each", "Food"),
    ]
    dates = _date_range("2023-01-01", "2024-12-31", freq="W")
    rows = []

    for item_id, item_name, category in items:
        reorder_pt = random.randint(20, 80)
        for region in REGIONS:
            for date in dates:
                stock = random.randint(10, 150)
                rows.append({
                    "item_id": item_id,
                    "item_name": item_name,
                    "category": category,
                    "region": region,
                    "date": date,
                    "stock_units": stock,
                    "reorder_point": reorder_pt,
                    "below_reorder": stock < reorder_pt,
                    "waste_units": random.randint(0, max(1, int(stock * 0.05))),
                    "shrinkage_pct": round(np.random.uniform(0.005, 0.04), 4),
                })

    df = pd.DataFrame(rows)
    logger.info("Generated inventory: %d rows", len(df))
    return df


def generate_customer_feedback(n_records: int = 600) -> pd.DataFrame:
    """Generate synthetic customer feedback records.

    Args:
        n_records: Number of feedback entries to generate.

    Returns:
        DataFrame with ratings, categories, themes, and store context.
    """
    themes: dict = {
        5: ["Excellent service!", "Barista remembered my order.", "Fastest drive-thru I've been to.",
            "My drink was perfect.", "Love the new seasonal menu.", "App order was ready early.",
            "Staff was incredibly friendly and welcoming."],
        4: ["Good experience overall.", "Drink was good but had a short wait.", "Usually consistent quality.",
            "Clean store and helpful staff.", "Mobile app worked great today."],
        3: ["Average visit, nothing special.", "Drink was okay but took longer than usual.",
            "Store was a bit crowded.", "Barista seemed new, but got it right eventually."],
        2: ["Wrong drink made twice.", "Long wait with no update.", "Store was not clean.",
            "App charged me but order wasn't ready.", "Out of my usual syrup again."],
        1: ["Waited 25 minutes for a latte.", "Rude staff member.", "Drink was completely wrong.",
            "Dirty tables and counters.", "App crashed during payment."],
    }
    feedback_categories = ["Service Speed", "Drink Quality", "Staff Friendliness",
                           "Cleanliness", "App Experience", "Food Quality", "Loyalty Rewards"]

    rows = []
    dates = _date_range("2023-01-01", "2024-12-31", freq="D")

    for i in range(n_records):
        region = random.choice(REGIONS)
        store_id = _make_store_id(region, random.randint(1, 10))
        rating = int(np.clip(np.random.normal(4.1, 0.9), 1, 5))
        rows.append({
            "feedback_id": f"FB-{i + 1:05d}",
            "store_id": store_id,
            "region": region,
            "date": random.choice(dates),
            "rating": rating,
            "category": random.choice(feedback_categories),
            "comment": random.choice(themes[rating]),
            "resolved": rating >= 3 or random.random() > 0.3,
            "channel": random.choice(["App", "In-Store Kiosk", "Email Survey", "Google Review"]),
        })

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    logger.info("Generated customer_feedback: %d rows", len(df))
    return df


# ── Markdown report generators ─────────────────────────────────────────────────

def generate_q4_2024_report() -> str:
    """Return Markdown text for the Q4 2024 executive summary report."""
    return """\
# Starbucks Q4 2024 Executive Summary

## Overview

Starbucks delivered record Q4 2024 performance with consolidated net revenues of **$9.7 billion**,
representing 8.2% year-over-year growth. Comparable store sales increased 6% globally, driven by
a 4% increase in average ticket size and a 2% increase in transactions.

## Key Financial Metrics

| Metric | Q4 2024 | Q4 2023 | YoY Change |
|---|---|---|---|
| Net Revenue | $9.7B | $8.96B | +8.2% |
| Operating Income | $1.54B | $1.38B | +11.6% |
| EPS (Diluted) | $1.22 | $1.06 | +15.1% |
| Comparable Store Sales | +6.0% | +5.0% | +1.0pp |
| Active Starbucks Rewards Members | 34.3M | 32.6M | +5.2% |

## Regional Performance

### West Coast
West Coast remains the highest-revenue region with $420M in quarterly revenue.
Digital orders reached 48% of total transactions, the highest of any region.
Nineteen new stores opened, bringing the total to 2,240 locations.

### Pacific Northwest
The company's home market posted 9.5% comparable growth, buoyed by Reserve Roastery traffic
and strong cold beverage attachment rates (61%). Drive-thru average service time improved
to 198 seconds from 217 seconds a year prior.

### Northeast
Densely populated urban markets in New York and Boston drove strong mobile-order adoption (44%).
Five new Reserve format stores opened in Manhattan, contributing premium ticket averages of $11.40.

### Southeast
Fastest-growing region by new store count (23 openings in Q4). Customer satisfaction averaged
4.38/5.0, a 12-basis-point improvement quarter-over-quarter. Nashville and Charlotte markets
exceeded revenue targets by 14%.

### Midwest
Stable performance with 8.9% market share. Drive-thru format dominates at 72% of locations.
Oat milk adoption reached 22% of milk-based beverages, up from 17% a year ago.

## Product & Menu Highlights

The **Pumpkin Spice Latte** seasonal launch generated $68M in incremental revenue during
the first four weeks, a new launch record.  Cold beverages accounted for 55% of all drink
revenue, continuing a multi-year trend away from hot beverages.

The **Iced Brown Sugar Oat Shaken Espresso** ranked as the #1 ordered beverage on the mobile
app for the third consecutive quarter.

Starbucks food attachment rate reached a record 22% of transactions, aided by the new
**Impossible Breakfast Sandwich** and expanded **Sous Vide Egg Bites** flavour lineup.

## Digital & Loyalty

Starbucks Rewards members now account for **59%** of US company-operated tender, up from 55%
in Q4 2023. The refreshed app with AI-powered personalisation increased average order value
among active members by $0.80.

Mobile order and pay reached **32%** of transactions across all US company-operated stores.

## Operational Efficiency

- Drive-thru average wait time: **204 seconds** (target: <240 seconds)
- Order accuracy rate: **98.2%** (up 0.4pp YoY)
- Waste as percentage of COGS: **2.1%** (down from 2.5%)
- Partner (employee) turnover: **52%** annualised (industry avg: 70%)

## Outlook — Q1 2025

Management guides Q1 2025 comparable store sales growth of 5–7% and EPS of $1.15–$1.25.
Key catalysts include the winter seasonal lineup, continued Reserve format expansion,
and the rollout of AI-driven drive-thru upsell prompts to 4,500 locations.
"""


def generate_market_expansion_report() -> str:
    """Return Markdown text for the market expansion analysis report."""
    return """\
# Starbucks Market Expansion Analysis — 2024–2026

## Executive Summary

This report analyses expansion opportunities across under-penetrated US markets and
evaluates international growth vectors through 2026. Current US store count stands at
16,800 company-operated and 6,200 licensed locations.

## Domestic Expansion Opportunities

### Tier-2 City Strategy

Analysis of population density, household income, and current Starbucks penetration
reveals significant whitespace in mid-size markets:

| Market | Population | Existing Stores | Opportunity Score |
|---|---|---|---|
| Raleigh, NC | 467,000 | 28 | 9.2/10 |
| Boise, ID | 235,000 | 14 | 8.8/10 |
| Colorado Springs, CO | 478,000 | 22 | 8.6/10 |
| Richmond, VA | 226,000 | 18 | 8.1/10 |
| Spokane, WA | 220,000 | 12 | 8.7/10 |

Recommended entry format for Tier-2 cities: **Drive-Thru+Cafe** hybrid, which delivers
30% higher revenue per square foot than standalone café formats in suburban settings.

### Drive-Thru Densification

The Southwest and Mountain regions are under-indexed on drive-thru relative to
comparable suburban demographics. An analysis of 340 candidate sites in Arizona, Nevada,
Colorado, and New Mexico identified 85 high-confidence drive-thru opportunities with
projected average unit volume (AUV) of $1.6M annually.

### Campus & Healthcare Channel

University campuses with 10,000+ students and hospital campuses with 500+ daily staff
represent a significantly underpenetrated licensed channel.  Current penetration is 34%
of target campuses. Expansion to 60% penetration by 2026 would add approximately
$380M in licensed revenue.

## Format Innovation

### Pickup-Only Stores
Following the success of the New York Pickup concept (AUV 40% above comp set),
Starbucks plans 300 additional urban pickup-only locations through 2025, targeting
office districts with high mobile-order rates (>45%).

### Starbucks Reserve Roastery Expansion
Two additional Roastery locations are planned — one in Austin, TX (opening Q2 2025)
and one in Washington DC (Q4 2025). Each Roastery generates approximately $9M AUV.

## International Growth Vectors

### China
Despite macroeconomic headwinds, Starbucks China represents the largest long-term growth
opportunity. Current store count: 6,800. Target by 2026: 9,000. Key markets include
lower-tier cities (Tier 3/4) where penetration is <1 store per 200,000 residents.

### India
India partnership with Tata Consumer Products currently operates 390 stores.
Target: 1,000 stores by 2028. Localised menu adaptations (Masala Chai Latte, Tandoori
Paneer Sandwich) have driven 22% higher ticket averages versus global menu at comparable
income demographics.

### Southeast Asia
Vietnam (+18% SSS in 2024), Thailand (+14% SSS) and the Philippines (+21% SSS) are
top performers. Combined store target: +340 net new through 2026.

## Competitive Landscape

| Competitor | US Locations | Key Threat |
|---|---|---|
| Dunkin' | 9,500 | Value positioning in drive-thru |
| McDonald's McCafé | 13,000 | Breakfast daypart overlap |
| Dutch Bros | 900 | Gen-Z drive-thru growth brand |
| Peet's Coffee | 350 | Premium positioning overlap |
| Local Independents | ~35,000 | Specialty beverage innovation |

Starbucks retains differentiation advantages in: loyalty program depth, mobile-order
infrastructure, brand premium, and barista customisation capability.

## Risk Factors

1. **Real estate cost inflation** in urban core markets (+12% YoY) compresses new-unit returns.
2. **Labour availability** remains the #1 operational constraint in Pacific Northwest and Northeast.
3. **Regulatory environment** — proposed minimum wage legislation in 8 states could add $0.20–0.35 to COGS per transaction.
4. **Consumer trade-down risk** in a softer macroeconomic environment, particularly for $7+ beverages.

## Recommendations

- Accelerate Tier-2 city drive-thru programme; approve 85 Southwest/Mountain sites.
- Expand campus/healthcare licensed channel sales team by 40 FTEs.
- Launch 300 urban pickup stores by end of 2025.
- Increase China new-store pace to 600/year, focused on Tier 3/4 cities.
- Establish India 1,000-store target with dedicated supply-chain infrastructure.
"""


def generate_operational_efficiency_report() -> str:
    """Return Markdown text for the operational efficiency and sustainability report."""
    return """\
# Starbucks Operational Efficiency & Sustainability Report — 2024

## Introduction

This report details Starbucks' progress against its operational efficiency targets for 2024,
including drive-thru performance, waste reduction, digital integration, and sustainability goals.

## Drive-Thru Performance

Drive-thru represents 53% of US company-operated revenue and is the most critical customer
touchpoint for speed-of-service metrics.

### 2024 Drive-Thru KPIs

| Metric | Target | Actual | Status |
|---|---|---|---|
| Average Wait Time (seconds) | < 240 | 204 | ✓ Achieved |
| Order Accuracy Rate | > 98% | 98.2% | ✓ Achieved |
| Speed of Service — AM Peak | < 270s | 261s | ✓ Achieved |
| Speed of Service — PM Peak | < 300s | 287s | ✓ Achieved |
| Digital Menu Board Rollout | 100% | 94% | ⚠ In Progress |

### AI-Powered Upsell at Drive-Thru

Pilot of AI-driven digital menu boards at 1,200 drive-thru locations showed:
- **+$0.62** average ticket increase per transaction
- **18%** improvement in food attachment rate
- Customer satisfaction scores unchanged (no perception of pressure upselling)

Full rollout to 4,500 US drive-thru stores is planned for Q2 2025.

## Mobile Order & Pay

Mobile Order & Pay (MOP) reached **32%** of US transactions, up from 27% in 2023.
Peak MOP usage is 7–9 AM at urban pickup and café formats (45–51% of transactions).

### Operational Impact of MOP

- Reduces average cashier transaction time by 38 seconds.
- Increases barista pressure during peak: 43% of stores report MOP backlog as top
  operational challenge.
- Starbucks Connect station redesign (tested in 400 stores) reduces MOP backlog
  incidents by 31%; full rollout targeted for mid-2025.

## Inventory & Supply Chain

### Waste Reduction Progress

| Category | 2022 Waste % | 2023 Waste % | 2024 Waste % | Target 2025 |
|---|---|---|---|---|
| Dairy | 3.8% | 3.1% | 2.4% | 2.0% |
| Food | 6.2% | 5.5% | 4.8% | 4.0% |
| Syrups & Sauces | 1.2% | 1.0% | 0.8% | 0.7% |
| Whole Bean Coffee | 0.9% | 0.7% | 0.5% | 0.4% |

Overall food and beverage waste reduced to **2.1% of COGS** in 2024, down from 2.5%
in 2023, avoiding approximately $48M in annualised write-off costs.

### Supplier Diversification

Following the 2023 oat milk supply constraint (which caused 6-week regional outages),
Starbucks added two additional certified oat milk suppliers and now holds 8 weeks of
strategic inventory across three regional distribution centres.

## Partner (Employee) Operations

### Staffing & Turnover

- Annualised store partner turnover: **52%** (industry benchmark: 68–72%)
- Average tenure of shift supervisors: 2.8 years (up from 2.3 years in 2022)
- Partner-to-transaction ratio maintained at 1:38 across all formats

### Training Investment

Starbucks invested **$210M** in partner training and development in 2024.
The Barista Basics digital certification programme reduced new-hire ramp time from
6 weeks to 4.5 weeks and improved beverage accuracy scores by 12%.

## Sustainability Initiatives

### Resource Consumption

| Resource | 2023 Actual | 2024 Actual | 2030 Target |
|---|---|---|---|
| Water per beverage (oz) | 28.4 | 26.1 | 22.0 |
| Energy per sq ft (kWh/yr) | 94.2 | 89.7 | 75.0 |
| Waste to landfill (%) | 34% | 29% | 10% |
| Reusable cup transactions (%) | 3.2% | 4.8% | 25% |

### Sustainable Sourcing

- **99%** of coffee ethically sourced under C.A.F.E. Practices standards.
- **78%** of cups are now made with 20% post-consumer recycled fibre.
- Regenerative agriculture pilot launched with 12 coffee farms in Colombia and Ethiopia.

### Plant-Based Menu Expansion

Plant-based beverage customisations (oat, almond, coconut, soy milk) now represent
**34%** of all milk-based drinks, up from 28% in 2022. Each plant-based substitution
reduces carbon footprint of the beverage by an estimated 0.22 kg CO₂e.

## Technology & Infrastructure

### Point-of-Sale Modernisation

Completed rollout of next-generation POS terminals to all US company-operated stores.
New terminals reduce transaction processing time by 1.2 seconds on average and support
tap-to-pay and wearable payment devices.

### Predictive Ordering System

Machine-learning-based demand forecasting deployed in 8,200 stores. The system
predicts hourly demand by SKU with 91% accuracy, reducing both stockouts and overproduction.
Estimated annualised savings: **$34M** in waste and $12M in emergency re-orders.

## 2025 Operational Priorities

1. Complete AI-powered drive-thru upsell rollout (4,500 stores, Q2 2025).
2. Deploy Starbucks Connect station redesign to all high-MOP urban locations.
3. Achieve food waste target of 4.0% through expanded predictive ordering.
4. Increase reusable cup programme transactions to 8% via incentive redesign.
5. Complete LEED Gold certification for 500 additional stores.
"""


# ── save helpers ──────────────────────────────────────────────────────────────

def save_csv(df: pd.DataFrame, filename: str) -> None:
    """Write a DataFrame to a CSV file in the output directory.

    Args:
        df: DataFrame to serialise.
        filename: Destination filename (no path).
    """
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    logger.info("Saved %s (%d rows, %d cols)", path, len(df), len(df.columns))


def save_markdown(content: str, filename: str) -> None:
    """Write a Markdown string to the reports directory.

    Args:
        content: Markdown text.
        filename: Destination filename (no path).
    """
    path = REPORTS_DIR / filename
    path.write_text(content, encoding="utf-8")
    logger.info("Saved %s (%d chars)", path, len(content))


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Generate all synthetic datasets and Markdown reports."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== Generating CSV datasets ===")
    save_csv(generate_sales_data(n_stores=80),         "sales_data.csv")
    save_csv(generate_store_performance(n_stores=80),  "store_performance.csv")
    save_csv(generate_product_catalog(),               "product_catalog.csv")
    save_csv(generate_regional_summary(),              "regional_summary.csv")
    save_csv(generate_inventory(),                     "inventory.csv")
    save_csv(generate_customer_feedback(n_records=600),"customer_feedback.csv")

    logger.info("=== Generating Markdown reports ===")
    save_markdown(generate_q4_2024_report(),              "Q4_2024_Executive_Summary.md")
    save_markdown(generate_market_expansion_report(),     "Market_Expansion_Analysis.md")
    save_markdown(generate_operational_efficiency_report(),"Operational_Efficiency_Report.md")

    # ── summary ──────────────────────────────────────────────────────────────
    csv_files  = list(OUTPUT_DIR.glob("*.csv"))
    md_files   = list(REPORTS_DIR.glob("*.md"))
    total_rows = sum(len(pd.read_csv(f)) for f in csv_files)

    logger.info("=== Generation complete ===")
    logger.info("CSV files   : %d  (%d total rows)", len(csv_files), total_rows)
    logger.info("MD reports  : %d", len(md_files))
    logger.info("Output dir  : %s", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
