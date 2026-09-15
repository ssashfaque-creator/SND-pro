"""Parse sales-team shop-wise targets (quota / plan).

This file is not a forecast. Enterprise S&D / SVP stacks keep three numbers:

* Actual — billed secondary
* Baseline — statistical Expected (run-rate, seasonality, pace)
* Target — the quota the sales team wrote

We never replace Expected with Target. The extract is an SSRS tablix keyed
by shop *name* (no POP code on the sample). Matching is conservative:
whales (>= 1 MT) require an exact folded name in the same city + distributor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from sndintel.io_utils import cell_str, norm_key, read_raw_table
from sndintel.storage import utcnow

FIELD_ID_MAP = {
    "txt_cregion": "zone",
    "txt_carea": "city",
    "txt_cdistributor_name": "distributor",
    "txt_cdistrib": "distributor",
    "txt_cdsr_name": "dsr_name",
    "txt_cdsr_na": "dsr_name",
    "txt_cdsr": "dsr_name",
    "txt_cpop_name": "store_name",
    "txt_cpop_na": "store_name",
    "txt_cpop_co": "store_id",
    "txt_cpop_c": "store_id",
    "txt_target_uom": "_skip",
    "uval_target_uom": "target_mt",
    "uval_target": "target_mt",
    "txt_calendar": "period_hint",
    "txt_calendar_year": "year",
    "txt_calendar_month": "month",
}

HUMAN_HEADER_MAP = {
    "region": "zone",
    "zone": "zone",
    "area": "city",
    "city": "city",
    "distributor": "distributor",
    "distributor name": "distributor",
    "dsr": "dsr_name",
    "dsr name": "dsr_name",
    "pop name": "store_name",
    "shop": "store_name",
    "shop name": "store_name",
    "outlet name": "store_name",
    "pop code": "store_id",
    "store id": "store_id",
    "target": "target_mt",
    "target uom": "target_mt",
    "target mt": "target_mt",
    "uom": "target_mt",
}

STOPWORDS = {
    "ks",
    "kiryana",
    "store",
    "stores",
    "general",
    "trader",
    "traders",
    "and",
    "the",
    "super",
    "mart",
    "shop",
    "outlet",
    "co",
    "company",
    "ent",
    "enterprises",
    "enterprise",
}

WHALE_MT = 1.0
TARGET_COLUMNS = [
    "store_id",
    "store_name",
    "distributor",
    "dsr_name",
    "zone",
    "city",
    "section",
    "target_mt",
    "match_method",
    "match_score",
    "source_file",
    "ingested_at",
]


@dataclass
class TargetParseReport:
    strategy: str = "unknown"
    source_file: str = ""
    n_raw_rows: int = 0
    n_clean_rows: int = 0
    n_matched: int = 0
    n_unmatched: int = 0
    matched_mt: float = 0.0
    book_mt: float = 0.0
    warnings: list[str] = field(default_factory=list)


def parse_shop_targets(path: str | Path) -> tuple[pd.DataFrame, TargetParseReport]:
    path = Path(path)
    raw = read_raw_table(path)
    report = TargetParseReport(source_file=str(path), n_raw_rows=int(len(raw)))
    header_row, mapping = _detect_headers(raw)
    if mapping and "target_mt" in mapping.values() and "store_name" in mapping.values():
        report.strategy = "ssrs_or_headers"
        body = raw.iloc[header_row + 1 :].copy()
        projected = pd.DataFrame({role: body[col].values for col, role in mapping.items() if col in body.columns})
    else:
        report.strategy = "positional"
        projected = _positional(raw)
        if projected.empty:
            raise ValueError(
                f"Could not detect shop-wise target columns in {path.name}. "
                "Expected region, area/city, distributor, DSR, shop name, target MT."
            )
    out = _clean_targets(projected)
    out["source_file"] = path.name
    out["ingested_at"] = utcnow()
    report.n_clean_rows = int(len(out))
    report.book_mt = float(pd.to_numeric(out.get("target_mt"), errors="coerce").fillna(0).sum()) if not out.empty else 0.0
    if report.n_clean_rows == 0:
        report.warnings.append("No numeric shop targets in the file.")
    return out, report


def match_shop_targets(targets: pd.DataFrame, stores: pd.DataFrame) -> tuple[pd.DataFrame, TargetParseReport]:
    """Attach store_id where the join is unique and conservative."""
    report = TargetParseReport()
    if targets is None or targets.empty:
        return pd.DataFrame(columns=TARGET_COLUMNS), report
    work = targets.copy()
    for col in ("store_name", "distributor", "dsr_name", "city", "zone", "store_id", "section"):
        if col not in work.columns:
            work[col] = ""
        work[col] = work[col].map(cell_str)
    work["target_mt"] = pd.to_numeric(work.get("target_mt"), errors="coerce")
    work = work.loc[work["target_mt"].notna()].copy()
    report.n_clean_rows = int(len(work))
    report.book_mt = float(work["target_mt"].fillna(0).sum())
    if stores is None or stores.empty:
        work["store_id"] = work["store_id"].where(work["store_id"].map(_looks_like_pop), "")
        work["match_method"] = work["store_id"].map(lambda s: "pop_code" if s else "unmatched")
        work["match_score"] = work["match_method"].map(lambda m: 1.0 if m == "pop_code" else 0.0)
        report.n_unmatched = int((work["match_method"] == "unmatched").sum())
        report.n_matched = int(len(work) - report.n_unmatched)
        report.warnings.append("No universe in the warehouse — targets stored unmatched.")
        return _finalize(work), report

    uni = stores.copy()
    for col in ("store_name", "distributor", "dsr_name", "city", "zone", "store_id", "section"):
        if col not in uni.columns:
            uni[col] = ""
        uni[col] = uni[col].map(cell_str)
    uni = uni.drop_duplicates("store_id")
    uni["_city_k"] = uni["city"].map(fold_place)
    uni["_dist_k"] = uni["distributor"].map(fold_name)
    uni["_dsr_k"] = uni["dsr_name"].map(fold_name)
    uni["_name_k"] = uni["store_name"].map(fold_name)
    uni["_core"] = uni["store_name"].map(core_key)

    work["_city_k"] = work["city"].map(fold_place)
    work["_dist_k"] = work["distributor"].map(fold_name)
    work["_dsr_k"] = work["dsr_name"].map(fold_name)
    work["_name_k"] = work["store_name"].map(fold_name)
    work["_core"] = work["store_name"].map(core_key)

    given_pop = work["store_id"].map(_looks_like_pop)
    live_ids = set(uni["store_id"].astype(str))
    pop_ok = given_pop & work["store_id"].isin(live_ids)

    methods = pd.Series("", index=work.index)
    scores = pd.Series(0.0, index=work.index)
    ids = work["store_id"].where(pop_ok, "")
    methods = methods.mask(pop_ok, "pop_code")
    scores = scores.mask(pop_ok, 1.0)

    pending = work.index[~pop_ok]
    indexes = {
        "city_dist_dsr_name": _unique_index(uni, ["_city_k", "_dist_k", "_dsr_k", "_name_k"]),
        "city_dist_name": _unique_index(uni, ["_city_k", "_dist_k", "_name_k"]),
        "city_name": _unique_index(uni, ["_city_k", "_name_k"]),
        "city_dist_core": _unique_index(uni, ["_city_k", "_dist_k", "_core"]),
        "city_core": _unique_index(uni, ["_city_k", "_core"]),
    }
    ladders = [
        ("city_dist_dsr_name", ["_city_k", "_dist_k", "_dsr_k", "_name_k"], "city_dist_dsr_name", 0.98, False),
        ("city_dist_name", ["_city_k", "_dist_k", "_name_k"], "city_dist_name", 0.92, False),
        ("city_name", ["_city_k", "_name_k"], "city_name", 0.82, True),
        ("city_dist_core", ["_city_k", "_dist_k", "_core"], "city_dist_core", 0.75, True),
        ("city_core", ["_city_k", "_core"], "city_core", 0.65, True),
    ]
    for idx_name, keys, method, score, skip_whale in ladders:
        if pending.empty:
            break
        lookup = indexes[idx_name]
        if not lookup:
            continue
        hit_ids = []
        hit_ok = []
        for i in pending:
            row = work.loc[i]
            if skip_whale and float(row.get("target_mt") or 0) >= WHALE_MT:
                hit_ids.append("")
                hit_ok.append(False)
                continue
            if method.endswith("core") and not row["_core"]:
                hit_ids.append("")
                hit_ok.append(False)
                continue
            key = tuple(str(row[k]) for k in keys)
            sid = lookup.get(key)
            hit_ids.append(sid or "")
            hit_ok.append(bool(sid))
        ok = pd.Series(hit_ok, index=pending)
        if ok.any():
            ids.loc[pending[ok]] = pd.Series(hit_ids, index=pending)[ok]
            methods.loc[pending[ok]] = method
            scores.loc[pending[ok]] = score
            pending = pending[~ok]

    work["store_id"] = ids.astype(str).replace({"nan": "", "None": ""})
    work["match_method"] = methods.where(work["store_id"] != "", "unmatched")
    work["match_score"] = scores.where(work["store_id"] != "", 0.0)
    # Live universe wins for roll-up identity so city/DSR spelling matches scorecards.
    live = uni.set_index("store_id")
    matched_mask = work["store_id"].astype(str).ne("")
    for col in ("city", "distributor", "dsr_name", "zone", "section"):
        if col not in live.columns:
            continue
        mapped = work["store_id"].map(live[col])
        work.loc[matched_mask, col] = mapped.loc[matched_mask].fillna(work.loc[matched_mask, col])
    report.n_matched = int((work["match_method"] != "unmatched").sum())
    report.n_unmatched = int((work["match_method"] == "unmatched").sum())
    report.matched_mt = float(work.loc[work["match_method"] != "unmatched", "target_mt"].sum())
    coverage = (report.matched_mt / report.book_mt) if report.book_mt else 0.0
    if coverage < 0.6 and report.book_mt >= 1:
        report.warnings.append(
            f"Only {coverage:.0%} of plan tons matched to the universe. "
            "Names on the target sheet do not always equal POP names — check Warehouse → Shop plan."
        )
    return _finalize(work), report


def fold_name(value: Any) -> str:
    text = norm_key(value)
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def fold_place(value: Any) -> str:
    text = fold_name(value)
    aliases = {
        "abbotabad": "abbottabad",
        "abbottabad": "abbottabad",
        "rahimyar khan": "rahim yar khan",
        "r y khan": "rahim yar khan",
    }
    return aliases.get(text, text)


def core_key(value: Any) -> str:
    toks = [t for t in fold_name(value).split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(sorted(toks))


def _looks_like_pop(value: Any) -> bool:
    text = cell_str(value).upper().replace(" ", "")
    return bool(re.match(r"^T\d{7,}$", text))


def _unique_index(uni: pd.DataFrame, keys: list[str]) -> dict[tuple[str, ...], str]:
    if uni is None or uni.empty:
        return {}
    work = uni.dropna(subset=keys)
    if work.empty:
        return {}
    grouped = work.groupby(keys, dropna=False)["store_id"].agg(["nunique", "first"])
    grouped = grouped[grouped["nunique"] == 1]
    out = {}
    for key, row in grouped.iterrows():
        if not isinstance(key, tuple):
            key = (key,)
        key_t = tuple("" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v) for v in key)
        if any(not k for k in key_t):
            continue
        out[key_t] = str(row["first"])
    return out


def _detect_headers(raw: pd.DataFrame) -> tuple[int, dict[int, str]]:
    scan = min(12, len(raw))
    best: tuple[int, dict[int, str]] = (0, {})
    for i in range(scan):
        mapping: dict[int, str] = {}
        for col in raw.columns:
            key = norm_key(raw.iat[i, col] if col < raw.shape[1] else "")
            key = key.replace(" ", "_")
            role = FIELD_ID_MAP.get(key)
            if not role:
                human = HUMAN_HEADER_MAP.get(norm_key(raw.iat[i, col] if col < raw.shape[1] else ""))
                role = human
            if role and role != "_skip" and col not in mapping:
                # First numeric target column wins; skip the TARGET_UOM label column.
                if role == "target_mt" and "target_mt" in mapping.values():
                    continue
                mapping[col] = role
        roles = set(mapping.values())
        if "store_name" in roles and "target_mt" in roles and len(roles) >= 4:
            return i, mapping
        if len(roles) > len(best[1]):
            best = (i, mapping)
    return best


def _positional(raw: pd.DataFrame) -> pd.DataFrame:
    """Region, area, distributor, DSR, shop, UOM label, target."""
    if raw is None or raw.empty or raw.shape[1] < 6:
        return pd.DataFrame()
    start = 1 if any("txt_" in cell_str(raw.iat[0, c]).lower() for c in range(min(7, raw.shape[1]))) else 0
    body = raw.iloc[start:]
    cols = list(body.columns)
    projected = pd.DataFrame(
        {
            "zone": body[cols[0]] if len(cols) > 0 else "",
            "city": body[cols[1]] if len(cols) > 1 else "",
            "distributor": body[cols[2]] if len(cols) > 2 else "",
            "dsr_name": body[cols[3]] if len(cols) > 3 else "",
            "store_name": body[cols[4]] if len(cols) > 4 else "",
        }
    )
    # Target is the first numeric column after the name (skip the TARGET_UOM label).
    target = None
    for col in cols[5:]:
        series = pd.to_numeric(body[col], errors="coerce")
        if series.notna().sum() >= max(3, int(0.3 * len(body))):
            target = series
            break
    if target is None:
        return pd.DataFrame()
    projected["target_mt"] = target.values
    return projected


def _clean_targets(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in ("zone", "city", "distributor", "dsr_name", "store_name", "store_id", "section"):
        if col not in work.columns:
            work[col] = ""
        work[col] = work[col].map(cell_str)
    work["target_mt"] = pd.to_numeric(work.get("target_mt"), errors="coerce")
    name = work["store_name"].str.lower().str.strip()
    drop = name.isin({"", "pop name", "shop", "nan", "none", "na", "n/a", "null", "#n/a", "-"})
    drop = drop | name.str.contains("total", na=False)
    label = work["store_name"].str.upper()
    drop = drop | label.eq("TARGET_UOM")
    work = work.loc[~drop & work["target_mt"].notna()].copy()
    return work.reset_index(drop=True)


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in TARGET_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out[TARGET_COLUMNS]
