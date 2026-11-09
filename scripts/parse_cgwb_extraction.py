"""Parse the CGWB 'National Compilation on Dynamic Ground Water Resources' PDF
into data/raw/district_extraction.csv.

Usage:
    python scripts/parse_cgwb_extraction.py <path_to_pdf>
    python scripts/parse_cgwb_extraction.py <path_to_pdf> --out data/raw/district_extraction.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import pdfplumber

ROOT = Path(__file__).resolve().parents[1]

ALIASES = {
    "Cuddapah": "Y.S.R Kadapa", "Jehanabad": "JAHANABAD", "Purnia": "Purnea",
    "Janjgir - Champa": "JANJGIR-CHAMP", "Dang": "DANGS", "Sonipat": "SONEPAT(RTK)",
    "Rajauri": "RAJOURI", "Kodarma": "KODERMA", "Palamau": "Palamu",
    "Bangalore Rural": "Bengaluru (Rural)", "Bangalore Urban": "Bengaluru (Urban)",
    "Belgaum": "Belagavi", "Chamarajanagara": "CHAMARAJANAGA",
    "Chikkamagaluru": "CHIKMAGALUR", "Dakshin Kannada": "Dakshina Kannada",
    "Davanagere": "DAVANGERE", "Shivamogga": "SHIMOGA", "Uttara Kannada": "UTTAR KANNADA",
    "Kannur": "CANNUR", "Kasaragod": "KASARGOD", "Buldana": "BULDHANA",
    "Sindudurg": "SINDHUDURG", "Anugul": "ANGUL", "Balangir": "BOLANGIR",
    "Jagatsinghapur": "JAGATSINGHAPU", "Jajapur": "JAJPUR",
    "Nabarangapur": "NAWARANGPUR", "Sonapur": "Subarnapur", "Mukatsar": "MUKTSAR",
    "Sbs Nagar": "SBS Nagar", "Kancheepuram": "KANCHIPURAM",
    "Ramanathapuram": "RAMANATHAPURA", "Tiruchirappalli": "TIRUCHIRAPPAL",
    "Tiruppur": "TIRUPUR", "Tiruvannamalai": "TIRUVANNAMALA",
    "Mahbubnagar": "MAHABUBNAGAR", "Budaun": "BADAUN",
    "Jyotiba Phule Nagar": "Amroha", "Raibareli": "RAE BARELI",
    "Sant Kabir Nagar": "SANT KABIR NGR", "Siddharthnagar": "SIDDHARTH NGR",
    "Udham Singh Nagar": "UDHAM SINGH N", "Darjiling": "DARJEELING",
    "Maldah": "MALDA", "Puruliya": "PURULIA", "Fatehgarh Sahib": "FATEHGARH SAH",
    "Ashok Nagar": "ASHOKNAGAR(GNA", "Pratapgarh": "PRATAPGARH(CHT",
    "Cuddapah": "Y.S.R Kadapa", "Gaurella Pendra Marwahi": "GAURELA-PENDRA-MARWAHI",
    "Ganganagar": "SRI GANGANAGAR", "Praygraj": "Prayagraj",
    "Anantapur": "Ananthapuramu", "Nellore": "Sri Potti Sriramulu Nellore",
    "Bangalore Rural": "Bengaluru (Rural)", "Bangalore Urban": "Bengaluru (Urban)",
    "Belgaum": "Belagavi", "Bijapur": "Vijayapura", "Mysore": "Mysuru",
    "Trichur": "Thrissur", "Mumbai City": "Mumbai", "Mumbai Suburban": "Mumbai Sub",
    "Baleshwar": "Balasore", "Baudh": "Boudh", "Debagarh": "Deogarh",
    "Kendujhar": "Keonjhar", "Khordha": "Khurda", "Sonapur": "Subarnapur",
    "Ferozepur": "Firozpur", "Mohali": "SAS Nagar", "Ropar": "Rupnagar",
    "Nilgiris": "The Nilgiris", "Tuticorin": "Thoothukudi",
    "Warangal Rural": "Warangal", "Warangal Urban": "Warangal",
    "Allahabad": "Prayagraj", "Kheri": "Lakhimpur Kheri", "Mau": "Maunath Bhanjan",
    "Cooch Behar": "Koch Bihar", "East Medinipur": "Purba Medinipur",
    "Howrah": "Haora", "North Dinajpur": "Uttar Dinajpur",
    "South Dinajpur": "Dakshin Dinajpur", "West Medinipur": "Paschim Medinipur",
    "Dholpur": "Dhaulpur", "Gurugram": "Gurgaon", "Nuh": "Mewat",
    "North Goa": "Goa North", "Pashchimi Singhbhum": "West Singhbhum",
    "Purbi Singhbhum": "East Singhbhum", "Koriya": "Korea",
    "Narsimhapur": "Narsinghpur", "Dohad": "Dahod",
    "Pashchim Champaran": "West Champaran", "Purba Champaran": "East Champaran",
    "Purnia": "Purnea", "Jehanabad": "JAHANABAD", "Purnia": "Purnea",
}


def norm(s):
    return re.sub(r"[^A-Z]", "", str(s).upper())


def to_num(x):
    if x is None:
        return None
    s = str(x).replace(",", "").replace("\n", " ").strip()
    s2 = re.sub(r"[^0-9.\-]", "", s)
    if s2 in ("", ".", "-", "-."):
        return None
    try:
        return float(s2)
    except ValueError:
        return None


def is_district_row(row):
    if not row or row[0] is None:
        return False
    s = str(row[0]).strip()
    return s.isdigit()


def parse_pdf(pdf_path):
    inv_aliases = {v: k for k, v in ALIASES.items()}
    dist = pd.read_csv(ROOT / "data/processed/district_metrics.csv")
    rows_out = []
    with pdfplumber.open(pdf_path) as pdf:
        for pg in pdf.pages:
            for t in (pg.extract_tables() or []):
                if not t or len(t) < 8:
                    continue
                if len(t[0]) != 16:
                    continue
                flat = " ".join(str(c) for row in t for c in (row or []) if c).lower()
                if "name of district" not in flat and "stage of" not in flat:
                    continue
                state = None
                for r in t[:4]:
                    cells = [c for c in (r or []) if c]
                    if len(cells) == 1 and len(cells[0].strip()) > 2 and not cells[0].strip().isdigit():
                        txt = cells[0].strip().upper()
                        if not any(k in txt for k in ["DYNAMIC", "INDIA", "RESOURCES", "GROUND WATER", "2025", "STATE-WISE", "STATEWISE", "IN BCM"]):
                            state = cells[0].strip().title()
                            break
                if state is None:
                    continue
                for r in t:
                    if not is_district_row(r) or len(r) < 16:
                        continue
                    name = (r[1] or "").replace("\n", " ").strip()
                    if not name or name.lower().startswith("total"):
                        continue
                    # match to our district
                    cand = dist[dist.state == state]
                    k = norm(name)
                    matched_name = None
                    a = inv_aliases.get(name)
                    if a:
                        for d in cand.district:
                            if norm(a) == norm(d):
                                matched_name = d
                                break
                    if matched_name is None:
                        for d in cand.district:
                            if norm(d) == k:
                                matched_name = d
                                break
                    if matched_name is None:
                        keys = [norm(d) for d in cand.district]
                        mm = difflib.get_close_matches(k, keys, n=1, cutoff=0.85)
                        if mm:
                            idx = keys.index(mm[0])
                            matched_name = list(cand.district)[idx]
                    if matched_name is None:
                        continue
                    rows_out.append({
                        "state": state,
                        "district": matched_name,
                        "recharge_ham": to_num(r[6]),
                        "extractable_ham": to_num(r[8]),
                        "extraction_ham": to_num(r[12]),
                        "extraction_irrigation_ham": to_num(r[9]),
                        "extraction_industrial_ham": to_num(r[10]),
                        "extraction_domestic_ham": to_num(r[11]),
                        "stage_pct": to_num(r[15]),
                        "net_availability_ham": to_num(r[14]),
                    })
    return pd.DataFrame(rows_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="path to CGWB National Compilation PDF")
    ap.add_argument("--out", default=str(ROOT / "data" / "raw" / "district_extraction.csv"))
    args = ap.parse_args()

    df = parse_pdf(args.pdf)
    df = df.drop_duplicates(subset=["state", "district"], keep="first")

    def cat(p):
        if pd.isna(p):
            return None
        if p > 100:
            return "Over-exploited"
        if p >= 90:
            return "Critical"
        if p >= 70:
            return "Semi-critical"
        return "Safe"

    df["category_cgwb"] = df["stage_pct"].apply(cat)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} districts -> {out_path}")


if __name__ == "__main__":
    main()