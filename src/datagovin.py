"""Client for the data.gov.in (OGD Platform India) API.

All datasets on data.gov.in are exposed at::

    https://api.data.gov.in/resource/{resource_id}?api-key=...&format=json

A free API key is required (register at https://data.gov.in/). Pass it via the
``DATAGOVIN_API_KEY`` environment variable or a ``config/api_key.txt`` file.

Usage::

    from datagovin import DataGovIn
    client = DataGovIn()                       # reads key from env/file
    df = client.fetch(resource_id, filters={"State": "Punjab"}, max_records=5000)
    fields = client.fields(resource_id)        # list available columns
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://api.data.gov.in/resource"
LISTS_URL = "https://api.data.gov.in/lists"
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
API_KEY_FILE = CONFIG_DIR / "api_key.txt"


def _load_api_key() -> str:
    key = os.environ.get("DATAGOVIN_API_KEY")
    if key:
        return key.strip()
    if API_KEY_FILE.exists():
        return API_KEY_FILE.read_text().strip()
    raise RuntimeError(
        "No data.gov.in API key found. Set DATAGOVIN_API_KEY or create "
        f"{API_KEY_FILE} with your key (register free at https://data.gov.in/)."
    )


class DataGovIn:
    def __init__(self, api_key: Optional[str] = None, verify_ssl: bool = True):
        self.api_key = api_key or _load_api_key()
        self.verify_ssl = verify_ssl

    # ------------------------------------------------------------------ discovery
    def search(self, keyword: str, max_pages: int = 20, page_size: int = 10) -> List[Dict[str, Any]]:
        """Best-effort keyword search over the resource catalog.

        The catalog endpoint is slow/unreliable, so this paginates the (unfiltered)
        list and filters titles client-side. Returns up to ``max_pages * page_size``
        candidate resources whose title contains ``keyword``.
        """
        out: List[Dict[str, Any]] = []
        kw = keyword.lower()
        offset = 0
        for _ in range(max_pages):
            try:
                r = requests.get(
                    LISTS_URL,
                    params={"api-key": self.api_key, "format": "json",
                            "limit": page_size, "offset": offset},
                    timeout=30, verify=self.verify_ssl,
                )
                r.raise_for_status()
                data = r.json()
            except Exception:
                break
            recs = data.get("records", [])
            if not recs:
                break
            for rec in recs:
                title = (rec.get("title") or "").lower()
                if kw in title:
                    out.append({"index_name": rec.get("index_name"),
                                "title": rec.get("title"),
                                "org": rec.get("org"),
                                "sector": rec.get("sector"),
                                "updated_date": rec.get("updated_date")})
            offset += page_size
            if offset >= int(data.get("total", 0)):
                break
            time.sleep(0.2)
        return out

    def fields(self, resource_id: str) -> List[Dict[str, str]]:
        """Return the field (column) definitions for a resource."""
        data = self._get(resource_id, params={"limit": 1})
        return data.get("fields", [])

    # ------------------------------------------------------------------ fetch
    def fetch(self, resource_id: str, filters: Optional[Dict[str, str]] = None,
              max_records: int = 100000, page_size: int = 100) -> pd.DataFrame:
        """Paginate through a resource and return all records as a DataFrame."""
        records: List[Dict[str, Any]] = []
        offset = 0
        while offset < max_records:
            params: Dict[str, Any] = {
                "api-key": self.api_key, "format": "json",
                "offset": offset, "limit": page_size,
            }
            if filters:
                for field, value in filters.items():
                    params[f"filters[{field}]"] = value
            data = self._get(resource_id, params=params)
            if data.get("status") not in ("ok", None):
                raise RuntimeError(f"API error: {data.get('message')}")
            batch = data.get("records", [])
            if not batch:                       # off-by-one guard on last page
                break
            records.extend(batch)
            total = int(data.get("total", 0) or 0)
            offset += page_size
            if total and offset >= total:
                break
            time.sleep(0.2)
        return pd.DataFrame(records)

    def _get(self, resource_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{BASE_URL}/{resource_id}"
        for attempt in range(3):
            try:
                r = requests.get(url, params=params, timeout=40,
                                 verify=self.verify_ssl)
                r.raise_for_status()
                return r.json()
            except requests.exceptions.SSLError:
                r = requests.get(url, params=params, timeout=40, verify=False)
                r.raise_for_status()
                return r.json()
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")