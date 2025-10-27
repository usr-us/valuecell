import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import aiofiles
import aiohttp
from agno.agent import Agent
from agno.models.google import Gemini
from agno.models.openrouter import OpenRouter
from edgar import Company
from edgar.entity.filings import EntityFilings

from valuecell.utils.path import get_knowledge_path

from .knowledge import insert_md_file_to_knowledge
from .schemas import (
    AShareFilingMetadata,
    AShareFilingResult,
    SECFilingMetadata,
    SECFilingResult,
)


def _ensure_list(value: str | Sequence[str] | None) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _parse_date(d: str | date | None) -> Optional[date]:
    if d is None:
        return None
    if isinstance(d, date):
        return d
    # try common formats
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(d, fmt).date()
        except ValueError:
            continue
    raise ValueError(
        f"Invalid date format: {d}. Expect YYYY-MM-DD, YYYY/MM/DD, or YYYYMMDD."
    )


async def _write_and_ingest(
    filings: Iterable,
    knowledge_dir: Path,
) -> List[SECFilingResult]:
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    results: List[SECFilingResult] = []
    for filing in filings:
        filing_date: str = filing.filing_date.strftime("%Y-%m-%d")
        period_of_report: str = getattr(filing, "period_of_report", "")
        # Convert to markdown; fall back to string if markdown unavailable
        try:
            content: str = filing.document.markdown()
        except Exception:
            try:
                content = str(filing.document)
            except Exception:
                content = ""
        doc_type: str = filing.form
        company_name: str = filing.company

        orig_doc = filing.document.document
        # build stable markdown filename using suffix replacement, keep base name only
        md_doc = Path(orig_doc).with_suffix(".md").name
        file_name = f"{doc_type}_{md_doc}"
        path = knowledge_dir / file_name
        metadata = SECFilingMetadata(
            doc_type=doc_type,
            company=company_name,
            period_of_report=period_of_report,
            filing_date=filing_date,
        )
        async with aiofiles.open(path, "w", encoding="utf-8") as file:
            await file.write(content)

        result = SECFilingResult(file_name, path, metadata)
        results.append(result)

        await insert_md_file_to_knowledge(
            name=file_name, path=path, metadata=metadata.__dict__
        )

    return results


async def fetch_periodic_sec_filings(
    cik_or_ticker: str,
    forms: List[str] | str = "10-Q",
    year: Optional[int | List[int]] = None,
    quarter: Optional[int | List[int]] = None,
    limit: int = 10,
):
    """Fetch periodic SEC filings (10-K/10-Q) and ingest into knowledge.

    - Designed for regular, scheduled reports with filing_date year/quarter filters (edgar API behavior).
    - If year is omitted, fetch latest filings via latest(limit) ordered by filing_date, constrained by forms. If quarter is provided, year must also be provided.

    Date concept guidance:
    - Filing date (filing_date): When the filing was submitted to the SEC. edgar filters by filing_date for year/quarter.
    - Period of report (period_of_report): The reporting period end date covered by the document (fiscal year/quarter-end). It may differ from filing_date.
    - Fiscal vs calendar: Users saying "Q3/FY" usually refer to period_of_report; however, the year/quarter parameters passed to edgar here filter by filing_date.

    Args:
        cik_or_ticker: CIK or ticker symbol (no quotes or backticks).
        forms: "10-K", "10-Q" or a list of these. Defaults to "10-Q".
        year: Single year or list of years to include (by filing_date). When omitted, the tool returns the latest filings using `limit`.
        quarter: Single quarter (1-4) or list of quarters (by filing_date). Requires `year` to be provided.
        limit: When `year` is omitted, number of latest filings to return (by filing_date). Defaults to 10.

    Returns:
        List[SECFilingResult]
    """
    req_forms = set(_ensure_list(forms)) or {"10-Q"}
    company = Company(cik_or_ticker)

    # If year is omitted, use latest(limit). Quarter without year is not supported.
    if year is None:
        if quarter is not None:
            raise ValueError(
                "quarter requires year to be specified for periodic filings"
            )
        filings = company.get_filings(form=list(req_forms)).latest(limit)
        if isinstance(filings, EntityFilings):
            items = list(filings)
        else:
            items = [filings]
        return await _write_and_ingest(items, Path(get_knowledge_path()))

    filings = company.get_filings(form=list(req_forms), year=year, quarter=quarter)

    return await _write_and_ingest(filings, Path(get_knowledge_path()))


async def fetch_event_sec_filings(
    cik_or_ticker: str,
    forms: List[str] | str = "8-K",
    start_date: Optional[str | date] = None,
    end_date: Optional[str | date] = None,
    limit: int = 10,
):
    """Fetch event-driven filings (e.g., 8-K, Forms 3/4/5) with optional date-range and limit.

    Args:
        cik_or_ticker: CIK or ticker symbol (no quotes or backticks).
        forms: One or more of ["8-K", "3", "4", "5"]. Defaults to "8-K".
        start_date: Inclusive start date (YYYY-MM-DD or date).
        end_date: Inclusive end date (YYYY-MM-DD or date).
        limit: Maximum number of filings to fetch after filtering. Defaults to 10.
    (Note: The tool will always ingest written markdown into the knowledge base.)

    Returns:
        List[SECFilingResult]
    """
    sd = _parse_date(start_date)
    ed = _parse_date(end_date)
    if sd and ed and sd > ed:
        raise ValueError("start_date cannot be after end_date")

    req_forms = set(_ensure_list(forms)) or {"8-K"}
    company = Company(cik_or_ticker)

    # If no date range specified, leverage edgar's latest(count) for efficiency
    if not sd and not ed:
        filings = company.get_filings(form=list(req_forms)).latest(limit)
        if isinstance(filings, EntityFilings):
            items = list(filings)
        else:
            items = [filings]
        return await _write_and_ingest(items, Path(get_knowledge_path()))

    # Otherwise, fetch and filter by filing_date range
    filings = company.get_filings(form=list(req_forms))
    if isinstance(filings, EntityFilings):
        items = list(filings)
    else:
        items = [filings]

    filtered: List = []
    for f in items:
        f_date = f.filing_date
        if sd and f_date < sd:
            continue
        if ed and f_date > ed:
            continue
        filtered.append(f)

    # Sort desc and apply limit
    filtered.sort(key=lambda f: f.filing_date, reverse=True)
    if limit is not None and limit > 0:
        filtered = filtered[:limit]

    return await _write_and_ingest(filtered, Path(get_knowledge_path()))


async def web_search(query: str) -> str:
    """Search web for the given query and return a summary of the top results.

    Args:
        query: The search query string.

    Returns:
        A summary of the top search results.
    """

    if os.getenv("WEB_SEARCH_PROVIDER", "google").lower() == "google" and os.getenv(
        "GOOGLE_API_KEY"
    ):
        return await _web_search_google(query)

    model = OpenRouter(id="perplexity/sonar", max_tokens=None)
    response = await Agent(model=model).arun(query)
    return response.content


async def _web_search_google(query: str) -> str:
    """Search Google for the given query and return a summary of the top results.

    Args:
        query: The search query string.

    Returns:
        A summary of the top search results.
    """
    model = Gemini(id="gemini-2.5-flash", search=True)
    response = await Agent(model=model).arun(query)
    return response.content


def _normalize_stock_code(stock_code: str) -> str:
    """Normalize stock code format"""
    # Remove possible prefixes and suffixes, keep only digits
    code = re.sub(r"[^\d]", "", stock_code)
    # Ensure it's a 6-digit number
    if len(code) == 6:
        return code
    elif len(code) < 6:
        return code.zfill(6)
    else:
        return code[:6]


async def _write_and_ingest_a_share(
    filings_data: List[dict],
    knowledge_dir: Path,
) -> List[AShareFilingResult]:
    """Write A-share filing data to files and import to knowledge base"""
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    results: List[AShareFilingResult] = []

    for filing_data in filings_data:
        # Build file name
        stock_code = filing_data["stock_code"]
        doc_type = filing_data["doc_type"]
        period = filing_data["period_of_report"]
        file_name = f"AShare_{stock_code}_{doc_type}_{period}.md"
        path = knowledge_dir / file_name

        # Use complete content returned from _fetch_announcement_content
        content = filing_data.get(
            "content",
            f"""# {filing_data["company"]} ({stock_code}) {doc_type}

## Basic Information
- **Company Name**: {filing_data["company"]}
- **Stock Code**: {stock_code}
- **Exchange**: {filing_data["market"]}
- **Report Type**: {doc_type}
- **Report Period**: {period}
- **Filing Date**: {filing_data["filing_date"]}

## Filing Content
{filing_data.get("announcement_title", "Filing content is being processed...")}

---
*Data Source: CNINFO*
""",
        )

        # Write to file
        async with aiofiles.open(path, "w", encoding="utf-8") as file:
            await file.write(content)

        # Create metadata
        metadata = AShareFilingMetadata(
            doc_type=doc_type,
            company=filing_data["company"],
            stock_code=stock_code,
            market=filing_data["market"],
            period_of_report=period,
            filing_date=filing_data["filing_date"],
        )

        # Create result object
        result = AShareFilingResult(file_name, path, metadata)
        results.append(result)

        # Import to knowledge base
        await insert_md_file_to_knowledge(
            name=file_name, path=path, metadata=metadata.__dict__
        )

    return results


async def _fetch_cninfo_data(
    stock_code: str, report_types: List[str], years: List[int], limit: int
) -> List[dict]:
    """Fetch real A-share filing data from CNINFO API

    Args:
        stock_code: Normalized stock code
        report_types: List of report types
        years: List of years
        limit: Maximum number of records to fetch

    Returns:
        List[dict]: List of filing data
    """

    # CNINFO API configuration
    base_url = "http://www.cninfo.com.cn/new/hisAnnouncement/query"

    # Request headers configuration
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Connection": "keep-alive",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Host": "www.cninfo.com.cn",
        "Origin": "http://www.cninfo.com.cn",
        "Referer": "http://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search&lastPage=index",
        "X-Requested-With": "XMLHttpRequest",
    }

    # Report type mapping (supports both English and Chinese for backward compatibility)
    category_mapping = {
        "annual": "category_ndbg_szsh",
        "semi-annual": "category_bndbg_szsh",
        "quarterly": "category_sjdbg_szsh",
    }

    # Determine exchange
    column = "szse" if stock_code.startswith(("000", "002", "300")) else "sse"

    filings_data = []
    current_year = datetime.now().year
    target_years = (
        years if years else [current_year, current_year - 1, current_year - 2]
    )

    async with aiohttp.ClientSession() as session:
        for report_type in report_types:
            if len(filings_data) >= limit:
                break

            category = category_mapping.get(report_type, "category_ndbg_szsh")

            # Build time range
            for target_year in target_years:
                if len(filings_data) >= limit:
                    break

                # Set search time range
                start_date = f"{target_year}-01-01"
                end_date = f"{target_year + 1}-01-01"
                se_date = f"{start_date}~{end_date}"

                # Build request parameters
                # Build orgId based on stock code
                if stock_code.startswith(("000", "002", "300")):
                    # SZSE stocks
                    org_id = f"gssz{stock_code.zfill(7)}"  # Pad to 7 digits
                    plate = "sz"
                else:
                    # SSE stocks
                    org_id = f"gssh{stock_code.zfill(7)}"  # Pad to 7 digits
                    plate = "sh"

                form_data = {
                    "pageNum": "1",
                    "pageSize": "30",
                    "column": column,
                    "tabName": "fulltext",
                    "plate": plate,
                    "stock": f"{stock_code},{org_id}",
                    "searchkey": "",
                    "secid": "",
                    "category": f"{category};",
                    "trade": "",
                    "seDate": se_date,
                    "sortName": "",
                    "sortType": "",
                    "isHLtitle": "true",
                }

                try:
                    async with session.post(
                        base_url, headers=headers, data=form_data
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            announcements = result.get("announcements", [])

                            if announcements is None:
                                continue

                            for announcement in announcements:
                                if len(filings_data) >= limit:
                                    break

                                # Extract filing information
                                filing_info = {
                                    "stock_code": announcement.get(
                                        "secCode", stock_code
                                    ),
                                    "company": announcement.get("secName", ""),
                                    "market": "SZSE" if column == "szse" else "SSE",
                                    "doc_type": report_type,
                                    "period_of_report": f"{target_year}",
                                    "filing_date": announcement.get("adjunctUrl", "")[
                                        10:20
                                    ]
                                    if announcement.get("adjunctUrl")
                                    else f"{target_year}-04-30",
                                    "announcement_id": announcement.get(
                                        "announcementId", ""
                                    ),
                                    "announcement_title": announcement.get(
                                        "announcementTitle", ""
                                    ),
                                    "org_id": announcement.get("orgId", ""),
                                    "content": "",  # Will fetch detailed content in subsequent steps
                                }

                                # Fetch detailed content
                                content = await _fetch_announcement_content(
                                    session, filing_info
                                )
                                filing_info["content"] = content

                                filings_data.append(filing_info)

                except Exception as e:
                    print(
                        f"Error fetching {stock_code} {report_type} {target_year} data: {e}"
                    )
                    continue

    return filings_data


async def _fetch_announcement_content(
    session: aiohttp.ClientSession, filing_info: dict
) -> str:
    """Fetch detailed content of announcement

    Args:
        session: aiohttp session
        filing_info: Filing information dictionary

    Returns:
        str: Announcement content
    """
    try:
        # CNINFO announcement detail API
        detail_url = "http://www.cninfo.com.cn/new/announcement/bulletin_detail"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        }

        params = {
            "announceId": filing_info.get("announcement_id", ""),
            "flag": "true",
            "announceTime": filing_info.get("filing_date", ""),
        }

        async with session.post(detail_url, headers=headers, params=params) as response:
            if response.status == 200:
                result = await response.json()

                # Build filing content
                content = f"""# {filing_info["company"]} ({filing_info["stock_code"]}) {filing_info["doc_type"]}

## Basic Information
- **Company Name**: {filing_info["company"]}
- **Stock Code**: {filing_info["stock_code"]}
- **Exchange**: {filing_info["market"]}
- **Report Type**: {filing_info["doc_type"]}
- **Report Period**: {filing_info["period_of_report"]}
- **Filing Date**: {filing_info["filing_date"]}

## Filing Content

{filing_info.get("announcement_title", "")}

## Financial Data
*Note: Detailed financial data needs to be extracted from PDF files, basic information is shown here*

PDF File Link: {result.get("fileUrl", "Not available")}

---
*Data Source: CNINFO*
"""
                return content

    except Exception as e:
        print(f"Error fetching announcement details: {e}")

    # Return basic content
    return f"""# {filing_info["company"]} ({filing_info["stock_code"]}) {filing_info["doc_type"]}

## Basic Information
- **Company Name**: {filing_info["company"]}
- **Stock Code**: {filing_info["stock_code"]}
- **Exchange**: {filing_info["market"]}
- **Report Type**: {filing_info["doc_type"]}
- **Report Period**: {filing_info["period_of_report"]}
- **Filing Date**: {filing_info["filing_date"]}

## Filing Content

{filing_info.get("announcement_title", "")}

---
*Data Source: CNINFO*
"""


async def fetch_a_share_filings(
    stock_code: str,
    report_types: List[str] | str = "annual",
    year: Optional[int | List[int]] = None,
    limit: int = 10,
) -> List[AShareFilingResult]:
    """Fetch A-share filing data from CNINFO and import to knowledge base

    Args:
        stock_code: Stock code (e.g.: 000001, 600036, etc.)
        report_types: Report types, options: "annual", "semi-annual", "quarterly" or Chinese "年报", "半年报", "季报". Default is "annual"
        year: Year filter, can be a single year or list of years. If not provided, fetch latest reports
        limit: Maximum number of records to fetch, default 10

    Returns:
        List[AShareFilingResult]: List of A-share filing results

    Examples:
        # Fetch latest annual report of Ping An Bank
        await fetch_a_share_filings("000001", "annual", limit=1)

        # Fetch 2023 annual and semi-annual reports of Kweichow Moutai
        await fetch_a_share_filings("600519", ["annual", "semi-annual"], year=2023)
    """

    # Normalize stock code
    normalized_code = _normalize_stock_code(stock_code)

    # Normalize report types
    report_types_list = _ensure_list(report_types)
    if not report_types_list:
        report_types_list = ["annual"]

    # Normalize years
    years_list = []
    if year is not None:
        if isinstance(year, int):
            years_list = [year]
        else:
            years_list = list(year)

    # Fetch real data from CNINFO
    filings_data = await _fetch_cninfo_data(
        normalized_code, report_types_list, years_list, limit
    )

    # Write to files and import to knowledge base
    knowledge_dir = Path(get_knowledge_path())
    return await _write_and_ingest_a_share(filings_data, knowledge_dir)
