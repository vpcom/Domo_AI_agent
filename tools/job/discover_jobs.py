import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import requests
import yaml

from assistant.config import get_config_path, get_display_path, get_job_search_config
from tools.job.models import JobState
from tools.job.text_normalization import normalize_job_posting_text

REQUEST_TIMEOUT = 30
ROLE_KEYWORDS = [
    "full stack",
    "fullstack",
    "software engineer",
    "backend",
    "frontend",
]
EXPANDED_ROLE_KEYWORDS = ROLE_KEYWORDS + ["engineer", "developer"]
GREENHOUSE_COMPANIES = [
    "airtable",
    "affirm",
    "asana",
    "benchling",
    "brex",
    "canva",
    "checkr",
    "coinbase",
    "datadog",
    "discord",
    "docker",
    "duolingo",
    "fivetran",
    "flexport",
    "gusto",
    "hashicorp",
    "instacart",
    "loom",
    "miro",
    "openphone",
    "pinterest",
    "plaid",
    "reddit",
    "retool",
    "rippling",
    "robinhood",
    "snyk",
    "snowflake",
    "sourcegraph",
    "stripe",
    "webflow",
    "whatnot",
    "ziphq",
]
LEVER_COMPANIES = [
    "applyboard",
    "bitwarden",
    "branch",
    "calendly",
    "carta",
    "circleci",
    "clearco",
    "clari",
    "coursera",
    "drata",
    "eventbrite",
    "gocardless",
    "heap",
    "harness",
    "intercom",
    "mixpanel",
    "mural",
    "pagerduty",
    "palantir",
    "patreon",
    "posthog",
    "postman",
    "procore",
    "quora",
    "ro",
    "samsara",
    "sentry",
    "tripactions",
    "udemy",
    "verkada",
    "vidyard",
    "whatnot",
    "zapier",
]
ASHBY_COMPANIES = [
    "anthropic",
    "arc",
    "baseten",
    "captions",
    "cartesia",
    "character",
    "clay",
    "cognition",
    "cursor",
    "decagon",
    "elevenlabs",
    "exa",
    "fermat",
    "glean",
    "helicone",
    "harvey",
    "mercor",
    "mistral",
    "modal",
    "numeric",
    "openai",
    "perplexity",
    "pinecone",
    "poolside",
    "pylon",
    "reka",
    "runway",
    "stackadapt",
    "twelve-labs",
    "vercel",
    "windsurf",
]


def run(state: JobState) -> None:
    print(f"[discover] start target_folder={state.raw_file.parent}")
    config = load_inputs_config()
    jobs = discover_jobs_from_config(config)
    selected_job = jobs[0]
    print(
        f"[discover] selected company={selected_job['company']} "
        f"title={selected_job['title']} location={selected_job['location']} "
        f"source={selected_job['source']}"
    )
    save_discovered_job(state.raw_file, state.metadata_file, selected_job)


def load_inputs_config(path: str | None = None) -> dict:
    if path:
        with open(path, encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        loaded_from = path
    else:
        config = get_job_search_config()
        loaded_from = get_display_path(get_config_path())

    print(
        f"[discover] loaded config path={loaded_from} "
        f"role={config.get('role')} location={config.get('location')} "
        f"sources={config.get('sources', [])} "
        f"max_jobs={config.get('max_jobs', 1)} "
        f"max_results_per_source={config.get('max_results_per_source', 5)} "
        f"max_company_attempts_per_source={config.get('max_company_attempts_per_source', 'all')}"
    )
    return config


def discover_jobs_from_config(config: dict) -> List[Dict[str, str]]:
    role = config["role"].lower()
    location = config["location"].lower()
    sources = config.get("sources", [])
    max_results_per_source = config.get("max_results_per_source", 5)
    max_jobs = config.get("max_jobs", 1)
    max_company_attempts_per_source = config.get("max_company_attempts_per_source")
    source_companies = build_source_companies(config)

    return discover_jobs(
        role,
        location,
        sources,
        max_results_per_source,
        max_jobs,
        source_companies,
        max_company_attempts_per_source,
    )


def save_discovered_job(raw_file: Path, metadata_file: Path, job: Dict[str, str]) -> None:
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)

    normalized_description = normalize_job_posting_text(job["description"])
    raw_file.write_text(normalized_description, encoding="utf-8")
    job_to_save = dict(job)
    job_to_save["description"] = normalized_description
    with open(metadata_file, "w", encoding="utf-8") as handle:
        json.dump(job_to_save, handle, indent=2)
    print(f"[discover] wrote raw_file={raw_file}")
    print(f"[discover] wrote metadata_file={metadata_file}")


def build_job_folder_path(job: Dict[str, str], root: Path, date_prefix: str) -> Path:
    company = sanitize_path_component(job["company"]) or "Unknown Company"
    title = sanitize_path_component(job["title"]) or "Unknown Role"
    base_name = f"{date_prefix} - {company} - {title}"
    candidate = root / base_name
    suffix = 2

    while candidate.exists():
        candidate = root / f"{base_name} ({suffix})"
        suffix += 1

    return candidate


def discover_jobs(
    role: str,
    location: str,
    sources: List[str],
    max_results_per_source: int,
    max_jobs: int,
    source_companies: Dict[str, List[str]],
    max_company_attempts_per_source: Optional[int],
) -> List[Dict[str, str]]:
    source_handlers: Dict[str, Callable[[str, int], List[Dict[str, str]]]] = {
        "greenhouse": fetch_greenhouse_jobs,
        "lever": fetch_lever_jobs,
        "ashby": fetch_ashby_jobs,
    }
    all_jobs: List[Dict[str, str]] = []
    seen_urls = set()

    print(
        f"[discover] search role={role} location={location} "
        f"sources={sources} max_jobs={max_jobs} "
        f"max_results_per_source={max_results_per_source} "
        f"max_company_attempts_per_source={max_company_attempts_per_source or 'all'}"
    )

    for source_name in sources:
        fetch_jobs = source_handlers.get(source_name)
        if fetch_jobs is None:
            print(f"[discover] skip unknown source={source_name}")
            continue

        companies = source_companies.get(source_name, [])
        if max_company_attempts_per_source is not None:
            companies = companies[:max_company_attempts_per_source]

        if not companies:
            print(f"[discover] skip source={source_name} no companies configured")
            continue

        print(f"[discover] source={source_name} companies={companies}")
        for company in companies:
            try:
                print(f"[discover] searching {source_name}:{company}")
                jobs = fetch_jobs(company, max_results_per_source)
                print(
                    f"[discover] fetched source={source_name} company={company} "
                    f"jobs={len(jobs)}"
                )
            except requests.RequestException as error:
                print(
                    f"[discover] fetch failed source={source_name} "
                    f"company={company} error={error}"
                )
                continue

            for job in jobs:
                candidate_key = build_candidate_key(job)
                if candidate_key in seen_urls:
                    continue

                seen_urls.add(candidate_key)
                all_jobs.append(job)

    print(f"[discover] collected_jobs={len(all_jobs)}")
    if not all_jobs:
        raise ValueError("No jobs fetched from ATS sources")

    ranked_jobs = select_best_jobs(all_jobs, role, location, max_jobs)
    print(f"[discover] candidates found: {len(ranked_jobs)}")
    print(
        f"[discover] selected: {ranked_jobs[0]['title']} "
        f"(score={ranked_jobs[0]['search_score']}, strategy={ranked_jobs[0]['search_strategy']})"
    )
    return ranked_jobs


def build_source_companies(config: dict) -> Dict[str, List[str]]:
    configured_companies = config.get("companies", {})
    defaults = {
        "greenhouse": GREENHOUSE_COMPANIES,
        "lever": LEVER_COMPANIES,
        "ashby": ASHBY_COMPANIES,
    }
    source_companies: Dict[str, List[str]] = {}

    for source_name, default_companies in defaults.items():
        companies = list(configured_companies.get(source_name, [])) + list(default_companies)
        source_companies[source_name] = dedupe_preserve_order(companies)

    print(f"[discover] source_companies={source_companies}")
    return source_companies


def fetch_greenhouse_jobs(company: str, max_results: int) -> List[Dict[str, str]]:
    url = f"https://boards-api.greenhouse.io/v1/boards/{company}/jobs"
    response = requests.get(
        url,
        params={"content": "true"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    jobs = []
    for item in response.json().get("jobs", [])[:max_results]:
        jobs.append(
            {
                "title": item.get("title", ""),
                "company": company,
                "location": item.get("location", {}).get("name", ""),
                "description": item.get("content", "") or "",
                "url": item.get("absolute_url", ""),
                "source": "greenhouse",
            }
        )

    return jobs


def fetch_lever_jobs(company: str, max_results: int) -> List[Dict[str, str]]:
    url = f"https://api.lever.co/v0/postings/{company}?mode=json"
    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    jobs = []
    for item in response.json()[:max_results]:
        jobs.append(
            {
                "title": item.get("text", ""),
                "company": company,
                "location": item.get("categories", {}).get("location", ""),
                "description": item.get("descriptionPlain")
                or item.get("description", "")
                or "",
                "url": item.get("hostedUrl", ""),
                "source": "lever",
            }
        )

    return jobs


def fetch_ashby_jobs(company: str, max_results: int) -> List[Dict[str, str]]:
    url = f"https://api.ashbyhq.com/posting-api/job-board/{company}"
    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    payload = response.json()

    items = payload.get("jobs") or payload.get("jobPostings") or payload.get("openings") or []
    jobs = []
    for item in items[:max_results]:
        location = item.get("location")
        if isinstance(location, dict):
            location = location.get("name", "")

        jobs.append(
            {
                "title": item.get("title", ""),
                "company": company,
                "location": location or "",
                "description": item.get("descriptionPlain")
                or item.get("descriptionHtml")
                or item.get("description")
                or "",
                "url": item.get("jobUrl") or item.get("applyUrl") or item.get("url", ""),
                "source": "ashby",
            }
        )

    return jobs


def select_best_jobs(
    jobs: List[Dict[str, str]],
    role: str,
    location: str,
    max_jobs: int,
) -> List[Dict[str, str]]:
    strategies = [
        ("strict", ROLE_KEYWORDS, True, False),
        ("ignore_location", ROLE_KEYWORDS, False, False),
        ("expanded_role", EXPANDED_ROLE_KEYWORDS, False, False),
        ("remote_only", [], False, True),
    ]
    fallback_errors: List[str] = []

    for strategy_name, role_keywords, require_location, remote_only in strategies:
        candidates = [
            decorate_job_with_score(job, role, location, strategy_name)
            for job in jobs
            if is_relevant(
                job,
                role,
                location,
                role_keywords,
                require_location=require_location,
                remote_only=remote_only,
            )
        ]
        candidates.sort(key=lambda job: job["search_score"], reverse=True)
        print(
            f"[discover] candidates found: {len(candidates)} "
            f"strategy={strategy_name}"
        )
        if candidates:
            return candidates[:max_jobs]

        fallback_errors.append(strategy_name)

    raise ValueError(
        "No matching jobs found after fallbacks: "
        + ", ".join(fallback_errors)
    )


def is_relevant(
    job: Dict[str, str],
    role: str,
    location: str,
    role_keywords: List[str],
    require_location: bool,
    remote_only: bool,
) -> bool:
    title = normalize_text(job["title"])
    loc = normalize_text(job["location"])
    normalized_location = normalize_text(location)
    normalized_role = normalize_text(role)

    if remote_only:
        return "remote" in loc

    role_match = normalized_role in title or any(keyword in title for keyword in role_keywords)
    if not role_match:
        return False

    if not require_location:
        return True

    location_keywords = [normalized_location, "remote", "switzerland"]
    return any(keyword and keyword in loc for keyword in location_keywords)


def score(job: Dict[str, str], role: str, location: str) -> int:
    score_value = 0
    title = normalize_text(job["title"])
    loc = normalize_text(job["location"])
    normalized_role = normalize_text(role)
    normalized_location = normalize_text(location)

    if normalized_role in title:
        score_value += 5

    if "full stack" in title or "fullstack" in title:
        score_value += 3

    if "software engineer" in title:
        score_value += 2

    if "backend" in title:
        score_value += 1

    if "frontend" in title:
        score_value += 1

    if normalized_location and normalized_location in loc:
        score_value += 4

    if "switzerland" in loc:
        score_value += 2

    if "remote" in loc:
        score_value += 2

    if "senior" in title:
        score_value -= 1

    return score_value


def decorate_job_with_score(
    job: Dict[str, str],
    role: str,
    location: str,
    strategy_name: str,
) -> Dict[str, str]:
    scored_job = dict(job)
    scored_job["search_score"] = score(job, role, location)
    scored_job["search_strategy"] = strategy_name
    return scored_job


def build_candidate_key(job: Dict[str, str]) -> str:
    if job.get("url"):
        return job["url"]

    return "|".join(
        [
            job.get("source", ""),
            job.get("company", ""),
            job.get("title", ""),
            job.get("location", ""),
        ]
    )


def normalize_text(value: str) -> str:
    lowered = value.lower().replace("/", " ").replace("-", " ")
    return re.sub(r"\s+", " ", lowered).strip()


def dedupe_preserve_order(values: List[str]) -> List[str]:
    deduped = []
    seen = set()

    for value in values:
        if value in seen:
            continue
        deduped.append(value)
        seen.add(value)

    return deduped


def sanitize_path_component(value: str) -> str:
    collapsed = re.sub(r"\s+", " ", value).strip()
    cleaned = re.sub(r"[^A-Za-z0-9 ._-]", "", collapsed)
    return cleaned[:80].strip(" ._-")
