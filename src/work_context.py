from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from typing import Any

import requests

from src.config import AppConfig


class WorkContextError(RuntimeError):
    pass


WIKIPEDIA_API_TEMPLATE = "https://{language}.wikipedia.org/w/api.php"

CONTENT_TYPE_HINTS = {
    "책": "book novel",
    "소설": "novel book",
    "영화": "film movie",
    "뮤지컬": "musical",
    "연극": "play theatre",
    "논문": "paper research",
    "기사": "article",
}

CONTENT_TYPE_MARKERS = {
    "책": ("책", "book", "novel"),
    "소설": ("소설", "novel", "book"),
    "영화": ("영화", "film", "movie"),
    "뮤지컬": ("뮤지컬", "musical"),
    "연극": ("연극", "play", "theatre", "theater"),
    "논문": ("논문", "paper", "research"),
    "기사": ("기사", "article"),
}

KOREAN_CONTENT_TYPE_HINTS = {
    "책": "책",
    "소설": "소설",
    "영화": "영화",
    "뮤지컬": "뮤지컬",
    "연극": "연극",
    "논문": "논문",
    "기사": "기사",
}

UNHELPFUL_SECTION_HEADINGS = {
    "References",
    "External links",
    "See also",
    "Notes",
    "Further reading",
    "Bibliography",
    "Sources",
}


def fetch_wikipedia_context(
    title: str,
    config: AppConfig,
    *,
    content_type: str = "",
    creator: str = "",
) -> dict[str, Any] | None:
    if not config.wikipedia_context_enabled:
        return None

    title = title.strip()
    if not title:
        return None

    language = _safe_language(config.wikipedia_language)
    api_url = WIKIPEDIA_API_TEMPLATE.format(language=language)
    headers = {"User-Agent": config.wikipedia_user_agent}
    queries = _candidate_queries(
        title,
        content_type=content_type,
        creator=creator,
        language=language,
    )
    if language == "en" and _contains_hangul(title):
        queries = [query for query in queries if not _contains_hangul(query)]

    last_error: Exception | None = None
    for query in queries:
        try:
            search_results = _search_wikipedia(
                api_url,
                query,
                headers=headers,
                timeout=config.wikipedia_timeout_seconds,
            )
            if not search_results:
                continue
            best_result, best_score = _select_best_search_result(
                search_results,
                title=title,
                content_type=content_type,
                creator=creator,
            )
            if best_score < 100:
                continue
            page = _fetch_page_extract(
                api_url,
                int(best_result["pageid"]),
                headers=headers,
                timeout=config.wikipedia_timeout_seconds,
            )
            extract = _clean_extract(str(page.get("extract") or ""))
            if not extract:
                continue
            extract = truncate_text(extract, config.wikipedia_store_max_chars)
            return {
                "source": f"Wikipedia {language.upper()}",
                "language": language,
                "query": query,
                "page_id": int(page.get("pageid") or best_result["pageid"]),
                "title": str(page.get("title") or best_result.get("title") or ""),
                "page_url": str(page.get("fullurl") or ""),
                "description": _clean_snippet(str(best_result.get("snippet") or "")),
                "extract": extract,
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        except (requests.RequestException, ValueError, KeyError) as exc:
            last_error = exc

    if language == "en" and _contains_hangul(title):
        try:
            english_title = _find_english_title_from_korean_wikipedia(
                title,
                config,
                content_type=content_type,
                creator=creator,
            )
            if english_title:
                page = _fetch_page_extract_by_title(
                    api_url,
                    english_title,
                    headers=headers,
                    timeout=config.wikipedia_timeout_seconds,
                )
                extract = _clean_extract(str(page.get("extract") or ""))
                if extract:
                    extract = truncate_text(extract, config.wikipedia_store_max_chars)
                    return {
                        "source": f"Wikipedia {language.upper()}",
                        "language": language,
                        "query": english_title,
                        "page_id": int(page.get("pageid") or 0),
                        "title": str(page.get("title") or english_title),
                        "page_url": str(page.get("fullurl") or ""),
                        "description": "",
                        "extract": extract,
                        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    }
        except (requests.RequestException, ValueError, KeyError) as exc:
            last_error = exc

    if last_error is not None:
        raise WorkContextError(f"Wikipedia 작품 정보를 가져오지 못했습니다: {last_error}") from last_error
    return None


def format_external_context_for_prompt(
    external_context: dict[str, Any] | None,
    *,
    max_chars: int = 7_000,
) -> str:
    if not has_external_context(external_context):
        return ""

    context = external_context or {}
    extract = truncate_text(str(context.get("extract") or ""), max_chars)
    if not extract:
        return ""

    source = str(context.get("source") or "Wikipedia")
    title = str(context.get("title") or "")
    url = str(context.get("page_url") or "")
    fetched_at = str(context.get("fetched_at") or "")
    return f"""
[외부 작품 맥락: {source}]
제목: {title}
출처: {url}
조회 시각: {fetched_at}

주의:
- 이 자료는 작품 전체 정보를 포함할 수 있다.
- 사용자 답변과 감상 상태에서 드러난 범위까지만 질문과 메모에 사용한다.
- 사용자 감상을 대신 쓰지 말고, 작품 식별과 장면/인물/배경 이해를 보조하는 데만 사용한다.

내용:
{extract}
""".strip()


def has_external_context(external_context: dict[str, Any] | None) -> bool:
    if not isinstance(external_context, dict):
        return False
    return bool(str(external_context.get("extract") or "").strip())


def external_context_source_label(external_context: dict[str, Any] | None) -> str:
    if not has_external_context(external_context):
        return ""
    context = external_context or {}
    title = str(context.get("title") or "").strip()
    source = str(context.get("source") or "Wikipedia").strip()
    if title:
        return f"{source} - {title}"
    return source


def truncate_text(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}\n\n[이하 생략]"


def _candidate_queries(
    title: str,
    *,
    content_type: str,
    creator: str,
    language: str = "en",
) -> list[str]:
    if language == "ko":
        hint = KOREAN_CONTENT_TYPE_HINTS.get(content_type.strip(), "")
    else:
        hint = CONTENT_TYPE_HINTS.get(content_type.strip(), "")
    raw_queries = []
    for variant in _title_variants(title):
        raw_queries.extend(
            [
                " ".join(part for part in [variant, creator.strip(), hint] if part).strip(),
                " ".join(part for part in [variant, hint] if part).strip(),
                variant,
            ]
        )
    queries: list[str] = []
    seen: set[str] = set()
    for query in raw_queries:
        normalized = query.lower()
        if query and normalized not in seen:
            seen.add(normalized)
            queries.append(query)
    return queries


def _search_wikipedia(
    api_url: str,
    query: str,
    *,
    headers: dict[str, str],
    timeout: int,
) -> list[dict[str, Any]]:
    response = requests.get(
        api_url,
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srnamespace": 0,
            "srlimit": 5,
            "format": "json",
            "utf8": 1,
        },
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("query", {}).get("search", [])
    return results if isinstance(results, list) else []


def _fetch_page_extract(
    api_url: str,
    page_id: int,
    *,
    headers: dict[str, str],
    timeout: int,
) -> dict[str, Any]:
    response = requests.get(
        api_url,
        params={
            "action": "query",
            "prop": "extracts|info",
            "pageids": page_id,
            "explaintext": 1,
            "exsectionformat": "plain",
            "inprop": "url",
            "format": "json",
        },
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    page = next(iter(payload.get("query", {}).get("pages", {}).values()), {})
    return page if isinstance(page, dict) else {}


def _fetch_page_extract_by_title(
    api_url: str,
    title: str,
    *,
    headers: dict[str, str],
    timeout: int,
) -> dict[str, Any]:
    response = requests.get(
        api_url,
        params={
            "action": "query",
            "prop": "extracts|info",
            "titles": title,
            "explaintext": 1,
            "exsectionformat": "plain",
            "inprop": "url",
            "redirects": 1,
            "format": "json",
        },
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    page = next(iter(payload.get("query", {}).get("pages", {}).values()), {})
    return page if isinstance(page, dict) else {}


def _select_best_search_result(
    results: list[dict[str, Any]],
    *,
    title: str,
    content_type: str,
    creator: str,
) -> tuple[dict[str, Any], float]:
    if not results:
        raise ValueError("Wikipedia 검색 결과가 없습니다.")

    title_norm = _normalize_search_text(title)
    title_variants = _title_variants(title)
    title_compacts = [_normalize_compact_text(variant) for variant in title_variants]
    creator_norm = _normalize_search_text(creator)
    creator_compact = _normalize_compact_text(creator)
    markers = CONTENT_TYPE_MARKERS.get(content_type.strip(), ())

    def score_result(index: int, item: dict[str, Any]) -> float:
        item_title = str(item.get("title") or "")
        snippet = _clean_snippet(str(item.get("snippet") or ""))
        title_text = _normalize_search_text(item_title)
        title_compact = _normalize_compact_text(item_title)
        combined = _normalize_search_text(f"{item_title} {snippet}")
        combined_compact = _normalize_compact_text(f"{item_title} {snippet}")
        score = 100.0 - index
        title_match = _title_match_score(title_compact, title_compacts)
        if title_match <= 0:
            score -= 80.0
        else:
            score += title_match
        if title_norm and title_text == title_norm:
            score += 15.0
        if title_norm and title_text.startswith(title_norm):
            score += 8.0
        for marker in markers:
            marker_norm = _normalize_search_text(marker)
            marker_compact = _normalize_compact_text(marker)
            if marker_norm in title_text:
                score += 30.0
            elif marker_norm in combined:
                score += 10.0
            elif marker_compact and marker_compact in combined_compact:
                score += 8.0
        if creator_norm and creator_norm in combined:
            score += 8.0
        elif creator_compact and creator_compact in combined_compact:
            score += 8.0
        if "disambiguation" in title_text or "may refer to" in combined:
            score -= 50.0
        if title_text.startswith("list of"):
            score -= 80.0
        return score

    best_index, best_item = max(
        enumerate(results),
        key=lambda pair: score_result(pair[0], pair[1]),
    )
    return best_item, score_result(best_index, best_item)


def _find_english_title_from_korean_wikipedia(
    title: str,
    config: AppConfig,
    *,
    content_type: str,
    creator: str,
) -> str:
    api_url = WIKIPEDIA_API_TEMPLATE.format(language="ko")
    headers = {"User-Agent": config.wikipedia_user_agent}
    for query in _candidate_queries(
        title,
        content_type=content_type,
        creator=creator,
        language="ko",
    ):
        results = _search_wikipedia(
            api_url,
            query,
            headers=headers,
            timeout=config.wikipedia_timeout_seconds,
        )
        if not results:
            continue
        best_result, best_score = _select_best_search_result(
            results,
            title=title,
            content_type=content_type,
            creator=creator,
        )
        if best_score < 100:
            continue
        english_title = _fetch_english_langlink(
            api_url,
            int(best_result["pageid"]),
            headers=headers,
            timeout=config.wikipedia_timeout_seconds,
        )
        if english_title:
            return english_title
    return ""


def _fetch_english_langlink(
    api_url: str,
    page_id: int,
    *,
    headers: dict[str, str],
    timeout: int,
) -> str:
    response = requests.get(
        api_url,
        params={
            "action": "query",
            "prop": "langlinks",
            "pageids": page_id,
            "lllang": "en",
            "format": "json",
        },
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    page = next(iter(payload.get("query", {}).get("pages", {}).values()), {})
    links = page.get("langlinks") if isinstance(page, dict) else []
    if not isinstance(links, list) or not links:
        return ""
    return str(links[0].get("*") or links[0].get("title") or "").strip()


def _clean_extract(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped in UNHELPFUL_SECTION_HEADINGS:
            break
        if not stripped and (not cleaned or not cleaned[-1]):
            continue
        cleaned.append(stripped)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()


def _clean_snippet(snippet: str) -> str:
    no_tags = re.sub(r"<[^>]+>", "", snippet)
    return html.unescape(no_tags).strip()


def _safe_language(language: str) -> str:
    normalized = re.sub(r"[^a-z-]+", "", language.strip().lower())
    return normalized or "en"


def _normalize_search_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _normalize_compact_text(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", str(value or "").strip().lower())


def _title_match_score(candidate_compact: str, title_compacts: list[str]) -> float:
    if not candidate_compact:
        return 0.0
    best = 0.0
    for title_compact in title_compacts:
        if not title_compact:
            continue
        if candidate_compact == title_compact:
            best = max(best, 35.0)
        elif candidate_compact.startswith(title_compact):
            best = max(best, 28.0)
        elif title_compact in candidate_compact:
            best = max(best, 22.0)
        elif candidate_compact in title_compact:
            best = max(best, 12.0)
    return best


def _contains_hangul(value: str) -> bool:
    return bool(re.search(r"[가-힣]", value))


def _title_variants(title: str) -> list[str]:
    variants = [title.strip()]
    for match in re.findall(r"\(([^()]+)\)", title):
        stripped = match.strip()
        if stripped:
            variants.append(stripped)
    without_parentheses = re.sub(r"\([^()]+\)", " ", title)
    without_parentheses = re.sub(r"\b\d+\s*부\b", " ", without_parentheses)
    without_parentheses = re.sub(r"\s+", " ", without_parentheses).strip()
    if without_parentheses:
        variants.append(without_parentheses)

    cleaned: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        key = variant.lower()
        if variant and key not in seen:
            seen.add(key)
            cleaned.append(variant)
    return cleaned
