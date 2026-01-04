"""
AI Safety Newsletter Generator API
Fetches, filters, and generates newsletters focused on AI safety and security concerns.
"""
import json
import re
import requests
import streamlit as st
from datetime import datetime
from dateutil.parser import isoparse
from openai import OpenAI
from typing import Any

from constants import AI_SAFETY_TERMS, AI_ORGANIZATIONS

# Initialize clients
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
groq_client = OpenAI(
    api_key=st.secrets["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)

SERPER_URL = "https://google.serper.dev/news"
MAX_PAGES = 5  # Maximum pages to fetch per query

# Provider options
PROVIDERS = {
    "OpenAI": "openai",
    "Groq (Free)": "groq",
}

# Models by provider
OPENAI_MODELS = {
    "gpt-4o-mini": "GPT-4o Mini — Fast & cheap ($0.15/1M in)",
    "gpt-4o": "GPT-4o — Best quality ($2.50/1M in)",
    "gpt-4-turbo": "GPT-4 Turbo — High quality ($10/1M in)",
}

GROQ_MODELS = {
    "llama-3.3-70b-versatile": "Llama 3.3 70B — Best quality (free)",
    "llama-3.1-8b-instant": "Llama 3.1 8B — Fast (free)",
    "mixtral-8x7b-32768": "Mixtral 8x7B — Good balance (free)",
    "gemma2-9b-it": "Gemma 2 9B — Fast (free)",
}

# Default models per provider
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "groq": "llama-3.3-70b-versatile",
}

# Time frame options for news search
TIME_FRAMES = {
    "Past 24 hours": "qdr:d",
    "Past week": "qdr:w",
    "Past month": "qdr:m",
}


def get_models_for_provider(provider: str) -> dict[str, str]:
    """Return available models for the given provider."""
    if provider == "groq":
        return GROQ_MODELS
    return OPENAI_MODELS


def get_client(provider: str) -> OpenAI:
    """Return the appropriate client for the provider."""
    if provider == "groq":
        return groq_client
    return openai_client


def log(message: str) -> None:
    """Print debug message to terminal."""
    print(f"[AI-Safety] {message}")


def is_ai_safety_related(title: str, description: str) -> bool:
    """Check if article content matches AI safety related terms."""
    text = f"{title or ''} {description or ''}".lower()
    return any(term in text for term in AI_SAFETY_TERMS)


def fetch_news_from_serper(query: str, time_frame: str = "qdr:d") -> list[dict[str, Any]]:
    """
    Fetch ALL news articles from Serper API using pagination.
    Returns filtered list of article dictionaries.
    """
    all_articles = []
    headers = {
        "X-API-KEY": st.secrets["SERPER_API_KEY"],
        "Content-Type": "application/json"
    }

    for page in range(1, MAX_PAGES + 1):
        payload = json.dumps({
            "q": query,
            "location": "United States",
            "page": page,
            "tbs": time_frame
        })

        response = requests.post(SERPER_URL, headers=headers, data=payload, timeout=30)
        if response.status_code != 200:
            log(f"ERROR: Serper API failed for '{query}' page {page}: {response.status_code}")
            break

        raw_articles = response.json().get("news", [])
        if not raw_articles:
            break  # No more results

        # Filter and format this page
        filtered = _filter_and_format_articles(raw_articles)
        all_articles.extend(filtered)

        log(f"'{query}' page {page}: {len(raw_articles)} raw -> {len(filtered)} relevant")

        # If we got fewer than expected, probably no more pages
        if len(raw_articles) < 10:
            break

    return all_articles


def _filter_and_format_articles(articles: list[dict]) -> list[dict[str, Any]]:
    """Filter articles for AI safety relevance and format them."""
    filtered = []
    for article in articles:
        title = article.get("title", "")
        description = article.get("snippet", "")

        if not is_ai_safety_related(title, description):
            continue

        pub_date = _parse_date(article.get("date", ""))
        filtered.append({
            "title": title,
            "url": article.get("link", ""),
            "description": description,
            "publishedAt": pub_date.isoformat() if pub_date else "",
            "source": article.get("source", "")
        })
    return filtered


def _parse_date(date_str: str):
    """Safely parse ISO date string."""
    try:
        return isoparse(date_str)
    except (ValueError, TypeError):
        return None


def _chunk_list(items: list, chunk_size: int):
    """Yield successive chunks from a list."""
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def _format_articles_for_prompt(articles: list[dict]) -> str:
    """Format articles/stories into a text block for LLM prompts."""
    lines = []
    for i, art in enumerate(articles, start=1):
        # Handle both article format (source string) and story format (sources array)
        if 'sources' in art and isinstance(art['sources'], list):
            # Story format with multiple sources
            sources_text = ", ".join(
                f"{s.get('source', 'Unknown')} ({s.get('url', '')})"
                for s in art['sources']
            )
            lines.append(f"""
[{i}] "{art.get('title', '')}"
Sources: {sources_text}
Published: {art.get('publishedAt', '')}
Summary: {art.get('description', '')}""")
        else:
            # Original article format
            lines.append(f"""
[{i}] "{art.get('title', '')}"
Source: {art.get('source', '')} | Published: {art.get('publishedAt', '')}
Summary: {art.get('description', '')}
Link: {art.get('url', '')}""")
    return "\n".join(lines)


def _format_stories_for_prompt(stories: list[dict]) -> str:
    """Format stories (with multiple sources) for newsletter prompt."""
    lines = []
    for i, story in enumerate(stories, start=1):
        sources_text = " | ".join(
            f"[{s['source']}]({s['url']})" for s in story.get("sources", [])
        )
        lines.append(f"""
[{i}] "{story['title']}"
Summary: {story['description']}
Published: {story.get('publishedAt', '')}
Sources: {sources_text}""")
    return "\n".join(lines)


def reduce_articles_batch(
        items: list[dict],
        provider: str = "openai",
        model: str | None = None
) -> list[dict]:
    """Use LLM to filter, prioritize, and merge articles/stories."""
    if model is None:
        model = DEFAULT_MODELS[provider]

    client = get_client(provider)
    items_text = _format_articles_for_prompt(items)

    # Detect if we're processing stories (already have sources array) or articles
    is_stories = any('sources' in item for item in items)
    item_type = "stories" if is_stories else "articles"

    prompt = f"""You filter and merge news {item_type} for an AI safety newsletter.

TASK: 
1. Group {item_type} about the SAME story/event together
2. Select 10-20 most important UNIQUE stories about AI safety
3. For each story, combine ALL source links from merged items

MERGE RULES:
- Items about the same event = ONE story with combined sources
- Example: "Grok generates harmful images" from multiple sources = ONE story
- Pick the best title and description, combine ALL source URLs

OUTPUT FORMAT - JSON array of story objects:
[
  {{
    "title": "Best headline for the story",
    "description": "Best 1-2 sentence summary",
    "publishedAt": "2024-01-03T...",
    "sources": [
      {{"source": "Reuters", "url": "https://..."}},
      {{"source": "CNBC", "url": "https://..."}}
    ]
  }}
]

PRIORITIZE stories about:
- AI safety research, alignment, existential risk
- AI security threats, vulnerabilities, misuse
- AI governance, regulation, policy
- Major AI capabilities with safety implications
- AI ethics, bias, societal impact

EXCLUDE:
- Routine product announcements without safety implications
- Clickbait or low-quality sources
- Items only tangentially related to AI

{item_type.upper()} TO PROCESS:
{items_text}

OUTPUT: Valid JSON array only, no other text."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You merge news items about same events. Output valid JSON only."},
            {"role": "user", "content": prompt}
        ]
    )

    result = _extract_json_array(response.choices[0].message.content)
    log(f"[{provider}/{model}] Merged {len(items)} {item_type} -> {len(result)} stories")
    return result


def _extract_json_array(content: str) -> list[dict]:
    """Extract JSON array from LLM response."""
    content = content.strip()
    match = re.search(r"\[.*]", content, re.DOTALL)
    if not match:
        raise ValueError("No JSON array found in response")
    return json.loads(match.group(0))


def get_all_articles(
        search_terms: list[str],
        time_frame: str = "qdr:d",
        provider: str = "openai",
        filter_model: str | None = None
) -> list[dict]:
    """
    Fetch articles for all search terms, deduplicate, and reduce via LLM.
    """
    if filter_model is None:
        filter_model = DEFAULT_MODELS[provider]

    log(f"Starting search with {len(search_terms)} terms (timeframe: {time_frame})...")
    log(f"Filter: {provider}/{filter_model}")
    all_articles = []

    # Fetch from each search term (with pagination)
    for term in search_terms:
        all_articles.extend(fetch_news_from_serper(term, time_frame))

    # Add general AI safety searches
    safety_queries = ["AI safety", "AI risk", "AI regulation", "AI alignment"]
    log(f"Adding {len(safety_queries)} safety-specific queries...")
    for query in safety_queries:
        all_articles.extend(fetch_news_from_serper(query, time_frame))

    log(f"Total collected articles: {len(all_articles)}")

    # Deduplicate by URL first (exact duplicates)
    unique = _deduplicate_by_url(all_articles)
    log(f"After URL dedup: {len(unique)} unique articles")

    if not unique:
        log("No articles found!")
        return []

    # Reduce in chunks via LLM (handles semantic merging)
    chunk_size = 30
    if len(unique) <= chunk_size:
        merged = reduce_articles_batch(unique, provider, filter_model)
    else:
        merged = []
        chunks = list(_chunk_list(unique, chunk_size))
        log(f"Processing {len(chunks)} chunks...")
        for i, chunk in enumerate(chunks, 1):
            log(f"Processing chunk {i}/{len(chunks)}...")
            merged.extend(reduce_articles_batch(chunk, provider, filter_model))

    # Final merge pass if we had multiple chunks
    if len(merged) > 25:
        log(f"Final merge pass on {len(merged)} stories...")
        merged = reduce_articles_batch(merged, provider, filter_model)

    log(f"Final story count: {len(merged)}")
    return merged


def _deduplicate_by_url(articles: list[dict]) -> list[dict]:
    """Remove duplicate articles based on URL."""
    seen = set()
    unique = []
    for article in articles:
        url = article.get("url", "").lower().strip()
        if url and url not in seen:
            seen.add(url)
            unique.append(article)
    return unique


def _get_period_label(time_frame: str) -> str:
    """Get human-readable label for the time period."""
    labels = {
        "qdr:d": "today",
        "qdr:w": "this week",
        "qdr:m": "this month",
    }
    return labels.get(time_frame, "recently")


def create_newsletter_prompt(stories: list[dict], time_frame: str = "qdr:d") -> str:
    """Create the prompt for generating the final newsletter."""
    stories.sort(key=lambda x: x.get("publishedAt", ""), reverse=True)
    stories_text = _format_stories_for_prompt(stories)
    orgs_text = ", ".join(AI_ORGANIZATIONS[:15])

    current_date = datetime.now().strftime("%B %d, %Y")
    period_label = _get_period_label(time_frame)

    # Number stories for tracking
    story_numbers = ", ".join(f"[{i+1}]" for i in range(len(stories)))

    return f"""You are an expert AI safety newsletter writer.

TODAY'S DATE: {current_date}
PERIOD: {period_label}

TASK: Create a newsletter from {len(stories)} stories. Each story has an ID number.

⚠️ CRITICAL - NO DUPLICATES:
- You have {len(stories)} stories numbered {story_numbers}
- Each story ID can appear in ONLY ONE section of the newsletter
- Once you use story [3] in "Critical Developments", you CANNOT use it again in any other section
- Before writing each section, check which story IDs you already used
- DIFFERENT HEADLINES about the SAME TOPIC (e.g., "Grok controversy") = SAME story, use only once

PROCESS:
1. First, identify which stories are about the SAME underlying event
2. Group them and pick the BEST section for each unique event
3. Write the newsletter using each unique event only ONCE

EXACT FORMAT FOR EACH STORY (copy this format precisely):
- **Headline Here** — Summary sentence here. Sources: [Source1](url1) | [Source2](url2)

RULES:
- Start each story with "- **" (dash, space, double asterisk)
- NO brackets around the headline
- Headline in bold, then " — " (space, em dash, space), then summary
- End with ". Sources: " then linked sources separated by " | "
- One blank line between stories

NEWSLETTER TEMPLATE:

---

**Subject:** [AI Safety Weekly] {current_date} — Top headline teaser here

**Preview:** Brief 100-140 character summary of the top stories.

---

## 📋 Executive Summary

Write 3-4 sentences summarizing key themes. Do not list individual stories.

---

## 🚨 Critical Developments

- **Story Headline** — One to two sentence summary of the story. Sources: [Source1](url) | [Source2](url)

---

## 📊 Top Stories

- **Story Headline** — One to two sentence summary of the story. Sources: [Source1](url) | [Source2](url)

- **Story Headline** — One to two sentence summary of the story. Sources: [Source1](url)

- **Story Headline** — One to two sentence summary of the story. Sources: [Source1](url) | [Source2](url)

---

## 🔬 Safety & Alignment Research

- **Story Headline** — Summary. Sources: [Source1](url)

- **Story Headline** — Summary. Sources: [Source1](url) | [Source2](url)

---

## ⚠️ Security & Threats

- **Story Headline** — Summary. Sources: [Source1](url)

---

## 🏛️ Governance & Regulation

- **Story Headline** — Summary. Sources: [Source1](url) | [Source2](url)

---

## 💡 Industry Developments

- **Story Headline** — Summary. Sources: [Source1](url)

---

PRIORITY ORGANIZATIONS: {orgs_text}

STORIES TO USE (each ID appears ONCE in newsletter):
{stories_text}

OUTPUT: Newsletter only. Each story appears in exactly ONE section."""


def _clean_newsletter_formatting(content: str) -> str:
    """Fix common formatting issues in generated newsletter."""
    lines = content.split('\n')
    cleaned = []

    for line in lines:
        # Fix broken bold formatting in story lines
        if line.strip().startswith('- **') or line.strip().startswith('**'):
            # Remove any ** that appear after the headline start but before —
            # Find the em dash
            if ' — ' in line:
                parts = line.split(' — ', 1)
                headline_part = parts[0]
                rest = parts[1] if len(parts) > 1 else ''

                # Clean headline: should be "- **Title**" or "**Title**"
                # Remove stray ** from inside headline
                if headline_part.startswith('- **'):
                    prefix = '- **'
                    title = headline_part[4:]
                elif headline_part.startswith('**'):
                    prefix = '**'
                    title = headline_part[2:]
                else:
                    prefix = ''
                    title = headline_part

                # Remove any ** from inside the title
                title = title.replace('**', '')
                # Remove brackets around title
                title = title.strip('[]')

                # Reconstruct
                line = f"{prefix}{title}** — {rest}"

        cleaned.append(line)

    return '\n'.join(cleaned)


def generate_newsletter(
        stories: list[dict],
        time_frame: str = "qdr:d",
        provider: str = "openai",
        model: str | None = None
) -> str:
    """Generate the final newsletter using specified provider and model."""
    if model is None:
        model = DEFAULT_MODELS[provider]

    client = get_client(provider)
    log(f"Generating newsletter from {len(stories)} stories...")
    log(f"Generation: {provider}/{model}")
    prompt = create_newsletter_prompt(stories, time_frame)

    system_msg = f"""You write AI safety newsletters with STRICT formatting:

FORMAT each story EXACTLY like this:
- **Headline Here** — Summary sentence here. Sources: [Source1](url) | [Source2](url)

RULES:
1. Start with "- **" (dash space asterisks)
2. Headline text, then closing "**", then " — "
3. NO asterisks or brackets inside the headline text
4. One blank line between each story
5. Each of the {len(stories)} stories appears in only ONE section
6. Same event = ONE entry (combine sources)"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ]
    )

    raw_content = response.choices[0].message.content
    cleaned = _clean_newsletter_formatting(raw_content)
    log("Newsletter generated and cleaned successfully!")
    return cleaned
