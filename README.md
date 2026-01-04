# AI Safety Newsletter Generator

A Streamlit app that generates curated newsletters focused on **AI safety**, **security concerns**, and **existential risk** developments.

## What It Does

1. **Searches** for news from major AI labs (OpenAI, Anthropic, DeepMind, etc.) and safety organizations
2. **Filters** articles using 60+ AI safety related keywords
3. **Reduces** articles via GPT to select the most important 8-15 stories
4. **Generates** a formatted newsletter with executive summary and categorized sections

## Newsletter Sections

- 📋 **Executive Summary** — 30-second overview of key developments
- 🚨 **Critical Developments** — Breaking news with immediate safety implications
- 📊 **Top Stories** — Most significant developments
- 🔬 **Safety & Alignment Research** — Technical safety work
- ⚠️ **Security & Threats** — Misuse, vulnerabilities
- 🏛️ **Governance & Regulation** — Policy developments
- 💡 **Industry Developments** — Company news with safety relevance

## Setup

1. Create virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create `.streamlit/secrets.toml`:
   ```toml
   OPENAI_API_KEY = "your-openai-key"
   SERPER_API_KEY = "your-serper-key"
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

## API Keys Required

- **OpenAI API Key** — For GPT-4o (article filtering and newsletter generation)
- **Serper API Key** — For Google News search ([serper.dev](https://serper.dev))

## Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit UI |
| `api.py` | News fetching, filtering, and newsletter generation |
| `constants.py` | AI organizations list and safety keywords |
