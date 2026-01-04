"""AI Safety Newsletter Generator - Streamlit App"""
import streamlit as st

from api import (
    generate_newsletter,
    get_all_articles,
    get_models_for_provider,
    TIME_FRAMES,
    PROVIDERS,
    DEFAULT_MODELS,
)
from constants import DEFAULT_ORGS_INPUT

st.set_page_config(
    page_title="AI Safety Newsletter Generator",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI Safety Newsletter Generator")

st.markdown("""
Generate a curated newsletter covering **AI safety**, **security concerns**, 
and **existential risk** developments. Tracks major AI labs, safety research 
organizations, and regulatory bodies.
""")

st.divider()

# Settings row 1: Time period and Provider
col_time, col_provider = st.columns(2)

with col_time:
    time_frame_label = st.selectbox(
        "Time period:",
        options=list(TIME_FRAMES.keys()),
        index=0,
        help="Select how far back to search for news"
    )
time_frame = TIME_FRAMES[time_frame_label]

with col_provider:
    provider_label = st.selectbox(
        "AI Provider:",
        options=list(PROVIDERS.keys()),
        index=1,  # Default to Groq (free)
        help="OpenAI (paid) or Groq (free, uses Llama/Mixtral)"
    )
provider = PROVIDERS[provider_label]

# Get models for selected provider
available_models = get_models_for_provider(provider)
model_names = list(available_models.keys())
model_labels = list(available_models.values())
default_model = DEFAULT_MODELS[provider]

# Model selection
col_check, _ = st.columns([1, 1])
with col_check:
    same_model = st.checkbox("Use same model for both tasks", value=True)

if same_model:
    col_single, _ = st.columns([1, 1])
    with col_single:
        selected_idx = st.selectbox(
            "Model:",
            options=range(len(model_names)),
            format_func=lambda i: model_labels[i],
            index=model_names.index(default_model) if default_model in model_names else 0,
            help="Model used for both filtering and newsletter generation"
        )
    filter_model = model_names[selected_idx]
    generation_model = model_names[selected_idx]
else:
    col_filter, col_gen = st.columns(2)
    with col_filter:
        filter_idx = st.selectbox(
            "Filter model:",
            options=range(len(model_names)),
            format_func=lambda i: model_labels[i],
            index=model_names.index(default_model) if default_model in model_names else 0,
            help="Model for filtering/reducing articles"
        )
    with col_gen:
        gen_idx = st.selectbox(
            "Generation model:",
            options=range(len(model_names)),
            format_func=lambda i: model_labels[i],
            index=model_names.index(default_model) if default_model in model_names else 0,
            help="Model for writing the newsletter"
        )
    filter_model = model_names[filter_idx]
    generation_model = model_names[gen_idx]

st.divider()

orgs_input = st.text_area(
    "Organizations & Search Terms (comma-separated):",
    value=DEFAULT_ORGS_INPUT,
    height=100,
    help="Enter AI companies, research labs, or search terms to track"
)

col1, col2 = st.columns([1, 4])
with col1:
    generate_btn = st.button("🚀 Generate Newsletter", type="primary")

if generate_btn:
    search_terms = [t.strip() for t in orgs_input.split(",") if t.strip()]
    
    if not search_terms:
        st.error("Please enter at least one search term.")
    else:
        # Show selected configuration
        provider_icon = "🆓" if provider == "groq" else "💰"
        if same_model:
            st.info(f"{provider_icon} **{provider_label}** | Model: **{filter_model}**")
        else:
            st.info(f"{provider_icon} **{provider_label}** | Filter: **{filter_model}** | Gen: **{generation_model}**")
        
        with st.spinner(f"🔍 Fetching AI safety news ({time_frame_label.lower()})..."):
            articles = get_all_articles(search_terms, time_frame, provider, filter_model)
        
        if not articles:
            st.warning("No relevant articles found. Try different search terms or time period.")
        else:
            # Count total sources across all stories
            total_sources = sum(len(s.get("sources", [])) for s in articles)
            st.success(f"Found {len(articles)} unique stories ({total_sources} total sources)")
            
            with st.spinner("✍️ Generating newsletter..."):
                newsletter = generate_newsletter(articles, time_frame, provider, generation_model)
            
            st.divider()
            
            # Show stories in expander
            with st.expander(f"📰 Source Stories ({len(articles)})", expanded=False):
                for story in articles:
                    sources = story.get("sources", [])
                    if sources:
                        links = " | ".join(f"[{s['source']}]({s['url']})" for s in sources)
                        st.markdown(f"- **{story['title']}** — {links}")
                    else:
                        # Fallback for old format
                        st.markdown(f"- [{story['title']}]({story.get('url', '#')})")
            
            # Show newsletter
            st.subheader("📧 Generated Newsletter")
            st.markdown(newsletter)
            
            # Download button
            st.download_button(
                label="📋 Download Newsletter",
                data=newsletter,
                file_name="ai_safety_newsletter.md",
                mime="text/markdown"
            )
