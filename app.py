import os
import json
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import requests
import streamlit as st
from collections import defaultdict
import random
import re

# --------- Config & Secrets ---------
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", os.getenv("NEWS_API_KEY", ""))
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
EMAIL_SENDER = st.secrets.get("EMAIL_SENDER", os.getenv("EMAIL_SENDER", ""))
EMAIL_APP_PASSWORD = st.secrets.get("EMAIL_APP_PASSWORD", os.getenv("EMAIL_APP_PASSWORD", ""))

NEWS_API_URL = "https://newsapi.org/v2/everything"

# --------- Source filters ---------
SOURCE_DOMAINS = {
    "TechCrunch": "techcrunch.com",
    "Axios": "axios.com",
    "WSJ": "wsj.com",
    "New York Times": "nytimes.com",
    "Wired": "wired.com",
    "The Verge": "theverge.com",
    "MIT Technology Review": "technologyreview.com",
    "Forbes": "forbes.com",
    "Bloomberg": "bloomberg.com",
    "Financial Times": "ft.com",
    "Business Insider": "businessinsider.com",
    "VentureBeat": "venturebeat.com",
    "CNBC": "cnbc.com",
    "Reuters": "reuters.com",
    "BBC News": "bbc.com",
    "The Guardian": "theguardian.com",
    "Nature": "nature.com",
    "Scientific American": "scientificamerican.com",
    "Ars Technica": "arstechnica.com"
}

domains = ",".join(SOURCE_DOMAINS)

url = (
    f"https://newsapi.org/v2/everything?"
    f"q=AI&"
    f"language=en&"
    f"domains={domains}&"
    f"sortBy=publishedAt&"
    f"apiKey={NEWS_API_KEY}"
)

response = requests.get(url)
data = response.json()

# Old Code
# def format_newsletter(articles):
#     email_body = "<h1>📰 Your AI News Digest</h1>"
#     for article in articles:
#         email_body += f"""
#         <h2>{article['title']}</h2>
#         <p><strong>Summary:</strong> {article['description']}</p>
#         <p><strong>Tag:</strong> {article['tag']}</p>
#         <p><a href="{article['url']}">Read more</a></p>
#         <hr>
#         """
#     return email_body

# def send_email(recipients, articles):
#     from_email = "nikhila.isukapally@gmail.com"  # Your sending email
#     password = "cciy vdnd ukuw okjc"  # Use an app password (not your normal password)

#     # Compose email
#     msg = MIMEMultipart('alternative')
#     msg['Subject'] = "Your AI News Digest"
#     msg['From'] = from_email
#     msg['To'] = ", ".join(recipients)

#     # Attach newsletter content
#     html_content = format_newsletter(articles)
#     msg.attach(MIMEText(html_content, 'html'))

#     # Send via Gmail SMTP
#     with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
#         server.login(from_email, password)
#         server.sendmail(from_email, recipients, msg.as_string())

# def fetch_news(query):
#     url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&apiKey={NEWS_API_KEY}"
#     response = requests.get(url)
#     data = response.json()
#     return data['articles'][:3]  # Limit to 3 articles for demo
    

# def summarize_article(article):
#     # Few-shot examples for prompting
#     few_shot_example = """
#     Example Article:
#     Title: AI Revolutionizes Healthcare
#     Description: A new AI model helps diagnose diseases faster and more accurately.
#     Output:
#     {
#       "title": "AI Revolutionizes Healthcare",
#       "summary": "AI accelerates disease diagnosis with improved accuracy.",
#       "why_it_matters": "Enhances patient care and reduces diagnostic errors.",
#       "tag": "AI & Healthtech"
#     }
#     """

#     # Construct the prompt
#     prompt = f"""
#     {few_shot_example}

#     Now summarize this article:
#     Title: {article['title']}
#     Description: {article['description']}
#     Tag: {article['tag']}

#     Provide output in the same JSON format.
#     """

#     # OpenAI call
#     response = openai.Completion.create(
#         engine="gpt-4o",
#         prompt=prompt,
#         max_tokens=300,
#         temperature=0.7
#     )

#     # Extract JSON output from response
#     return response.choices[0].text.strip()


# # Streamlit UI
# st.title("GenAI News Summarizer")

# tags = [
#     "AI & Infrastructure",
#     "AI & Wearables",
#     "AI & Healthtech",
#     "Agentic AI",
#     "Product Innovation & Technology",
#     "Sustainability & Energy in AI",
#     "Technology & AI Efficiency",
#     "Governance & Security",
#     "AI Innovation & Enterprise Solutions",
#     "AI & Emerging Technologies"
# ]

# selected_tags = st.multiselect("Select AI Topics:", tags)

# # Store selected articles
# selected_articles = []

# if selected_tags:
#     for tag in selected_tags:
#         st.subheader(f"News for: {tag}")
#         articles = fetch_news(tag)
#         for idx, article in enumerate(articles):
#             # Display article info
#             st.write(f"**{article['title']}**")
#             st.write(article['description'])
#             st.write(article['url'])
#             # Add checkbox
#             selected = st.checkbox("Select this article", key=f"{tag}_{idx}")
#             if selected:
#                 selected_articles.append({
#                     "title": article['title'],
#                     "description": article['description'],
#                     "url": article['url'],
#                     "tag": tag
#                 })
#             st.write("---")

# # Display selected articles summary
# if selected_articles:
#     st.subheader("Selected Articles for Newsletter:")
#     for article in selected_articles:
#         st.write(f"**{article['title']}**")
#         st.write(article['description'])
#         st.write(article['url'])
#         st.write(f"*Tag: {article['tag']}*")
#         st.write("---")

#     # Email input
#     st.subheader("Send Newsletter")
#     emails = st.text_input("Enter recipient email addresses (comma-separated):")

#     # Send button
#     if st.button("Send Newsletter"):
#         if emails:
#             recipient_list = [email.strip() for email in emails.split(",")]

#             # Call the email sending function (to be implemented below)
#             send_email(recipient_list, selected_articles)

#             st.success(f"Newsletter sent to: {', '.join(recipient_list)}")
#         else:
#             st.error("Please enter at least one valid email address.")



#new Code


# Optional: map UI tags to tighter NewsAPI queries
TAG_QUERY = {
    "AI & Infrastructure": '"artificial intelligence" AND (infrastructure OR datacenter OR "data center" OR GPUs OR chips OR NVIDIA OR AMD OR Intel OR ARM OR TSMC OR cloud OR hyperscale OR networking OR "AI supercomputer")',

    "AI & Wearables": '("AI" OR "artificial intelligence") AND (wearable OR smartwatch OR smart ring OR "health sensor" OR fitness tracker OR biosensor OR "smart glasses" OR "Apple Watch" OR Fitbit OR Oura)',

    "AI & Healthtech": '("AI" OR "artificial intelligence") AND (health OR hospital OR clinical OR diagnostics OR radiology OR healthtech OR telemedicine OR "drug discovery" OR genomics OR biotech OR "patient monitoring")',

    "Agentic AI": '(agentic OR "AI agents" OR "autonomous agents" OR tooluse OR "tool use" OR multi-agent OR orchestration OR planner OR executor OR "AI workflows")',

    "Product Innovation & Technology": '("AI" OR "machine learning") AND (product OR feature OR launch OR roadmap OR platform OR prototype OR innovation OR "proof of concept" OR beta OR release OR integration)',

    "Sustainability & Energy in AI": '("AI" OR datacenter OR "data center") AND (energy OR sustainability OR PUE OR emissions OR efficiency OR carbon OR renewable OR "green AI" OR climate OR solar OR wind OR recycling)',

    "Technology & AI Efficiency": '("AI" OR "machine learning") AND (efficiency OR optimization OR throughput OR latency OR inference OR scalability OR acceleration OR "low power" OR performance OR tuning)',

    "Governance & Security": '("AI" OR model OR LLM) AND (governance OR policy OR safety OR security OR compliance OR watermarking OR regulation OR legislation OR copyright OR risk OR trust OR bias OR fairness OR audit)',

    "AI Innovation & Enterprise Solutions": '("AI" OR "machine learning") AND (enterprise OR SaaS OR adoption OR deployment OR productivity OR workflow OR automation OR ERP OR CRM OR "business process" OR ROI)',

    "AI & Emerging Technologies": '("AI" OR "machine learning") AND (quantum OR robotics OR AR OR VR OR edge OR on-device OR 5G OR IoT OR blockchain OR metaverse OR nanotech OR "brain-computer interface")',
}


# --------- Utilities ---------
def build_query(tag: str) -> str:
    return TAG_QUERY.get(tag, tag)

def _normalize_article(a, tag):
    return {
        "title": a.get("title") or "Untitled",
        "description": a.get("description") or (a.get("content") or "")[:240],
        "url": a.get("url"),
        "source": (a.get("source") or {}).get("name"),
        "publishedAt": a.get("publishedAt"),
        "tag": tag,
        "image_url": a.get("urlToImage"),
    }

def _dedupe(articles):
    seen = set()
    out = []
    for a in articles:
        key = (str(a.get("url") or "") or "").strip().lower() or (a["title"].strip().lower())
        if key and key not in seen:
            seen.add(key)
            out.append(a)
    return out

def _iso_to_dt(art):
    s = art.get("publishedAt")
    try:
        return datetime.fromisoformat((s or "").replace("Z", "+00:00"))
    except Exception:
        return datetime.min  # puts unknown dates at the end

def diversify_articles(articles, max_per_source=2, limit=None):
    # group by source
    grouped = defaultdict(list)
    for a in articles:
        src = (a.get("source") or "unknown").lower()
        grouped[src].append(a)

    # deterministically sort within each source by date desc
    for src in grouped:
        grouped[src] = sorted(grouped[src], key=_iso_to_dt, reverse=True)

    # deterministic source order (alphabetical)
    diversified = []
    for src in sorted(grouped.keys()):
        diversified.extend(grouped[src][:max_per_source])

    # final deterministic sort by date desc
    diversified = sorted(diversified, key=_iso_to_dt, reverse=True)

    # apply overall limit only if provided
    if limit is not None:
        return diversified[:limit]
    return diversified


@st.cache_data(ttl=900)  # cache for 15 minutes
def fetch_news(tag: str, page_size: int = 10, lookback_days: int = 21, domains: tuple[str, ...] | None = None):
    q = build_query(tag)
    params = {
        "q": q,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "from": (datetime.utcnow() - timedelta(days=lookback_days)).date().isoformat(),
        "searchIn": "title,description,content",
        "apiKey": NEWS_API_KEY,
    }
    if domains:
        params["domains"] = ",".join(domains)

    r = requests.get(NEWS_API_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    articles = [_normalize_article(a, tag) for a in data.get("articles", []) if a.get("url")]
    # only keep articles where title includes "AI"
    # articles = [a for a in articles if "AI" in (a.get("title") or "").lower()]
    articles = [
    a for a in articles
    if re.search(r"\bAI\b", (a.get("title") or ""), flags=re.IGNORECASE)]
    articles = _dedupe(articles)
    articles = diversify_articles(articles, max_per_source=2, limit=None)
    return articles

    # return _dedupe(articles)[:3]


def filter_already_seen(articles, seen_urls: set):
    """Remove articles whose URL already appeared in previous categories."""
    unique = []
    for a in articles:
        url = (a.get("url") or "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        unique.append(a)
    return unique


def summarize_article_openai(article: dict) -> dict:
    """
    Optional: Summarize an article into:
      {title, summary, why_it_matters, tag}
    Uses OpenAI Chat Completions. If no OPENAI_API_KEY, returns pass-through.
    """
    if not OPENAI_API_KEY:
        # graceful fallback: return the original with no 'why it matters'
        return {
            "title": article["title"],
            "summary": article["description"] or "",
            "why_it_matters": None,
            "tag": article["tag"],
        }

    # Using the "openai" SDK v1 style
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        few_shot = {
            "title": "AI Revolutionizes Healthcare",
            "summary": "AI accelerates disease diagnosis with improved accuracy.",
            "why_it_matters": "Enhances patient care and reduces diagnostic errors.",
            "tag": "AI & Healthtech"
        }
        user_payload = {
            "title": article["title"],
            "description": article["description"] or "",
            "tag": article["tag"]
        }

        messages = [
            {"role": "system", "content": "You produce short, factual JSON summaries for an AI newsletter."},
            {"role": "user", "content": f"Example Output:\n{json.dumps(few_shot)}"},
            {"role": "user", "content": f"Summarize in the SAME JSON keys:\n{json.dumps(user_payload)}"},
        ]

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
            max_tokens=250,
        )
        text = resp.choices[0].message.content.strip()
        start = text.find("{")
        end = text.rfind("}")
        parsed = json.loads(text[start:end+1])
        # Ensure keys exist
        return {
            "title": parsed.get("title", article["title"]),
            "summary": parsed.get("summary", article["description"] or ""),
            "why_it_matters": parsed.get("why_it_matters"),
            "tag": parsed.get("tag", article["tag"]),
        }
    except Exception as e:
        # On any error, gracefully fall back
        return {
            "title": article["title"],
            "summary": article["description"] or "",
            "why_it_matters": None,
            "tag": article["tag"],
        }

def _pairs(seq, n=2):
    """Yield items in chunks of n (for 2-column rows)."""
    chunk = []
    for x in seq:
        chunk.append(x)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def format_newsletter(articles):
    # Image dimensions for email cards
    IMAGE_W = 300
    IMAGE_H = 200

    # Outer container table (email-safe)
    html = [
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        'style="border-collapse:collapse;font-family:Arial, Helvetica, sans-serif;color:#222;">',
        '<tr><td style="padding:16px 0;"><h1 style="margin:0;font-size:22px;">📰 Your AI News Digest</h1></td></tr>'
    ]

    # Render two cards per row
    for row in _pairs(articles, 2):
        html.append('<tr>')

        for a in row:
            img   = a.get("image_url")
            tag   = a.get("tag") or ""
            title = a.get("title") or "Untitled"
            desc  = a.get("summary") or a.get("description") or ""
            why   = a.get("why_it_matters")
            url   = a.get("url") or "#"

            html.append(
                f'''<td width="50%" valign="top" style="padding:12px;">
                      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" 
                             style="border-collapse:collapse;border:1px solid #eee;border-radius:8px;overflow:hidden;">
                        <tr>
                          <td style="padding:0;">
                            {f'<img src="{img}" alt="" style="display:block;width:{IMAGE_W}px;height:{IMAGE_H}px;object-fit:cover;border-bottom:1px solid #eee;" />' if img else ''}
                          </td>
                        </tr>
                        <tr>
                          <td style="padding:12px 14px;">
                            {f'<div style="font-size:12px;color:#3b5bfd;margin:0 0 6px 0;">{tag}</div>' if tag else ''}
                            <h3 style="margin:0 0 10px 0;font-size:18px;line-height:1.3;">{title}</h3>
                            <p style="margin:0 0 10px 0;font-size:14px;line-height:1.5;">
                              <strong>Summary:</strong> {desc}
                            </p>
                            {f'<p style="margin:0 0 12px 0;font-size:14px;line-height:1.5;"><strong>Why it matters:</strong> {why}</p>' if why else ''}
                            <p style="margin:0;font-size:14px;">
                              <a href="{url}" style="color:#1a73e8;text-decoration:none;">Read More</a>
                            </p>
                          </td>
                        </tr>
                      </table>
                    </td>'''
            )

        # If odd number of articles, add an empty cell to keep layout clean
        if len(row) == 1:
            html.append('<td width="50%" style="padding:12px;"></td>')

        html.append('</tr>')

    html.append('</table>')
    return ''.join(html)


def send_email(recipients, articles):
    """
    Sends HTML email via Gmail SMTP using app password.
    """
    if not EMAIL_SENDER or not EMAIL_APP_PASSWORD:
        raise RuntimeError("Email credentials missing. Set EMAIL_SENDER and EMAIL_APP_PASSWORD in Streamlit secrets.")

    msg = MIMEMultipart('alternative')
    msg['Subject'] = "Your AI News Digest"
    msg['From'] = EMAIL_SENDER
    msg['To'] = ", ".join(recipients)

    html_content = format_newsletter(articles)
    msg.attach(MIMEText(html_content, 'html'))

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(EMAIL_SENDER, EMAIL_APP_PASSWORD)
        server.sendmail(EMAIL_SENDER, recipients, msg.as_string())

# --------- Streamlit UI ---------
st.set_page_config(page_title="GenAI News Summarizer", page_icon="📰")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Noto Sans', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📰 GenAI News Summarizer")

# # Secrets status UI (small helper)
# with st.expander("⚙️ Configuration status"):
#     st.write(f"NEWS_API_KEY set: {'✅' if NEWS_API_KEY else '❌'}")
#     st.write(f"OPENAI_API_KEY set (optional): {'✅' if OPENAI_API_KEY else '❌'}")
#     st.write(f"EMAIL_SENDER set: {'✅' if EMAIL_SENDER else '❌'}")
#     st.write(f"EMAIL_APP_PASSWORD set: {'✅' if EMAIL_APP_PASSWORD else '❌'}")
#     st.caption("Tip: put these in .streamlit/secrets.toml")

tags = [
    "AI & Infrastructure",
    "AI & Wearables",
    "AI & Healthtech",
    "Agentic AI",
    "Product Innovation & Technology",
    "Sustainability & Energy in AI",
    "Technology & AI Efficiency",
    "Governance & Security",
    "AI Innovation & Enterprise Solutions",
    "AI & Emerging Technologies"
]

selected_tags = st.multiselect("Select AI Topics:", tags)

if not NEWS_API_KEY:
    st.warning("Add your NEWS_API_KEY to proceed.")
    st.stop()

# Let user pick trusted sources
source_names = list(SOURCE_DOMAINS.keys())
selected_sources = st.multiselect(
    "Limit to trusted sources (domains):",
    source_names,
    default=source_names  # preselect all four; change as you like
)
if not selected_sources:
    st.warning("Please select at least one trusted source to fetch news.")
    st.stop()

selected_domains = [SOURCE_DOMAINS[name] for name in selected_sources]


# Store selected articles in session state
if "selected_articles" not in st.session_state:
    st.session_state.selected_articles = []

# === NEW: track seen URLs across categories ===
seen_urls = set()

if selected_tags:
    for tag in selected_tags:
        st.subheader(f"News for: {tag}")
        try:
            # Fetch more articles so filtering still leaves enough variety
            raw_articles = fetch_news(tag, domains=tuple(selected_domains), page_size=20)

            # Drop any articles already shown in previous categories
            articles = filter_already_seen(raw_articles, seen_urls)[:3]  # keep top 3 per tag after filtering

            if not articles:
                st.info("No fresh, unique articles found for this tag.")
                continue

            # for idx, article in enumerate(articles):
            #     st.write(f"**{article['title']}**")

            #     if article.get("source") or article.get("publishedAt"):
            #         # format date if available
            #         pub_date_str = ""
            #         if article.get("publishedAt"):
            #             try:
            #                 pub_date = datetime.fromisoformat(article["publishedAt"].replace("Z", "+00:00"))
            #                 pub_date_str = pub_date.strftime("%b %d, %Y")  # e.g. "Sep 22, 2025"
            #             except Exception:
            #                 pub_date_str = article["publishedAt"]

            #         st.caption(f"Source: {article.get('source', 'Unknown')} • Published: {pub_date_str}")


            #     st.write(article.get("description", ""))
            #     st.write(article.get("url", ""))

            #     # Optional: show image if NewsAPI provides one
            #     if article.get("image_url"):
            #         st.image(article["image_url"], use_container_width=True)

            #     selected = st.checkbox("Select this article", key=f"{tag}_{idx}")

            #     if selected:
            #         # Avoid duplicates in selected_articles
            #         if article["url"] not in [a.get("url") for a in st.session_state.selected_articles]:
            #             st.session_state.selected_articles.append(article)

            #     st.write("---")
            for idx, article in enumerate(articles):
                col1, col2 = st.columns(2, gap="large")

                url = article.get("url")
                # Stable key tied to URL so it persists across reruns
                ckey = f"pick_{tag}_{abs(hash(url))}"

                # Initialize checkbox state from selection basket (only once)
                if ckey not in st.session_state:
                    st.session_state[ckey] = any(url == a.get("url") for a in st.session_state.selected_articles)

                with col1:
                    st.subheader(article.get("title", "Untitled"))

                    # Pretty date
                    pub_date_str = ""
                    if article.get("publishedAt"):
                        try:
                            pub_date = datetime.fromisoformat(article["publishedAt"].replace("Z", "+00:00"))
                            pub_date_str = pub_date.strftime("%b %d, %Y")
                        except Exception:
                            pub_date_str = article["publishedAt"]

                    if article.get("source") or pub_date_str:
                        st.caption(f"Source: {article.get('source', 'Unknown')} • Published: {pub_date_str or 'Unknown date'}")

                    st.write(article.get("description", ""))
                    if url:
                        st.markdown(f"[Read more]({url})")

                    checked = st.checkbox("Select this article", key=ckey)

                    # Sync checkbox with the selection basket (add/remove)
                    in_basket = any(url == a.get("url") for a in st.session_state.selected_articles)
                    if checked and not in_basket and url:
                        st.session_state.selected_articles.append(article)
                    elif (not checked) and in_basket:
                        st.session_state.selected_articles = [a for a in st.session_state.selected_articles if a.get("url") != url]

                with col2:
                    if article.get("image_url"):
                        st.image(article["image_url"], use_container_width=True)

                st.write("---")


        except requests.HTTPError as http_err:
            st.error(f"NewsAPI error: {http_err}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")


# # Show selected articles & allow summarization
# if st.session_state.selected_articles:
#     st.subheader("Selected Articles for Newsletter")
#     for a in st.session_state.selected_articles:
#         st.write(f"**{a['title']}**")
#         st.write(a.get("description") or a.get("summary") or "")
#         st.write(a['url'])
#         st.write(f"*Tag: {a['tag']}*")
#         st.write("---")
# Show how many articles were selected
if st.session_state.selected_articles:
    st.success(f"✅ You have selected {len(st.session_state.selected_articles)} articles.")


    col1, col2 = st.columns(2)
    with col1:
        # if st.button("🧠 Generate Summaries (OpenAI)"):
        #     updated = []
        #     for art in st.session_state.selected_articles:
        #         s = summarize_article_openai(art)
        #         updated.append({
        #             **art,
        #             "title": s.get("title", art["title"]),
        #             "summary": s.get("summary", art.get("description")),
        #             "why_it_matters": s.get("why_it_matters"),
        #         })
        #     st.session_state.selected_articles = updated
        #     st.success("Summaries generated.")
        if st.button("🗑️ Clear Selected"):
            st.session_state.selected_articles = []
            st.info("Selection cleared.")

    with col2:

    # Email UI
        st.subheader("Send Newsletter")
        emails = st.text_input("Recipient emails (comma-separated)")

    if st.button("📧 Send Newsletter"):
        if not emails.strip():
            st.error("Please enter at least one recipient email.")
        else:
            recipients = [e.strip() for e in emails.split(",") if e.strip()]
            try:
                send_email(recipients, st.session_state.selected_articles)
                st.success(f"Newsletter sent to: {', '.join(recipients)}")
            except Exception as e:
                st.error(f"Email failed: {e}")

