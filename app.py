import json
import re
from typing import Dict, List, Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components


# ----------------------------
# App / Theme Configuration
# ----------------------------
st.set_page_config(
    page_title="ArchitectAI — System Design Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark purple custom theme styling
st.markdown(
    """
    <style>
    :root {
        --bg: #0f0a1f;
        --card: #1a1233;
        --muted: #b9a9e8;
        --text: #f3edff;
        --accent: #8b5cf6;
        --accent2: #a78bfa;
        --border: #2d1f55;
    }

    .stApp {
        background: radial-gradient(circle at top right, #1a1233 0%, #0f0a1f 45%);
        color: var(--text);
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, h4, h5 {
        color: var(--text) !important;
    }

    .muted {
        color: var(--muted);
        font-size: 0.95rem;
    }

    .card {
        background: rgba(26, 18, 51, 0.85);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    }

    .badge {
        display: inline-block;
        background: linear-gradient(90deg, var(--accent), var(--accent2));
        color: #fff;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }

    div[data-testid="stSidebar"] {
        background: #120b26;
        border-right: 1px solid var(--border);
    }

    .stButton > button {
        background: linear-gradient(90deg, #7c3aed, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }

    .stTextArea textarea, .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #130d2a !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: #170f31;
        border: 1px solid var(--border);
        border-radius: 10px;
        color: #e7dcff;
        padding: 8px 12px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #7c3aed, #8b5cf6);
        border: none;
        color: #fff !important;
    }

    code {
        color: #f3edff !important;
        background: #251848 !important;
    }

    .small-note {
        color: #b9a9e8;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Ollama helpers
# ----------------------------
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3"


def call_ollama(
    prompt: str,
    system: str = "You are a senior software architect.",
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    temperature: float = 0.2,
    max_tokens: int = 1500,
) -> str:
    """
    Calls local Ollama chat API and returns content text.
    Compatible with /api/chat endpoint.
    """
    url = f"{ollama_url}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    try:
        r = requests.post(url, json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        # Typical structure:
        # {"message": {"role": "assistant", "content": "..."}}
        return data.get("message", {}).get("content", "").strip()
    except requests.RequestException as e:
        return f"❌ Error contacting Ollama at `{ollama_url}`: {e}"
    except Exception as e:
        return f"❌ Unexpected error: {e}"


def extract_mermaid_block(text: str) -> str:
    """
    Extract Mermaid code from markdown fence if present.
    Falls back to full text if no fence found.
    """
    pattern = r"```mermaid\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def render_mermaid(mermaid_code: str, height: int = 520) -> None:
    """
    Renders Mermaid diagram via HTML component.
    """
    html = f"""
    <div id="container" style="background:#130d2a;border:1px solid #2d1f55;border-radius:12px;padding:12px;">
      <pre class="mermaid" style="background:transparent;color:#f3edff;">{mermaid_code}</pre>
    </div>

    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({{
        startOnLoad: true,
        theme: "dark",
        securityLevel: "loose"
      }});
    </script>
    """
    components.html(html, height=height, scrolling=True)


def extract_json_array(text: str) -> List[Dict[str, Any]]:
    """
    Tries to extract JSON array from model output.
    Useful for tech stack structured parsing.
    """
    # Try fenced JSON block first
    fence_match = re.search(r"```json\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidate = fence_match.group(1).strip() if fence_match else text

    # Attempt direct parse
    try:
        data = json.loads(candidate)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # Try extracting first [ ... ] block
    bracket_match = re.search(r"(\[\s*{.*}\s*\])", candidate, re.DOTALL)
    if bracket_match:
        try:
            data = json.loads(bracket_match.group(1))
            if isinstance(data, list):
                return data
        except Exception:
            pass

    return []


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown("## 🧠 ArchitectAI")
    st.markdown('<div class="muted">System Design Assistant powered by Ollama + Streamlit</div>', unsafe_allow_html=True)
    st.markdown("---")

    ollama_url = st.text_input("Ollama URL", value=DEFAULT_OLLAMA_URL)
    model_name = st.text_input("Model", value=DEFAULT_MODEL)
    st.markdown('<div class="small-note">No API key required. Ensure Ollama is running locally.</div>', unsafe_allow_html=True)

    if st.button("Check Ollama Connection"):
        try:
            ping = requests.get(f"{ollama_url}/api/tags", timeout=20)
            if ping.status_code == 200:
                models = [m.get("name", "unknown") for m in ping.json().get("models", [])]
                st.success("Connected to Ollama ✅")
                if models:
                    st.write("Available models:", ", ".join(models))
            else:
                st.error(f"Ollama reachable but returned status {ping.status_code}")
        except Exception as e:
            st.error(f"Connection failed: {e}")

st.title("🧠 ArchitectAI — System Design Assistant")
st.caption("Analyze requirements, generate architecture, write ADRs, compare stacks, estimate scalability, and bootstrap IaC.")


tabs = st.tabs(
    [
        "1) Requirements Analyzer",
        "2) Architecture Generator",
        "3) ADR Writer",
        "4) Tech Stack Selector",
        "5) Scalability Estimator",
        "6) IaC Generator",
    ]
)

# ----------------------------
# 1) Requirements Analyzer
# ----------------------------
with tabs[0]:
    st.subheader("Requirements Analyzer")
    st.write("Paste functional and non-functional requirements. AI extracts drivers, constraints, quality attributes, and summary.")

    req_input = st.text_area(
        "Requirements Input",
        height=260,
        placeholder="Example:\nFunctional: users can upload photos, search by tags...\nNon-functional: p95 latency < 300ms, 99.95% uptime, GDPR compliance...",
    )

    col_a, col_b = st.columns([1, 4])
    with col_a:
        run_req = st.button("Analyze Requirements", key="analyze_req")
    with col_b:
        st.markdown('<span class="small-note">Tip: Include scale targets, compliance needs, and integration constraints.</span>', unsafe_allow_html=True)

    if run_req:
        if not req_input.strip():
            st.warning("Please provide requirements.")
        else:
            prompt = f"""
Analyze these software system requirements and return a structured analysis in markdown.

Requirements:
{req_input}

Please include:
1) Functional requirements (bullet list)
2) Non-functional requirements (bullet list)
3) Key architectural drivers
4) Constraints and assumptions
5) Quality attributes with specific measurable targets where possible:
   - Performance
   - Scalability
   - Security
   - Reliability
   - Maintainability
6) Risks and unknowns
7) Structured summary table (Category, Details, Priority)
"""
            with st.spinner("Analyzing requirements..."):
                out = call_ollama(prompt, model=model_name, ollama_url=ollama_url)

            st.markdown("### Structured Requirements Summary")
            st.markdown(out)

# ----------------------------
# 2) Architecture Generator
# ----------------------------
with tabs[1]:
    st.subheader("Architecture Generator (C4-style)")
    st.write("Describe the system and get Context, Container, and Component level architecture + Mermaid diagram.")

    arch_input = st.text_area(
        "System Description",
        height=220,
        placeholder="Describe domain, users, external systems, constraints, expected load, and critical quality attributes...",
    )

    run_arch = st.button("Generate Architecture", key="gen_arch")

    if run_arch:
        if not arch_input.strip():
            st.warning("Please describe your system.")
        else:
            prompt = f"""
You are a principal architect. Generate a C4-style architecture description for this system:

{arch_input}

Output in markdown with these sections:
1) Context Level
2) Container Level
3) Component Level
4) Data Flow
5) Security & Reliability Considerations

Then provide a Mermaid diagram in a fenced code block using ```mermaid ... ```
Prefer a graph TD/flowchart syntax.
"""
            with st.spinner("Generating architecture..."):
                out = call_ollama(prompt, model=model_name, ollama_url=ollama_url)

            st.markdown("### C4-style Architecture Description")
            st.markdown(out)

            st.markdown("### Mermaid Diagram")
            mermaid_code = extract_mermaid_block(out)
            if mermaid_code:
                render_mermaid(mermaid_code, height=560)

# ----------------------------
# 3) ADR Writer
# ----------------------------
with tabs[2]:
    st.subheader("ADR Writer")
    st.write("Describe a decision scenario and generate a complete Architecture Decision Record in Markdown.")

    adr_input = st.text_area(
        "Decision Scenario",
        height=220,
        placeholder="Example: Choose between monolith and microservices for a rapidly growing B2B SaaS with strict compliance...",
    )

    run_adr = st.button("Generate ADR", key="gen_adr")
    if run_adr:
        if not adr_input.strip():
            st.warning("Please provide an architecture decision scenario.")
        else:
            prompt = f"""
Write a complete Architecture Decision Record (ADR) in markdown.

Scenario:
{adr_input}

Use this exact structure:
- Title
- Status
- Context
- Decision
- Consequences
- Alternatives Considered

Also include:
- Decision drivers
- Trade-offs
- Follow-up actions
"""
            with st.spinner("Writing ADR..."):
                out = call_ollama(prompt, model=model_name, ollama_url=ollama_url)

            st.markdown("### Generated ADR")
            st.markdown(out)

# ----------------------------
# 4) Tech Stack Selector
# ----------------------------
with tabs[3]:
    st.subheader("Tech Stack Selector")
    st.write("Describe requirements and receive 3 stack options with comparative scoring and radar chart.")

    stack_input = st.text_area(
        "System Requirements for Stack Selection",
        height=220,
        placeholder="Include team size, timeline, budget sensitivity, expected traffic, ecosystem preference, etc...",
    )

    run_stack = st.button("Recommend Tech Stacks", key="gen_stack")

    if run_stack:
        if not stack_input.strip():
            st.warning("Please provide system requirements.")
        else:
            prompt = f"""
Based on these requirements, propose exactly 3 technology stack options.

Requirements:
{stack_input}

Return:
1) Short explanation for each option
2) JSON array only with this schema:
[
  {{
    "name": "Option name",
    "description": "Short description",
    "scalability": 1-10,
    "cost": 1-10,
    "dev_speed": 1-10,
    "community": 1-10,
    "maturity": 1-10
  }}
]

Ensure valid JSON.
"""
            with st.spinner("Selecting tech stacks..."):
                out = call_ollama(prompt, model=model_name, ollama_url=ollama_url)

            st.markdown("### AI Recommendation")
            st.markdown(out)

            stacks = extract_json_array(out)
            if stacks:
                df = pd.DataFrame(stacks)
                expected_cols = ["name", "scalability", "cost", "dev_speed", "community", "maturity"]
                if all(c in df.columns for c in expected_cols):
                    st.markdown("### Comparison Table")
                    st.dataframe(df[expected_cols], use_container_width=True)

                    categories = ["scalability", "cost", "dev_speed", "community", "maturity"]

                    fig = go.Figure()
                    for _, row in df.iterrows():
                        values = [row[c] for c in categories]
                        values += values[:1]
                        theta = categories + categories[:1]
                        fig.add_trace(
                            go.Scatterpolar(
                                r=values,
                                theta=theta,
                                fill="toself",
                                name=row["name"],
                            )
                        )

                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 10],
                                gridcolor="#3a2a6a",
                                linecolor="#3a2a6a",
                            ),
                            bgcolor="#120b26",
                        ),
                        showlegend=True,
                        paper_bgcolor="#120b26",
                        plot_bgcolor="#120b26",
                        font=dict(color="#f3edff"),
                        title="Tech Stack Radar Comparison",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Could not parse full scoring schema from model response.")
            else:
                st.info("No valid JSON array detected for radar chart. Try again or refine prompt.")

# ----------------------------
# 5) Scalability Estimator
# ----------------------------
with tabs[4]:
    st.subheader("Scalability Estimator")
    st.write("Input expected load and get infra estimates, bottlenecks, and scaling strategies.")

    col1, col2, col3 = st.columns(3)
    with col1:
        dau = st.number_input("Daily Active Users (DAU)", min_value=0, value=100000, step=1000)
    with col2:
        rps = st.number_input("Peak Requests Per Second (RPS)", min_value=0, value=1500, step=50)
    with col3:
        data_tb = st.number_input("Data Size (TB)", min_value=0.0, value=5.0, step=0.5)

    traffic_notes = st.text_area("Additional Context (optional)", height=120, placeholder="Read/write ratio, global regions, latency target, workload pattern...")

    run_scale = st.button("Estimate Scalability Plan", key="gen_scale")
    if run_scale:
        prompt = f"""
Estimate infrastructure and scalability strategy for:
- DAU: {dau}
- Peak RPS: {rps}
- Data size (TB): {data_tb}
- Additional context: {traffic_notes}

Return in markdown:
1) Capacity assumptions
2) Estimated infrastructure needs (compute, DB, cache, queue, CDN, storage)
3) Potential bottlenecks
4) Horizontal and vertical scaling strategies
5) Reliability and failover recommendations
6) Cost-awareness tips
7) Immediate next steps checklist
"""
        with st.spinner("Estimating scalability..."):
            out = call_ollama(prompt, model=model_name, ollama_url=ollama_url)

        st.markdown("### Scalability Estimation")
        st.markdown(out)

# ----------------------------
# 6) IaC Generator
# ----------------------------
with tabs[5]:
    st.subheader("IaC Generator")
    st.write("Describe your infrastructure and generate Docker Compose or Terraform HCL stubs with comments.")

    iac_type = st.selectbox("Choose IaC Output", ["Docker Compose", "Terraform HCL"])
    iac_input = st.text_area(
        "Infrastructure Description",
        height=220,
        placeholder="Example: 3-tier web app with Nginx, API, Postgres, Redis; or AWS VPC + ECS + RDS + ALB...",
    )

    run_iac = st.button("Generate IaC Stub", key="gen_iac")
    if run_iac:
        if not iac_input.strip():
            st.warning("Please provide infrastructure details.")
        else:
            if iac_type == "Docker Compose":
                prompt = f"""
Generate a docker-compose.yml stub with explanatory comments for this infrastructure:

{iac_input}

Requirements:
- Valid YAML
- Include common services, networks, volumes
- Add comments explaining why each section exists
- Keep placeholders for environment variables
- After code, add a short "How to run" section
"""
            else:
                prompt = f"""
Generate a Terraform HCL stub with explanatory comments for this infrastructure:

{iac_input}

Requirements:
- Use provider + core resources only (skeleton)
- Add comments explaining each resource
- Keep variables/placeholders clear
- Include outputs section
- After code, add a short "How to apply" section
"""

            with st.spinner("Generating IaC..."):
                out = call_ollama(prompt, model=model_name, ollama_url=ollama_url)

            st.markdown("### Generated IaC")
            st.markdown(out)

st.markdown("---")
st.markdown(
    '<div class="small-note">Built with Streamlit + Ollama (llama3). Mermaid rendering uses st.components HTML embedding.</div>',
    unsafe_allow_html=True,
)