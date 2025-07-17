import replicate
import streamlit as st

# REPLICATE API key (set this in your Streamlit secrets)
REPLICATE_API_TOKEN = st.secrets["replicate_api_token"]

st.set_page_config(page_title="Legislation Review & Process Mapper", layout="wide")
st.title("Legislation Review and Business Process Mapper (Claude via Replicate)")

st.write("""
Upload legislative or requirements text. This app will:
1. Extract and summarize a **Program Glossary** of important terms.
2. Generate a **business process map** for each user type or role, based on the legislation.

Glossary example sections:
- General Terms
- Technology & Systems
- Project & Delivery
- Stakeholders & Governance
- Legislation & Policy
- Processes & Methods

The process maps show step-by-step activities for each user type/role.
""")

uploaded_file = st.file_uploader("Upload a text file", type="txt")

def chunk_text(text, max_chars=5000):
    paragraphs = text.split('\n')
    chunks, chunk = [], ""
    for para in paragraphs:
        if len(chunk) + len(para) <= max_chars:
            chunk += para + '\n'
        else:
            chunks.append(chunk)
            chunk = para + '\n'
    if chunk:
        chunks.append(chunk)
    return chunks

def call_claude_replicate(prompt, max_tokens=800, temperature=0.3):
    output = replicate.run(
        # Use the appropriate Claude model version! E.g.
        "anthropic/claude-3-haiku:latest",
        input={
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        api_token=REPLICATE_API_TOKEN
    )
    # Output is a generator; join chunks
    return "".join([str(part) for part in output])

def glossary_chunk_claude(chunk):
    prompt = f"""You are a senior business analyst.
Review the legislative or requirements text below:

{chunk}

Extract a **Program Glossary**: list important and program-relevant terms, acronyms, and key phrases.

For each, provide:
- The term or acronym
- A 1-2 sentence plain-language, contextual definition

**Group terms under appropriate section headings** (e.g., General Terms, Technology & Systems, Project & Delivery, Stakeholders & Governance, Legislation & Policy, Processes & Methods).

Format:

# [Section Name]
- **Term**: Definition

Omit terms not important to this program or legislation.
"""
    return call_claude_replicate(prompt, max_tokens=800)

def process_map_chunk_claude(chunk):
    prompt = f"""You are a business process analyst.
Review the text below:

{chunk}

Identify and list all user types/roles (e.g., Customer, Staff, Regulator).

For **each user type**, create a clear **high-level, step-by-step business process map**:
- Number all main steps and state plainly
- Include decisions, key inputs/outputs, and interactions if mentioned
- Organize under clear user/role headings

Format:

# [User Type/Role]
1. [Step description]
    - Decision: [if any]
    - Input: [if any]
    - Output: [if any]
    - Interacts with: [role, if relevant]

Map only explicit or clearly implied user types present in the text.
"""
    return call_claude_replicate(prompt, max_tokens=1200)

def consolidate_outputs(outputs, mode="glossary"):
    if mode == "glossary":
        prompt = ("Combine, de-duplicate and rewrite the following **Program Glossaries** into one, organizing with appropriate section headings. "
                  "Use clear glossary formatting.\n\nTEXT:\n" + "\n".join(outputs))
    else:
        prompt = ("Merge the following **process maps**. Combine duplicate user roles, and present for each a single, clear set of sequenced steps."
                  "\nTEXT:\n" + "\n".join(outputs))
    return call_claude_replicate(prompt, max_tokens=1800)

def iterative_merge(list_to_merge, mode, group_size=5):
    if len(list_to_merge) == 1:
        return list_to_merge[0]
    summaries = []
    for i in range(0, len(list_to_merge), group_size):
        group = list_to_merge[i:i+group_size]
        merged = consolidate_outputs(group, mode)
        summaries.append(merged)
    return iterative_merge(summaries, mode, group_size)

if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    st.header("Original Content")
    st.write(f"File contains {len(content)} characters. Showing first 3000 chars below:\n\n")
    st.text_area("Preview", content[:3000] + ("..." if len(content) > 3000 else ""), height=300)

    if st.button("Generate Glossary and Business Process Maps"):
        st.info("Large files will be processed in chunks and summarized in steps.")
        with st.spinner("Extracting glossary and process maps..."):
            chunks = chunk_text(content, max_chars=4000)
            all_glossaries, all_process_maps = [], []

            for i, chunk in enumerate(chunks):
                st.write(f"Processing chunk {i+1}/{len(chunks)}")
                glossary = glossary_chunk_claude(chunk)
                process_map = process_map_chunk_claude(chunk)
                all_glossaries.append(glossary)
                all_process_maps.append(process_map)

            st.success("Initial extraction complete! Consolidating outputs...")

            final_glossary = iterative_merge(all_glossaries, mode="glossary", group_size=5)
            final_process_maps = iterative_merge(all_process_maps, mode="process_map", group_size=5)

            st.subheader("Extracted Program Glossary")
            st.write(final_glossary)

            st.subheader("Business Process Maps by User Type")
            st.write(final_process_maps)
