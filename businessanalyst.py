import streamlit as st
import openai

openai.api_key = st.secrets["api_key"]

st.set_page_config(page_title="Program Glossary Extractor", layout="wide")

st.title("Wazzup!!! BA Arvee - Program Glossary Generator")

st.write("""
Upload a text file (project brief, business requirement, etc).
This app will extract and summarize a **Program Glossary** of important terms, grouped under clear headings such as:
- General Terms
- Technology & Systems
- Project & Delivery
- Stakeholders & Governance
- Legislation & Policy
- Processes & Methods
Each term is briefly defined in plain language.
""")

uploaded_file = st.file_uploader("Choose a text file", type="txt")

def chunk_text(text, max_chars=5000):
    paragraphs = text.split('\n')
    chunks = []
    chunk = ""
    for para in paragraphs:
        if len(chunk) + len(para) <= max_chars:
            chunk += para + '\n'
        else:
            chunks.append(chunk)
            chunk = para + '\n'
    if chunk:
        chunks.append(chunk)
    return chunks

def glossary_chunk(chunk):
    prompt = f"""
You are an experienced business analyst extracting important terminology from program documentation.

Carefully review the following text:

{chunk}

Identify **important program terms, acronyms, and key phrases** relevant to IT, projects, transformation, or legislation.

For each term, provide:
 - The term or acronym
 - A 1-2 sentence plain-language definition (contextualized to this material if possible)

Organize the glossary into appropriately-titled sections to aid readability (e.g., General Terms, Technology & Systems, Project & Delivery, Stakeholders & Governance, Legislation & Policy, Processes & Methods).

Format:

# [Section Name]
- **Term:** Definition

Only extract and define significant and program-relevant terms or concepts.
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def consolidate_glossaries(glossaries):
    prompt = f"""
You are an expert business analyst.
Below are extracted program glossary sections from different parts of a requirement or legislative document.

Combine and rewrite them as a **single, de-duplicated, comprehensive Program Glossary**.
- Organize the glossary under clear, logical section headings (e.g., General Terms, Technology & Systems, Project & Delivery, Stakeholders & Governance, Legislation & Policy, Processes & Methods).
- For each term under each section, provide a short, context-relevant definition.

Only include important and program-relevant terms.

TEXT:
{' '.join(glossaries)}
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1800,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def iterative_glossary(requirements_list, group_size=5):
    if len(requirements_list) == 1:
        return requirements_list[0]
    summaries = []
    for i in range(0, len(requirements_list), group_size):
        group = requirements_list[i:i+group_size]
        merged = consolidate_glossaries(group)
        summaries.append(merged)
    return iterative_glossary(summaries, group_size)

if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    st.header("Original Content")
    st.write(f"File contains {len(content)} characters. Showing first 3000 chars below:\n\n")
    st.text_area("Preview", content[:3000] + ("..." if len(content) > 3000 else ""), height=300)

    if st.button("Generate Program Glossary"):
        st.info("Large files will be processed in chunks and summarized in steps.")
        with st.spinner("Extracting program glossary..."):

            # Split content into manageable pieces
            chunks = chunk_text(content, max_chars=4000)
            all_glossaries = []

            for i, chunk in enumerate(chunks):
                st.write(f"Processing chunk {i+1}/{len(chunks)}")
                glossary = glossary_chunk(chunk)
                all_glossaries.append(glossary)

            st.success("Initial extraction complete. Consolidating outputs...")

            # Hierarchical summarization
            final_glossary = iterative_glossary(all_glossaries, group_size=5)

            st.subheader("Extracted Program Glossary")
            st.write(final_glossary)
