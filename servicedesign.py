import replicate
import streamlit as st 
# import os
r_token = st.secrets["REPLICATE_API_TOKEN"]

# Set your Replicate API token in Streamlit secrets
# REPLICATE_API_TOKEN = st.secrets["replicate_api_token"]

st.set_page_config(page_title="Business Process Mapper (Claude via Replicate)", layout="wide")
st.title("Business Process Mapper by User Type (Claude via Replicate)")

st.write("""
**Upload legislation or requirements text. This app will:**
- Generate a clear, high-level, step-by-step **business process map for each user type or role** found in the document.
- The process maps group steps by each identified user type/role (e.g., customer, staff member, regulator).

Output shows:
- User/role heading
- Each main step with decisions, inputs, outputs, and interactions (if present)
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

def call_claude_replicate(prompt, max_tokens=1200, temperature=0.3):
    output = replicate.run(
        "anthropic/claude-3.7-sonnet",
        input={
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        api_token=r_token
    )
    return "".join([str(part) for part in output])

def process_map_chunk_claude(chunk):
    prompt = f"""You are a business process analyst.
Review the text below:

{chunk}

Identify the main user types/roles (e.g., Customer, Staff, Regulator).

For **each user type**, create a clear, high-level, step-by-step business process map:
- Number each main step and state plainly
- Include decisions, key inputs/outputs, and interactions if mentioned
- Organize under a bold user/role heading

Format:

# [User Type/Role]
1. [Step description]
    - Decision: [if any]
    - Input: [if any]
    - Output: [if any]
    - Interacts with: [role, if relevant]

Map only explicit or clearly implied user types present in the text.
"""
    return call_claude_replicate(prompt)

def consolidate_process_maps(process_maps):
    prompt = (
        "You are a business analyst. Merge the following process maps. "
        "Combine process steps for duplicate roles and present each role with a single clear set of sequenced steps."
        "\nTEXT:\n" + "\n".join(process_maps)
    )
    return call_claude_replicate(prompt, max_tokens=1800)

def iterative_merge(list_to_merge, group_size=5):
    if len(list_to_merge) == 1:
        return list_to_merge[0]
    summaries = []
    for i in range(0, len(list_to_merge), group_size):
        group = list_to_merge[i:i+group_size]
        merged = consolidate_process_maps(group)
        summaries.append(merged)
    return iterative_merge(summaries, group_size)

if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    st.header("Original Content")
    st.write(f"File contains {len(content)} characters. Showing first 3000 chars below:\n\n")
    st.text_area("Preview", content[:3000] + ("..." if len(content) > 3000 else ""), height=300)

    if st.button("Generate Business Process Maps by User Type"):
        st.info("Large files will be processed in chunks and summarized.")
        with st.spinner("Extracting process maps..."):
            chunks = chunk_text(content, max_chars=4000)
            all_process_maps = []

            for i, chunk in enumerate(chunks):
                st.write(f"Processing chunk {i+1}/{len(chunks)}")
                process_map = process_map_chunk_claude(chunk)
                all_process_maps.append(process_map)

            st.success("Initial extraction complete. Consolidating outputs...")

            final_process_maps = iterative_merge(all_process_maps, group_size=5)

            st.subheader("Business Process Maps by User Type")
            st.write(final_process_maps)
