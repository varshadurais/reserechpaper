
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import re

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Research Paper Answer Bot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Research Paper Answer Bot")
st.write("Upload a research paper and ask questions about it.")

# LOAD AI MODEL
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


# TEXT CHUNKING
def split_text(text, chunk_size=900, overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# PDF EXTRACTION WITH PAGE NUMBERS
def extract_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages, len(reader.pages)


# CREATE PAGE-AWARE CHUNKS
def create_chunks(pages):
    all_chunks = []

    for page in pages:
        chunks = split_text(page["text"])

        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "page": page["page"]
            })

    return all_chunks


# COSINE SIMILARITY SEARCH
def search_chunks(question, chunks, embeddings, top_k=3):

    query_embedding = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = np.dot(embeddings, query_embedding)

    indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in indices:
        results.append({
            "text": chunks[index]["text"],
            "page": chunks[index]["page"],
            "score": float(scores[index]),
            "index": int(index)
        })

    return results


# EXTRACTIVE ANSWER GENERATION
def generate_answer(question, results):

    if not results:
        return "Answer not found in the uploaded paper.", []

    if results[0]["score"] < 0.25:
        return "Answer not found in the uploaded paper.", []

    candidates = []

    # EXTRACT SENTENCES
    for result in results:

        sentences = re.split(
            r'(?<=[.!?])\s+',
            result["text"]
        )

        for sentence in sentences:

            sentence = sentence.strip()

            # Ignore very short sentences
            if len(sentence.split()) < 6:
                continue

            # Ignore incomplete sentences
            if sentence[-1] not in ".!?":
                continue

            candidates.append({
                "text": sentence,
                "page": result["page"]
            })

    if not candidates:
        return "Answer not found in the uploaded paper.", []

    # CREATE SENTENCE EMBEDDINGS
    sentence_texts = [
        item["text"] for item in candidates
    ]

    sentence_embeddings = model.encode(
        sentence_texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    question_embedding = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # SENTENCE SIMILARITY
    scores = np.dot(
        sentence_embeddings,
        question_embedding
    )

    ranked = np.argsort(scores)[::-1]

    selected = []
    used_sentences = set()

    # SELECT RELEVANT SENTENCES
    for index in ranked:

        if scores[index] < 0.25:
            continue

        sentence = candidates[index]["text"]
        normalized = sentence.lower().strip()

        # Remove duplicates
        if normalized in used_sentences:
            continue

        used_sentences.add(normalized)

        selected.append({
            "text": sentence,
            "page": candidates[index]["page"],
            "score": float(scores[index])
        })

        if len(selected) == 3:
            break

    if not selected:
        return "Answer not found in the uploaded paper.", []

    # JOIN EXACT PDF SENTENCES
    answer = " ".join(
        item["text"] for item in selected
    )

    return answer, selected


# PDF UPLOAD
uploaded_file = st.file_uploader(
    "Upload Research Paper PDF",
    type=["pdf"]
)

if uploaded_file:

    st.success("PDF uploaded successfully!")

    try:

        # EXTRACT PDF
        pages, total_pages = extract_pdf(uploaded_file)

        full_text = "\n".join(
            page["text"] for page in pages
        )

        if not full_text.strip():
            st.error("No readable text found in PDF.")
            st.stop()

        # PAPER INFORMATION
        st.subheader("📊 Paper Information")

        chunks = create_chunks(pages)

        col1, col2, col3 = st.columns(3)

        col1.metric("Number of Pages", total_pages)
        col2.metric("Extracted Characters", len(full_text))
        col3.metric("Number of Chunks", len(chunks))

        # VIEW EXTRACTED TEXT
        with st.expander("📖 View Extracted Text"):

            st.text_area(
                "Extracted Text",
                full_text,
                height=300
            )

        # VIEW CHUNKS
        with st.expander("🧩 View Text Chunks"):

            for i, chunk in enumerate(chunks):

                st.write(f"Chunk {i + 1}")
                st.caption(f"Page {chunk['page']}")
                st.write(chunk["text"])
                st.divider()

        # CREATE EMBEDDINGS
        with st.spinner("Creating AI embeddings..."):

            embeddings = model.encode(
                [chunk["text"] for chunk in chunks],
                convert_to_numpy=True,
                normalize_embeddings=True
            )

        st.success("Paper processed successfully!")

        # ASK QUESTION
        st.subheader("🔎 Ask a Question")

        question = st.text_input(
            "Enter your question:",
            placeholder="What is the main objective of this paper?"
        )

        if question.strip():

            with st.spinner("Searching for the best answer..."):

                results = search_chunks(
                    question,
                    chunks,
                    embeddings,
                    top_k=3
                )

                answer, selected_sentences = generate_answer(
                    question,
                    results
                )

            # SEARCH SIMILARITY
            st.subheader("📊 Search Confidence")

            score = max(
                0,
                min(results[0]["score"], 1)
            ) if results else 0

            st.progress(score)

            st.write(
                f"Best semantic similarity: {score:.2%}"
            )

            st.caption(
                "Similarity is a retrieval score, "
                "not verified answer accuracy."
            )

            # FINAL ANSWER
            st.subheader("🤖 Answer")

            st.info(answer)

            # SOURCE REFERENCES
            if selected_sentences:

                st.subheader("📚 Answer Sources")

                for item in selected_sentences:

                    st.write(
                        f"📄 Page {item['page']}"
                    )

                    st.write(item["text"])

            # RELEVANT CHUNKS
            if results:

                st.subheader("📌 Most Relevant Information")

                for i, result in enumerate(results):

                    with st.expander(
                        f"Chunk {result['index'] + 1} | "
                        f"Page {result['page']} | "
                        f"Similarity: {result['score']:.2%}"
                    ):

                        st.write(result["text"])

    except Exception as e:

        st.error(
            f"Something went wrong: {str(e)}"
        )
