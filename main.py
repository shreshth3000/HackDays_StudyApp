import os
import re
import json
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    store = FAISS.from_texts(chunks, embedding=embeddings)
    store.save_local("faiss_index")
    st.session_state.vector_ready = True

def ensure_vector_store_ready():
    if not st.session_state.get("vector_ready"):
        st.error("Please upload and process PDFs first in either mode.")
        st.stop()

def get_conversational_chain():
    template = """
Answer the question as detailed as possible from the provided context.
If the answer is not in the context, say "answer is not available in the context".
Context:
{context}

Question:
{question}

Answer:
"""
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", client=genai, temperature=0.3)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "upload some pdfs and ask me a question"}]

def user_input(user_question):
    ensure_vector_store_ready()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question)
    chain = get_conversational_chain()
    return chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

def generate_mcqs_from_context(num_questions):
    ensure_vector_store_ready()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search("generate quiz questions from the material", k=6)
    context = " ".join([d.page_content for d in docs])
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", client=genai, temperature=0.25)
    prompt = f"Generate exactly {num_questions} multiple choice questions from the context below. Format each question as:\nQ: <text>\nA) <optA>\nB) <optB>\nC) <optC>\nD) <optD>\nCorrect: <A/B/C/D>\nContext:\n{context[:12000]}"
    resp = model.invoke(prompt).content
    return resp

def parse_mcqs(raw):
    items = []
    pattern = r"Q[:\s]*(.*?)\nA\)[\s]*(.*?)\nB\)[\s]*(.*?)\nC\)[\s]*(.*?)\nD\)[\s]*(.*?)\nCorrect[:\s]*([ABCD])"
    matches = re.findall(pattern, raw, flags=re.DOTALL)
    for m in matches:
        q, a, b, c, d, corr = m
        opts = [a.strip(), b.strip(), c.strip(), d.strip()]
        items.append({"question": q.strip(), "options": opts, "answer": corr.strip()})
    if items:
        return items
    json_try = None
    try:
        json_try = json.loads(raw)
    except:
        json_try = None
    if isinstance(json_try, list):
        parsed = []
        for p in json_try:
            if "question" in p and "options" in p and "answer" in p:
                parsed.append({"question": p["question"].strip(), "options": [o.strip() for o in p["options"]], "answer": p["answer"]})
        if parsed:
            return parsed
    blocks = re.split(r'\n(?=Q\d*[:.\s])', raw)
    parsed = []
    for block in blocks:
        qmatch = re.search(r'Q[:\s]*(.*)', block)
        if not qmatch:
            continue
        qtext = qmatch.group(1).splitlines()[0].strip()
        opts = re.findall(r'^[A-D][\)\.\s-]\s*(.+)$', block, flags=re.M)
        corr = None
        cm = re.search(r'(Correct|Answer)[:\s]*([A-D0-3])', block, flags=re.I)
        if cm:
            c = cm.group(2).strip().upper()
            if c in "ABCD":
                corr = c
            else:
                try:
                    idx = int(c)
                    corr = "ABCD"[idx] if 0 <= idx < 4 else None
                except:
                    corr = None
        if len(opts) >= 4 and corr:
            parsed.append({"question": qtext, "options": opts[:4], "answer": corr})
    return parsed

def generate_flashcards_from_context(num_cards):
    ensure_vector_store_ready()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search("extract key facts for flashcards", k=6)
    context = " ".join([d.page_content for d in docs])
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", client=genai, temperature=0.2)
    prompt = f"Generate {num_cards} concise flashcards from the context. Return JSON array: [{{'q':'...','a':'...'}}, ...]. Context:\n{context[:12000]}"
    resp = model.invoke(prompt).content
    parsed = None
    try:
        parsed = json.loads(resp)
    except:
        parsed = None
    if isinstance(parsed, list):
        cards = []
        for p in parsed:
            if "q" in p and "a" in p:
                cards.append({"q": p["q"].strip(), "a": p["a"].strip()})
        if cards:
            return cards
    lines = re.split(r'\n+', resp.strip())
    cards = []
    for line in lines:
        m = re.match(r'^\s*Q[:\-\s]*(.*?)\s*[:\-]\s*A[:\-\s]*(.*)$', line)
        if m:
            cards.append({"q": m.group(1).strip(), "a": m.group(2).strip()})
    return cards

def generate_adaptive_quiz(missed_texts, num_questions):
    ensure_vector_store_ready()
    context_parts = []
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    for t in missed_texts:
        docs = db.similarity_search(t, k=4)
        context_parts.extend([d.page_content for d in docs])
    context = " ".join(context_parts)
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", client=genai, temperature=0.3)
    prompt = f"Create {num_questions} focused multiple choice questions (format as Q:, A), B), C), D), Correct:) that target the following student weaknesses or topics:\n{json.dumps(missed_texts)}\nUse this context:\n{context[:12000]}"
    resp = model.invoke(prompt).content
    return resp

def clear_revise_state():
    for k in ["quiz_questions", "quiz_started", "quiz_submitted", "quiz_answers", "quiz_score", "flashcards", "adaptive_history"]:
        st.session_state.pop(k, None)

def revise_ui():
    st.title("Revise")
    ensure_vector_store_ready()
    if "quiz_questions" not in st.session_state:
        st.session_state.quiz_questions = []
    if "quiz_started" not in st.session_state:
        st.session_state.quiz_started = False
    if "flashcards" not in st.session_state:
        st.session_state.flashcards = []
    col1, col2, col3 = st.columns([2,2,1])
    with col1:
        num_questions = st.number_input("Number of questions", min_value=1, max_value=20, value=5, step=1, key="revise_num")
    with col2:
        if st.button("Generate Quiz"):
            with st.spinner("Generating quiz..."):
                st.session_state.flashcards = []
                raw = generate_mcqs_from_context(int(st.session_state.revise_num))
                parsed = parse_mcqs(raw)
                if parsed:
                    st.session_state.quiz_questions = parsed
                    st.session_state.quiz_started = True
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_answers = {}
                else:
                    st.error("Could not generate parseable questions from the provided documents/text.")
    with col2:
        if st.button("Generate Flashcards"):
            with st.spinner("Generating flashcards..."):
                st.session_state.quiz_questions = []
                st.session_state.quiz_started = False
                st.session_state.quiz_submitted = False
                st.session_state.quiz_answers = {}
                cards = generate_flashcards_from_context(10)
                if cards:
                    st.session_state.flashcards = cards
                else:
                    st.error("Could not generate flashcards.")
    with col3:
        if st.button("Clear Revise Section"):
            clear_revise_state()
    if st.session_state.get("flashcards"):
        st.subheader("Flashcards")
        for i, c in enumerate(st.session_state.flashcards):
            with st.expander(f"{i+1}. {c['q']}"):
                st.write(c["a"])
    if st.session_state.get("quiz_started") and st.session_state.get("quiz_questions"):
        answers = {}
        for i, q in enumerate(st.session_state.quiz_questions):
            st.subheader(f"Q{i+1}: {q['question']}")
            answers[i] = st.radio("", q["options"], key=f"rev_q_{i}", index=st.session_state.get("quiz_answers", {}).get(i, 0))
        if st.button("Submit Quiz"):
            score = 0
            misses = []
            stored = {}
            for i, q in enumerate(st.session_state.quiz_questions):
                sel = answers.get(i)
                stored[i] = sel
                correct_idx = ord(q["answer"]) - ord("A")
                correct_opt = q["options"][correct_idx] if 0 <= correct_idx < len(q["options"]) else None
                if sel == correct_opt:
                    score += 1
                else:
                    misses.append(q["question"])
            st.session_state.quiz_score = score
            st.session_state.quiz_submitted = True
            st.session_state.quiz_answers = stored
            st.session_state.adaptive_history = st.session_state.get("adaptive_history", []) + misses
    if st.session_state.get("quiz_submitted"):
        st.write(f"Score: {st.session_state.quiz_score}/{len(st.session_state.quiz_questions)}")
        for i, q in enumerate(st.session_state.quiz_questions):
            user_sel = st.session_state.quiz_answers.get(i)
            correct_idx = ord(q["answer"]) - ord("A")
            correct_opt = q["options"][correct_idx] if 0 <= correct_idx < len(q["options"]) else "Unknown"
            if user_sel == correct_opt:
                st.write(f"Q{i+1}: Correct — {correct_opt}")
            else:
                st.write(f"Q{i+1}: Incorrect. Your answer: {user_sel if user_sel else 'No answer'} | Correct: {correct_opt}")
        if st.button("Generate Adaptive Quiz"):
            misses = st.session_state.get("adaptive_history", [])
            if not misses:
                st.info("No missed topics recorded yet.")
            else:
                raw = generate_adaptive_quiz(misses, min(5, len(misses)))
                parsed = parse_mcqs(raw)
                if parsed:
                    st.session_state.quiz_questions = parsed
                    st.session_state.quiz_started = True
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_answers = {}
                else:
                    st.error("Could not generate adaptive quiz.")
    if st.session_state.get("flashcards") or st.session_state.get("quiz_questions"):
        st.write("Revise state persists until you clear the Revise Section.")

def recommender_ui():
    st.title("Elective Recommender")
    uploaded = st.file_uploader("Upload your electives list, syllabus, and structure (exactly 3 PDFs)", type=["pdf"], accept_multiple_files=True)
    if uploaded and len(uploaded) == 3 and st.button("Process Files"):
        text = get_pdf_text(uploaded)
        chunks = get_text_chunks(text)
        get_vector_store(chunks)
        st.session_state.elective_text = text
        st.success("Processed successfully.")
    st.write("Chat with the recommender below:")
    if "recommender_msgs" not in st.session_state:
        st.session_state.recommender_msgs = [{"role": "assistant", "content": "Upload your elective files and tell me your preferences."}]
    for msg in st.session_state.recommender_msgs:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    if query := st.chat_input("Type your message..."):
        st.session_state.recommender_msgs.append({"role": "user", "content": query})
        with st.chat_message("assistant"):
            ensure_vector_store_ready()
            model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", client=genai, temperature=0.5)
            context = st.session_state.get("elective_text", "")
            prompt = f"You are an academic advisor. Use the syllabus and elective info to recommend electives. Infer and reason; do not reply 'answer is not available in the context'. Elective info:\n{context}\nStudent query:\n{query}\nProvide structured recommendations with course name, domain, reasons, and prerequisites."
            resp = model.invoke(prompt).content
            st.write(resp)
        st.session_state.recommender_msgs.append({"role": "assistant", "content": resp})

def shared_upload_ui():
    st.sidebar.subheader("Upload and Process PDFs (Shared Context)")
    pdf_docs = st.sidebar.file_uploader("Upload PDF Files", accept_multiple_files=True)
    if st.sidebar.button("Submit & Process"):
        raw = get_pdf_text(pdf_docs)
        chunks = get_text_chunks(raw)
        get_vector_store(chunks)
        st.success("Shared context processed successfully.")

def chat_ui():
    st.title("Meet DTUChattraAssistant, your personal study buddy...")
    st.sidebar.button("Clear Chat History", on_click=clear_chat_history)
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "upload some pdfs and ask me a question"}]
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])
    if prompt := st.chat_input():
        if prompt.strip().lower() == "/revise":
            st.session_state.mode = "Revise"
            return
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp = user_input(prompt)
                if isinstance(resp, dict) and "output_text" in resp:
                    out = "".join(resp["output_text"])
                else:
                    out = str(resp)
                st.write(out)
        st.session_state.messages.append({"role": "assistant", "content": out})

def main():
    st.set_page_config(page_title="Gemini PDF Chatbot", layout="wide")
    if "mode" not in st.session_state:
        st.session_state.mode = "Chatbot"
    st.sidebar.title("Menu")
    shared_upload_ui()
    mode = st.sidebar.selectbox("Select Mode", ["Chatbot", "Revise", "Elective Recommender"], index=["Chatbot", "Revise", "Elective Recommender"].index(st.session_state.mode) if st.session_state.mode in ["Chatbot", "Revise", "Elective Recommender"] else 0)
    st.session_state.mode = mode
    if mode == "Chatbot":
        chat_ui()
    elif mode == "Revise":
        revise_ui()
    elif mode == "Elective Recommender":
        recommender_ui()

if __name__ == "__main__":
    main()
