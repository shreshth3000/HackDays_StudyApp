import os
import re
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
            text += page.extract_text()
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
    Context:\n{context}\n
    Question:\n{question}\n
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

def generate_quiz(num_questions):
    ensure_vector_store_ready()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search("generate quiz questions from the material")
    context = " ".join([d.page_content for d in docs])
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", client=genai, temperature=0.3)
    prompt = f"""
    Generate {num_questions} multiple choice questions from the following context.
    Each question must be formatted as:
    Q: <question text>
    A) <option>
    B) <option>
    C) <option>
    D) <option>
    Correct: <A/B/C/D>
    Context:
    {context[:12000]}
    """
    result = model.invoke(prompt).content
    pattern = r"Q:\s*(.*?)\nA\)\s*(.*?)\nB\)\s*(.*?)\nC\)\s*(.*?)\nD\)\s*(.*?)\nCorrect:\s*([ABCD])"
    matches = re.findall(pattern, result, re.DOTALL)
    questions = []
    for m in matches:
        q, a, b, c, d, correct = m
        questions.append({"question": q.strip(), "options": [a.strip(), b.strip(), c.strip(), d.strip()], "answer": correct.strip()})
    if not questions:
        return None
    return questions

def reset_quiz():
    for key in ["quiz_started", "quiz_questions", "quiz_submitted", "quiz_answers", "quiz_score"]:
        if key in st.session_state:
            del st.session_state[key]

def quiz_ui():
    st.title("Quiz")
    if "quiz_started" not in st.session_state:
        st.session_state.quiz_started = False
    if "quiz_questions" not in st.session_state:
        st.session_state.quiz_questions = []
    if not st.session_state.quiz_started:
        num_questions = st.number_input("How many questions would you like?", min_value=1, max_value=10, step=1)
        if st.button("Generate Quiz"):
            with st.spinner("Generating quiz..."):
                qs = generate_quiz(num_questions)
                if qs:
                    st.session_state.quiz_questions = qs
                    st.session_state.quiz_started = True
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_answers = {}
                else:
                    st.error("Could not generate parseable questions from the provided documents/text.")
    else:
        answers = {}
        for i, q in enumerate(st.session_state.quiz_questions):
            st.subheader(f"Q{i+1}: {q['question']}")
            answers[i] = st.radio("Choose one:", q["options"], key=f"q{i}", index=st.session_state.quiz_answers.get(i, None))
        if st.button("Submit Quiz"):
            score = 0
            for i, q in enumerate(st.session_state.quiz_questions):
                user_ans = answers.get(i)
                correct_index = ord(q["answer"]) - ord("A")
                if user_ans == q["options"][correct_index]:
                    score += 1
            st.session_state.quiz_score = score
            st.session_state.quiz_submitted = True
            st.session_state.quiz_answers = {i: q["options"].index(answers[i]) if answers[i] else None for i, q in enumerate(st.session_state.quiz_questions)}
        if st.session_state.get("quiz_submitted"):
            st.write(f"Score: {st.session_state.quiz_score}/{len(st.session_state.quiz_questions)}")
            for i, q in enumerate(st.session_state.quiz_questions):
                user_ans_idx = st.session_state.quiz_answers.get(i)
                correct_idx = ord(q["answer"]) - ord("A")
                user_ans = q["options"][user_ans_idx] if user_ans_idx is not None else "No answer"
                correct_ans = q["options"][correct_idx]
                if user_ans == correct_ans:
                    st.success(f"Q{i+1}: Correct — {q['answer']}) {correct_ans}")
                else:
                    st.error(f"Q{i+1}: Incorrect. Correct Answer: {q['answer']}) {correct_ans}")
        st.button("Reset Quiz", on_click=reset_quiz)

def recommender_ui():
    st.title("Elective Recommender")
    uploaded = st.file_uploader("Upload your electives list, syllabus, and structure", type=["pdf"], accept_multiple_files=True)
    if uploaded and st.button("Process Files"):
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
            prompt = f"""
            You are an academic advisor recommending electives to a student.
            Use the following syllabus and elective information to give the best possible recommendations.
            Never reply "answer is not available in the context" — instead, infer or reason from the given material.
            Make sure recommendations are based on prerequisites, topical similarity, and student interests.

            Elective and Syllabus Information:
            {context}

            Student Query or Preferences:
            {query}

            Respond with clear, structured recommendations, including:
            1. Course name
            2. Department or domain
            3. Key reasons for recommendation
            4. Any prerequisite or duplication issues
            """
            response = model.invoke(prompt)
            text = response.content
            st.write(text)
        st.session_state.recommender_msgs.append({"role": "assistant", "content": text})

def shared_upload_ui():
    st.sidebar.subheader("Upload and Process PDFs (Shared Context)")
    pdf_docs = st.sidebar.file_uploader("Upload PDF Files", accept_multiple_files=True)
    if st.sidebar.button("Submit & Process"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Shared context processed successfully.")

def chat_ui():
    st.title("Meet DTUChattraAssistant, your personal study buddy...")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "upload some pdfs and ask me a question"}]
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    if prompt := st.chat_input():
        if prompt.strip().lower() == "/questions":
            st.session_state.mode = "Questions (/questions)"
            st.rerun()
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp = user_input(prompt)
                full_resp = resp["output_text"]
                st.write(full_resp)
        st.session_state.messages.append({"role": "assistant", "content": full_resp})

def main():
    st.set_page_config(page_title="Gemini PDF Chatbot", layout="wide")
    st.sidebar.title("Menu")    
    shared_upload_ui()
    mode = st.sidebar.radio("Select Mode", ["Chatbot", "Questions (/questions)", "Elective Recommender"])
    st.sidebar.button("Clear Chat History", on_click=clear_chat_history)
    if mode == "Chatbot":
        chat_ui()
    elif mode == "Questions (/questions)":
        quiz_ui()
    elif mode == "Elective Recommender":
        recommender_ui()

if __name__ == "__main__":
    main()
