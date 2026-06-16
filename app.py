"""
HRAssistant-Agent — Streamlit chat UI.

Run with:  streamlit run app.py
"""
import streamlit as st

import config
from src.agent import HRAgent
from src.database import EmployeeDB
from src.llm import LLMClient, LLMError
from src.rag import RAGPipeline, load_embeddings

st.set_page_config(page_title=config.PAGE_TITLE, page_icon=config.PAGE_ICON, layout="wide")


@st.cache_resource(show_spinner="Loading embedding model…")
def get_embeddings():
    return load_embeddings()


@st.cache_resource(show_spinner="Loading employee database…")
def get_db():
    return EmployeeDB()


@st.cache_resource(show_spinner="Loading RAG pipeline…")
def get_rag():
    # Starts empty — policy documents come from user PDF uploads.
    return RAGPipeline(get_embeddings())


def get_agent():
    return HRAgent(llm=LLMClient(), rag=get_rag(), db=get_db())


def sidebar(db: EmployeeDB, rag: RAGPipeline) -> str | None:
    st.sidebar.header("👤 Employee")
    employees = db.list_employees()
    labels = ["(none)"] + [f"{e['EmpID']} — {e['Name']}" for e in employees]
    choice = st.sidebar.selectbox("Identify yourself for personalized answers", labels)
    emp_id = None if choice == "(none)" else choice.split(" — ")[0]

    if emp_id:
        emp = db.get_employee(emp_id)
        st.sidebar.caption(f"**{emp['Name']}** · {emp['Role']} · {emp['Department']}")
        bal = db.get_leave_balance(emp_id)
        c1, c2, c3 = st.sidebar.columns(3)
        c1.metric("Casual", bal["CasualLeave"])
        c2.metric("Sick", bal["SickLeave"])
        c3.metric("Earned", bal["EarnedLeave"])

    st.sidebar.divider()
    st.sidebar.header("📄 Policy documents")
    indexed = st.session_state.get("indexed_docs", [])
    if indexed:
        st.sidebar.caption("Indexed: " + ", ".join(indexed))
    else:
        st.sidebar.caption("No documents indexed yet — upload a PDF to enable policy answers.")

    uploads = st.sidebar.file_uploader(
        "Upload policy documents (PDF)", type=["pdf"], accept_multiple_files=True
    )
    if uploads and st.sidebar.button("Index uploaded documents"):
        with st.spinner("Indexing…"):
            for f in uploads:
                if f.name in indexed:
                    continue
                if rag.index_uploaded_file(f):
                    indexed.append(f.name)
            st.session_state.indexed_docs = indexed
        st.sidebar.success(f"Indexed {len(indexed)} document(s).")
        st.rerun()

    st.sidebar.divider()
    if st.sidebar.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

    return emp_id


def main():
    st.title("💼 HRAssistant-Agent")
    st.caption("Ask about HR policies, your leave balance, your team — or all at once.")

    try:
        agent = get_agent()
    except LLMError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:  # noqa: BLE001
        st.error(f"Failed to initialize the app: {e}")
        st.stop()

    emp_id = sidebar(get_db(), agent.rag)

    if not agent.rag.is_ready:
        st.info(
            "📄 No policy documents indexed yet. Upload a **PDF** in the sidebar to "
            "enable policy (RAG) answers. Employee-record questions work without uploads."
        )

    st.session_state.setdefault("messages", [])
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask an HR question…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    result = agent.answer(prompt, emp_id, st.session_state.messages[:-1])
                    answer = result["answer"]
                    st.markdown(answer)
                    st.caption(f"Route: {result['route']}")
                except Exception as e:  # noqa: BLE001
                    answer = f"Sorry, I hit an error: {e}"
                    st.error(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
