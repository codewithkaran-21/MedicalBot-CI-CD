from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt
from pydantic import BaseModel
import os
from dotenv import load_dotenv


class AgentState(BaseModel):
    input: str
    context: str | None = None
    answer: str | None = None
    reflection: str | None = None


load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

embeddings = download_hugging_face_embeddings()
index_name = "medibot"


docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
)


def plan_node(state: AgentState) -> AgentState:
    """Decide whether retrieval is needed."""
    query = state.input.lower()
    needs_retrieval = any(word in query for word in ["disease", "symptom", "treatment", "medicine", "drug", "diagnosis"])

    if needs_retrieval:
        state.context = "RETRIEVE"
    else:
        state.context = "NO_RETRIEVE"

    return state


def retrieve_node(state: AgentState) -> AgentState:
    """Retrieve relevant context using the retriever."""
    if state.context == "RETRIEVE":
        docs = retriever.get_relevant_documents(state.input)
        combined_context = "\n\n".join([doc.page_content for doc in docs])
        state.context = combined_context
    else:
        state.context = "No retrieval needed."
    return state


def answer_node(state: AgentState) -> AgentState:
    """Generate answer using LLM."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    chain = create_stuff_documents_chain(llm, prompt)

    if state.context and state.context != "No retrieval needed.":
        response = chain.invoke({
            "input": state.input,
            "context": state.context
        })
        answer = response.get("output_text") or response.get("answer", "Sorry, I couldn’t find that information.")
    else:
        response = llm.invoke(state.input)
        answer = response.content if hasattr(response, "content") else str(response)

    state.answer = answer
    return state


def reflect_node(state: AgentState) -> AgentState:
    """Reflect on the answer: simple relevance validation."""
    reflection_prompt = (
        f"Question: {state.input}\n"
        f"Answer: {state.answer}\n\n"
        "Does the answer correctly and directly address the question? "
        "Respond with 'Yes' or 'No' and a short reason."
    )
    reflection = llm.invoke(reflection_prompt)
    state.reflection = reflection.content if hasattr(reflection, "content") else str(reflection)
    return state



def create_medibot_agent():
    graph = StateGraph(AgentState)

    
    graph.add_node("planner", plan_node)
    graph.add_node("retriever_node", retrieve_node)
    graph.add_node("answer_generator", answer_node)
    graph.add_node("reflector", reflect_node)

    
    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever_node")
    graph.add_edge("retriever_node", "answer_generator")
    graph.add_edge("answer_generator", "reflector")
    graph.add_edge("reflector", END)

    return graph.compile()
