import streamlit as st
import random
from typing import List, Dict, Any, Tuple
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.retrievers import BM25Retriever
from langchain.retrievers.multi_query import MultiQueryRetriever
import random
from typing import List, Dict, Any, Tuple
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.retrievers import BM25Retriever
from langchain.retrievers.multi_query import MultiQueryRetriever
class CarInsuranceAssistant:
    def __init__(self, knowledge_base_path: str, groq_api_key: str,
                 chunk_size: int = 1500, chunk_overlap: int = 300):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm = ChatGroq(
            model_name="llama3-8b-8192",
            temperature=0.2,
            max_tokens=1000,
            groq_api_key=groq_api_key
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            cache_folder="./embedding_cache"
        )
        self.knowledge_base = self._load_and_process_knowledge_base(knowledge_base_path)
        self.setup_guardrails()
        self.setup_conversational_elements()
        self.setup_rag_pipeline()
        self.conversation_history = []

    def _load_and_process_knowledge_base(self, knowledge_base_path: str) -> List[Document]:
        with open(knowledge_base_path, 'r', encoding='utf-8') as file:
            raw_text = file.read()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        texts = text_splitter.split_text(raw_text)
        documents = [Document(page_content=t) for t in texts]
        return documents

    def setup_rag_pipeline(self):
        self.vectorstore = FAISS.from_documents(self.knowledge_base, self.embeddings)
        self.bm25_retriever = BM25Retriever.from_documents(self.knowledge_base)
        self.multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=self.vectorstore.as_retriever(),
            llm=self.llm
        )
        self._setup_retrieval_qa_chain()

    def _setup_retrieval_qa_chain(self):
        qa_template = """
        You are an advanced car insurance information assistant designed to provide comprehensive, detailed, and clear explanations about auto insurance.
        Context Guidelines:
        - Use the provided context to generate thorough and informative responses about car insurance
        - If context is insufficient, provide general, well-informed auto insurance explanations
        - Break down complex car insurance terms and concepts
        - Provide practical examples related to vehicle coverage and claims
        - Balance technical accuracy with easy-to-understand explanations
        Context: {context}
        Question: {question}
        Response Requirements:
        1. Explain car insurance terms, policies, and processes clearly
        2. Use simple language for technical auto insurance terminology
        3. Never provide specific policy recommendations or legal advice
        4. Maintain a professional and helpful tone
        5. Acknowledge the complexity of auto insurance policies when relevant
        6. Focus on general car insurance education and information
        7. Encourage consultation with insurance providers for specific coverage questions
        8. Explain concepts thoroughly but concisely
        9. Address common misconceptions
        10. Provide context for terms and processes
        Note: Focus on general, educational information and encourage consultation with providers for specific advice.
        Response:
        """
        PROMPT = PromptTemplate(template=qa_template, input_variables=["context", "question"])
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 10}
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

    def setup_guardrails(self):
        self.sensitive_topics = [
            'fraud', 'legal advice', 'specific policy recommendations',
            'accident details', 'financial details', 'claims disputes',
            'immediate coverage needs', 'emergency situations'
        ]
        self.emergency_response = """
        For car insurance emergencies or accidents, please:
        1. Ensure everyone's safety and call emergency services if needed
        2. Document the accident scene with photos and notes
        3. Contact your car insurance provider's emergency hotline
        4. Exchange information with other involved parties
        5. Do not admit fault or discuss liability
        """

    def setup_conversational_elements(self):
        self.greeting_patterns = [
            'hi', 'hello', 'hey', 'good morning', 'good afternoon',
            'good evening', 'help', 'car insurance help'
        ]
        self.greeting_responses = [
            "Hello! I'm your car insurance assistant. How can I help you understand your coverage better?",
            "Hi there! I can help explain car insurance concepts. What would you like to know?",
            "Welcome! I'm here to help with your car insurance questions. What can I explain for you?"
        ]

    def chat(self, user_input: str) -> Dict[str, Any]:
        if user_input.lower() in self.greeting_patterns:
            response = random.choice(self.greeting_responses)
            self.conversation_history.append({"user": user_input, "assistant": response})
            return {
                "response": response,
                "history": self._format_history()
            }
        if any(topic in user_input.lower() for topic in self.sensitive_topics):
            if "emergency" in user_input.lower() or "accident" in user_input.lower():
                response = self.emergency_response
            else:
                response = ("I apologize, but I cannot provide specific advice on this topic. "
                            "Please consult with your insurance provider for guidance.")
            self.conversation_history.append({"user": user_input, "assistant": response})
            return {
                "response": response,
                "history": self._format_history()
            }
        result = self.qa_chain({"query": user_input})
        response = result["result"]
        self.conversation_history.append({"user": user_input, "assistant": response})
        return {
            "response": response,
            "history": self._format_history()
        }

    def _format_history(self) -> str:
        formatted = []
        for entry in self.conversation_history[-5:]:
            formatted.append(f"User: {entry['user']}")
            formatted.append(f"Assistant: {entry['assistant']}\n")
        return "\n".join(formatted)

def run_car_insurance_assistant():
    st.header("Car Insurance Assistant")
    assistant = CarInsuranceAssistant(
        knowledge_base_path="Basic Coverage Types.txt",
        groq_api_key="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"
    )
    user_input = st.text_input("Ask a question about car insurance:")
    if user_input:
        result = assistant.chat(user_input)
        st.write("Assistant Response:", result["response"])
        st.write("Conversation History:", result["history"])