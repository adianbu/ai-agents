import os
import sys
import argparse
import textwrap
from typing import List, Dict, Any, Optional
import logging

# Langchain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms.openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_LM_STUDIO_URL = "http://localhost:1234/v1"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

class LMStudioLLM(OpenAI):
    """Custom LLM class for LM Studio API."""
    
    def __init__(
        self,
        model: str = "qwen2.5-coder-32b-instruct",
        base_url: str = DEFAULT_LM_STUDIO_URL,
        api_key: str = "lm-studio",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ):
        # Pass the LM Studio URL as OpenAI base
        super().__init__(
            model=model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

class PDFRAGAgent:
    """RAG Agent for PDF documents using Langchain and LM Studio."""
    
    def __init__(
        self,
        lm_studio_url: str = DEFAULT_LM_STUDIO_URL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        vector_store_dir: str = "./chroma_db",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        temperature: float = 0.7
    ):
        self.lm_studio_url = lm_studio_url
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store_dir = vector_store_dir
        self.embedding_model = embedding_model
        self.temperature = temperature
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize LLM
        self.llm = LMStudioLLM(
            base_url=lm_studio_url,
            temperature=temperature
        )
        
        # Try to load existing vector store
        try:
            self.vector_store = Chroma(
                persist_directory=vector_store_dir,
                embedding_function=self.embeddings
            )
            logger.info(f"Loaded existing vector store from {vector_store_dir}")
        except Exception as e:
            logger.info(f"No existing vector store found at {vector_store_dir} or error loading it: {e}")
            self.vector_store = None
    
    def ingest_pdf(self, pdf_path: str) -> int:
        """Process a PDF file and add its content to the vector store."""
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return 0
        
        try:
            # Load the PDF
            logger.info(f"Loading PDF: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Check if we got any content
            if not documents:
                logger.warning(f"No content extracted from {pdf_path}")
                return 0
            
            logger.info(f"Extracted {len(documents)} pages from {pdf_path}")
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks from {pdf_path}")
            
            # Add metadata to track source
            for chunk in chunks:
                if "source" not in chunk.metadata:
                    chunk.metadata["source"] = pdf_path
            
            # Create or update vector store
            if self.vector_store is None:
                logger.info(f"Creating new vector store at {self.vector_store_dir}")
                self.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    persist_directory=self.vector_store_dir
                )
            else:
                logger.info(f"Adding to existing vector store")
                self.vector_store.add_documents(chunks)
            
            # Persist the vector store
            self.vector_store.persist()
            
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Error ingesting PDF {pdf_path}: {str(e)}")
            return 0
    
    def ingest_directory(self, directory_path: str) -> int:
        """Process all PDF files in a directory."""
        if not os.path.isdir(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return 0
        
        try:
            # Use DirectoryLoader to load all PDFs
            loader = DirectoryLoader(
                directory_path,
                glob="**/*.pdf",  # Load all PDFs including in subdirectories
                loader_cls=PyPDFLoader
            )
            
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from PDFs in {directory_path}")
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks from PDFs in {directory_path}")
            
            # Create or update vector store
            if self.vector_store is None:
                logger.info(f"Creating new vector store at {self.vector_store_dir}")
                self.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    persist_directory=self.vector_store_dir
                )
            else:
                logger.info(f"Adding to existing vector store")
                self.vector_store.add_documents(chunks)
            
            # Persist the vector store
            self.vector_store.persist()
            
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Error ingesting directory {directory_path}: {str(e)}")
            return 0
    
    def create_qa_chain(self):
        """Create a question-answering chain using the vector store for retrieval."""
        if self.vector_store is None:
            logger.error("No vector store available. Please ingest documents first.")
            return None
        
        # Create a retriever
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 5}  # Return top 5 most relevant chunks
        )
        
        # Create a custom prompt template
        template = """
        You are a helpful assistant that accurately answers questions based on the provided context from PDF documents.
        
        CONTEXT:
        {context}
        
        QUESTION:
        {question}
        
        INSTRUCTIONS:
        - Answer the question based ONLY on the information in the CONTEXT.
        - If the answer cannot be found in the context, say that you don't know based on the available information.
        - Do not make up information that is not supported by the context.
        - Cite the relevant parts of the context that support your answer.
        - Be concise and to the point.
        
        YOUR ANSWER:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # "stuff" puts all retrieved docs into a single prompt
            retriever=retriever,
            return_source_documents=True,  # Return source documents for citation
            chain_type_kwargs={"prompt": prompt}
        )
        
        return qa_chain
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question based on the ingested PDFs."""
        qa_chain = self.create_qa_chain()
        if qa_chain is None:
            return {
                "answer": "Error: No documents have been ingested yet. Please add some PDFs first.",
                "sources": []
            }
        
        try:
            # Get answer from the chain
            result = qa_chain({"query": question})
            
            # Extract source documents for citation
            source_documents = result.get("source_documents", [])
            sources = []
            
            # Format sources
            for doc in source_documents:
                source = doc.metadata.get("source", "Unknown source")
                # Get just the filename from the path
                if isinstance(source, str) and os.path.exists(source):
                    source = os.path.basename(source)
                
                page = doc.metadata.get("page", "Unknown page")
                content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                
                sources.append({
                    "source": source,
                    "page": page,
                    "content_preview": content_preview
                })
            
            return {
                "answer": result["result"],
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": []
            }
    
    def chat_loop(self):
        """Interactive chat loop for answering questions."""
        print("\n=== PDF RAG Agent (Langchain + LM Studio) ===")
        print("Type 'exit' or 'quit' to end the conversation.")
        print("Type 'ingest <pdf_path>' to add a PDF.")
        print("Type 'ingest_dir <directory_path>' to add all PDFs in a directory.")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ["exit", "quit"]:
                    print("Exiting. Goodbye!")
                    break
                
                elif user_input.lower().startswith("ingest "):
                    pdf_path = user_input[7:].strip()
                    if os.path.isfile(pdf_path):
                        chunks = self.ingest_pdf(pdf_path)
                        print(f"Added {chunks} chunks from {pdf_path}")
                    else:
                        print(f"File not found: {pdf_path}")
                
                elif user_input.lower().startswith("ingest_dir "):
                    dir_path = user_input[11:].strip()
                    if os.path.isdir(dir_path):
                        chunks = self.ingest_directory(dir_path)
                        print(f"Added {chunks} chunks from PDFs in {dir_path}")
                    else:
                        print(f"Directory not found: {dir_path}")
                
                else:
                    print("\nProcessing your question...")
                    response = self.answer_question(user_input)
                    answer = response["answer"]
                    sources = response["sources"]
                    
                    # Print the answer
                    print(f"\nAssistant: {textwrap.fill(answer, width=100)}")
                    
                    # Print sources if available
                    if sources:
                        print("\nSources:")
                        for i, source in enumerate(sources):
                            print(f"  {i+1}. {source['source']} (Page {source['page']})")
            
            except KeyboardInterrupt:
                print("\nExiting. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="PDF RAG Agent using Langchain and LM Studio")
    parser.add_argument("--pdf", type=str, help="Path to a PDF file to ingest")
    parser.add_argument("--pdf_dir", type=str, help="Path to a directory of PDF files to ingest")
    parser.add_argument("--lm_studio_url", type=str, default=DEFAULT_LM_STUDIO_URL, help="URL for LM Studio API")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="Size of text chunks")
    parser.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Overlap between text chunks")
    parser.add_argument("--vector_store_dir", type=str, default="./chroma_db", help="Directory for vector store")
    parser.add_argument("--embedding_model", type=str, default=DEFAULT_EMBEDDING_MODEL, help="Embedding model to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for text generation")
    parser.add_argument("--question", type=str, help="Question to answer (skip for interactive mode)")
    
    args = parser.parse_args()
    
    # Initialize the agent
    agent = PDFRAGAgent(
        lm_studio_url=args.lm_studio_url,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        vector_store_dir=args.vector_store_dir,
        embedding_model=args.embedding_model,
        temperature=args.temperature
    )
    
    # Ingest PDF(s) if specified
    if args.pdf:
        agent.ingest_pdf(args.pdf)
    
    if args.pdf_dir:
        agent.ingest_directory(args.pdf_dir)
    
    # Either answer a specific question or enter interactive mode
    if args.question:
        response = agent.answer_question(args.question)
        print(f"\nAnswer: {response['answer']}")
        
        if response["sources"]:
            print("\nSources:")
            for i, source in enumerate(response["sources"]):
                print(f"  {i+1}. {source['source']} (Page {source['page']})")
    else:
        agent.chat_loop()

if __name__ == "__main__":
    main()