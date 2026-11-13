import os

from getpass import getpass

from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# 1. CONSTRUCTION DU CONTEXTE
# ============================================================

def build_context(documents):

    context_parts = []


    for i, doc in enumerate(
        documents,
        start=1
    ):

        metadata = doc.metadata


        context = f"""
[Source {i}]

Company: {metadata["company"]}
Ticker: {metadata["ticker"]}
Date: {metadata["date"]}
Source: {metadata["source"]}
URL: {metadata["article_url"]}

Content:
{doc.page_content}
"""


        context_parts.append(
            context.strip()
        )


    return "\n\n".join(
        context_parts
    )


# ============================================================
# 2. PROMPT RAG
# ============================================================

rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a financial research assistant specialized
in Malaysian listed companies.

Answer the user's question using ONLY the information
contained in the provided context.

Rules:

1. Do not use outside knowledge.

2. Do not invent financial figures, events,
   dates or explanations.

3. If the available context is insufficient,
   explicitly say that the retrieved documents
   do not contain enough information.

4. Cite the supporting source using:
   [Source 1], [Source 2], etc.

5. Be precise and concise.

6. When several documents provide relevant information,
   combine them carefully.

7. Do not treat information from one company as
   information about another company.
"""
        ),

        (
            "human",
            """
QUESTION:

{question}


CONTEXT:

{context}


ANSWER:
"""
        )
    ]
)


# ============================================================
# 3. LLM
# ============================================================

load_dotenv()


api_key = os.getenv(
    "GOOGLE_API_KEY"
)


if not os.getenv(
    "GOOGLE_API_KEY"
):

    os.environ[
        "GOOGLE_API_KEY"
    ] = getpass(
        "Google API Key: "
    )


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# ============================================================
# 4. GENERATION CHAIN
# ============================================================

generation_chain = (
    rag_prompt
    | llm
    | StrOutputParser()
)