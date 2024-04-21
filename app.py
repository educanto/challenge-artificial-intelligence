import streamlit as st
from dotenv import load_dotenv

from langchain.prompts import (ChatPromptTemplate, MessagesPlaceholder,
                               PromptTemplate)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import (LLMChain, StuffDocumentsChain,
                              ConversationalRetrievalChain)
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, \
    create_retrieval_chain

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader

load_dotenv()

user_name = 'You'
bot_name = 'AI Tutor+'
avatars = {"human": user_name, "ai": bot_name}


st.set_page_config(
    page_title=f"{bot_name}"
)
st.subheader(f":book: Embarque na jornada do conhecimento com o {bot_name}!")

st.info(
    (
        "Essa ferramenta é capaz de compreender o seu nível de inglês e criar "
        "lições personlizadas para você. "
    ),
    icon="ℹ️",
)

if st.button("Limpar histórico"):
    st.session_state.llm_chain.memory.clear()
    st.session_state.llm_chain.memory.add_ai_message("How can I help you?")
    st.balloons()

if "docs_summarization" not in st.session_state:

    # 1 - Sumarization
    #
    # ref: https://python.langchain.com/docs/use_cases/summarization/
    #
    # Goes through the resources folder and generates a list of topics with
    # the content they cover. The process respects the token limit of the LLM
    # context window. Therefore, the topic generation is done recursively until
    # all documents have been covered.
    #
    # This may take a few seconds, but it only runs once on chat startup.
    # Alternatively, it can be adapted to generate the topics externally.


    llm_summarization = ChatOpenAI(model="gpt-3.5-turbo", temperature=0,
                                   streaming=False)

    # Map
    map_template = """A seguir está um conjunto de documentos
{docs}
Com base nesta lista de documentos, identifique os principais temas.
Resposta útil:"""

    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm_summarization, prompt=map_prompt)

    # Reduce
    reduce_template = """A seguir está um conjunto de resumos:
{docs}
Utilize-os para destilar em um resumo final e consolidado dos principais 
temas.
Resposta útil:"""

    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=llm_summarization, prompt=reduce_prompt)

    # Takes a list of documents, combines them into a single string, and
    # passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="docs"
    )

    # Combines and iteratively reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=4000,
    )

    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )

    loader = DirectoryLoader('resources/', exclude=["*.mp4", "*.json"],
                             use_multithreading=True)
    docs = loader.load()

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=10
    )
    split_docs = text_splitter.split_documents(docs)

    st.chat_message(avatars['human']).write(split_docs)

    docs_summarization = map_reduce_chain.run(split_docs)

    st.session_state.docs_summarization = docs_summarization


if "llm_chain" not in st.session_state:

    # 2 - Interactive Learning Conversational Chat
    #
    # ref: https://python.langchain.com/docs/use_cases/question_answering/chat_history/
    #
    # Chat that interacts with the user to identify topics that need to be
    # taught and present lessons.
    #
    # Receives the list of topics obtained from documents as a context for the
    # topic selection. Contains a chain that identifies the current topic of
    # discussion and generates a query to search for relevant information in
    # the documents, which are also inserted in the chat context.

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, streaming=False)

    embedding = OpenAIEmbeddings()

    vectorstore = Chroma(persist_directory='db_docs',
                         embedding_function=embedding)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    memory = StreamlitChatMessageHistory(key="chat_messages")
    memory.add_ai_message("Olá! Eu sou uma IA que cria lições personalizadas"
                          "com base no seu nível de conhecimento em "
                          "conteúdos do meu escopo. Vamos começar? ")

    st.session_state.memory = memory

    # Contextualize history
    contextualize_q_system_prompt = """Dado um histórico de chat de uma 
conversa sobre temas do conhecimento e a última entrada do usuário (que pode 
fazer referência ao contexto no histórico), formule uma frase independente 
que represente o assunto do último tópico que está sendo discutido. NÃO 
responda à entrada do usuário, apenas a reformule se necessário.
Caso ainda não tenha sido abordado nenhum tópico em específico, retorne uma 
frase vazia.
"""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm_summarization, retriever, contextualize_q_prompt
    )

    # Chat
    chat_system_prompt = """Você é uma IA conversacional da área do ensino. 
Seu objetivo é interagir com o usuário sobre os temas do seu conhecimento 
fornecidos abaixo, identificando o nível de conhecimento dele sobre cada 
tema e criando lições personalizadas para ele sobre o tema em questão. Após 
fornecer as lições de um tema, pergunte se o usuário ficou com dúvidas. Se 
sim, responda. Se não, troque o tópico e repita o processo.

Temas:

{context}
"""
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", chat_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    chat_chain = create_stuff_documents_chain(llm, chat_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever,
                                       chat_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: memory,
        input_messages_key="input",
        history_messages_key="history"
    )

    conversational_rag_chain.get_graph().print_ascii()

    st.session_state.llm_chain = conversational_rag_chain

    ##################

for msg in st.session_state.memory.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_input := st.chat_input("Digite aqui"):
    st.chat_message("human").write(user_input)

    config = {"configurable": {"session_id": "any"}}
    response = st.session_state.llm_chain.invoke(
        {"input": user_input},
        config)

    st.chat_message(bot_name).write(response.content)
