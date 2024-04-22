import os

import streamlit as st
from dotenv import load_dotenv

from langchain.prompts import (ChatPromptTemplate, MessagesPlaceholder,
                               PromptTemplate)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import (create_history_aware_retriever,
                              create_retrieval_chain, LLMChain,
                              StuffDocumentsChain,
                              MapReduceDocumentsChain, ReduceDocumentsChain)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory)
from langchain_community.document_loaders import DirectoryLoader

load_dotenv()

user_name = 'You'
bot_name = 'AI Tutor+'
avatars = {"human": user_name, "ai": bot_name}
dir_vectorstore = 'db_docs'
max_memory = 16
first_ai_message = ("Olá! Eu sou uma IA que cria lições personalizadas com "
                    "base no seu nível de conhecimento em conteúdos do meu "
                    "escopo. Vamos começar? ")

st.set_page_config(
    page_title=f"{bot_name}"
)
st.subheader(f":book: Embarque na jornada do conhecimento com o {bot_name}!")

st.info(
    (
        "Essa ferramenta é capaz de compreender o seu nível de conhecimento "
        "em diversos temas e criar lições personalizadas para você. "
    ),
    icon="ℹ️",
)




def clear_memory():
    st.session_state.memory.messages = []
    st.session_state.memory.add_ai_message(first_ai_message)
    st.session_state.interface_memory.messages = []
    st.session_state.interface_memory.add_ai_message(first_ai_message)
    st.balloons()


if st.button("Limpar histórico"):
    clear_memory()

if "llm_chain" not in st.session_state:

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

    for key in st.session_state.keys():
        del st.session_state[key]

    llm_summarization = ChatOpenAI(model="gpt-3.5-turbo", temperature=0,
                                   streaming=False)

    # Map
    map_template = (
        "A seguir está um pedaço de documento: "

        "\n\n{docs} "

        "\n\nCom base nesta lista de documentos, identifique os "
        "principais temas, em tópicos curtos, com poucas "
        "palavras. Limite de 1 a 3 tópicos. "
        "Seja breve na descrição dos tópicos e atenha-se apenas "
        "ao conteúdo dos documentos. "

        "\nResposta: "
    )

    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm_summarization, prompt=map_prompt)

    # Reduce
    reduce_template = (
        "A seguir está um conjunto de tópicos de diversos "
        "documentos: "

        "\n\n{docs} "

        "\n\nCom base nesses tópicos, crie uma nova lista de "
        "tópicos resumindo todo o conjunto fornecido. Limite "
        "de 5 a 10 tópicos. Não crie tópicos repetitivos. "
        "Dê pouca ênfase em tópicos destoantes do contexto "
        "geral. "

        "\nResposta:"
    )

    # maps each document to an individual summary using a chain
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=llm_summarization, prompt=reduce_prompt)

    # combine those summaries into a single global summary
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="docs")

    # combines and iteratively reduces the mapped documents
    # if the cumulative number of tokens in our mapped documents exceeds 4000
    # tokens, then we’ll recursively pass in the documents in batches of
    # < 4000 tokens to our StuffDocumentsChain
    reduce_documents_chain = ReduceDocumentsChain(
        # this is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # if documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # the maximum number of tokens to group documents into.
        token_max=4000,
    )

    # combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # map chain
        llm_chain=map_chain,
        # reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # the variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # return the results of the map steps in the output
        return_intermediate_steps=False,
    )

    # obtains information from various files, including PDFs and images
    loader = DirectoryLoader('resources/', exclude=["*.mp4", "*.json"],
                             use_multithreading=True)
    docs = loader.load()

    # split documents into smaller parts
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=600, chunk_overlap=0
    )
    split_docs = text_splitter.split_documents(docs)

    # create a vector database for similarity search
    embedding = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=split_docs,
                                        embedding=embedding)
    retriever = vectorstore.as_retriever(
        search_kwargs={'k': 3}
    )

    # list of topics from documents
    docs_summarization = map_reduce_chain.run(split_docs)

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

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, streaming=False)

    memory = StreamlitChatMessageHistory(key="chat_messages")
    memory.add_ai_message(first_ai_message)

    st.session_state.memory = memory

    interface_memory = StreamlitChatMessageHistory()
    interface_memory.add_ai_message(first_ai_message)
    st.session_state.interface_memory = interface_memory

    ### Contextualize question ###
    contextualize_q_system_prompt = (
        "Dado o histórico, identifique qual é o tema mais recente que está "
        "sendo discutido, dentre a lista de temas abaixo. "
        "Caso o histórico contenha apenas saudações ou apresentações "
        "retorne apenas: 'Saudações'. "
        "\nNÃO responda à entrada do usuário. NÃO faça perguntas. NÃO "
        "interaja com o usuário. Dê uma resposta curta contendo apenas o "
        "tema da lista. "

        "\n\nTemas:"
        "\n\n0. Saudações"
        f"\n{docs_summarization}"
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            ("system", contextualize_q_system_prompt)
        ]
    )
    # make a rephrasing of the input query to the retriever, so that the
    # retrieval incorporates the context of the conversation
    history_aware_retriever = create_history_aware_retriever(
        llm_summarization, retriever, contextualize_q_prompt
    )

    ### Answer question ###
    chat_system_prompt = (
        "Você é uma IA conversacional da área do ensino. Seu objetivo é "
        "interagir com o usuário sobre os temas do seu conhecimento "
        "fornecidos abaixo. Identifique primeiro o nível de conhecimento "
        "dele sobre cada tema com uma pergunta e então crie lições "
        "personalizadas para que ele aprenda sobre o tema em questão. "

        "\nPrimeiro, verifique se algum tema já foi proposto ao usuário e se "
        "ele já mostrou seu nível de conhecimento. Caso não, proponha um "
        "tema e procure saber o nível de conhecimento do usuário sobre "
        "aquele tema. Caso sim, crie lições com base no nível do usuário "
        "sobre o tema baseado no contexto fornecido abaixo. SEMPRE ESCOLHA "
        "UM TEMA E AVALIE O NÍVEL DO USUÁRIO ANTES DE APLICAR A LIÇÃO. "
        "O contexto não deve ser levado em conta na escolha do tema, "
        "apenas a lista de temas."
        "\nApós cada lição, troque o tema, comece avaliando o usuário sobre "
        "o novo tema e forneceça lições. JAMAIS deixe a conversa acabar e "
        "não pare de propor novas lições. "

        "\nTemas: "

        f"\n\n{docs_summarization}"

        "\n\nContexto: "
        "\n\n{context} "
    )
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", chat_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    # receives the retrieved context, the list of topics, history and user
    # input to create an answer
    chat_chain = create_stuff_documents_chain(llm, chat_prompt)

    # combine the chains
    rag_chain = create_retrieval_chain(history_aware_retriever,
                                       chat_chain)

    # deal with message history update
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: memory,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # variable to be maintained at each streamlit execution
    st.session_state.llm_chain = conversational_rag_chain

    ##################

# fill in chat messages
for msg in st.session_state.interface_memory.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

# runs when the user sends his message
if user_input := st.chat_input("Digite aqui"):
    # put the user's message in the chat
    st.chat_message("human").write(user_input)
    # put the user's message in the interface memory
    st.session_state.interface_memory.add_user_message(user_input)

    # run the chain
    config = {"configurable": {"session_id": "any"}}
    response = st.session_state.llm_chain.invoke(
        {"input": user_input},
        config)

    # debug information for devs
    st.write(response)
    # put the AI message in the chat
    st.chat_message(bot_name).write(response['answer'])
    # put the AI message in the interface memory
    st.session_state.interface_memory.add_ai_message(response['answer'])

    # removes two messages from chain memory since two new ones are inserted
    # in each iteration
    if len(st.session_state.memory.messages) > max_memory:
        st.session_state.memory.messages.pop(0)
        st.session_state.memory.messages.pop(0)
