import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import AIMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, \
    MessagesPlaceholder, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains.llm import LLMChain


bot_name = 'AI Tutor+'

load_dotenv()


def create_chat_first_message():
    return AIMessage("Olá! Vamos aprender sobre a língua inglesa?")


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
    st.session_state.llm_chain.memory.chat_memory.messages = \
        [create_chat_first_message()]
    st.balloons()


if "llm_chain" not in st.session_state:

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, streaming=False)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "Você é um chabot conversando com uma pessoa. Você tem "
                "domínio sobre a língua inglesa e quer ajudar o usuário a "
                "aprender o idioma. Faça perguntas para identificar o nível "
                "de conhecimento do usuário e crie lições sobre o tema com "
                "base no nível dele. Não deixe a conversa acabar e sempre "
                "proponha novas lições, se adaptando ao nível do usuário. "
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    memory = ConversationBufferMemory(memory_key="chat_history",
                                      return_messages=True)
    memory.chat_memory.messages = [create_chat_first_message()]

    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory
    )

    st.session_state.llm_chain = llm_chain

if "llm_chain" in st.session_state:
    for msg in st.session_state.llm_chain.memory.chat_memory.messages:
        owner = bot_name if isinstance(msg, AIMessage) else 'User'
        with st.chat_message(name=owner):
            st.write(msg.content)


if user_input := st.chat_input("Digite aqui"):

    with st.chat_message(name="User"):
        st.write(user_input)

    with st.chat_message(name=bot_name):
        response = st.session_state.llm_chain.invoke({"input": user_input})

        st.write(response)
        st.write(response['text'])
