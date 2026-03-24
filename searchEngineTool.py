from re import search
from langchain_ollama import ChatOllama
import streamlit as st
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_classic.agents import AgentType, initialize_agent

# Tool Usage Visibility and Real-time streaming
from langchain_classic.callbacks import StreamlitCallbackHandler


arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_char_max = 200)
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_char_max = 200)

arxiv = ArxivQueryRun(api_wrapper = arxiv_wrapper)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search_tool = DuckDuckGoSearchRun(name = "Browse")


st.title("Chatbot with Tools")

st.sidebar.title("Settings")

if "messages" not in st.session_state:
    st.session_state['messages'] = [
        {
            "role":"Assistant",
            "content":"Hi, I am chatbot who can search the web. How can I help You"

        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt:= st.chat_input(placeholder="Ex. What is LLM?") :# By Default a Text Message

    st.session_state.messages.append({"role":"user", "content":prompt})
    st.chat_message("user").write(prompt)

    llm = ChatOllama(model="phi3:mini", model_provider = "ollama", streaming = True)
    tools = [search_tool, wiki, arxiv]
    search_agent = initialize_agent(tools, llm, agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors = True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks = [st_cb])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)
