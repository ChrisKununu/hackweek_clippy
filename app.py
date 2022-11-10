import streamlit as st
import json
from haystack.document_stores import OpenSearchDocumentStore
from haystack.pipelines import ExtractiveQAPipeline
from haystack.schema import Document
from haystack.nodes import BM25Retriever, FARMReader
import pandas as pd
from tqdm import tqdm
import yaml
from annotated_text import annotation
from markdown import markdown
from PIL import Image

# Adjust to a question that you would like users to see in the search bar when they load the UI:
DEFAULT_QUESTION_AT_STARTUP = "Was sind die häufigsten Fehler in einer Bewerbung?"
DEFAULT_ANSWER_AT_STARTUP = "None"

# Sliders
DEFAULT_DOCS_FROM_RETRIEVER = 3
DEFAULT_NUMBER_OF_ANSWERS = 3


def set_state_if_absent(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


def main():
    image = Image.open('img/clippy_image.png')

    st.set_page_config(page_title="Clippy Demo", page_icon=image)

    # Persistent state
    set_state_if_absent("question", DEFAULT_QUESTION_AT_STARTUP)
    set_state_if_absent("answer", DEFAULT_ANSWER_AT_STARTUP)

    # Small callback to reset the interface in case the text of the question changes
    def reset_results(*args):
        st.session_state.answer = None
        st.session_state.results = None
        st.session_state.raw_json = None

    # load settings and mappings for index

    with open('config/credentials.yaml') as f:
        credentials = yaml.load(f, Loader=yaml.FullLoader)

    with open('config/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    document_store = OpenSearchDocumentStore(host=credentials['host'],
                                             username=credentials['user'],
                                             password=credentials['password'],
                                             index=config['index_name'])

    # Sidebar

    st.sidebar.image(image, caption=None)
    st.sidebar.header("Options")
    top_k_reader = st.sidebar.slider(
        "Max. number of answers",
        min_value=1,
        max_value=10,
        value=DEFAULT_NUMBER_OF_ANSWERS,
        step=1,
        on_change=reset_results,
        )
    top_k_retriever = st.sidebar.slider(
        "Max. number of documents from retriever",
        min_value=1,
        max_value=10,
        value=DEFAULT_DOCS_FROM_RETRIEVER,
        step=1,
        on_change=reset_results,
        )

    with st.spinner(
            "🧠 &nbsp;&nbsp; Setting up the system... \n "
            ):
        retriever = BM25Retriever(document_store=document_store)
        reader = FARMReader(model_name_or_path=config['model_name'], use_gpu=True)
        # Title
        st.write("# Clippy Demo - Explore the blog")
        st.markdown(
            """
    This demo takes its data from news.kununu.com pages crawled in November 2022.

    Ask any question on this topic and see if Clippy can find the correct answer to your query!

    *Note: do not use keywords, but full-fledged questions.* The demo is not optimized to deal with keyword queries and might misunderstand you.
    """,
            unsafe_allow_html=True,
            )

        search_term = st.text_input(
            value=st.session_state.question,
            max_chars=100,
            on_change=reset_results,
            label='Enter your Question:',
            label_visibility="hidden",
            )

        pipe = ExtractiveQAPipeline(reader, retriever)

    if search_term:
        with st.spinner(
                "🧠 &nbsp;&nbsp; Performing neural search on documents... \n "
                ):

            prediction = pipe.run(
                query=search_term,
                params={"Retriever": {"top_k": top_k_retriever}, "Reader": {"top_k": top_k_reader}}
                )

            for i in range(len(prediction['answers'])):
                answer = prediction['answers'][i].answer
                context = prediction['answers'][i].context
                start_idx = context.find(answer)
                end_idx = start_idx + len(answer)
                score = round(prediction['answers'][i].score * 100, 2)
                document_id = prediction['answers'][i].document_id
                doc = [x for x in prediction['documents'] if x.id == document_id][0]
                meta = doc.meta
                url, title = meta['link'], meta['title']
                source = f"[{title}]({url})"

                if score > 0.7:
                    # Hack due to this bug: https://github.com/streamlit/streamlit/issues/3190
                    st.write(
                        markdown(context[:start_idx] + str(annotation(answer, "ANSWER", "#8ef")) + context[end_idx:]),
                        unsafe_allow_html=True,
                        )
                    st.markdown(f"**Relevance:** {score} -  **Source:** {source}")
                else:
                    st.info(
                        "🤔 &nbsp;&nbsp; Clippy is unsure whether any of the documents contain an answer to your question. Try to reformulate it!"
                        )
                    st.write("**Relevance:** ", score)


if __name__ == '__main__':
    main()



