import streamlit as st
import json
from haystack.document_stores import OpenSearchDocumentStore
from haystack.pipelines import ExtractiveQAPipeline
from haystack.schema import Document
from haystack.nodes import BM25Retriever, FARMReader
import pandas as pd
from tqdm import tqdm
import yaml


def search_result(i: int, answer: str, context: str, score: float, meta: dict, offsets_in_context) -> str:
    span_start = '<span style="border-radius: 0.4rem; color: white;font-weight: 700; padding: 0.1rem; margin-bottom: 0.5rem;background-color: lightblue;">'
    span_end = '</span>'
    span_answer = "<span style='padding-left: 1rem;font-weight: 100;font-size:70%;'> Answer </span>"
    f_context = context[:offsets_in_context.start] + span_start + answer + span_answer + span_end + context[offsets_in_context.end:]
    return f"""
        <div style="font-size:95%;">
            {i + 1}.
            <a href="{meta['link']}">
                {meta['title']}
            </a>
        </div>
        <div style="font-size:95%;">
            <div style="color:white;font-size:95%;">
                {'Answer: '}&nbsp;
            </div>
        </div>
        <div style="font-size:95%;">
                <div style="font-size:95%;">
                {f_context}
            </div>
        </div>
        <div style="font-size:95%;">
            <div style="color:grey;font-size:95%;">
                {meta['pubDate']}
            </div>
            <div style="color:grey;font-size:95%;">
                Author: {meta['author']} &nbsp;
            </div>
            <div style="float:left;">
                Score: {score} &nbsp;
            </div>
        </div>
    """


def main():
    # load settings and mappings for index

    with open('config/credentials.yaml') as f:
        credentials = yaml.load(f, Loader=yaml.FullLoader)

    with open('config/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    document_store = OpenSearchDocumentStore(host=credentials['host'],
                                             username=credentials['user'],
                                             password=credentials['password'],
                                             index=config['index_name'])

    retriever = BM25Retriever(document_store=document_store)
    reader = FARMReader(model_name_or_path="deepset/gelectra-large-germanquad", use_gpu=True)
    st.title('Ask the kununu blog:')
    search_term = st.text_input('Enter your Question:')
    pipe = ExtractiveQAPipeline(reader, retriever)

    if search_term:
        prediction = pipe.run(
            query=search_term,
            params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
            )

        for i in range(len(prediction['answers'])):
            answer = prediction['answers'][i].answer
            context = prediction['answers'][i].context
            offsets_in_context = prediction['answers'][i].offsets_in_context[0]


            score = prediction['answers'][i].score

            document_id = prediction['answers'][i].document_id
            doc = [x for x in prediction['documents'] if x.id == document_id][0]
            meta = doc.meta

            if score > 0.7:
                st.markdown(search_result(i, answer, context, score, meta, offsets_in_context), unsafe_allow_html=True)


if __name__ == '__main__':
    main()

