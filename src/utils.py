from haystack.document_stores import OpenSearchDocumentStore
from haystack.schema import Document
import pandas as pd
from tqdm import tqdm
import yaml


def generate_documents(data: pd.DataFrame,
                       min_words: int) -> list:
    documents = []
    for idx, item in tqdm(data.T.items()):
        meta = item.drop(labels=['content']).to_dict()
        for text in item.content.split('\n\n'):
            if len(text.split()) > min_words and '?' not in text:
                document = Document(content=text, meta=meta, id_hash_keys=['content'])
                documents.append(document)
    return documents


def create_index(data: pd.DataFrame,
                 min_words: int) -> None:

    with open('config/credentials.yaml') as f:
        credentials = yaml.load(f, Loader=yaml.FullLoader)

    with open('config/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    document_store = OpenSearchDocumentStore(host=credentials['host'],
                                             username=credentials['user'],
                                             password=credentials['password'],
                                             index=config['index_name'])
    # delete all documents just to be safe
    document_store.delete_index(config['index_name'])

    # preprocess input data for indexing
    print("Generating Documents from the files...")
    documents = generate_documents(data=data, min_words=min_words)

    # write the dicts containing documents to Opensearch
    print("Writing Documents to the Document store...")
    document_store.write_documents(documents, index=config['index_name'])

    document_store.describe_documents(index=config['index_name'])


if __name__ == '__main__':
    create_index()
