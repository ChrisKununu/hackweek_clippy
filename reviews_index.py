from haystack.document_stores import OpenSearchDocumentStore
from haystack.schema import Document
import pandas as pd
import yaml
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning


requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

with open('./config/config.yaml', mode='r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

with open('./config/credentials.yaml', mode='r') as file:
    credentials = yaml.load(file, Loader=yaml.FullLoader)


relevant_cols = [
    'text_s1_original',
    'profile_ebp_status',
    'profile_industry_id',
    'profile_weighted_total',
    'review_url',
    'industry_name']

questions = [
        's1',
        's2',
        's3',
        's4',
        's5',
        's6',
        's7',
        's8',
        's9',
        's10',
        's11',
        's12',
        's13']

questions_mapping = [
        'Vorgesetztenverhalten',
        'Kollegenzusammenhalt',
        'Interessante Aufgaben',
        'Arbeitsatmosphäre',
        'Kommunikation',
        'Gleichberechtigung',
        'Karriere/Weiterbildung',
        'Gehalt/Sozialleistungen',
        'Arbeitsbedingungen',
        'Work-Life-Balance',
        'Image',
        'Umgang mit älteren Kollegen',
        'Umwelt-/Sozialbewusstsein']

mapping = dict(zip(questions, questions_mapping))


def extract_document(
    data: pd.DataFrame = None,
    question: str | None = 's1',
    profile_uuid: str = '304de031-0ca6-45fe-b57e-99209d358551',
    min_length: int = 20,
    mapping: dict = mapping,
    columns: list = relevant_cols) -> list[Document]:

    """
    ingest Dataframe from load_reviews.py and create a list of OS documents
    @param data: review data
    @param question: which text to extract
    @param profile_uuid: which profile to filter on
    @param min_length: minimum words
    @param columns: columns to add
    """

    columns.append(f's_{question[1:]}')
    columns.append(f'text_{question}')
    columns.append(f'text_{question}_original')

    print(columns)
    print(data.head())
    # filter data
    data = data.loc[
    (data['profile_uuid'] == profile_uuid) &
    (data[f'text_{question}_len'] >= min_length) &
    (data[f'text_{question}_language'] == 'de') &
    (data[f'text_{question}_language_confidence'] >= .75),
    columns]

    if data.empty:
        return None

    result = list()

    for index, values in data.to_dict(orient='index').items():
        d = Document(
            content=values[f'text_{question}_original'],
            meta={
                'stars': values[f's_{question[1:]}'],
                'ebp': values['profile_ebp_status'],
                'industry': values['industry_name'],
                'review_url': values['review_url'],
                'category': mapping[question],
                'stars_profile': values['profile_weighted_total']})

        result.append(d)

        # todo: paragraph split

    return result


if __name__ == '__main__':
    from pprint import pprint

    ds = OpenSearchDocumentStore(
        host=credentials['host'],
        username=credentials['user'],
        password=credentials['password'],
        index=config['review_index'])

    ds.delete_index(config['review_index'])

    ds = OpenSearchDocumentStore(
        host=credentials['host'],
        username=credentials['user'],
        password=credentials['password'])

    with open('./data/processed_reviews.parquet', mode='rb') as file:
        df = pd.read_parquet(file)

    with open('./industry_mapping.csv', mode='r') as file:
        df_industry = pd.read_csv(file)
        df_industry.rename(columns={i: f'industry_{i}' for i in df_industry.columns}, inplace=True)

    df = pd.merge(
        left=df.loc[~df['profile_industry_id'].isna()],
        right=df_industry[['industry_id', 'industry_name']],
        how='inner',
        left_on='profile_industry_id',
        right_on='industry_id')


    for i in questions:

        print(f'processing texts f{mapping[i]}')
        extract = extract_document(df, question=i)

        if extract:
            pprint(extract[0].to_dict())
            ds.write_documents(extract, index=config['review_index'])

    ds.describe_documents(index=config['review_index'])

