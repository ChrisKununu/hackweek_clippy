import os
import re
import pandas as pd
import fasttext
from nltk.corpus import stopwords


model_path = "./models/lid.176.bin"
model = fasttext.load_model(model_path)
parquet_path = os.path.join(os.curdir, 'data/reviews.parquet')
query_path = os.path.join(os.curdir, 'queries/reviews.sql')
data_path = os.path.abspath('data/')

text_star_map = {
    None: '[No Filter]',
#    nan: 'prak_job',
    'text_s1': 's_1',
    'text_s10': 's_10',
    'text_s11': 's_11',
    'text_s12': 's_12',
    'text_s13': 's_13',
    'text_s2': 's_2',
    'text_s3': 's_3',
    'text_s4': 's_4',
    'text_s5': 's_5',
    'text_s6': 's_6',
    'text_s7': 's_7',
    'text_s8': 's_8',
    'text_s9': 's_9'}

text_mapping = {
    None: '[No Filter]',
#    nan: 'Hat in diesem Jahr ein Praktikum abgeschlossen',
    'text_s1': 'Vorgesetztenverhalten',
    'text_s10': 'Work-Life-Balance',
    'text_s11': 'Image',
    'text_s12': 'Umgang mit älteren Kollegen',
    'text_s13': 'Umwelt-/Sozialbewusstsein',
    'text_s2': 'Kollegenzusammenhalt',
    'text_s3': 'Interessante Aufgaben',
    'text_s4': 'Arbeitsatmosphäre',
    'text_s5': 'Kommunikation',
    'text_s6': 'Gleichberechtigung',
    'text_s7': 'Karriere/Weiterbildung',
    'text_s8': 'Gehalt/Sozialleistungen',
    'text_s9': 'Arbeitsbedingungen'}


def load_reviews_gbq(path: os.PathLike = query_path) -> pd.DataFrame:
    with open(path, mode='r') as query_file:
        reviews_query = query_file.read()
    df = pd.read_gbq(reviews_query, project_id='kununu-dwh', use_bqstorage_api=True)

    return df


def load_reviews_local(path: os.PathLike = parquet_path) -> pd.DataFrame:
    df = pd.read_parquet(path=path)

    return df


def process_reviews(
        data: pd.DataFrame,
        save_parts: bool = True) -> pd.DataFrame:

    # set convenient url
    data['review_url'] = 'www.kununu.com' + data['profile_url_short'] + '/bewertung/' + data['review_uuid']

    for column in filter(lambda x: x.startswith('text_s') | (x == 'improvement_text'), data.columns):
        print(f'processing column: {column}')

        # first copy over the unprocessed text
        data[f'{column}_original'] = data[column]

        # lower, remove special chars
        data[column] = data[column].str.lower()
        data[column] = data[column].str.replace(pat=r'[0-9]', repl='', regex=True)
        data[column] = data[column].str.replace(pat=r'ß', repl='ss', regex=True)
        # todo: before removing non-whitespace, split multi-paragraph answers into PARTS (for topic detection
        data[column] = data[column].str.replace(pat=r'\W+', repl=' ', regex=True)

        sw = stopwords.words('german')
        sw.remove('nicht')
        #    sw += stopwords.words('english')

        # add z and b for all the various spellings of z.b.
        sw += ['z', 'b']

        sw_replacemap = {k: '' for k in sw}
        sw_regex = '|'.join(r'\b%s\b' % re.escape(s) for s in sw_replacemap)

        # remove all stopwords
        data[column] = data[column].str.replace(pat=sw_regex, repl='', regex=True)
        # trim, split, rejoin to remove redundant spaces
        data[column] = data[column].str.strip().str.split().str.join(' ')

        data[f'{column}_len'] = data[f'{column}_original'].str.split(' ').str.len()
        data[f'{column}_language'] = data[column].apply(lambda x: model.predict(x) if x is not None else None)
        data[f'{column}_language_confidence'] = data[f'{column}_language'].apply(lambda x: x[1][0] if x is not None else None)

        data[f'{column}_language'] = data[f'{column}_language'].apply(lambda x: x[0][0].split('__')[-1] if x is not None else None)

        if save_parts:
            print(f'saving data for column: {column} to path: {data_path}/{column}.parquet')

            part = data.copy()
            part = part.dropna(subset=f'{column}')

            part = part.rename(
                columns={
                    f'{column}_len': 'number_words',
                    f'{column}_language': 'language',
                    f'{column}_language_confidence': 'language_confidence',
                    f'{column}': 'text',
                    f'{column}_original': 'text_original'})

            part_columns = [
                    'review_uuid',
                    'review_url',
                    'number_words',
                    'text',
                    'text_original',
                    'language',
                    'language_confidence']

            if column in text_star_map.keys():
                print(f'stars available for column: {column} adding star column: {text_star_map[column]}')
                part_columns.append(text_star_map[column])
		# todo: rename column to 'stars' for reasons

            with open(os.path.join(data_path, f'{column}.parquet'), mode='wb') as part_file:
                part[part_columns].to_parquet(part_file)

            del part

    return data


def load_opensearch(index_name: str = None):
    pass


if __name__ == '__main__':

    print('loading')

    pd.set_option(
        'display.width', 5000,
        'display.max_columns', 60,
        'display.max_colwidth', 80)

    test_data = load_reviews_gbq()

    print('processing')
    test_data = process_reviews(test_data)
#    print(test_data.head())

    with open('./data/processed_reviews.parquet', mode='wb') as file:
        test_data.to_parquet(file)
