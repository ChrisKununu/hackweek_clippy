import xml.etree.ElementTree

import pandas as pd
import re
import xml.etree.ElementTree as ET
import yaml
from src.utils import create_index


def parse_xml(path: str) -> xml.etree.ElementTree.Element:
    tree = ET.parse(path)
    root = tree.getroot()
    return root


def load_data(r: xml.etree.ElementTree.Element) -> list:
    data = []
    for x in r[0].findall('item'):
        title = x.find('title').text
        html_string = x[6].text
        result = re.sub(r'<.*?>', '', html_string)
        results = result.split('\n\n\n\n')
        results = [x.replace("\n", "") for x in results]
        results = "\n\n".join(results)
        data.append({
                             "title": title,
                             "content": results,
                             "pubDate": x.find('pubDate').text,
                             "link": x[4].text,
                             "author": x[3].text})
    return data


if __name__ == '__main__':

    with open('config/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    file_path = config['input_file_path']
    root = parse_xml(file_path)
    blog_data = load_data(root)
    df = pd.DataFrame(blog_data)
    print(f"Loaded {df.shape[0]} Articles!")

    with open(config['output_file_path'], mode='wb') as file:
        df.to_parquet(file)

    create_index(data=df, min_words=12)

