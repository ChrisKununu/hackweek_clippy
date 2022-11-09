import pandas as pd
import re
import xml.etree.ElementTree as ET


def parse_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    return root


def load_data(r):
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
                             "link": x.find('link').text,
                             "author": x[3].text})
    return data


if __name__ == '__main__':
    file_path = 'data/blog_data/kununublog.WordPress.2022-11-08.xml'
    root = parse_xml(file_path)
    blog_data = load_data(root)
    df = pd.DataFrame(blog_data)
    print(f"Loaded {df.shape[0]} Articles!")

    with open('data/blog_data/kununublog.parquet', mode='wb') as file:
        df.to_parquet(file)

