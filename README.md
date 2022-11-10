# hackweek_clippy
Make semantic search service based on kununu blog articles

## How to run the search engine
1. Spin up an OpenSearch instance. Instructions can be found [here](https://opensearch.org/docs/latest/opensearch/install/docker/).
2. Adapt the `config/config.yaml` to your project needs.
3. In order to start *Clippy -- the semantic search service* run `create_index.py` which loads the blog data from 
`data/kununublog.WordPress.2022-11-08.xml`, preprocesses the texts and creates an index.
4. Finally, run `streamlit run app.py`.