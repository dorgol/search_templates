import json
import pandas as pd
import os
import yaml
from src.get_data import SingleQuery

with open('config.yaml') as f:
    config = yaml.safe_load(f)


def fetch_data(query_path, data_path, reload=False, pp_function=None):
    """
            This function checks if the tags data exists, if not it queries it
            :return: DataFrame. df of tags
            """
    is_exist = os.path.isfile(data_path)
    if is_exist and not reload:
        rows = []
        with open('/Users/dgoldenberg/PycharmProjects/encode_templates/' + data_path, 'r') as file:
            for line in file:
                rows.append(json.loads(line))
        df = pd.DataFrame(rows)
        return df
    else:
        sq = SingleQuery(path_to_query=query_path)
        if pp_function is not None:
            sq.post_process = pp_function
        sq.run_query(result="bq", dataset_id="templates_analysis", table_id="tags_originals",
                     bucket_name="templates_analysis", file_name="tags_originals.json")


fetch_data(config['paths']['ORIGINAL_QUERY_PATH'], config['paths']['TAGS_PATH'], reload=True)
