import pandas as pd
from google.cloud import bigquery
from google.cloud import storage
import yaml
import typing
import pathlib
import os.path

with open('config.yaml') as f:
    config = yaml.safe_load(f)


class SingleQuery:
    """
    A class for running a single query on BigQuery.

    Args:
        path_to_query (str or pathlib.Path): The path to the SQL file containing the query.

    Attributes:
        client (google.cloud.bigquery.client.Client): A client object for interacting with BigQuery.
        path (str or pathlib.Path): The path to the SQL file containing the query.
        query (str): The contents of the SQL file.

    Methods:
        run_query(result: str) -> Union[pd.DataFrame, None]:
            Runs the query on BigQuery and returns the result as a DataFrame or None.
        post_process(df: pd.DataFrame) -> pd.DataFrame:
            A static method for performing post-processing on the query result.

    Example:
        >>> from src.get_data import SingleQuery
        >>> query = SingleQuery('queries/templates_with_features.txt')
        >>> result = query.run_query('df')
        >>> processed_result = SingleQuery.post_process(result)
    """

    def __init__(self, path_to_query: typing.Union[str, pathlib.Path]):
        """
        Initializes a new SingleQuery object.

        Args:
            path_to_query (str or pathlib.Path): The path to the SQL file containing the query.
        """
        self.client = bigquery.Client(location=config['cloud']['CLOUD_LOCATION'],
                                      project=config['cloud']['PROJECT'])
        self.path = path_to_query
        with open(self.path, "r") as f:
            self.query = f.read()

    def run_query(self, result: str, dataset_id: str = None, table_id: str = None,
                  bucket_name: str = None, file_name: str = None) -> typing.Union[pd.DataFrame, None]:
        """
        Runs the query on BigQuery and returns the result as a DataFrame or None.

        Args:
            result (str): Either "df" to return a DataFrame or "bq" to just run the query.

        Returns:
            Union[pd.DataFrame, None]: If result is "df", returns the query result as a pandas DataFrame.
                If result is "bq", returns None.

        Raises:
            ValueError: If result is not "df" or "bq".
            :param file_name:
            :param bucket_name:
            :param result:
            :param table_id:
            :param dataset_id:
        """
        # Get the BigQuery job done
        job_config = bigquery.QueryJobConfig(destination=f"{self.client.project}.{dataset_id}.{table_id}")
        query_job = self.client.query(
            self.query,
            location=config['cloud']['CLOUD_LOCATION'],
            job_config=job_config
        )
        if result == "bq":
            query_job.result()
            if query_job.error_result is not None:
                print(f"Error occurred: {query_job.error_result['message']}")
            else:
                print(f"Query results saved to: {dataset_id}.{table_id}")

            # save results to gcs
            destination_uri = "gs://{}/{}".format(bucket_name, file_name)
            dataset_ref = bigquery.DatasetReference(self.client.project, dataset_id)
            table_ref = dataset_ref.table(table_id)
            job_config = bigquery.job.ExtractJobConfig()
            job_config.destination_format = bigquery.DestinationFormat.NEWLINE_DELIMITED_JSON

            extract_job = self.client.extract_table(
                table_ref,
                destination_uri,
                job_config=job_config,
                # Location must match that of the source table.
                location="US",
            )  # API request
            extract_job.result()  # Waits for job to complete.

            gcs_client = storage.Client(project=config['cloud']['PROJECT'])
            bucket = gcs_client.bucket(bucket_name)
            blob = bucket.blob(file_name)
            # blob.download_to_filename(config['paths']['TEMPLATES_PATH'])
            blob.download_to_filename('data/' + file_name)

        elif result == "df":
            df = query_job.to_dataframe()
            df = self.post_process(df)
            return df
        else:
            raise ValueError(
                'result must be either "bq" for running BigQuery action or "df" for returning DataFrame'
            )

    @staticmethod
    def post_process(df: pd.DataFrame) -> pd.DataFrame:
        """
        A static method for performing post-processing on the query result. Can be override

        Args:
            df (pd.DataFrame): The query result as a pandas DataFrame.

        Returns:
            pd.DataFrame: The processed DataFrame.
        """
        return df


def fetch_data(query_path, data_path, reload=False, pp_function=None):
    """
            This function checks if the tags data exists, if not it queries it
            :return: DataFrame. df of tags
            """
    is_exist = os.path.isfile(data_path)
    if is_exist and not reload:
        df = pd.read_csv(data_path, header=0)
        return df
    else:
        sq = SingleQuery(path_to_query=query_path)
        if pp_function is not None:
            sq.post_process = pp_function
        sq.run_query(result="bq", dataset_id="templates_analysis", table_id="templates",
                     bucket_name="templates_analysis", file_name="outputs.json")


def main(reload=True):
    def _post_process(df):
        import ast
        import re

        def _parse_lists(x):
            x = str(x)
            x = re.sub(r'(?<=\[)\s+', '', x)
            x = re.sub(r' +', ',', x)
            x = re.sub(r'\n', '', x)
            x = ast.literal_eval(x)
            return x

        new_df = df
        new_df['feature_types'] = new_df['feature_types'].apply(_parse_lists)
        new_df['start_times'] = new_df['start_times'].apply(_parse_lists)
        new_df['feature_durations'] = new_df['feature_durations'].apply(_parse_lists)

        new_df = new_df.explode(['feature_types', 'start_times', 'feature_durations'])
        return new_df

    fetch_data(config['paths']['TEMPLATES_QUERY_PATH'], config['paths']['TEMPLATES_PATH'], pp_function=_post_process,
               reload=reload)


if __name__ == '__main__':
    main()
