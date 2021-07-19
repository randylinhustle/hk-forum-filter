import pandas as pd
import os
import re
from typing import Union, List
from sklearn.feature_extraction.text import CountVectorizer


def filter_csv(
    input_filepath: str,
    output_filepath: str = "",
    title_keywords: Union[str, List] = [],
    msg_keywords: Union[str, List] = [],
    getPost: bool = True,
    categories: Union[str, List] = [],
    low_memory=False,
):
    """Filter a lihkg csv by keywords in title and category.

    Args:
        input_filepath (str): The path of the lihkg csv file.
        output_filepath (str): The save path of the filtered csv file.
        keywords (str or List): The keywords to filter.
        categories (str or List): The categories in lihkg to look for.
        low_memory (bool): If True, csv will be read in chunks to reduce
            memory usage.

    Returns:
        DataFrame: if low_memory is False.
    """
    print("Handling", input_filepath)
    os.makedirs(
        os.path.normpath(os.path.dirname(output_filepath)), exist_ok=True
    )
    if low_memory:
        header = True
        for df in pd.read_csv(
            input_filepath, encoding="utf-8-sig", chunksize=2000000, lineterminator="\n"
        ):
            df = filter_df_by_category(df=df, categories=categories)
            df = filter_df_by_title(df=df, keywords=title_keywords)
            df = filter_df_by_msg(
                df=df, keywords=msg_keywords, getPost=getPost
            )
            df.to_csv(
                output_filepath, header=header, encoding="utf-8-sig", mode="a"
            )
            header = False
        print("Dropping duplicates..")
        remove_duplicates([output_filepath], "post_id")
    else:
        df = pd.read_csv(input_filepath, encoding="utf-8-sig", lineterminator="\n")
        df = filter_df_by_category(df=df, categories=categories)
        df = filter_df_by_title(df=df, keywords=title_keywords)
        df = filter_df_by_msg(df=df, keywords=msg_keywords, getPost=getPost)
        df = df.drop_duplicates("post_id")
        return df


def filter_df_by_category(df: pd.DataFrame, categories: Union[str, List] = []):
    """
    Args:
        df (DataFrame): The DataFrame object to filter.
        categories (str, List[str]): The categories in lihkg to look for.

    Return:
        DataFrame
    """
    if isinstance(categories, str):
        categories = [categories]
    if categories:
        df = df[df["category"].isin(categories)]
    return df


def filter_df_by_title(df: pd.DataFrame, keywords: Union[str, List] = []):
    """
    Args:
        df (DataFrame): The DataFrame object to filter.
        keywords(str or List): The keywords to filter.
        category (str): The categories in lihkg to look for.

    Return:
        DataFrame

    """
    if isinstance(keywords, list):
        keywords = "|".join(keywords)

    # remove na in 'thread_title' field
    df = df[~df["thread_title"].isnull()]

    df = df[df["thread_title"].str.contains(keywords, case=False, regex=True)]
    return df


def filter_df_by_msg(
    df: pd.DataFrame, keywords: Union[str, List] = [], getPost: bool = True
):
    """
    Args:
        df (DataFrame): The DataFrame object to filter.
        keywords(str or List): The keywords to filter.
        getPost (bool): If True, return only the posts which contain the
            keywords. Otherwise, it will return the threads (and posts of it)
            which contain all those posts with the keyword. (i.e. if one post
            of a thread contains a keyword, all the posts of the thread will be
            returned.)

    Return:
        DataFrame

    """
    if isinstance(keywords, list):
        keywords = "|".join(keywords)

    # remove na in 'msg' field
    _df = df[~df["msg"].isnull()]
    _df = _df[_df["msg"].str.contains(keywords, case=False, regex=True)]

    if getPost:
        return _df
    else:
        return df[df.thread_id.isin(_df.thread_id)]


def getTopNFreqWords(csv, topn, ngrams, vocab, output_filepath):
    def preprocess(raw_html):
        cleanr = re.compile("<.*?>")
        cleantext = re.sub(cleanr, "", raw_html)
        cleantext = "".join(cleantext.split())
        return cleantext

    df = pd.read_csv(csv, encoding="utf-8-sig", usecols=["msg"])
    df = df.dropna()

    c = CountVectorizer(
        ngram_range=ngrams,
        analyzer="char",
        preprocessor=preprocess,
        max_features=topn,
        vocabulary=vocab if vocab else None,
    )

    # input to fit_transform() should be an iterable with strings
    words = c.fit_transform(df["msg"])

    # a dict where keys are terms and values are indices in the feature matrix
    vocab = c.vocabulary_

    count_values = words.sum(axis=0)

    df = pd.DataFrame(
        sorted(
            [
                {"freq": count_values[0, i], "ngram": k}
                for k, i in vocab.items()
            ],
            key=lambda x: x["freq"],
            reverse=True,
        )
    )

    if output_filepath:
        df.to_csv(output_filepath, index=False, encoding="utf-8-sig")
    else:
        return df


def remove_duplicates(csvs, field):
    """
    Args:
        csvs (List): The csv filenames.
        field (str): The fieldname in the csv to remove duplicates

    Return:
        DataFrame
    """
    _ids = set()
    unique_df = pd.DataFrame()
    for csv in csvs:
        dfs = pd.read_csv(csv, encoding="utf-8-sig", chunksize=2000000)
        for df in dfs:
            df = df[~df[field].isin(_ids)]
            unique_df = pd.concat([unique_df, df], axis=0)
            _ids.update(df[field])
    return unique_df
