import pandas as pd

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


class GephiPreparation:
    @staticmethod
    def prepare_text(texts: pd.Series, out_path: str):
        """_summary_
        :param texts: _description_
        :type texts: pd.Series
        :param out_path: _description_
        :type out_path: str
        """
        # Convert a collection of text documents to a matrix of token counts
        cv = CountVectorizer(ngram_range=(1, 1), stop_words=stopwords.words("spanish"))
        # matrix of token counts
        X = cv.fit_transform(texts)
        Xc = X.T * X  # matrix manipulation
        Xc.setdiag(0)  # set the diagonals to be zeroes as it's pointless to be 1
        names = cv.get_feature_names_out()  # This are the entity names (i.e. keywords)
        df = pd.DataFrame(data=Xc.toarray(), columns=names, index=names)
        df.to_csv(out_path, sep=",")
