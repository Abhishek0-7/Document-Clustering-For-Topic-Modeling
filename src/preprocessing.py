import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

COMMON_WORDS = set("""
would could should one two also said say get go know think see come way time
people make good well back much first may new like use man day right work old
want however something take still even many really need look find give year point
different lot tell seem ask become fact hand high keep last leave move part place
put show small turn try follow mean help talk set end why call might next problem
group important often example always long feel life little public social such thing
though without world write young
""".split())

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens
              if token not in stop_words and len(token) > 2]
    
    return " ".join(tokens)

def advanced_preprocess_text(text):
    text = preprocess_text(text)
    tokens = [t for t in text.split() if t not in COMMON_WORDS]
    return " ".join(tokens)
