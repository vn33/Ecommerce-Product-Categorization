import streamlit as st
import joblib
import re, nltk
from nltk.corpus import stopwords
from nltk.tokenize import  RegexpTokenizer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')

# Load the model and vectorizer
best_ridge = joblib.load('best_ridge_model.pkl')
TfidfVec = joblib.load('tfidf_vectorizer.pkl')

# RegexpTokenizer
regexp = RegexpTokenizer("[\w']+")

# Stopwords
stops = stopwords.words("english")  # stopwords

# Common words in English
alphabets = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
             "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

# Prepositions
prepositions = ["about", "above", "across", "after", "against", "among",
                "around", "at", "before", "behind", "below", "beside", "between",
                "by", "down", "during", "for", "from", "in", "inside", "into", "near",
                "of", "off", "on", "out", "over", "through", "to", "toward", "under",
                "up", "with"]

# Less common prepositions
prepositions_less_common = ["aboard", "along", "amid", "as", "beneath", "beyond", "but",
                            "concerning", "considering", "despite", "except", "following",
                            "like", "minus", "onto", "outside", "per", "plus", "regarding",
                            "round", "since", "than", "till", "underneath", "unlike",
                            "until", "upon", "versus", "via", "within", "without"]

# Coordinating conjunctions
coordinating_conjunctions = ["and", "but", "for", "nor", "or", "so", "and", "yet"]

# Correlative conjunctions
correlative_conjunctions = ["both", "and", "either", "or", "neither", "nor", "not",
                            "only", "but", "whether", "or"]

# Subordinating conjunctions
subordinating_conjunctions = ["after", "although", "as", "as if", "as long as", "as much as",
                              "as soon as", "as though", "because", "before", "by the time",
                              "even if", "even though", "if", "in order that", "in case",
                              "in the event that", "lest", "now that", "once", "only", "only if",
                              "provided that", "since", "so", "supposing", "that", "than",
                              "though", "till", "unless", "until", "when", "whenever", "where",
                              "whereas", "wherever", "whether or not", "while"]

# Other words
others = ["ã", "å", "ì", "û", "ûªm", "ûó", "ûò", "ìñ", "ûªre", "ûªve", "ûª",
          "ûªs", "ûówe"]
# Additional stopwords
addstops = ["among", "get", "onto", "shall", "thrice", "thus", "twice", "unto", "us",
            "would"]
# Common words in ecommerce contexts
common_ecommerce_words = ["shop", "shops", "shopping", "buy", "genuine", "product",
                          "store", "stores", "day", "replacement", "good", "description",
                          "purchase", "purchases", "checkout", "cart", "details", "detail",
                          "discount", "discounts", "offer", "offers", "specification",
                          "deal", "deals", "sale", "sales", "item", "items",
                          "voucher", "vouchers", "coupon", "coupons",
                          "promo", "promos", "promotion", "promotions",
                          "buying", "selling", "seller", "sellers",
                          "buyer", "buyers", "payment", "payments",
                          "checkout", "free", "order", "orders", "available",
                          "return", "returns", "exchange", "exchanges",
                          "refund", "refunds", "customer", "customers",
                          "service", "services", "support", "feedback",
                          "review", "reviews", "rating", "ratings",
                          "online", "offline", "delivery", "shipping",
                          "shipped", "ship", "track", "tracking", "cash",
                          "payment", "prices", "price", "rs.", "rs", "select", "selected",
                          "transaction", "transactions", "secure", "key", "feature", "features",
                          "guarantee", "guaranteed", "fast", "quick",
                          "easy", "convenient", "reliable", "trustworthy",
                          "safe", "secure", "doorstep", "discounted",
                          "affordable", "cheap", "low", "high", "best",
                          "popular", "top", "quality", "brand", "brands",
                          "stock", "new", "latest", "trending", "hot", "exclusive"]

# since it is product categorization, platforms names won't help
ecommerce_platforms = ["flipkart", "amazon", "mintra", "snapdeal"]
allstops = stops + alphabets + prepositions + prepositions_less_common + coordinating_conjunctions + correlative_conjunctions + subordinating_conjunctions + others + addstops + common_ecommerce_words + ecommerce_platforms


# Define the text normalization functions
def convert_to_lowercase(text):
    return text.lower()


def remove_whitespace(text):
    return " ".join(text.split())


def remove_http(text):
    http = "https?://\S+|www\.\S+" # matching strings beginning with http (but not just "http")
    pattern = r"({})".format(http) # creating pattern
    return re.sub(pattern, "", text)


def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)


def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)


def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags = re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Lemmatization
# spacy_lemmatizer = spacy.load("en_core_web_sm", disable = ['parser', 'ner'])
#
#
# def text_lemmatizer(text):
#     text_spacy = " ".join([token.lemma_ for token in spacy_lemmatizer(text)])
#     #text_wordnet = " ".join([lemmatizer.lemmatize(word) for word in word_tokenize(text)]) # regexp.tokenize(text)
#     return text_spacy


def discard_non_alpha(text):
    return re.sub(r'[^a-zA-Z\s]', '', text)


def keep_pos(text):
    word_list_non_alpha = [word for word in regexp.tokenize(text) if word.isalpha()]
    text_non_alpha = " ".join(word_list_non_alpha)
    return text_non_alpha


def remove_stopwords(text):
    return " ".join([word for word in regexp.tokenize(text) if word not in allstops])


def text_normalizer(text):
    text = convert_to_lowercase(text)
    text = remove_whitespace(text)
    text = re.sub('\n', '', text)
    text = re.sub(r'\.com\b', '', text)
    text = re.sub('\[.*?\]', '', text)
    text = remove_http(text)
    text = remove_punctuation(text)
    text = remove_html(text)
    text = remove_emoji(text)
    # text = text_lemmatizer(text)
    text = discard_non_alpha(text)
    text = keep_pos(text)
    text = remove_stopwords(text)
    return text


# Define category mapping
category_mapping = {
    0: 'Clothing',
    1: 'Footwear',
    2: 'Pens & Stationery',
    3: 'Bags, Wallets & Belts',
    4: 'Home Decor & Festive Needs',
    5: 'Automotive',
    6: 'Tools & Hardware',
    7: 'Baby Care',
    8: 'Mobiles & Accessories',
    9: 'Watches',
    10: 'Toys & School Supplies',
    11: 'Jewellery',
    12: 'Kitchen & Dining',
    13: 'Computers'
}

# Streamlit app
st.title('Ecommerce Product Categorization')
st.write('Enter the product description to predict its category.')

input_text = st.text_area('Product Description')

if st.button('Predict'):
    # Normalize the input text
    normalized_text = text_normalizer(input_text)

    # Vectorize the input text
    input_tfidf = TfidfVec.transform([normalized_text])

    # Predict the category
    prediction = best_ridge.predict(input_tfidf)

    # Get the category name
    predicted_category = category_mapping[prediction[0]]

    st.write(f'The predicted category is: **{predicted_category}**')

