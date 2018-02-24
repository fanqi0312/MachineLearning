import jieba.analyse
import numpy
import pandas as pd

"""
数据问题：
1.非常多
"""
df_news = pd.read_table('./data/val.txt', names=['category', 'theme', 'URL', 'content'], encoding='utf-8')
# 删除有缺失数据
df_news = df_news.dropna()
print(df_news.head())

print(df_news.shape)  # (5000, 4)

# 要求是list格式
content = df_news.content.values.tolist()
print(content[1000])

content_S = []
for line in content:
    # 逐行分词
    current_segment = jieba.lcut(line)
    if len(current_segment) > 1 and current_segment != '\r\n':  # 换行符
        content_S.append(current_segment)

print(content_S[1000])

df_content = pd.DataFrame({'content_S': content_S})
print(df_content.head())

# 去停用词
stopwords = pd.read_csv("stopwords.txt", index_col=False, sep="\t", quoting=3, names=['stopword'], encoding='utf-8')
stopwords.head(20)


def drop_stopwords(contents, stopwords):
    contents_clean = []
    all_words = []
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
            all_words.append(str(word))
        contents_clean.append(line_clean)
    return contents_clean, all_words
    # print (contents_clean)


contents = df_content.content_S.values.tolist()
stopwords = stopwords.stopword.values.tolist()
contents_clean, all_words = drop_stopwords(contents, stopwords)

df_content = pd.DataFrame({'contents_clean': contents_clean})
df_content.head()

df_all_words = pd.DataFrame({'all_words': all_words})
df_all_words.head()

words_count = df_all_words.groupby(by=['all_words'])['all_words'].agg({"count": numpy.size})
words_count = words_count.reset_index().sort_values(by=["count"], ascending=False)
words_count.head()

index = 2400
print(df_news['content'][index])
content_S_str = "".join(content_S[index])

# 提取关键词
print("  ".join(jieba.analyse.extract_tags(content_S_str, topK=5, withWeight=False)))

from gensim import corpora
import gensim

# 做映射，相当于词袋
dictionary = corpora.Dictionary(contents_clean)
corpus = [dictionary.doc2bow(sentence) for sentence in contents_clean]

# 类似Kmeans自己指定K值，20类越准越好
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)

# 一号分类结果， 数据量越大越好，至少几十万。
print(lda.print_topic(1, topn=5))

for topic in lda.print_topics(num_topics=20, num_words=5):
    print(topic[1])

df_train = pd.DataFrame({'contents_clean': contents_clean, 'label': df_news['category']})
df_train.tail()

df_train.label.unique()

label_mapping = {"汽车": 1, "财经": 2, "科技": 3, "健康": 4, "体育": 5, "教育": 6, "文化": 7, "军事": 8, "娱乐": 9, "时尚": 0}
df_train['label'] = df_train['label'].map(label_mapping)
df_train.head()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df_train['contents_clean'].values, df_train['label'].values,
                                                    random_state=1)

# x_train = x_train.flatten()
x_train[0][1]

words = []
for line_index in range(len(x_train)):
    try:
        # x_train[line_index][word_index] = str(x_train[line_index][word_index])
        words.append(' '.join(x_train[line_index]))
    except:
        print(line_index, word_index)
words[0]

print(len(words))

classifier.score(vec.transform(test_words), y_test)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(analyzer='word', max_features=4000, lowercase=False)
vectorizer.fit(words)

from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
classifier.fit(vectorizer.transform(words), y_train)

classifier.score(vectorizer.transform(test_words), y_test)
