import tensorflow as tf

'''
padding 补零 加长
pooling 截断 缩短
'''

imbd = tf.keras.datasets.imdb
vocab_size = 10000  # 统计10000个词语
index_from = 3  # 索引偏移3
# [25000,] [25000,] [25000,] [25000,] X是变长样本 每个X长度不定
(X_train, y_train), (X_test, y_test) = imbd.load_data(num_words=vocab_size, index_from=index_from)
print(X_train[0])

# 词表 [88584,] 字典类型 word:index
word_index = imbd.get_word_index()
# index_from偏移3 校正
word_index = {k: (v + index_from) for k, v in word_index.items()}
# 加入特殊字符
word_index["<PAD>"] = 0  # 补0
word_index["<START>"] = 1  # 句首
word_index["<UNK>"] = 2  # 未找到
word_index["<END>"] = 3  # 句尾

# 词表反转
# word_index_reverse = dict([(v, k) for k, v in word_index.items()])
#
# # 把单词拼接为句子
# def decode_review(text_ids):
#     return " ".join([word_index_reverse.get(word_id, "<UNK>") for word_id in text_ids])  # 默认返回<UNK>
#
#
# print(decode_review(X_train[0]))

# padding
max_length = 256
X_train = tf.keras.preprocessing.sequence.pad_sequences(
    X_train,  # 待处理句子
    value=word_index["<PAD>"],  # 填充的值
    padding="post",  # 尾填充
    maxlen=max_length  # 最大长度
)
X_test = tf.keras.preprocessing.sequence.pad_sequences(
    X_test,  # 待处理句子
    value=word_index["<PAD>"],  # 填充的值
    padding="post",  # 尾填充
    maxlen=max_length  # 最大长度
)
print(X_train[0])

# 训练
embedding_dim = 16
batch_size = 128

model = tf.keras.Sequential()
# [vocab_size,embedding_dim] -> [batch_size,max_length,embedding_dim]
model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length))
# [batch_size,max_length,embedding_dim] -> [batch_size,embedding_dim]
model.add(tf.keras.layers.GlobalAvgPool1D())
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
model.summary()

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
model.fit(X_train, y_train, batch_size=batch_size, epochs=30, validation_split=0.2)

model.evaluate(X_test, y_test, batch_size=batch_size)
