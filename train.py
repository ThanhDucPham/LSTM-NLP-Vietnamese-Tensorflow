from NER.data_utils import prepare_data, score
from NER.ner_model import NERModel
from NER.config import Config

# create instance of config
config = Config()

# build model
model = NERModel(config)
model.build()

# create datasets
train_sents, train_labels, length_sentences = prepare_data(config.filename_train,config.vocab_words)
dev_sents, dev_labels, length_sentences_dev = prepare_data(config.filename_dev, config.vocab_words,colum=[0,3])
# train model
model.train(train_sents, train_labels, dev_sents,dev_labels)

# a = [0, 1, 2, 0, 1, 2,3]
# b =(get_chunks(a,config.vocab_tags))
# c = [0, 1, 1, 0, 0, 2,1]
# d =get_chunks(c,config.vocab_tags)
# print(b)
# print(d)
# re = score(a,c,config.vocab_tags)
# print(re)