import os


from NER.general_utils import get_logger
from NER.data_utils import get_word2vec_vocab


class Config():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()


    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        # self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags  = {'O': 0, 'B-ORG': 1, 'B-PER': 2, 'B-LOC': 3, 'I-ORG': 4, 'I-PER': 5, 'I-LOC': 6}
        self.idx2Tag =dict(zip(self.vocab_tags.values(),self.vocab_tags.keys()))
        # self.vocab_chars = load_vocab(self.filename_chars)
        print('check: ',self.filename_word2vec)
        self.nwords, self.vocab_words, self.id2Word, self.embeddings = get_word2vec_vocab(self.filename_word2vec)

        self.nwords     = len(self.vocab_words)
        # self.nchars     = len(self.vocab_chars)
        # self.ntags      = len(self.vocab_tags)
        self.ntags = 7
        # 2. get processing functions that map str -> id
        # self.processing_word = get_processing_word(self.vocab_words,
        #         self.vocab_chars, lowercase=True, chars=self.use_chars)
        # self.processing_tag  = get_processing_word(self.vocab_tags,
        #         lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings



    # general config
    dir_output = "./results/test/"
    dir_model  = dir_output + "model.weights/"
    path_log   = dir_output + "log.txt"


    # embeddings
    dim_word = 300
    dim_char = 100

    # word2vec files
    filename_word2vec = "/content/drive/My Drive/AI_COLAB/DATA/model_trained/VNCorpus_W2VN_big.bin"
    use_pretrained = True

    # dataset
    filename_dev = "/content/drive/My Drive/AI_COLAB/DATA/VLSP/dev.txt"
    filename_test = "/content/drive/My Drive/AI_COLAB/DATA/VLSP/test.txt"
    filename_train = '/content/drive/My Drive/AI_COLAB/DATA/VLSP/train.txt'

    # filename_dev = filename_test = filename_train = "data/test.txt" # test

    max_iter = None # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_words = "data/words.txt"
    filename_tags = "data/tags.txt"
    filename_chars = "data/chars.txt"

    # training option
    train_embeddings = False
    nepochs          = 15
    dropout          = 0.5
    batch_size       = 20
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = 3 # if negative, no clipping
    nepoch_no_imprv  = 3

    # model hyperparameters
    hidden_size_char = 100 # lstm on chars
    hidden_size_lstm = 300 # lstm on word embeddings

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True # if crf, training is 1.7x slower on CPU
    use_chars = False # if char embedding, training is 3.5x slower on CPU
