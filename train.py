"""
usaege
$ python train.py
File structure required - check the Config + *Mode classes.

The entry point for the main part of the project.
Check the train() method at the bottom for the actual logic.
"""
from datetime import timedelta
import json,logging,logging,gensim,time,string
import numpy as np
from gensim import corpora
import gensim.models.ldamulticore as models
from gensim.parsing.preprocessing import STOPWORDS
from util import *

english_alphabet = string.ascii_letters
#GLOBALS

# stores the configuration for the current run.
config = None
logger = logging.getLogger('app')

def inRange(index, r):
    """
    checks if the index is in the range @r.
    :param index:
    :param r: tuple (int,int)
    :return: bool
    """
    min_r,max_r = r
    return index >= min_r and index < max_r
def sizeOfRange(r):
    min_r, max_r = r
    return len(range(min_r, max_r))

class Config(object):
    MIXED_DOCUMENT_OVERALL_LENGTH = 200
    TEXT_LINE_SUM_OF_SCORES = 100000  # as from the project description
    SCALE_DOWN_FACTOR_TEXT_SCORE = 200
    VISUAL_WORD_DIM = 4096
    SCALE_DOWN_FACTOR_VISUAL_SCORE = 10
    VISUAL_WORD_ID_PREFIX = 'visiword_'
    TEXTUAL_WORD_ID_PREFIX = 'text_word_'
    WORKERS = 10
    lda_passes = 4
    lda_topics = 500
    visual_matrix_suffix = '.converted_to_numpy.npy'
    model_folder = 'data/models/'
    chunksize = 2000
    dict_size= 0
    best_n_visual_features= 100
    test_result_path = 'data/results/'
    test_result_file = None

# the Mode determines from where to take the data
class DevMode(Config):
    text_documents_file_path = 'data/main/DevData/parsed.text'
    image_vectors_file_path = 'data/main/DevData/parsed.visual'
    img_ids_file_path = 'data/main/DevData/scaleconcept16.teaser_dev.ImgID.txt'
    number_of_img_ids = 3339

class FullMode(Config):
    text_documents_file_path = 'data/main/Features/Textual/train_data.txt'
    image_vectors_file_path = 'data/main/Features/Visual/Visual/scaleconcept16_data_visual_vgg16-relu7.dfeat'
    img_ids_file_path = 'data/main/Features/scaleconcept16_ImgID.txt_Mod'
    number_of_img_ids = 510123

# different chunks of the input text
class AwsTrain(FullMode):
    range_of_documents_indeces = (0, 310112)
    dictionary_label = 'aws_train'

class DevTrain(DevMode):
    range_of_documents_indeces = (0, 3000)
    dictionary_label='dev_train'

class FullTrain(FullMode):
    range_of_documents_indeces = (0, 270000)
    dictionary_label = 'full_train'

class DevTest(DevMode):
    range_of_documents_indeces = (3000, 3340)
    dictionary_label = 'dev_test'

class FullTest(FullMode):
    range_of_documents_indeces = (0, 10)
    dictionary_label = 'full_test'

def wordIsGarbage(word):
    for char in word:
        if char not in english_alphabet:
            return True
    return False


def preprocessWordFromInputLine(word):
    word = word.strip()
    if word[-2:] == '\'s':
        word = word[0:len(word)-2]
    # now see if it is a number. if it is, return None
    if word in STOPWORDS:
        return None
    if wordIsGarbage(word):
        return None
    return word


def checkIfStringIsANumber(string):
    """
    this function is no longer used. left it just in case.
    used to check if the string can be casted to a number
    """
    result = False
    try:
        temp = float(string)
        result = True
    except:
        pass
    try:
        temp = int(string)
        result = True
    except:
        pass
    return result

def parseRawTextDocumentLine(line):
    """
    given a raw line representing a document (e.g. 000qUQAfomr0QAm4 100 pole 21113 fireman's 19956...)
    get the row_id  and an array with the words and their scores.

    :param line: whole line from train_data.txt
    :return: a tuple - doc_id(string), [(word, score),...]
    """
    splitted = line.split()
    id_from_row = splitted[0]
    number_of_words = int(splitted[1])

    # now let's genetarate the array with (word,score) tuples
    word_score_tuples = []

    # get the indeces for all words in the splitted array.
    # ignore the first and second indeces cuz they are for the img id and number of words
    for word_index in range(2, len(splitted), 2):
        word = preprocessWordFromInputLine(splitted[word_index])
        if not word:
            continue
        score = int(splitted[word_index+1])
        word_score_tuples.append((word,score))


    return id_from_row, word_score_tuples


def normalizeValuesForImgArray(arr, NORMALIZED_SUM):
    """
    Each position in the @arr marks a "word'. Since the vector_dim is the same of all vectors(usually 4096),
    we can interpret the position in the vector as a word. the value at that position shows the number of occurrances
    of the word. We want to normalize the values (the sum should be always be some constant).
    Then, we normalize again, but with respect to the "text word" counts from the document collection.
    This way, an occurrance count for a visual term or a textual one are in the same scale.
    :param [(v_word_label,float)] arr: the arr with (label,occurance) e.g [('visiword_0',1.3), ('visiword_1', 4)]
    :return: normalized occurrance counts for each visual word
    """
    # first normalize so that the sum of the visual terms ocuurances is a constant
    # TODO the normalization returns ints, which is lossy normalization. However, The error, on average, for each visiterm is below 1.

    self_normalize_sum = 3000
    labels, occurances = zip(*arr)
    sum = np.sum(occurances)
    diff = self_normalize_sum - sum
    # delta = 0
    delta = abs(diff) / config.VISUAL_WORD_DIM
    # now normalize in terms of the textual documents
    ratio = NORMALIZED_SUM / self_normalize_sum

    if diff > 0:
        occurances = (occurances + delta) * ratio
    else:
        occurances = (occurances - delta) * ratio
    rounded_up = np.around(occurances)
    # error = abs(np.sum(arr - rounded_up)) / vector_dim
    norm_occ = np.array(rounded_up, dtype=np.uint16)
    return list(zip(labels,norm_occ))


def processImg(floats):
    """
    the input is a vector like [wv1, wv2,...], where the value is the number of occurances of v1, v2...
    this method will use the position in the vector to create a label

    :param img_line:
    :return: list with (visi_word_label, occurance) tuples
    """

    v = [('%s%i'%(config.VISUAL_WORD_ID_PREFIX, index), v) for index, v in enumerate(floats)]
    return v


def combineTextAndImageVectors(word_score_tuples, visiterm_occurances_tuples):

    visiterm_occurances_tuples = sorted(visiterm_occurances_tuples, reverse=True, key=lambda tup: tup[1])
    visiterm_occurances_tuples = visiterm_occurances_tuples[0:config.MIXED_DOCUMENT_OVERALL_LENGTH // 2]
    word_score_tuples = word_score_tuples[0:config.MIXED_DOCUMENT_OVERALL_LENGTH//2]

    sum_of_text_occurances = sum([occ for _,occ in word_score_tuples])
    normalized_img = normalizeValuesForImgArray(visiterm_occurances_tuples,  sum_of_text_occurances)


    return word_score_tuples + normalized_img


def prepareTexts():
    """
    Reads the text documents and make a dict. The keys are the img ids, the values are array of (word, score) tuples,
    """
    imgid2wordscoretuple = {}

    with open(config.text_documents_file_path) as text_documents:
        # make a dict, where the key is the image id, and the val is an arr of word,score pairs
        for i, text_line in enumerate(text_documents):
            if not inRange(i, config.range_of_documents_indeces):
                continue
            img_id, word_score_tuples = parseRawTextDocumentLine(text_line)
            imgid2wordscoretuple[img_id] = word_score_tuples
    logger.info('done with the texts. size of imgid2wordscoretuple: %i' %len(imgid2wordscoretuple))
    return imgid2wordscoretuple


class MyCorpus(object):
    """
    This class is used to load the coprus.


    The iterator yields a single element - a document-image pair.
    """
    def __init__(self, visual_matrix, texts):
        self.visual_matrix = visual_matrix
        self.texts = texts


    def __iter__(self):
        """
        This method is important. It reads a textual document. It converts the document to a list of words,
        where each word is repeated according to it's score. The same is done for the accompying image of the textual document.
        Namely, the image vector is read(it's a BOVW). Using the position of the visual terms in the vector, they are converted to
        textual labels, and repeated based on the value of the visual term.
        Both arrays are concatenated, resulting in list with words in a common vocabulary, representing both the textual
        document and the image.
        :return: array with words(both textual and visual)in a common vocabulary (strings), representing the document-image pair.
        e.g. [('car',100), ('visual_term_1',3). //word(both textual or visiword) and score tuples
        """
        with open(config.img_ids_file_path) as img_ids:
            # read a img line, using it's img_id, get the textual document too
            # then combine them both in a common format (word,occurances), normalize
            # note that 'word' can be both textual term or a visual term
            for index, img_id in enumerate(img_ids):

                visiterms_occurances_pairs = processImg(self.visual_matrix[index])
                try:
                    words = self.texts[img_id.strip()]
                except KeyError:
                    # if we haven't read the document, we don't need the image vector
                    continue

                result = combineTextAndImageVectors(words, visiterms_occurances_pairs)
                yield result


class Dictionary(object):

    def __iter__(self):

        for index,line in enumerate(open(config.text_documents_file_path)):

            if not inRange(index, config.range_of_documents_indeces):
                continue
            _, word_and_scores = parseRawTextDocumentLine(line)
            parsed = [w for w, _ in word_and_scores]
            yield parsed

def get_visual_terms_labels(config):
    return ['%s%i'%(config.VISUAL_WORD_ID_PREFIX,index) for index in range(config.VISUAL_WORD_DIM)]


def createDictionary(extraLabel=""):
    # TODO in the report note the optimization done on the dict - it was ~700 000 workds, now ~90 000
    dic = Dictionary()
    d = corpora.Dictionary(dic)

    d.filter_extremes(no_below=10, no_above=0.6, keep_n=None)
    d.compactify()
    # add the visual terms as words in the vocabulary too
    d.add_documents([get_visual_terms_labels(config)])
    extraLabel = extraLabel+"_"+config.dictionary_label
    fName = 'data/dics/%s_%s.dict' % (pretty_current_time(), extraLabel)
    d.save(fName+'.bin')

    d.save_as_text(fName+'.txt')
    setLastDictFileName(fName+'.bin')
    logger.info('Dict created and saved to %s. Size: %i' % (fName, len(d)))
    return d


def train():
    global config
    config = AwsTrain()

    logger.info('MODE: ' + config.dictionary_label)
    visual_matrix = loadVisualMatrix(config)
    imgid2wordscoretuple = prepareTexts()

    #uncomment if loading previously loaded, and comment the next line
    # dictionary = corpora.Dictionary().load(getLastDictFileName())
    dictionary = createDictionary()
    config.dict_size=len(dictionary)
    logger.info('Dict read')

    #comment this if loading from a previously serialised corpus(much quicker)
    bow = BOW(dictionary=dictionary, input = MyCorpus(visual_matrix, imgid2wordscoretuple))
    corporaFname = 'data/corpora'+config.dictionary_label
    gensim.corpora.MmCorpus.serialize(corporaFname, bow)
    bow = gensim.corpora.MmCorpus(corporaFname)
    logger.info('Corpora read')

    topics = config.lda_topics
    passes = config.lda_passes
    # start training
    lda = models.LdaMulticore(corpus=bow, id2word=dictionary,
        num_topics=topics, passes=passes, chunksize=config.chunksize, workers=4)
    modelFname = config.model_folder+'lda_%i_topics_%i_passes_%s.%s.model'%(topics, passes, config.dictionary_label,pretty_current_time())
    # persist the model for later
    lda.save(modelFname)

if __name__ == '__main__':

    start = time.time()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    train()

    timed = timedelta(seconds=time.time()-start)
    print(timed)
    logExperiment(config, timed)



