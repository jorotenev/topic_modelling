from train import *
import logging
from util import *
logger = logging.getLogger('app')
class TestingCorpus(object):
    """
    this class's __iter__ is used when we are testing our model.

    it's __iter__ only returns (word,occurances) where word is a textual word.
    the class has an attribute for the img ids of the documents we've seen (ground truth)

    """
    def __init__(self):
        self.rangeOfDocsToBeRead = config.range_of_documents_indeces
        self.rowIds = []

    def __iter__(self):
        with open(config.text_documents_file_path) as text_documents:
            for i, text_line in enumerate(text_documents):
                if not inRange(i, self.rangeOfDocsToBeRead):
                    continue
                row_id, word_score_tuples = parseRawTextDocumentLine(text_line)

                # the index of the groundTruthImgIds will match with the yielded el, when this iter is enumarated
                self.rowIds.append(row_id)
                yield word_score_tuples


def getVisualWordProbabilitiesForTopic(topic_probabilities):
    """
    it's just a view of a single topic of the model - to access more quickly the probabilities of all visiterm in the whole topic
    
    given an array of (probability, words) pairs, look only for words which are actually labels for visual terms (e.g 'visiterm_1')
    get the id from the label and put it in a list as a tuple together with the probability of the visiwor.
    sort by the id(which is just an index with range 0-4096), then return list with just the probabilities.

    :return: array with probabilities, where the index in array, matches the id from the visual term label
    """
    prefix = config.VISUAL_WORD_ID_PREFIX
    result= [(int(w[len(prefix):]),p) for p,w in topic_probabilities if len(w) > len(prefix) and w[0:len(prefix)] == prefix]
    result.sort() # sort on first key, which is the visual_term_id
    return [p for i,p in result]


def findProbabilitiesVWordsGivenADocument(lda, topic_probabilities_for_doc, visi_word_probabilities):
    
    """
    Given a document, and a model, see the probabilities of all visual words.
    "Compute the probability of each visual term in the vocabulary, given document D,
    by marginalizing over the document topics z1:k"

    Args:
        : lda - the model
        : topic_probabilities_for_doc - arr of (topic_id, topic_prob) tuples, for the given document
        : visi_word_probabilities - numpy arr. height is the number of topics of the model. width is 4096.
            it's just a view of the model - to access more quickly the probabilities of all visiterm in a whole topic
    :return arr - len(arr)=4096. the probability of every visual term, given the document.

    """

    probs_of_vwords_given_a_document = np.zeros((config.VISUAL_WORD_DIM), dtype=np.float16)
    for index in range(config.VISUAL_WORD_DIM):
        probability_of_visi_term = 0

        for topic_id, topic_prob in topic_probabilities_for_doc:
            # topic_prob = P(z|d) - the probability of the topic, given the document

            # all_words_probs_for_topic = lda.show_topic(topic_id, topn=None)

            # P(w|z) - the probability of a visiword for a topic
            prob_of_vword_in_a_topic = visi_word_probabilities[topic_id][index]
            probability_of_visi_term = probability_of_visi_term + prob_of_vword_in_a_topic * topic_prob
            
        probs_of_vwords_given_a_document[index] = probability_of_visi_term
    return probs_of_vwords_given_a_document

def givenProbsFindIdOfBestMatch(visual_matrix, probs):
    
    """
    Find the most likely vwords based on the @probs and find in the @visual_matrix which row has the highes values for these vwords.
    Args:
        :visual_matrix - numpy arr with all the visual features for all images (number_of_imgs x 4096)
        :probs - (4096,) array with the probabilities of a vword
    """
    indeces_of_most_likely_words = np.argmax(probs)
    pass
def filterVisualProbabilities(lda, modelFname):
    fName = 'data/'+modelFname + '.visual_probabilities_filtered.npy'
    try:
        result = np.load(fName)
        logger.info("Loaded the filtered viusal words probability from .npy file")
        return result
    except:
        logger.info("No filtered visual words probability file found. Creating now.")
    result = []
    for topic_id in range(0, lda.num_topics):
        # this is the slow call. that's why we serialize the result...
        probabilities_for_topic = getVisualWordProbabilitiesForTopic(lda.show_topic(topic_id, topn = None))
        assert len(probabilities_for_topic) == config.VISUAL_WORD_DIM

        result.append(probabilities_for_topic)
    np.save(fName, result)
    return result


def test():
    # load all img ids (order same as in the visual matrix)
    img_ids = imgIdOfTestDocument(config)

    # this is just the visual features, but loaded in a numpy array. loading from a serialized file for efficiency
    visual_matrix = loadVisualMatrix(config)
    assert visual_matrix.shape[0] == len(img_ids)
    
    logger.info("Loaded all image ids. Size %i." % len(img_ids))

    # load the dict
    dictionary = gensim.corpora.Dictionary.load(getLastDictFileName())

    # the input  documents (they come as array of (word, score) tuples. in this case, the words are only textual words(as opposed to visual words))
    corpus = TestingCorpus()

    # generate bow representation of the input
    bows = BOW(dictionary=dictionary, input = corpus)

    modelFname = 'lda_100_topics_5_passes_4h58m_25Apr.model'
    modelPath = config.model_folder + modelFname
    
    lda = gensim.models.LdaModel.load(modelPath)
    logger.info("Started loading filtering the visual words probabilities...")
    # index is the topic id. the value is array of 4096 probabilities of each visiterm for the given topic.
    probabilities_of_visual_terms_in_topics = filterVisualProbabilities(lda, modelFname)
    logger.info('Filtered the probabilities for topics for visual words. Size: %i' %len(probabilities_of_visual_terms_in_topics))

    for index, bow in enumerate(bows):
        row_id = corpus.rowIds[index] # img/query id
        # for a given doc, get the probability distribution over the topics
        topic_probabilities_for_doc = lda.get_document_topics(bow, minimum_probability=0)
        img_id = findProbabilitiesVWordsGivenADocument(lda, topic_probabilities_for_doc, probabilities_of_visual_terms_in_topics)


if __name__ == '__main__':

    start = time.time()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    config = FullTest()
    
    test()

    timed = timedelta(seconds=time.time()-start)
    print(timed)
    logExperiment(config, timed)
