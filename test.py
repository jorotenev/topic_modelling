from train import *
import threading

import logging
from util import *
logger = logging.getLogger('app')
class TestingCorpus(object):
    """
    this class's __iter__ is used when we are testing our model.

    it's __iter__ only returns (word,occurances) where word is a textual word.
    the class has an attribute for the img ids of the documents we've seen (ground truth)

    """
    def __init__(self,fname):
        self.rangeOfDocsToBeRead = config.range_of_documents_indeces
        self.rowIds = []
        self.fname=fname

    def __iter__(self):
        with open(self.fname) as text_documents:
            for i, text_line in enumerate(text_documents):
                # if not inRange(i, self.rangeOfDocsToBeRead):
                #     continue
                row_id, word_score_tuples = parseRawTextDocumentLine(text_line)

                # the index of the rowIds will match with the yielded el, when this iter is enumarated
                self.rowIds.append(row_id)
                yield word_score_tuples




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

def givenProbsFindIndexAndScoreOfBestMatches(visual_matrix, probs):

    """
    Find the most likely vwords based on the @probs and find in the @visual_matrix which row has the highest values for these vwords.
    Args:
        :visual_matrix - numpy arr with all the visual features for all images (number_of_imgs x 4096)
        :probs - (4096,) array with the probabilities of a vword
    """
    indeces_of_top_n = probs.argsort()[-config.best_n_visual_features:][::-1]
    # get only the columns for which we care
    clipped_visual_features = visual_matrix[:, indeces_of_top_n]
    index_sum_tuples=[]
    for row_index in range(0,clipped_visual_features.shape[0]):
        if row_index>310112:
            index_sum_tuples.append((np.sum(clipped_visual_features[row_index,:]), row_index))
    index_sum_tuples.sort()
    index_sum_tuples.reverse()
    return [(index,s) for s, index in index_sum_tuples[0:config.best_n_visual_features]]


def filterVisualProbabilities(lda, modelFname):
    fName = 'data/'+modelFname + '.visual_probabilities_filtered.npy'
    try:
        result = np.load(fName)
        logger.info("Loaded the filtered visual words probability from .npy file")
        return result
    except:
        logger.info("No filtered visual words probability file found. Creating now.")
    result = []
    for topic_id in range(0, lda.num_topics):
        # this is the slow call. that's why we serialize the result...
        probabilities_for_topic = getVisualWordProbabilitiesForTopic(lda.show_topic(topic_id, topn = None), config)
        assert len(probabilities_for_topic) == config.VISUAL_WORD_DIM

        result.append(probabilities_for_topic)
    np.save(fName, result)
    return result


def findBestImages(model=None, doc_topic_probabilities=None, vwords_probs_for_all_topics=None, visual_matrix=None, img_ids=None):

    probs_of_vwords_given_a_document = findProbabilitiesVWordsGivenADocument(model,doc_topic_probabilities, vwords_probs_for_all_topics)
    best_img_indeces = givenProbsFindIndexAndScoreOfBestMatches(visual_matrix, probs_of_vwords_given_a_document)

    return [(img_ids[i],score) for i,score in best_img_indeces]


def test():
    # load all img ids (order same as in the visual matrix)
    img_ids = imgIdOfTestDocument(config)

    logger.info("Loaded all image ids. Size %i." % len(img_ids))

    # load the dict
    dictionary = gensim.corpora.Dictionary.load(getLastDictFileName())

    # the input  documents (they come as array of (word, score) tuples. in this case, the words are only textual words(as opposed to visual words))

    # generate bow representation of the input

    modelFname = 'lda_1000_topics_1_passes_aws_train.10h31m_28Apr.model'
    modelPath = config.model_folder + modelFname
    num_threads = 2
    result_file_base_name = config.test_result_path+'/'+str(num_threads) +'/'+ pretty_current_time() + modelFname+'_'
    config.test_result_file = config.test_result_path + pretty_current_time() + modelFname+'.test_result'
    lda = gensim.models.LdaModel.load(modelPath)
    logger.info("Started loading filtering the visual words probabilities...")
    # index is the topic id. the value is array of 4096 probabilities of each visiterm for the given topic.
    probabilities_of_visual_terms_in_topics = filterVisualProbabilities(lda, modelFname)
    logger.info('Filtered the probabilities for topics for visual words. Size: %i' %len(probabilities_of_visual_terms_in_topics))

    # this is just the visual features, but loaded in a numpy array. loading from a serialized file for efficiency
    visual_matrix = loadVisualMatrix(config)
    assert visual_matrix.shape[0] == len(img_ids)

    for thread_id in range(num_threads):
        thread = myThread(result_file_base_name, num_threads, thread_id,visual_matrix,lda,img_ids,probabilities_of_visual_terms_in_topics,dictionary)
        thread.start()



class myThread (threading.Thread):

    def __init__(self,
                 result_file_base_name,
                 all_threads_num,
                 thread_number,
                 visual_matrix,
                 lda,
                 img_ids,
                 probabilities_of_visual_terms_in_topics,
                 dictionary):
        threading.Thread.__init__(self)

        self.result_file_base_name = result_file_base_name
        self.all_threads_num = all_threads_num
        self.thread_number = thread_number
        self.visual_matrix=visual_matrix
        self.lda = lda
        self.img_ids= img_ids
        self.probabilities_of_visual_terms_in_topics= probabilities_of_visual_terms_in_topics
        self.dictionary= dictionary


    def run(self):
        fName='data/main/TestData/'+str(self.all_threads_num)+'/'+str(self.thread_number)+'.txt'
        corpus = TestingCorpus(fName)

        bows = BOW(dictionary=self.dictionary, input=corpus)

        fname = self.result_file_base_name+'_chunk' + str(self.thread_number)
        with open(fname, 'w') as resultFile:
            logger.info("Beginning testing NOW :)")
            for index, bow in enumerate(bows):

                row_id = corpus.rowIds[index]  # img/query id
                # for a given doc, get the probability distribution over the topics
                topic_probabilities_for_doc = self.lda.get_document_topics(bow, minimum_probability=0)
                best_img_ids = findBestImages(model=self.lda,
                                              doc_topic_probabilities=topic_probabilities_for_doc,
                                              vwords_probs_for_all_topics=self.probabilities_of_visual_terms_in_topics,
                                              visual_matrix=self.visual_matrix,
                                              img_ids=self.img_ids)
                write_result_line_of_test(resultFile, row_id, best_img_ids)


if __name__ == '__main__':

    start = time.time()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    config = FullTest()

    test()

    timed = timedelta(seconds=time.time()-start)
    print(timed)
    logExperiment(config, timed)
