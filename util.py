import logging,time,socket
import numpy as np
logger = logging.getLogger('app')
class BOW:

    def __init__(self, dictionary=None, input = None):
        self.dictionary = dictionary
        self.input = input

    def __iter__(self):

        for mixed_document in self.input:

            yield [(self.dictionary.token2id[w], occ) for (w, occ) in mixed_document if w in self.dictionary.token2id]

def loadVisualMatrix(config):
    """
    Create a numpy matrix with the size of the images of the documents we are testing
    :param heightOfMatrix:
    :return:
    """
    # logger.info("Starting to create a matrix of the visual features")
    expectedShape = (config.number_of_img_ids, config.VISUAL_WORD_DIM)

    fNameOfMatrix = config.image_vectors_file_path + config.visual_matrix_suffix
    try:
        loaded = np.load(fNameOfMatrix)
        assert loaded.shape == expectedShape
        logger.info(
            '*Loaded* the Visual Feature matrix with dimensions %s.' % str(loaded.shape))
        return loaded
    except:
        # execute the code below only if we couldnt load a seriliazed numpy version (which is FASTER)
        logger.info("Cannot find a serialized version of the matrix. Creating and serializingn the matrix now.")
        pass
    image_features_matrix = np.zeros(expectedShape, dtype=np.float16)

    with open(config.image_vectors_file_path) as image_vectors:
        _ = image_vectors.readline() # consume the header.

        for row_index, image_vector_line in enumerate(image_vectors):
            vals = image_vector_line.split()
            assert len(vals) == config.VISUAL_WORD_DIM + 1 # + 1 for the img_id
            try:
                image_features_matrix[row_index, :] = np.array([float(f) for f in vals[1:]], dtype=np.float64)
            except Exception as e:
                logger.info('Stop with creating the matrix. Exception: %s' % str(e))
                break
    logger.info('Created the Visual Feature matrix with dimensions %s.).' %(str(image_features_matrix.shape)))

    np.save(fNameOfMatrix, image_features_matrix)
    return image_features_matrix

def imgIdOfTestDocument(config):
    """
    :return: array of all img_ids. The order(read indeces of the array) is the same as the vectors in the Visual Matrix
    """
    img_ids_of_document = []
    # init the dict with all possible IDS
    with open(config.img_ids_file_path) as img_ids_file:
        for line in img_ids_file:
            img_ids_of_document.append(line.strip())

    return img_ids_of_document

def lineSep():
    return "-----------------------------"
def logExperiment(config, runtime):
    newline = '\n'
    with open('model_trainings.log','a') as f:
        f.write(lineSep()+newline)
        f.write(pretty_current_time()+newline)
        f.write(config.dictionary_label +' @ '+ socket.gethostname())
        f.write(newline)
        f.write('Range of docs: '+str(config.range_of_documents_indeces))
        f.write(newline)
        f.write('Dictionary size: '+str(config.dict_size))
        f.write(newline)
        f.write('VISUAL Scale down factor: '+ str(config.SCALE_DOWN_FACTOR_VISUAL_SCORE))
        f.write(newline)
        f.write('TEXTUAL Scale down factor: '+ str(config.SCALE_DOWN_FACTOR_TEXT_SCORE))
        f.write(newline)
        f.write('Mixed document length: '+ str(config.MIXED_DOCUMENT_OVERALL_LENGTH))
        f.write(newline)
        f.write('Workers: '+ str(getNumberOfWorkers()))
        f.write(newline)
        f.write('LDA Passes: '+ str(config.lda_passes))
        f.write(newline)
        f.write('LDA Topics: '+ str(config.lda_topics))
        f.write(newline)
        f.write('LDA Train Chunksize: '+ str(config.chunksize))
        f.write(newline)
        f.write('RUNTIME: '+ str(runtime))
        f.write(newline)
        f.write(lineSep())
        f.write(newline)

def pretty_current_time():
    return time.strftime('%lh%Mm_%d%b').strip()
def getLastDictFileName():
    with open('data/last_dict_fname.txt') as f:
        return f.readline().strip()
def setLastDictFileName(fName):
    with open('data/last_dict_fname.txt','w') as f:
        f.write(fName)
def getNumberOfWorkers():
    with open('.num_workers') as f:
        return int(f.readline().strip())