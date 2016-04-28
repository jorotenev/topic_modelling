import gensim ,time, logging, time
from datetime import timedelta

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('thisss')

corporaFname = 'data/corporaaws_train'
bow = gensim.corpora.MmCorpus(corporaFname)
logger.info("loaded")
start = time.time()
for a in bow:
    pass
logger.info('finished. printing results..')
timed = timedelta(seconds=time.time() - start)
print(timed)