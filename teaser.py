"""
This file is performing the Teaser excercise - 2.1.
It does the analogy solving - a : b = c : ?
and reports the successfully predicted anologies, out of all analogies.
"""
import datetime
import time

from gensim.models import Word2Vec



class ExperimentSettings:
    analogies_name = 'data/warm_up/questions-words.txt'
    debug = False
    restrict_vocab_nb = 80000


class GoogleNewsModel(ExperimentSettings):
    is_model_binary = True
    model_name = 'data/warm_up/GoogleNews-vectors-negative300.bin'
    lower_case_the_input = False


class GloveModel(ExperimentSettings):
    is_model_binary = False
    model_name = 'data/warm_up/glove.6B.50d.txt'
    lower_case_the_input = True

# change this to either GloveModel or GoogleNewsModel
config = GloveModel


def get_best_analogy(model, pos=[], neg=[]):
    """"
    e.g king - man + woman = ?
    Args:
        @param model - wword2vec model
        @param pos - e.g. ['king','woman']
        @param neg - e.g. ['man']
        @param limit (int) - limit the number of result tuples
    Returns:
        return - array of (word, similarity) tuples.
        e.g. [('queen', 0.2547094679191654),
             ('marries', 0.210394133486288),
            ('daughter', 0.20279386590540743)]
    """
    try:

        ignore = pos + neg

        result = model.most_similar(positive=pos, negative=neg, restrict_vocab=config.restrict_vocab_nb)
        #get just the word(the model returns a word,similarity tuple)
        result = [r[0] for r in result if r not in ignore]

        return result[0]
    except Exception as e:
        # KeyError if the model doesnt have a word
        if config.debug:
            print(e)
        return None


def perform_experiment(model, analogies=[]):
    """
    Given a word2vec model and analogies (with separated test queries and a correct response),
    run the test queries over the model and see if the output of the model matches the correct response.
    Args:
        @param model - word2vec trained model
        @param analogies - a list of dicts with keys: "test" - array of 3 test words; "result" - string, the expected result
    Returns
        return score - the correct number of guesses.
    """

    score = 0
    iterations = 0
    no_entries_words = 0 #number of words which are not in the vocab of the model

    for line in analogies:
        iterations = iterations + 1 #used when the debugger is ON to track progress

        input = line['test']
        correct_result = line['result']
        # as explained in the assignment - section 2.1
        # a : b = c : ?
        # king : man = queen : woman
        # queen - king + man
        # to find the ?, we need to do c - a + b

        if config.lower_case_the_input:
            input = [word.lower() for word in input]
        a, b, c = input

        res = get_best_analogy(model, pos=[c, b], neg=[a])
        if res:
            score = update_score_if_words_match(score, res, correct_result)
        else:
            no_entries_words = no_entries_words + 1
    print("No entries were found in our model for %i words." % no_entries_words)
    return score


# helpers
def update_score_if_words_match(current_score, correct_word, result_word):
    if correct_word.lower() == result_word.lower():
        current_score = current_score + 1
    return current_score


def read_analogies(fileName):
    with open(fileName) as f:
        content = f.readlines()
        return content


def prepare_analogies(raw_analogies):
    """
    get the raw_analogies, skip the irrelevant lines (e.g. beginning with ':' )
    the result is a list of dicts - each dict has keys - "test" and "result". "
    "test" has the words for which an analogy should be made. the correct analogy word("single ground truth answer") is the value of "result"
    Args:
        @param raw_analogies - a list of lines such as "Athens Greece Baghdad Iraq"
    Returns:
        return - array of dicts containg the input array for the query and the correct answer
    """
    analogies = []

    for line in raw_analogies:
        if line.startswith(':'):
            continue

        words = line.split(' ')
        words = [s.strip() for s in words]

        analogies.append({'test': words[:3], 'result': words[-1]})

    return analogies


def main():
    analogies_name = config.analogies_name
    model_name = config.model_name

    print("[%i Vocab]" % (config.restrict_vocab_nb or -1))
    print("Analogies: %s | Model: %s" % (analogies_name, model_name))

    start = now()
    analogies = read_analogies(analogies_name)
    analogies = prepare_analogies(analogies)

    model = Word2Vec.load_word2vec_format(model_name, binary=config.is_model_binary)
    model.init_sims(replace=True)
    model_loaded = now()
    print("Model loaded for [%s]" % str(model_loaded - start))

    correct_guesses = perform_experiment(model, analogies=analogies)
    end = now()

    result_ratio, result_percentage = experiment_result_str(correct_guesses, analogies)
    print("Correct number of predictions out of all predictions %s [%i%%]" % (result_ratio, result_percentage))
    print("It took %s to load the model. After that, it took %s to perform the check" % (
        delta_to_str(model_loaded - start), delta_to_str(end - model_loaded)))


def now():
    return datetime.datetime.fromtimestamp(time.time())


def delta_to_str(delta):
    s = delta.seconds
    hours, remainder = divmod(s, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "%sh:%sm:%ss" % (hours, minutes, seconds)


def experiment_result_str(correct_guesses, analogies):
    result_ratio = correct_guesses / len(analogies)
    result_ratio_str = "%i/%i" % (correct_guesses, len(analogies))
    result_percentage = int(result_ratio * 100)
    return result_ratio_str, result_percentage


if __name__ == "__main__":
    main()
