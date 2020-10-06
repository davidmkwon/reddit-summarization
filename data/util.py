import re
import random

import convokit
import matplotlib.pyplot as plt

def get_conv_lengths(corpus):
    '''
    Returns a list of conversation lengths in a given corpus
    '''
    conv_lengths = []
    for conv_id in corpus.get_conversation_ids():
        conv = corpus.get_conversation(conv_id)
        conv_lengths.append(len(conv.get_utterance_ids()))

    return conv_lengths

def find_links_conv(corpus):
    '''
    Some random helper method
    '''
    count = 0
    for conv_id in corpus.get_conversation_ids():
        conv = corpus.get_conversation(conv_id)
        for utt in conv.iter_utterances():
            if "http" in utt.text:
                count += 1
                if count == 2:
                    return conv

    return None

def mean_std(conv_lengths):
    '''
    Returns the mean and standard deviation in a list of
    conversation lengths
    '''
    mean_length = sum(conv_lengths) / len(conv_lengths)
    variance_length = sum(
        (conv_lengths[i] - mean_length)**2 \
        for i in range(len(conv_lengths))
    ) / len(conv_lengths)

    return mean_length, variance_length**0.5

def regex_filter(utt):
    '''
    Returns a filtered utterance text through a series
    of regex replacements.
    '''
    # set of regex patterns to check with--make sure these aren't too costly
    # r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
    res = utt.text
    regex_patterns = {
        r"(http|https)://.*\s" : " ",
        r"(http|https)://.*" : " ",
        "\[.*\]\(" : " ",
        r"\.(\n+)" : ". ",
        r"(\n+)" : ". "
    }
    for pattern, repl in regex_patterns.items():
        res = re.sub(pattern, repl, res)

    return res

def _name_and_text(utt):
    '''
    Lambda function used in below methods for determining
    what utterance details to include
    '''
    res = utt.speaker.id + ": " + regex_filter(utt)

    return res

def execute_regex(corpus):
    '''
    Mutate the corpus by executing the regex replacements
    on all utterances
    '''
    for utt_id in corpus.get_utterance_ids():
        utt = corpus.get_utterance(utt_id)
        utt_text = regex_filter(utt)
        utt.text = utt_text

def print_conversation_complete(conv, utt_info_func=_name_and_text):
    '''
    Print the conversation (speaker names and utterance texts)
    in its structured way
    '''
    conv.print_conversation_structure(utt_info_func=utt_info_func)

def _get_convo_helper(conv, root_id, indent, reply_to_dict, utt_info_func, limit):
    '''
    Recursive helper method for getting structure conversation
    '''
    if limit is not None:
        if conv.get_utterance(root_id).meta['order'] > limit:
            return ""

    child_utt_ids = [k for k, v in reply_to_dict.items() if v == root_id]
    sub_conv_str = ""
    for child_utt_id in child_utt_ids:
        sub_conv_str += _get_convo_helper(
            conv=conv, root_id=child_utt_id, indent=indent+4,
            reply_to_dict=reply_to_dict, utt_info_func=utt_info_func, limit=limit
        )

    conv_str = " "*indent + utt_info_func(conv.get_utterance(root_id)) + "\n\n"
    conv_str = conv_str + sub_conv_str
    return conv_str

def get_conversation_complete(conv, utt_info_func=_name_and_text, limit=None):
    '''
    Returns a string representation of the complete conversation
    (speaker names and utterance texts) in its structured way
    '''
    if limit is not None:
        for idx, utt in enumerate(conv.get_chronological_utterance_list()):
            utt.meta['order'] = idx + 1

    root_utt_id = [utt for utt in conv.iter_utterances() if utt.reply_to is None][0].id
    reply_to_dict = {utt.id: utt.reply_to for utt in conv.iter_utterances()}

    conv_str = _get_convo_helper(
        conv=conv, root_id=root_utt_id, indent=0, reply_to_dict=reply_to_dict,
        utt_info_func=utt_info_func, limit=limit
    )
    return conv_str

def plot_conv_lengths_histogram(conv_lengths, low, high, bins=30):
    '''
    Plots a histogram of the conv length distribution
    '''
    plt.hist(
        conv_lengths, bins=bins, range=(low, high)
    )
    plt.show()

def plot_conv_lengths_cdf(conv_lengths, low=0, high=float('inf')):
    '''
    Plots a cdf of the conv length distribution that falls between
    low and high, inclusive
    '''
    import numpy as np

    CX = np.array(conv_lengths)
    CX = CX[(CX >= low) & (CX <= high)]
    CX.sort()
    CY = np.array(CX)
    CY = CY / CY.sum()
    CY = np.cumsum(CY)
    plt.plot(CX, CY)
    plt.show()

def num_convs_between(conv_lengths, low, high):
    '''
    Returns the number of conversations that have lengths between
    low and high, inclusive
    '''
    num_convs = 0
    for conv_length in conv_lengths:
        if conv_length >= low and conv_length <= high:
            num_convs += 1

    return num_convs

def count_unclean_corpus(corpus):
    count = 0
    for conv in corpus.iter_conversations():
        if not conv.check_integrity(verbose=False):
            count += 1

    return count

def num_deleted_utterances(corpus):
    '''
    Count the total number of utterances that are "[deleted]"
    '''
    count = 0
    for conv in corpus.iter_conversations():
        for utt in conv.iter_utterances():
            if len(utt.text) == 9 and utt.text == "[deleted]":
                count += 1

    return count

def num_deleted_convs(corpus):
    '''
    Count the number of conversations that have a "[deleted]" utterance
    and the lengths of these conversations
    '''
    count = 0
    conv_lengths = []
    for conv in corpus.iter_conversations():
        for utt in conv.iter_utterances():
            if len(utt.text) == 9 and utt.text == "[deleted]":
                conv_lengths.append(len(conv.get_utterance_ids()))
                count += 1
                break

    return count, conv_lengths

def num_removed_convs(corpus):
    '''
    Count the number of conversations that have a "[removed]" utterance
    and the lengths of these conversations
    '''
    count = 0
    conv_lengths = []
    for conv in corpus.iter_conversations():
        for utt in conv.iter_utterances():
            if len(utt.text) == 9 and utt.text == "[removed]":
                conv_lengths.append(len(conv.get_utterance_ids()))
                count += 1
                break

    return count, conv_lengths

def num_deleted_root_utterances(corpus):
    '''
    Return the number of conversations where the root (first)
    comment is "[deleted]"
    '''
    idx = 0
    count = 0
    indexes = []
    for conv in corpus.iter_conversations():
        root_utt = [utt for utt in conv.iter_utterances() if utt.reply_to is None][0]
        if len(root_utt.text) == 9 and root_utt.text == "[deleted]":
            count += 1
            indexes.append(idx)
        idx += 1

    return count, indexes

def get_convs_with_length(corpus, length):
    '''
    Return a list of conversation objects with length = length
    '''
    res = []
    for conv in corpus.iter_conversations():
        if len(conv.get_utterance_ids()) == length:
            res.append(conv)

    return res

def get_conv_utt_lengths_avg(corpus, length):
    '''
    Returns a list of the average utterance length in conversations
    with length = length
    '''
    convs = get_convs_with_length(corpus, length)
    res = []
    for conv in convs:
        avg = 0
        for utt in conv.iter_utterances():
            avg += len(utt.text.split(" "))
        avg /= length
        res.append(avg)

    return res
