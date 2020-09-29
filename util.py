import re

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

def _name_and_text(utt):
    '''
    Lambda function used in below methods for determining
    what utterance details to include
    '''
    res = utt.speaker.id + ": " + utt.text

    # set of regex patterns to check with--make sure these aren't too costly
    # r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
    regex_patterns = {
        r"(http|https)://.*\s" : " ",
        r"(http|https)://.*" : " ",
        "\[.*\]\(" : " ",
        r".(\n+)" : ". ",
        r"(\n+)" : ". "
    }
    for pattern, repl in regex_patterns.items():
        res = re.sub(pattern, repl, res)

    return res

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

    conv_str = " "*indent + utt_info_func(conv.get_utterance(root_id)) + "\n"
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

def plot_conv_lengths(conv_lengths, low, high, bins=30):
    '''
    Plots a histogram of the conv length distribution
    '''
    plt.hist(
        conv_lengths, bins=bins, range=(low, high)
    )
    plt.show()
