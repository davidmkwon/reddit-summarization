from convokit import Corpus, download

# run if dataset is not downloaded yet:
#corpus = Corpus(download('subreddit-travel'))

data_path_home = '/Users/davidkwon/.convokit/downloads/subreddit-gatech'
data_path_remote = '/home/dkwon49/.convokit/downloads/subreddit-travel'
corpus = Corpus(data_path_remote)
