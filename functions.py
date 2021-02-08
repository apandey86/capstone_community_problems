#Contraction dictionary defined as a global variable
c_dict = {
	"ain't": "am not",
	"aren't": "are not",
	"can't": "cannot",
	"can't've": "cannot have",
	"'cause": "because",
	"cos": "because",
	"could've": "could have",
	"couldn't": "could not",
	"couldn't've": "could not have",
	"didn't": "did not",
	"doesn't": "does not",
	"don't": "do not",
	"hadn't": "had not",
	"hadn't've": "had not have",
	"hasn't": "has not",
	"haven't": "have not",
	"he'd": "he would",
	"he'd've": "he would have",
	"he'll": "he will",
	"he'll've": "he will have",
	"he's": "he is",
	"how'd": "how did",
	"how'd'y": "how do you",
	"how'll": "how will",
	"how's": "how is",
	"i'd": "I would",
	"I'd": "I would",
	"i'd've": "I would have",
	"I'd've": "I would have",
	"i'll": "I will",
	"I'll": "I will",
	"i'll've": "I will have",
	"I'll've": "I will have",
	"i'm": "I am",
	"I'm": "I am",
	"i've": "I have",
	"I've": "I have",
	"isn't": "is not",
	"it'd": "it had",
	"it'd've": "it would have",
	"it'll": "it will",
	"it'll've": "it will have",
	"it's": "it is",
	"let's": "let us",
	"ma'am": "madam",
	"mayn't": "may not",
	"might've": "might have",
	"mightn't": "might not",
	"mightn't've": "might not have",
	"must've": "must have",
	"mustn't": "must not",
	"mustn't've": "must not have",
	"needn't": "need not",
	"needn't've": "need not have",
	"o'clock": "of the clock",
	"oughtn't": "ought not",
	"oughtn't've": "ought not have",
	"shan't": "shall not",
	"sha'n't": "shall not",
	"shan't've": "shall not have",
	"she'd": "she would",
	"she'd've": "she would have",
	"she'll": "she will",
	"she'll've": "she will have",
	"she's": "she is",
	"should've": "should have",
	"shouldn't": "should not",
	"shouldn't've": "should not have",
	"so've": "so have",
	"so's": "so is",
	"that'd": "that would",
	"that'd've": "that would have",
	"that's": "that is",
	"there'd": "there had",
	"there'd've": "there would have",
	"there's": "there is",
	"they'd": "they would",
	"they'd've": "they would have",
	"they'll": "they will",
	"they'll've": "they will have",
	"they're": "they are",
	"they've": "they have",
	"to've": "to have",
	"wasn't": "was not",
	"we'd": "we had",
	"we'd've": "we would have",
	"we'll": "we will",
	"we'll've": "we will have",
	"we're": "we are",
	"we've": "we have",
	"weren't": "were not",
	"what'll": "what will",
	"what'll've": "what will have",
	"what're": "what are",
	"what's": "what is",
	"what've": "what have",
	"when's": "when is",
	"when've": "when have",
	"where'd": "where did",
	"where's": "where is",
	"where've": "where have",
	"who'll": "who will",
	"who'll've": "who will have",
	"who's": "who is",
	"who've": "who have",
	"why's": "why is",
	"why've": "why have",
	"will've": "will have",
	"won't": "will not",
	"won't've": "will not have",
	"would've": "would have",
	"wouldn't": "would not",
	"wouldn't've": "would not have",
	"y'all": "you all",
	"y'alls": "you alls",
	"y'all'd": "you all would",
	"y'all'd've": "you all would have",
	"y'all're": "you all are",
	"y'all've": "you all have",
	"you'd": "you had",
	"you'd've": "you would have",
	"you'll": "you you will",
	"you'll've": "you you will have",
	"you're": "you are",
	"you've": "you have"}
#############################################################Text cleaning functions###################################################
def preclean_only(df):
	'''
	This functioon will apply 7 precleaning steps on text data and will not perform
	any transformations applied to scraped tweets
	'''
	import copy
	from functools import reduce
	clean_df = df.copy()
	import re
	#Contraction
	c_re = re.compile('(%s)' % '|'.join(c_dict.keys()))
	def expand_contractions(sentence, c_re = c_re):
		def replace(match):
			return c_dict[match.group(0)]
		return c_re.sub(replace, sentence)

	#Trim repeated letters function
	def trim_repeated_letters(sentence):
		return(reduce(lambda x,y: x+y if x[-2:] != y*2 else x, sentence, ""))

	#Clean df with functions above, it must be noted that the order is of utmost importance here
	clean_df['clean_text'] = clean_df['text'].apply(lambda x: re.sub("@[\w]*","", x)) #User handles
	clean_df['clean_text'] = clean_df['clean_text'].apply(lambda x: re.sub("#[\w]*","", x)) #hash tags
	clean_df['clean_text'] = clean_df['clean_text'].apply(lambda x: re.sub("http\S+", "", x)) #urls
	clean_df['clean_text'] = clean_df['clean_text'].apply(lambda x: expand_contractions(x)) #expand contractions
	clean_df['clean_text'] = clean_df['clean_text'].apply(lambda x: re.sub("""['!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~']""",
																  " ", x)) #Punctuations
	clean_df['clean_text'] = clean_df['clean_text'].apply(lambda x: re.sub('[^A-Za-z]', ' ', x.lower())) #Numbers and lowercase
	#Repeated letters
	clean_df['clean_text'] = clean_df['clean_text'].apply(lambda x: trim_repeated_letters(x))
	return clean_df	
	

def clean_tweets_df(tweets_df):
	'''
	This function will take in tweets_df downloaded via twint and return a df
	with original text, clean text and date components. The cleaning steps addressed are as below:
	1. Remove user handles 
	2. Remove hashtags  
	3. Remove urls
	4. Adress contractions 
	5. Replace punctuations with whitespace 
	6. Remove numbers
	7. Trim repeated letters in a word. Only 2 repetitions within a word are allowed. 

	'''
	###################Extract date and tweet text and extract date components 
	import datetime as dt 
	import pandas as pd
	import copy
	from functools import reduce
	clean_df = tweets_df.copy()
	clean_df = tweets_df[['tweet','date']]

	#Rename tweet columnt to text to allow for cleaning function to work
	clean_df.rename(columns = {'tweet':'text'}, inplace = True)

	#convert date to datetime format
	clean_df['date'] = clean_df['date'].apply(lambda x: 
										dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
	#Create time based columns
	clean_df['hour'] = clean_df['date'].dt.hour
	clean_df['month'] = clean_df['date'].dt.month
	clean_df['day'] =  clean_df['date'].dt.day
	clean_df['year'] = clean_df['date'].dt.year
	clean_df['date'] = pd.to_datetime(clean_df[["year", "month", "day"]])



	##################Perform cleaning steps listed above
	import re	
	#Contraction
	c_re = re.compile('(%s)' % '|'.join(c_dict.keys()))
	def expand_contractions(sentence, c_re = c_re):
		def replace(match):
			return c_dict[match.group(0)]
		return c_re.sub(replace, sentence)

	#Trim repeated letters function
	def trim_repeated_letters(sentence):
		return(reduce(lambda x,y: x+y if x[-2:] != y*2 else x, sentence, ""))

	#Clean df with functions above, it must be noted that the order is of utmost importance here
	clean_df['clean_text'] = clean_df['text'].apply(lambda x: re.sub("@[\w]*","", x)) #User handles
	clean_df['clean_text'] = clean_df['clean_text'].apply(lambda x: re.sub("#[\w]*","", x)) #hash tags
	clean_df['clean_text'] = clean_df['clean_text'].apply(lambda x: re.sub("http\S+", "", x)) #urls
	clean_df['clean_text'] = clean_df['clean_text'].apply(lambda x: expand_contractions(x)) #expand contractions
	clean_df['clean_text'] = clean_df['clean_text'].apply(lambda x: re.sub("""['!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~']""",
																  " ", x)) #Punctuations
	clean_df['clean_text'] = clean_df['clean_text'].apply(lambda x: re.sub('[^A-Za-z]', ' ', x.lower())) #Numbers and lowercase
	#Repeated letters
	clean_df['clean_text'] = clean_df['clean_text'].apply(lambda x: trim_repeated_letters(x))
	return clean_df
	
##############################Function to visulaize top words from vectorizer
def visualize_vectorized_data(transformed_df,n_words):
	top_words = transformed_df.sum(axis = 0).sort_values(ascending = False)[:(n_words - 1)]
	#import dependancies
	import matplotlib.pyplot as plt
	#Visualize vectorized data
	plt.figure(figsize =(10,10))
	plt.barh(top_words.index, top_words.values)
	plt.xlabel("Count")
	plt.xticks(rotation = 90)
	plt.ylabel("Tokens")
	plt.show()
	
###################################Function to visulaize words from log reg and random forest
def visualize_log_reg_predictors(log_reg, X_train_transformed_df, n_words):
	import pandas as pd
	coef_df = pd.DataFrame(data = log_reg.coef_, columns = X_train_transformed_df.columns, 
						   index = ['coef'])

	coef_df_top_words = coef_df.sort_values(by ='coef', axis = 1, ascending = False).iloc[:,:n_words]
	#import dependancies
	import matplotlib.pyplot as plt
	#Visualize data
	plt.figure(figsize =(10,10))
	plt.barh(coef_df_top_words.loc['coef'].index, coef_df_top_words.loc['coef'].values)
	plt.xlabel("coefficients")
	plt.xticks(rotation = 90)
	plt.ylabel("Words")
	plt.title('Most predictive words for Log Reg model')
	plt.show()
#################################################################################################################	
def visualize_rand_forest_predictors(rand_forest, X_train_transformed_df, n_words):
	import pandas as pd	
	coef_df = pd.DataFrame(data = rand_forest.feature_importances_.reshape(1, len(X_train_transformed_df.columns)), 
						   columns = X_train_transformed_df.columns,
						   index = ['coef'])

	#Top words 
	coef_df_top_words = coef_df.sort_values(by ='coef', axis = 1, ascending = False).iloc[:,:n_words]

	#import dependancies
	import matplotlib.pyplot as plt
	#Visualize top words
	plt.figure(figsize =(10,10))
	plt.barh(coef_df_top_words.loc['coef'].index, coef_df_top_words.loc['coef'].values)
	plt.xlabel("coefficients")
	plt.xticks(rotation = 90)
	plt.ylabel("Words")
	plt.title('Most predictive words for Random forest model')
	plt.show()
###############################################################################################
def extract_data_pts(df, label, n_pts):
    df = df[df['label'] == label].sample(
    n = n_pts, replace = False, random_state = 7)
    return df
##################################################################################################	
def return_trnsfrmddf_visualize_words(X_train_transformed, X_test_transformed, fit,n_words):
	import pandas as pd
	import matplotlib.pyplot as plt
	X_train_transformed_df = pd.DataFrame(columns = fit.get_feature_names(),
										 data = X_train_transformed.toarray())
	X_test_transformed_df = pd.DataFrame(columns = fit.get_feature_names(),
										 data = X_test_transformed.toarray())
	#Most frequent words
	top_words = X_train_transformed_df.sum(axis = 0).sort_values(ascending = False)[:n_words - 1]

	#Visualize vectorized data
	plt.figure(figsize =(10,10))
	plt.barh(top_words.index, top_words.values)
	plt.xlabel("Count")
	plt.xticks(rotation = 90)
	plt.ylabel("Tokens")
	plt.show()
	#return dfs
	return X_train_transformed_df, X_test_transformed_df

###############################################################################################################
def visualize_entity(df, entity, n):
	import copy
	import pandas as pd
	import matplotlib.pyplot as plt
	import seaborn as sns
	#Extract top n words, by count, belonging to an enity
	top_n_words = df[df['entity'] == entity]['word'].value_counts()[:n-1]
	plt.figure(figsize=(10,5))
	sns.barplot(top_n_words.values, top_n_words.index, alpha=0.8)
	plt.title(str(entity) + " community is not happy about")
	plt.ylabel('Word from Tweet', fontsize=12)
	plt.xlabel('Count of Words', fontsize=12)
	plt.show()
	
	
##################################################################################################################
def vectorize_no_prd_filter(df_clean_text_col):
	#Vectorizer dependancies
	#Create a stemmer object and define stop words
	import pandas as pd
	import joblib
	bagofwords_150k = joblib.load('D:\\capstone_data\\processed_csv\\master_data_clean_sent140_cmplnts_mixed\\bagofwords_rm_prdcts_150K.pkl')
	vector = bagofwords_150k.transform(df_clean_text_col)
	vectorized_df = pd.DataFrame(columns = bagofwords_150k.get_feature_names(), data = vector.toarray())
	return vectorized_df
	
##################################################################################################################
def lda_bagofwords_topics(greivance_df_clean_text_col, n_components,max_iter):
	from sklearn.decomposition import LatentDirichletAllocation
	from sklearn.feature_extraction.text import CountVectorizer
	from nltk.corpus import stopwords
	ENGLISH_STOP_WORDS = stopwords.words('english')
	bagofwords = CountVectorizer(stop_words = ENGLISH_STOP_WORDS, min_df = 0.01, ngram_range =(1,3))
	greivance_transformed_bagofwords = bagofwords.fit_transform(greivance_df_clean_text_col)
	lda_bagofwords = LatentDirichletAllocation(n_components = n_components, max_iter = max_iter, random_state = 7)
	lda_bagofwords.fit(greivance_transformed_bagofwords)
	#Print modelled topics
	words = bagofwords.get_feature_names()
	topic_words_master = []
	for i, topic in enumerate(lda_bagofwords.components_):
		topic_words = [words[j] for j in topic.argsort()[: -11: -1]]
		for k in topic_words:
		    topic_words_master.append(k)
		print(f"Topic #{i} words: {topic_words}")
	return lda_bagofwords, greivance_transformed_bagofwords, topic_words_master




def mapped_topics_df(greivance_vectorized, lda_fit):
	from sklearn.decomposition import LatentDirichletAllocation
	from sklearn.feature_extraction.text import CountVectorizer
	from sklearn.feature_extraction.text import CountVectorizer
	import pandas as pd
	import numpy as np
	lda_output = lda_fit.transform(greivance_vectorized)
	topic_names = ["Topic"+ str(i) for i in range(lda_fit.n_components)]
	doc_names = ["Doc" + str(i) for i in range(lda_output.shape[0])]
	topics_df = pd.DataFrame(data = lda_output, columns = topic_names, index = doc_names )
	topics_df['dominant_topic'] = np.argmax(topics_df.values, axis = 1)
	return topics_df