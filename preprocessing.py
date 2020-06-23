from nltk.stem.porter import PorterStemmer  
import nltk  
import pandas as pd 
import re
from nltk.corpus import stopwords

def preprocessor(text):
    text=re.sub('<[^>]*>','',text)#移除HTML標記，#把<>里面的東西刪掉包括內容
    emotions = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    text=re.sub('[\W]+',' ',text.lower())+''.join(emotions).replace('-','')
    return text
	
def tokenizer(text):#提取詞匯
    return text.split()


def tokenizer_porter(text):#文本分詞並提取詞幹
    tokenizer_porter_text = [porter.stem(word) for word in text]
    #print("--------5(1)---------", tokenizer_porter_text)
    return tokenizer_porter_text


def FilterStopwords(text):
    #global filtered_sentence
    for word in text:
        if word in stop_words:
            text.remove(word)
    #print("--------6(1)---------",text)
    return text

	
if __name__ == "__main__":
    print("--------1---------")
	#
    df = pd.read_csv("C:/Users/YT/Desktop/MLfinalproject/aclImdb_v1/aclImdb.csv", encoding='utf-8',low_memory=False)
    print("--------2---------",df.shape[1],df['review'])
	
    df['review'] = df['review'].apply(preprocessor)
    print("--------3---------",df['review'])
	
    df['review'] = df['review'].apply(tokenizer)
    print("--------4---------",df['review'])
	
    porter = PorterStemmer()
    df['review'] = df['review'].apply(tokenizer_porter)
    print("--------5---------",df['review'])
	
	#只需downlod一次就好
	#nltk.download('stopwords')
	#停用詞移除(stop-word removal)，停用詞是文本中常見單不能有效判別信息的詞匯
    stop_words = stopwords.words('english') #獲得英文停用詞集
    #filtered_sentence = pd.DataFrame()
    df['review'] = df['review'].apply(FilterStopwords)
    print("--------6---------",df['review'])
	
    df.columns = ["review","sentiment"]
    print("--------7---------")
    df.to_csv("C:/Users/YT/Desktop/MLfinalproject/aclImdb_v1/cleaneddata.csv",index=False)
    print("--------8---------")