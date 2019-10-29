# Analyzing Toxic Comments Online

For this project, I will be classifying comments (found in a kaggle dataset) as 'toxic', 'severe toxic', 'obscene', 'threat', 'insult', and or 'identity hate' (or none of these). Any blog or site who has access to my model should be able to run the algorithm on their site to clear comments they deem innappropriate based on the categories above.

In this notebook, I walk through each step from the basic preprocessing to the more complex models I finally build out. Below I start by importing all libraries I need for the project:


```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

For Natural Language Processing: 


```python
import re  
import nltk

nltk.download('stopwords') # Downloading the words that are not relevant to exclude them from the data
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer # for stemming -- not sure if I will go through with this 
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /Users/anupaarora/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!



```python
import sys # In order to import tensorflow 
sys.executable

from keras.preprocessing.text import Tokenizer # For tokenization 
from keras.preprocessing.sequence import pad_sequences # To make all the sentences equal in length 
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation 
from keras import initializers, regularizers, constraints, optimizers, layers 
from keras.layers import Bidirectional, GlobalMaxPool1D 
from keras.models import Model
```

## Analyzing the Dataset

Below is the training set (all 95851 comments) that I am using to run my model on. As you can see each section has multiple different columns that can be classified as either a '0' or a '1'. 


```python
train_toxic = pd.read_csv("data/train_toxic.csv")
train_toxic
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>comment_text</th>
      <th>toxic</th>
      <th>severe_toxic</th>
      <th>obscene</th>
      <th>threat</th>
      <th>insult</th>
      <th>identity_hate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22256635</td>
      <td>Nonsense?  kiss off, geek. what I said is true...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>27450690</td>
      <td>"\n\n Please do not vandalize pages, as you di...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>54037174</td>
      <td>"\n\n ""Points of interest"" \n\nI removed the...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>77493077</td>
      <td>Asking some his nationality is a Racial offenc...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>79357270</td>
      <td>The reader here is not going by my say so for ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>82428052</td>
      <td>Fried chickens \n\nIs dat sum fried chickens?</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>87311443</td>
      <td>Why can you put English for example on some pl...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>114749757</td>
      <td>Guy Fawkes \n\nim a resident in bridgwater and...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>138560519</td>
      <td>as far as nicknames go this article is embarra...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>139353149</td>
      <td>Woodland Meadows\nGood to hear that you correc...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>150009866</td>
      <td>"\n\nWell I just finished a good bit of editin...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>152276337</td>
      <td>Discussion should take place on the article ta...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>153331729</td>
      <td>Uh oh, you called my bluff. I am intimidated b...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>153600803</td>
      <td>"\nWe should also contact the living descendan...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>154186883</td>
      <td>" May 2008 (UTC)\n\nNotability of Your New Hea...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>161894108</td>
      <td>"\n\nWhile I agree that this article isn't FA ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>169740962</td>
      <td>a Turkish citizen and him having received an a...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>178084608</td>
      <td>Please explain why censorship of quality addit...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>208005265</td>
      <td>In any case, this edit war will last forever. ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>224733383</td>
      <td>"\n\n ""Vandalism"" of George Washington \n\nW...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>225701312</td>
      <td>Why hasn't Alitalia been removed rom the allia...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>237279459</td>
      <td>"\n\n Another AfD stats example \n\nI hope you...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>254015372</td>
      <td>"\nI will ;). How about... ah, I've got nothin...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>280570905</td>
      <td>":I have moved some tedious detail in ""Survey...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>284253328</td>
      <td>@AnnieHall, what separates this from capitalis...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>287371884</td>
      <td>.  and its also not random, it was the first c...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>293668009</td>
      <td>"\nThe Graceful Slick....\nIs non other than a...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>300891545</td>
      <td>"====Regarding edits made during December 2 20...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>307040060</td>
      <td>"::The section is now called ""Discrepancies a...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>340940431</td>
      <td>"\n\n Smackdown! \n\nGood smackdown on Qatar, ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95821</th>
      <td>999674711016</td>
      <td>Gods. I'm an Anglo-Norman-ist, really, not an ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95822</th>
      <td>999677373048</td>
      <td>"==T. COTTON LETTER TO THE AYATOLLAHS OF IRAN=...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95823</th>
      <td>999693619845</td>
      <td>Much appreciated \n\nThank you! Yours is my fi...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95824</th>
      <td>999714130533</td>
      <td>I do apologize once more to you for my unkind ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95825</th>
      <td>999716611707</td>
      <td>It has been confirmed that Raul has joined FC ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95826</th>
      <td>999731184003</td>
      <td>"\nThat an article is ""more than a definition...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95827</th>
      <td>999741263322</td>
      <td>Calling someone the archetypical unscientific ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95828</th>
      <td>999744390044</td>
      <td>Just pointing out that if you intend to keep a...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95829</th>
      <td>999749161210</td>
      <td>I'm sorry for not noticing that on the article...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95830</th>
      <td>999776197636</td>
      <td>Can I just say, no-one cares about your opinio...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95831</th>
      <td>999781781533</td>
      <td>See Wikipedia:Administrators#Misuse_of_adminis...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95832</th>
      <td>999825778693</td>
      <td>December 2010\nPlease stop the foolish edits t...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95833</th>
      <td>999829405247</td>
      <td>Another Unblock Request \n\n 137.240.136.80</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95834</th>
      <td>999856640855</td>
      <td>so this can finally be over with</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95835</th>
      <td>999883234532</td>
      <td>Oh, okay. Fair enough, then. Thanks for making...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95836</th>
      <td>999887342595</td>
      <td>"\nI believe you're out of line. You're specul...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95837</th>
      <td>999888096837</td>
      <td>, 22 April 2007 (UTC)\n\nSecond-hander.  21:45</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95838</th>
      <td>999898414104</td>
      <td>"\nIt's staying. Let's move on.  Corbett "</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95839</th>
      <td>999903896729</td>
      <td>"\n\n Conflict of Interest \n\nPm_shef: This i...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95840</th>
      <td>999909788460</td>
      <td>"\nPerhaps the single most potent way to balan...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95841</th>
      <td>999912713635</td>
      <td>"\n Please don't bother. I was just wondering....</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95842</th>
      <td>999939579242</td>
      <td>The article The eighth sea has been speedily d...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95843</th>
      <td>999945355747</td>
      <td>Each alum agrees to how much information can b...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95844</th>
      <td>999950278781</td>
      <td>"\n\n Caucasion vs. white \n\nI noticed that t...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95845</th>
      <td>999976306590</td>
      <td>This culture allows people to hold their wives...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95846</th>
      <td>999977655955</td>
      <td>"\nI have discussed it, unlike most of those w...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95847</th>
      <td>999982426659</td>
      <td>ps. Almost forgot, Paine don't reply back to t...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95848</th>
      <td>999982764066</td>
      <td>Mamoun Darkazanli\nFor some reason I am unable...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95849</th>
      <td>999986890563</td>
      <td>Salafi would be a better term. It is more poli...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95850</th>
      <td>999988164717</td>
      <td>making wikipedia a better and more inviting pl...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>95851 rows × 8 columns</p>
</div>



### Checking for Null Values

Checking if any of the sections are null / I need to fill in and or change any data (it turns out I don't!) 


```python
train_toxic.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 95851 entries, 0 to 95850
    Data columns (total 8 columns):
    id               95851 non-null int64
    comment_text     95851 non-null object
    toxic            95851 non-null int64
    severe_toxic     95851 non-null int64
    obscene          95851 non-null int64
    threat           95851 non-null int64
    insult           95851 non-null int64
    identity_hate    95851 non-null int64
    dtypes: int64(7), object(1)
    memory usage: 5.9+ MB


### Counting Classifications

Now I want to count how many comments are classified under each category. (e.g. how many 'obscene' examples are there) -- I use the .sum() function 


```python
count = pd.DataFrame(columns=['Toxicity count', 'Severe_toxic count', 'Obscene count',
                             'Threat count', 'Insult count', 'Identity_hate count'])

count = count.append({'Toxicity count': train_toxic.toxic.sum(), 
                    'Severe_toxic count': train_toxic.severe_toxic.sum(),
                    'Obscene count': train_toxic.obscene.sum(),
                    'Threat count': train_toxic.threat.sum(),
                    'Insult count': train_toxic.insult.sum(),
                    'Identity_hate count': train_toxic.identity_hate.sum()}, 
                    ignore_index=True)
count
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Toxicity count</th>
      <th>Severe_toxic count</th>
      <th>Obscene count</th>
      <th>Threat count</th>
      <th>Insult count</th>
      <th>Identity_hate count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9237</td>
      <td>965</td>
      <td>5109</td>
      <td>305</td>
      <td>4765</td>
      <td>814</td>
    </tr>
  </tbody>
</table>
</div>



### Finding Correlation 

Trying to find the correlation between each of the target categories...


```python
corr = train_toxic.corr()
corr
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>toxic</th>
      <th>severe_toxic</th>
      <th>obscene</th>
      <th>threat</th>
      <th>insult</th>
      <th>identity_hate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id</th>
      <td>1.000000</td>
      <td>-0.000033</td>
      <td>0.004019</td>
      <td>0.005868</td>
      <td>-0.002872</td>
      <td>0.003664</td>
      <td>0.001950</td>
    </tr>
    <tr>
      <th>toxic</th>
      <td>-0.000033</td>
      <td>1.000000</td>
      <td>0.308810</td>
      <td>0.677491</td>
      <td>0.162967</td>
      <td>0.648330</td>
      <td>0.259124</td>
    </tr>
    <tr>
      <th>severe_toxic</th>
      <td>0.004019</td>
      <td>0.308810</td>
      <td>1.000000</td>
      <td>0.404540</td>
      <td>0.133469</td>
      <td>0.377450</td>
      <td>0.193385</td>
    </tr>
    <tr>
      <th>obscene</th>
      <td>0.005868</td>
      <td>0.677491</td>
      <td>0.404540</td>
      <td>1.000000</td>
      <td>0.149874</td>
      <td>0.744685</td>
      <td>0.287794</td>
    </tr>
    <tr>
      <th>threat</th>
      <td>-0.002872</td>
      <td>0.162967</td>
      <td>0.133469</td>
      <td>0.149874</td>
      <td>1.000000</td>
      <td>0.157534</td>
      <td>0.123971</td>
    </tr>
    <tr>
      <th>insult</th>
      <td>0.003664</td>
      <td>0.648330</td>
      <td>0.377450</td>
      <td>0.744685</td>
      <td>0.157534</td>
      <td>1.000000</td>
      <td>0.331922</td>
    </tr>
    <tr>
      <th>identity_hate</th>
      <td>0.001950</td>
      <td>0.259124</td>
      <td>0.193385</td>
      <td>0.287794</td>
      <td>0.123971</td>
      <td>0.331922</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Investigating Individual Comments

Here I investigate indvidual comments and attempt to on the surface look at the difference between threats, insults, and identity_hate by looking at comments from each section that contains '1' for that section. 


```python
threats = train_toxic.loc[train_toxic['threat'] == 1]
insults = train_toxic.loc[train_toxic['insult'] == 1]
identity_hate = train_toxic.loc[train_toxic['identity_hate'] == 1]

print("threats: ", threats.comment_text)
print("\n\ninsults: ", insults.comment_text)
print("\n\nidentity_hate: ", identity_hate.comment_text)
```

    threats:  981      "\n\n A knife for you! \n\n  A knife for you! ...
    1471     My personal opinion \n\nIs that you're a fucki...
    1563     I am going to shoot you in the head and laugh ...
    1703     Stop being a shithead, or I will rip off your ...
    2457     You. \n\nYou're such a sad person, the fact yo...
    2984     I AM ANGRY AT YOU BILLJ! YOU GOT PEOPLE BLOCKE...
    4034     You are such a fucking hypocrite. \n\nI hope y...
    4102     Who gives a shit? Anthony Bourdain is a tool. ...
    4180     Personal Attack Number 2 \n\nThis is another p...
    4285                               Graeme Bartlet MUST die
    4339     Wow dude.  As your physician I suggest you sli...
    4558     JD IS A RAGING HOMOSEXUAL FAGGOT WHO SHOULD BE...
    4847     You're a stupid cunt \n\nFuck you dumb arse, y...
    4909     Hello, I think you are a fuckwad. Eat shit and...
    5282     If you think SOAD is nu-metal.... \n\nYou are ...
    5562     Last warning, Mark! Anymore defiant arguments ...
    5773     I'ma smack ya upside da head wit a shovel \n\n...
    5924     Terrorize \nI will terrorise you for as long a...
    5940     Bertie, You're A Stupid Ignorant Cunt !!\n\nDo...
    7004               I wish you had died in that earthquake.
    7258     Eat shit you fucking arse rapping jew fucking ...
    7364     Bastard \n\nYou fat bastard! You have done it!...
    7406     CALTLAS IF YOU CONTINUE TO RVV IT, I WILL BLAN...
    7609     THAKSIN IS A FXXCKING THIEF AND SHOULD BE SHOT...
    7867     I'm gonna beat you to a bloody pulp then shoot...
    8115     I am going to kill you. I am going to murder you.
    8354     You do not deserve to be alive \n\nLook over y...
    9051     Do not ever send me a message again i will hun...
    9211     why good sir? Why? \n\nYou, sir, obviously do ...
    9318     "\n\nNew Award\n\nWell Done!!!\n\n  The IAmThe...
                                   ...                        
    86525    I am going to kill you. I am going to murder y...
    86946                             Drop dead \n\nDrop dead.
    87001    you see that kid talking to me i hate him he n...
    87288    "\n\n Niggard PLEASE! \n\nThe niggardly nigger...
    87617    I'll fucking eat your queef asshole \n\nIf you...
    87931    If no-one beats me to it, I'll knock something...
    87949    I am going to bash your skull in and urinate a...
    88074    If I hack your account, I am going to poop on ...
    88118    sonek \n\nput princes saly in sonek 4 or clay ...
    88177    My mate told me you were a fuckin homosexual. ...
    88640    I am going to shove a pineapple up your ass \n...
    89164    Your a twat, you just delete things that you d...
    89335    (http://wiki.uschallengecoins.com/Main_Page)\n...
    89764    I will murder your pets and slash your tires. ...
    90076    JEW \n\nGet the fuck out of here you jewish so...
    90289    Hello \nHear me dammed bastard, some images ar...
    90341    THIS MAN SHOULD BE SHOT FOR WAR CRIMES AND MURDER
    90715    FUCKING WANKER. FUCK OFF BEFORE I COME AND LEA...
    91018    You don't know what are real difficulties, lit...
    92220    Hi! I am back again!\nLast warning!\nStop undo...
    92566    Wtf, Why You deleting Maashel? Bad things will...
    92722    You... \n\nYou are attacking me again, beacuse...
    92765    "\n Ain't bothering me, I'm not required to si...
    93111    You are the troll, anti-Semite. \n\nAll anti-S...
    93118    HEY ASSHOLE I'LL FIND OUT WHERE YOU LIVE AND B...
    93535    hi were bp [PARODIE] \n\nhello were gonna kill...
    94193    mothafucka! \n\nYou son of a bitch, shup up!\n...
    94637    I'm going to track you down \n\nI am going to ...
    94815    JIM WALES MUST DIE!!!!!!!!!!!!  JIM WALES MUST...
    94991    SuPeRTR0LL WiLL LiVe FoReVeR!\niF You DoN'T Re...
    Name: comment_text, Length: 305, dtype: object
    
    
    insults:  32       "\nBan one side of an argument by a bullshit n...
    81       "\n\nIt was very constructive you are just ver...
    86       "\n\n You know what? Fuck you! I tried to be c...
    104      "\n\n Corrected spelling of ""recognised"" to ...
    122      your time....\n\nYou suck because you are RIGH...
    124                           YOU NEED A PUSSY, YOUR GEEK!
    160      loser, get a life\n\nyou must be fat, ugly and...
    213      F'UCK YOU ADMINISTRATORS (I WILL BE BACK TO TR...
    292      Jimbo is bending over backwards so hard to do ...
    307      Another stupid person who considers leaving a ...
    308      These Iranian bastards are at it again. Not on...
    330                        is the sock puppet of ,fuck off
    352      Idiots\n\nThe Barris batmobile did not have a ...
    436            you are a stupid ass hole thats my  message
    459                                  alex fuck you \n\n...
    480      AH HAAHHAHAHAH YAAA BITCH YAAAA!@!!!!!!AH HAAH...
    490                              fucking loser suck a dick
    511      WikiTheClown\nWelcome to Wikipedia. Although e...
    571      "\n\n The Great One! \nI dont really have a pr...
    577      STOP YOUR DAMN NONSENSE LUMINIFER!!!!!  YOU HA...
    587                          fuck you i will edit anything
    595      probably that you are a german... fucking germ...
    616      Show up again when you figure out how to rub t...
    626      and your a fucking fat wanker with no life. si...
    633      hey you dumb bitch if i cant spell saturn girl...
    665      Eat a Dick, Lucasbfr \n\nYo . i'm [[User:Cool ...
    674      who are you? \n\nwho the hell are you and why ...
    708                        Screw my Butt \n\nScrew my Butt
    714                     blow \n\nit out of your ass, spic.
    730      <- Someone doesn't listen to Propagandhi. I pi...
                                   ...                        
    95108    "\n\nFuck off you stupid fucking retard cunt.\...
    95157                            SO YOU ADMIT YOURE A FAG?
    95165    Congrats  \n\nCongrats that the wikidouchebags...
    95189    You, too can join the ranks of looking-to-be-o...
    95203    wikipedia search warrior4321 \n\nyour a dickhe...
    95220    FUCK U DENNIS BURN IN HELL YOU SON OF A BITCH ...
    95269    Fuck you 2.0 \n\nYou are an idiot and an admin...
    95282    What a retard, that . Shut your FUD up, you ar...
    95287    hey faggot. \n\nInsert non-formatted text here...
    95301    THESE WIKIPEDIA PEOPLE ARE LIARS AND FASCISTS ...
    95337    fuck wikipidia \n\nand fuck you.\n\nFuck this ...
    95357    Kind Sir, why dont you try to lick my butthole...
    95362    metalcore is NOT a punk genre you fucking idio...
    95370    Why are you harassing me, when you have never ...
    95398    Diem was a gay homosexual dictator \nThis fact...
    95399    The above user, who has failed to sign, is ign...
    95411                              your a peace of ass!!!!
    95412              Go fuck yourself and block me you cunt.
    95423    Yall are vandals\n\nI like Tyar and Stopdropro...
    95426    Wrestling with a pig \n\nWell, you know what t...
    95430    You are the biggest and dumbest idiot I have e...
    95498    There are some pathetic people on Wikipedia to...
    95517    Layne Staley\nwho told you that you crackhead,...
    95542    IN THE anus with a nigger dick in it.68.33.41.181
    95576           Try the official website then - shit head!
    95702    Licking my ass \nHahaha quit banning me you fa...
    95718    Suitcivil133 is a little bitch. he is a bandwa...
    95731                           get a proper job daft cunt
    95741    so, who the fuck gives a damn, you dipshitted ...
    95751    asshole and Richard Simmons's asshole, and ass...
    Name: comment_text, Length: 4765, dtype: object
    
    
    identity_hate:  308      These Iranian bastards are at it again. Not on...
    571      "\n\n The Great One! \nI dont really have a pr...
    595      probably that you are a german... fucking germ...
    889      "\n\n Contested deletion \n\nThis page should ...
    1013     Also, it's well known that slavic women don't ...
    1015     How can you defend those barbarian invaders? \...
    1172     i think this som bitch is fuckin nuts...i mean...
    1916     Fuck Wikipedia and fuck all the leftist, liber...
    1951     stop reverting my edit\nyour so fucken gay get...
    2111     He IS in Gay German Nazi Faggots. Please do no...
    2187     If you guys really discuss napoleon you need t...
    2297     Suck it!!! Asyou're used to do,ignorant Romani...
    2322                 LOL \n\nStop eating shit you nigger!!
    2467     Swedes are the racist trolls. They support the...
    2592     Fuck you you stupid and gay bastard who thinls...
    2776        MATE, are you jewish?\n\nGayness is in the air
    2798     NIGHTSTALLIONS FIANCE GOT FUCKED BY A NIGGER A...
    2817     What a faggot.  The man above says he's an adu...
    3003     "\n\nToo funny\n\ncorey suggests a famous peop...
    3062     It's \n\nGo Go Jewranger\nGo Go Jewranger\nHar...
    3071     Abrech is a stupid jew rat. and i hate jews. s...
    3136                            your fucking face. faggots
    3283     ur so immature fr3nzi3,im 17 and ur not evn de...
    3335     RE:Knowledge Generation Bureau \n\nWhy the fuc...
    3338     "\n\n Ah, typical drunken Irish bitch. If you ...
    3454     also do not vandilise streets when you feel li...
    3461                   nigger paedophile nigger paedophile
    3497     Yeah thats a really good way to get a life! Fu...
    3584     NEUTRALHOMER HATES BLACK PEOPLE! AND SO DOES G...
    3631                      Haha. Go ahead. Stupid American.
                                   ...                        
    91836                   Jimbo Wales is a gay little fucker
    91842    Fuck the shit out of you, Bharatveer! You dese...
    91928    Jusdafax is a little whiny bitch!!! Lick my sw...
    91949                          Andrew Kwon \n\nFuCKING FAG
    92182                      You are very gay \n\nYou're gay
    92324    Hey fuck you Hal Turner sent a death threat to...
    92411                    FUCK YOU CRAZY FAGOTS!Italic text
    92440    Fuck off you right wing nazi scum. BTW your cr...
    92488    Gay\nI would like to request the removal of th...
    92935    YOU DUMB ASS!!\nWhy the fuck would you delete ...
    93067    This indo-fijian dude is a helpful janitor, bu...
    93479    These hoe's lame on here on leaving this site ...
    93505    Hey\n\nI bet you Quinsareth are gay and like t...
    93560                      SCONES ARE COOL ARD YOU ARE GAY
    93655    HEY BITCH STOP TAKING DOWN OUR FUCKING EVENT P...
    93656               SOCALSUPEREAGLE IS FULL OF BLACK HATE!
    93758    Hell-bound fuckwit!\n\nHu,you are a fuckwit wh...
    93923                              n i g g e r f a g g o t
    93980      you fag \n\nWAAA WAAA I'M A HUGE GAY (says you)
    94000                       . USE THAT BANHAMMA NAO GAYBOY
    94205    You are homosexual. You fuck people through th...
    94218    Homo... \n\nDon't be gay- mind your own fuckin...
    94790           A pair of jew-hating weiner nazi schmucks.
    94806    the RepulicanSodomite has responded!!! praise ...
    95268           Shutup \n\nU ain't nobody, U ugly gay fag,
    95287    hey faggot. \n\nInsert non-formatted text here...
    95398    Diem was a gay homosexual dictator \nThis fact...
    95542    IN THE anus with a nigger dick in it.68.33.41.181
    95702    Licking my ass \nHahaha quit banning me you fa...
    95741    so, who the fuck gives a damn, you dipshitted ...
    Name: comment_text, Length: 814, dtype: object


## Natural Language Processing

The point of natural language processing is to clean up the text so only the relevant info is left that will allow the computer to better classify the data. For instance, irrelevant info such as the words 'a', 'the', 'your' etc. needs to be removed because it could effect the ultimate classification in a bad way. 

To do this I used the documentation from: https://pythonspot.com/nltk-stop-words/

In order to do natural language processing, I want to do the following steps:

1. Get rid of all punctuation and irrelevant words (called stop words such as 'a', 'the', 'your' etc.)
2. (Possible step -- won't do it) Make any words into its root word (e.g. loved will become love)
3. Making everything lowercase so capitals don't have a large effect on how it is classified

Here I am going to show an example with one comment and will later apply it to every comment:


```python
first_comment = train_toxic['comment_text'][0]
first_comment
```




    "Nonsense?  kiss off, geek. what I said is true.  I'll have your account terminated."




```python
comment = re.sub('[^a-zA-Z]', ' ', first_comment) # Taking out everything that is not a - z / A - Z and is not capital
comment = comment.lower() # making everything in the comment lowercase
comment = comment.split() # split the comment into different words so it becomes a list of the different words

# ps = PorterStemmer() (If I decided to do stemming I would add this)
```

Taking out 'stop words' which are basically irrelevant words that will confuse the machine


```python
comment = [word for word in comment if not word in set(stopwords.words('english'))] 
comment = ' '.join(comment) # joining them all together

print("Original Comment: ", first_comment)
print("New comment: ", comment) 
```

    Original Comment:  Nonsense?  kiss off, geek. what I said is true.  I'll have your account terminated.
    New comment:  nonsense kiss geek said true account terminated


## Cleaning up all the text 

Below I go through each comment and clean up all of the text to prepare it for training. 


```python
full_text = []

for i in range(len(train_toxic)):
    comment = re.sub('[^a-zA-Z]', ' ', train_toxic['comment_text'][i])
    comment = comment.lower() 
    comment = comment.split()
    
    comment = [word for word in comment if not word in set(stopwords.words('english'))]
    comment = ' '.join(comment)
    
    full_text.append(comment)
    
train_toxic['comment_text'] = full_text
```

The new cleaned up data:


```python
train_toxic
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>comment_text</th>
      <th>toxic</th>
      <th>severe_toxic</th>
      <th>obscene</th>
      <th>threat</th>
      <th>insult</th>
      <th>identity_hate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22256635</td>
      <td>nonsense kiss geek said true account terminated</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>27450690</td>
      <td>please vandalize pages edit w merwin continue ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>54037174</td>
      <td>points interest removed points interest sectio...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>77493077</td>
      <td>asking nationality racial offence wow aware bl...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>79357270</td>
      <td>reader going say ethereal vocal style dark lyr...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>82428052</td>
      <td>fried chickens dat sum fried chickens</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>87311443</td>
      <td>put english example players others people like</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>114749757</td>
      <td>guy fawkes im resident bridgwater go carnival ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>138560519</td>
      <td>far nicknames go article embarrassing human fi...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>139353149</td>
      <td>woodland meadows good hear corrected</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>150009866</td>
      <td>well finished good bit editing chance go taggi...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>152276337</td>
      <td>discussion take place article talk page user t...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>153331729</td>
      <td>uh oh called bluff intimidated number images p...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>153600803</td>
      <td>also contact living descendants adolf hitler o...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>154186883</td>
      <td>may utc notability new heart tag placed new he...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>161894108</td>
      <td>agree article fa material must suggest added c...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>169740962</td>
      <td>turkish citizen received award notability due ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>178084608</td>
      <td>please explain censorship quality additions to...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>208005265</td>
      <td>case edit war last forever want go tit tat pla...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>224733383</td>
      <td>vandalism george washington say changes consti...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>225701312</td>
      <td>alitalia removed rom alliance due piss poor cu...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>237279459</td>
      <td>another afd stats example hope mind letting kn...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>254015372</td>
      <td>ah got nothing know need random president usa ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>280570905</td>
      <td>moved tedious detail surveys helmet use cyclin...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>284253328</td>
      <td>anniehall separates capitalism state terrorism...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>287371884</td>
      <td>also random first clan guild star wars battlef...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>293668009</td>
      <td>graceful slick non ungraceful dick</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>300891545</td>
      <td>regarding edits made december utc thank experi...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>307040060</td>
      <td>section called discrepancies inaccuracies fine...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>340940431</td>
      <td>smackdown good smackdown qatar long al khalifa...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95821</th>
      <td>999674711016</td>
      <td>gods anglo norman ist really anglo saxonist mu...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95822</th>
      <td>999677373048</td>
      <td>cotton letter ayatollahs iran dwp including te...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95823</th>
      <td>999693619845</td>
      <td>much appreciated thank first star ever cheers ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95824</th>
      <td>999714130533</td>
      <td>apologize unkind insult traditional among iris...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95825</th>
      <td>999716611707</td>
      <td>confirmed raul joined fc schalke link http www...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95826</th>
      <td>999731184003</td>
      <td>article definition make encyclopaedia article ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95827</th>
      <td>999741263322</td>
      <td>calling someone archetypical unscientific econ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95828</th>
      <td>999744390044</td>
      <td>pointing intend keep arguing thing whole month...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95829</th>
      <td>999749161210</td>
      <td>sorry noticing article mea culpa still plan bu...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95830</th>
      <td>999776197636</td>
      <td>say one cares opinion also see member wikiproj...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95831</th>
      <td>999781781533</td>
      <td>see wikipedia administrators misuse administra...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95832</th>
      <td>999825778693</td>
      <td>december please stop foolish edits article chr...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95833</th>
      <td>999829405247</td>
      <td>another unblock request</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95834</th>
      <td>999856640855</td>
      <td>finally</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95835</th>
      <td>999883234532</td>
      <td>oh okay fair enough thanks making effort expla...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95836</th>
      <td>999887342595</td>
      <td>believe line speculating fruitless debates wou...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95837</th>
      <td>999888096837</td>
      <td>april utc second hander</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95838</th>
      <td>999898414104</td>
      <td>staying let move corbett</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95839</th>
      <td>999903896729</td>
      <td>conflict interest pm shef attempt resolve conc...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95840</th>
      <td>999909788460</td>
      <td>perhaps single potent way balance article line...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95841</th>
      <td>999912713635</td>
      <td>please bother wondering also telling reason us...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95842</th>
      <td>999939579242</td>
      <td>article eighth sea speedily deleted wikipedia ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95843</th>
      <td>999945355747</td>
      <td>alum agrees much information released alums ge...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95844</th>
      <td>999950278781</td>
      <td>caucasion vs white noticed change white caucas...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95845</th>
      <td>999976306590</td>
      <td>culture allows people hold wives hostage dowry...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95846</th>
      <td>999977655955</td>
      <td>discussed unlike revert heonsi pure sockpuppet...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95847</th>
      <td>999982426659</td>
      <td>ps almost forgot paine reply back shit want se...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95848</th>
      <td>999982764066</td>
      <td>mamoun darkazanli reason unable fix bold forma...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95849</th>
      <td>999986890563</td>
      <td>salafi would better term politically correct u...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95850</th>
      <td>999988164717</td>
      <td>making wikipedia better inviting place</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>95851 rows × 8 columns</p>
</div>



## Tokenization 

To go forward, there are two models I could prepare for. One is the LSTM model and the other is the 'Bag of Words' model. In order to prepare for both, I need to do the following now that my data is 'cleaned'. Simply, using a tokenizer allows me to divide my comments up into meaningful pieces. 

Here I start the process:


```python
targets = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
values = train_toxic[targets].values
training_setences = train_toxic["comment_text"]

tokenization = Tokenizer(num_words = 19500) # 19500 = num of unique words 
tokenization.fit_on_texts(list(training_setences)) 
training_after_tokenization = tokenization.texts_to_sequences(training_setences)
```

### Adjusting Sentence Length 

Now what I need to do is adjust the sentence length so each sentence is the same length so when I input it, it will work equally in the model and won't interefere with it. I can go about this in mutliple ways one of which is filling some sentences with 0's until it is the same length as others.

Therefore, I learned 'padding' which lets you make a max-length for some sentences (that are extremely long) and add 0's to sentences that are too short. I accomplish this through the following piece of code: 


```python
padding_training = pad_sequences(training_after_tokenization, 175)
```

# The Model

Below I outline all the models I use. Much of the info that I learned to understand what RNN's/LSTM's were and there function in Machine learning was taken from this blog: http://colah.github.io/posts/2015-08-Understanding-LSTMs/ (however, NO code was taken from this blog and used). 

## RNN + LSTM 

The first model I will use is a LSTM Neural Network which is based on the idea behind Reccuring Neural Networks and solves the vanishing gradient problem. 

**RNN's**: The whole concept of reccuring neural networks is to bascially remember what happened previously. So it is the same thing as a standard neural network but is able to remember what was in that neuron. This allows them to pass information onto themselves in the future and then remember things. For example, if you couldn't remember what happened in the previous presentation that would be pretty sad. That is your short term memory being enacted here and we want to use that same power with neural networks. Ultimately, it gives you previous contect. 

**The Vanishing Gradient Problem**: As we propogate the error through the network, it goes through many layers of neurons connected to themselves, it causes the gradient to decline rapidly meaning that the weights of the layers on the very far left are updated much slower than the weights on the far right creating a chain effect, effecting the whole thing. 

Basically, the vanishing gradient problem is where the RNN can remember things from the short term (and this is all that is necessary usually for a small amount of data) but as time goes on, it has trouble remembering things and slowly deteriates until it is eventually useless. 

**LSTM Solution**: In all practical terms, LSTM's are built specifically around long term memory. The way they accomplish this is very complex (see the blog cited above) and basically uses four neural networks or 'valves' in one to accomplish this. 

![LSTM Solution Example](Desktop/LSTM Example.png)

## Layers I will Create:

The final classes I am using to create the ultimate neural network (all of these have been imported above):

1. I will use the 'Dense' class to add the output layer
2. I will use the 'LSTM' class to add the LSTM layers (as discussed above)
3. I will use the 'Dropout' class to add dropout regularization 

Dropout Regularization basically allows me to reduce overfitting (only for neural networks) by dropping out units of the hidden layers and visible layers in the neural network. 

I also create standard input layers and embedding layers (a pretty confusing topic, here is a blog I found in addition to the LSTM one that is helpful: (https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/). 

Below I create it:  


```python
the_inputs = Input(shape=(175, )) # Input layer (175 = the maximum length for the size of a comment)
out = Embedding(19500, 125)(the_inputs) # Embedding layer -- to see the relevance and contect of any particular word (19500 is the max feautures)
out = LSTM(60, return_sequences=True, name='lstm_layer')(out) # Return sequences = True allows me to build more LSTM layers that can work together
out = GlobalMaxPool1D()(out) # To reshape the data from 3D to 2D
```

The Dropout Layer: The dropout layer drops out units of the hidden layers and visible layers in the neural network. After research, I found that the average amount to dropout was 10-20%, and after some tweaking (seen at the end) to see what works best, I found that the optimal dropout layer was 12% for my dataset. 


```python
out = Dropout(0.12)(out)
out = Dense(52, activation="relu")(out) # Found 50 was the best after extensive grid search
out = Dropout(0.12)(out)
out = Dense(6, activation="sigmoid")(out) # Using sigmoid to get binary classification on each of the labels
```

## The LSTM Final Model:

Finally, to create the model, I used binary crossentropy because I needed to classify things as either '0's or '1's. 


```python
LSTM_model = Model(inputs = the_inputs, outputs = out)
LSTM_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

Now while running it, after extensive grid search work, I found that the optimal batch size was 33 with the ideal number of epochs being 2. Validation split allowed me to split the training set into partly a test set so I can see how it performs as predictions. 


```python
LSTM_model.fit(padding_training, values, batch_size=33, epochs=2, validation_split=0.1)
```

    Train on 86265 samples, validate on 9586 samples
    Epoch 1/2
    86265/86265 [==============================] - 417s 5ms/step - loss: 0.1004 - acc: 0.9705 - val_loss: 0.0572 - val_acc: 0.9793
    Epoch 2/2
    86265/86265 [==============================] - 412s 5ms/step - loss: 0.0508 - acc: 0.9816 - val_loss: 0.0530 - val_acc: 0.9810





    <keras.callbacks.History at 0x1a34597470>



## Grid Search 

Below I show an example of my grid search function. I ran it much, much more extensively for many of the parameters (including the Dropout amounts, batch size, epochs, max sentence length etc.) I change the function each time I run it, and have made some changes to where you will have to tweak it a bit for it to properly run. The untweaked function can be seen here below: 


```python
from sklearn.model_selection import GridSearchCV

def run_grid_search(model):
    grid = GridSearchCV(model,
                         param_grid={'epochs': [1, 2, 3, 4],
                                      'batch_size' : [31, 32, 33, 34]},
                         scoring='accuracy',
                         cv=3)

    grid.fit(the_inputs, out)
    grid_results = grid.cv_results_ 
    zipped_results = list(zip(grid_results["params"], grid_results["mean_test_score"])) 
    zipped_results_sorted = sorted(zipped_results, key=lambda tmp : tmp[1])[::-1]
    for element in zipped_results_sorted: 
        print(element[1], element[0])

    print('The parameters of the best model are: ') 
    print(grid.best_params_)
```

# Conclusions

In the end, I learned a lot about understanding my data, natural language processing, using tokenization, using RNN models, the vanishing gradient problem and how LSTM solves that, and using grid search to effectively change the hyper parameter in my model. 

Ultimately, after extensive grid search work, I was able to get my model up to a 98% prediction accuracy on the *test set* I split it into! I learned a lot of new things and dipped my feet into complex models within deep learning...I hope to continue on this path. 



 
