from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model

from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.utils import get_custom_objects


from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import load_model






import numpy as np
import pickle


max_length=26



# Open the file in read mode
with open('all_captions.pkl', 'rb') as f:
  # Load the data
  all_captions = pickle.load(f)

# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
print("captions loaded and tokenizer initialized")

# Register custom initializers
get_custom_objects().update({'Orthogonal': Orthogonal})
vgg_model = VGG16(weights=None, include_top=True)
vgg_model.load_weights("vgg16.h5")

# restructure the model
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
print("vgg16 loaded")
print(vgg_model.summary())


#model = load_model('model.h5', custom_objects={'LSTM': keras.layers.LSTM})
#model = load_model('model.h5', compile=False)
model = load_model('model.keras')
#model = load_model('model.h5', custom_objects={'CustomLayer': CustomLayer})


print("Ai model loaded")
print(model.summary())




def idx_to_word(integer, tokenizer):
    
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None



# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    
    
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break

    return in_text




def generate_caption_ar (image_path) :

    
    
    
    # load image
    image = load_img(image_path, target_size=(224, 224))
    # convert image pixels to numpy array
    image = img_to_array(image)
    # reshape data for model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # preprocess image for vgg
    image = preprocess_input(image)
    # extract features
    feature = vgg_model.predict(image, verbose=0)
    # predict from the trained model
    result =predict_caption(model, feature, tokenizer, max_length)   
    result = result.replace('startseq ', '').replace(' endseq', '')
    return result 

#print(generate_caption_ar("./test.jpg"))