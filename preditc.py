from keras.models import load_model 
from keras.preprocessing import image
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pandas as pd
import os,glob


PATH = os.getcwd()
model = load_model('model/cifar10')


encoder = LabelBinarizer().fit(["upright", "rotated_left", "rotated_right", "upside_down"])

encoder_from_pos = LabelBinarizer().fit([0,1,2,3])

X_test = []
X_test_path = glob.glob(PATH + '/test/*.jpg')
for sample in X_test_path:
        x = image.load_img(sample, target_size=(32,32,3))
        X_test.append(image.img_to_array(x))

fn = [x.split('/')[-1] for x in X_test_path]
y_test = model.predict(np.array(X_test))

y_test =  np.argmax(y_test, axis=1)
decoded_y = encoder.inverse_transform(encoder_from_pos.transform(y_test))
df = pd.DataFrame(decoded_y)
df.columns = ['label']
df['fn'] = fn
print("?")
df.to_csv('test.preds.csv')
