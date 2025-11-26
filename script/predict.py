
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np


# load model
model = load_model("fake_image_detector.h5")

# image path
img_path = "test.jpg"  # replace with any image you want to test

# preprocess
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# predict
prediction = model.predict(img_array)[0][0]

if prediction > 0.5:
    print("✅ Real Image")
else:
    print("❌ Fake Image")