from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = FastAPI(title="MNIST Digit Classifier API")


model = tf.keras.models.load_model("mnist_cnn_model.h5")

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = image.reshape(1, 28, 28, 1)
    return image

@app.get("/")
def root():
    return {"message": "MNIST Digit Classifier API is running"}

@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = preprocess_image(image_bytes)

    predictions = model.predict(img)
    predicted_class = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    return {
        "predicted_digit": predicted_class,
        "confidence": round(confidence, 4)
    }
