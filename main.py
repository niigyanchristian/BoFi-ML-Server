from fastapi import FastAPI, HTTPException
import requests
from keras.models import load_model
from PIL import Image
import numpy as np
import io
import pandas as pd

app = FastAPI()

# Load the model
# model = load_model('app/jaundice_classification_model.h5')
model = load_model('jaundice_classification_model_3.h5')

# Define image size and preprocessing function
img_width, img_height = 100, 100

def prepare_image(img: Image.Image) -> np.array:
    img = img.resize((img_width, img_height))
    img = img.convert("RGB")
    img = np.array(img)
    img = img / 255.0  # Normalize the image to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# class ImageUrl(BaseModel):
#     image_url: str

@app.post("/predict")
async def predict(image_url: dict):
    df = pd.DataFrame(image_url,index=range(1))
    print(df.iloc[0].image_url)
    image_url = df.iloc[0].image_url
    try:
        # Download the image from the URL
        response = requests.get(image_url)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
        
        # Preprocess the image
        prepared_img = prepare_image(img)
        
        # Make prediction
        prediction = model.predict(prepared_img)
        prediction_class = 'normal' if prediction[0][0] >= 0.52 else 'jaundice'
        
        return {
            "filename": image_url,  # Extract filename from URL
            "prediction": prediction_class,
            "confidence": float(prediction[0][0])
        }
    
    except requests.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the Jaundice Classification API!"}