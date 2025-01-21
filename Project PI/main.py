from typing import Union
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import httpx
import my_module
import os

app = FastAPI()

# CSV file path
csv_file_path = "class_data.csv"
label_mapping = my_module.read_label_mappings_from_csv(csv_file_path)

# Template folder path
templates = Jinja2Templates(directory=".")

# Mount the uploads directory to serve static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post('/predict')
async def predict(request: Request, file: UploadFile):
    contents = await file.read()
    
    file_location = f"uploads/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(contents)
        
    #Convert image to base 64
    image_base64 = my_module.image_to_base64(contents)

    response = httpx.post('http://localhost:8601/v1/models/my_simple_model:predict', json=image_base64)

    predictions = response.json().get('predictions', [])

    translated_predictions = []
    accuracy = None

    for prediction in predictions:
        # Extract accuracy
        accuracy = prediction[0] * 100  # Convert to percentage format
        
        # Assuming prediction is a single value for sigmoid output
        if prediction[0] > 0.5:
            translated_prediction = label_mapping[0]  # rotten
        else:
            translated_prediction = label_mapping[1]  # fresh
        
        translated_predictions.append(translated_prediction)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "predictions": translated_predictions[0]["name"],
        "accuracy": f"{accuracy:.2f}%",  
        "image_url": f"/uploads/{file.filename}",
    })

@app.get("/version")
async def check_model():
    response = httpx.get('http://localhost:8601/v1/models/my_simple_model/versions/2')

    return response.json()