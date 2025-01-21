from io import BytesIO
from PIL import Image
import numpy as np
import base64
import csv
import cv2

# Function to read label mappings from CSV file
def read_label_mappings_from_csv(csv_file_path):
    label_mapping = {}
    with open(csv_file_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            index = int(row['Index'])
            name = row['Name']
            label_mapping[index] = {"name": name}
    return label_mapping

def image_to_base64(image):
    # Load and preprocess the image
    image = Image.open(BytesIO(image))
    image = image.resize((100, 100))  # Resize to 100x100 as in the Colab code
    image_array = np.array(image)
    
    # Convert to RGB if not already
    if image_array.shape[2] == 4:  # If the image has an alpha channel
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
    elif image_array.shape[2] == 1:  # If the image is grayscale
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)

    # Normalize the pixel values to the range [0, 1]
    image_array = image_array / 255.0

    # If your model expects a specific input shape, adjust accordingly
    # For example, a common shape is (1, height, width, channels)
    image_array = np.expand_dims(image_array, axis=0)

    # Convert to list (or use base64 if preferred)
    image_list = image_array.tolist()

    # Alternatively, you can use base64 encoding
    _, buffer = cv2.imencode('.jpg', (image_array[0] * 255).astype(np.uint8))
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    # Create the payload
    payload = {
        "instances": image_list  # or [{"b64": image_base64}]
    }

    return payload