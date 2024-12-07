import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from deepface import DeepFace

# Constants
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Definition (same as in training script)
def create_model():
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return model

# Load the trained model
def load_model(model_path):
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Preprocess image for inference
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

# Perform inference on a single face
def infer(model, face_image):
    with torch.no_grad():
        outputs = model(preprocess_image(face_image))
        _, preds = torch.max(outputs, 1)
    return preds.item()

# Detect faces and perform mask classification
def detect_and_classify(image_path, mask_model):
    # Read the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces using RetinaFace through DeepFace
    faces = DeepFace.extract_faces(img_path=image_path, detector_backend='retinaface')

    # Convert to PIL Image for drawing
    pil_image = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_image)

    # Try to load a font, use default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    label_map = {0: 'With Mask', 1: 'Without Mask', 2: 'Mask Worn Incorrectly'}
    color_map = {0: (0, 255, 0), 1: (255, 0, 0), 2: (255, 255, 0)}  # Green, Red, Yellow

    for face in faces:
        # Extract face ROI
        x = face['facial_area']['x']
        y = face['facial_area']['y']
        w = face['facial_area']['w']
        h = face['facial_area']['h']
        face_roi = pil_image.crop((x, y, x+w, y+h))

        # Perform inference
        prediction = infer(mask_model, face_roi)
        label = label_map[prediction]
        color = color_map[prediction]

        # Draw bounding box and label
        draw.rectangle([(x, y), (x+w, y+h)], outline=color, width=2)
        draw.text((x, y-25), label, font=font, fill=color)

    return pil_image

# Main execution
def main():
    model_path = '/content/drive/MyDrive/c/archive/mask_detection_model.pth'  # Path to your saved model
    image_path = '/content/drive/MyDrive/c/archive/images/maksssksksss10.png'  # Path to the test image

    # Load the mask detection model
    mask_model = load_model(model_path)

    # Perform detection and classification
    result_image = detect_and_classify(image_path, mask_model)

    # Save and show the result
    result_image.save('result.jpg')
    result_image.show()

if __name__ == '__main__':
    main()