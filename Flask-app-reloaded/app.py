from flask import Flask, render_template, request, jsonify
import cv2
import torch
import numpy as np
from torchvision import transforms
import mediapipe as mp
import torch
import torch.nn as nn

app = Flask(__name__)

# Load your custom models
class HandSignClassifier(nn.Module):
    def __init__(self, num_classes):
        super(HandSignClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model_c = HandSignClassifier(num_classes=36)
model_c.load_state_dict(torch.load('../hand_sign_classifier_final.pth', map_location=torch.device('cpu')))
model_c.eval()

model_v = HandSignClassifier(num_classes=13)
model_v.load_state_dict(torch.load("../hand_sign_classifier_best_vowel-bs16.pth", map_location=torch.device('cpu')))
model_v.eval()

# Class mapping for consonants (starting from 0)
consonants_mapping_nepali = {
    0: 'क', 1: 'ख', 2: 'ग', 3: 'घ', 4: 'ङ', 5: 'च',
    6: 'छ', 7: 'ज', 8: 'झ', 9: 'ञ', 10: 'ट', 11: 'ठ',
    12: 'ड', 13: 'ढ', 14: 'ण', 15: 'त', 16: 'थ', 17: 'द',
    18: 'ध', 19: 'न', 20: 'प', 21: 'फ', 22: 'ब', 23: 'भ',
    24: 'म', 25: 'य', 26: 'र', 27: 'ल', 28: 'व', 29: 'श',
    30: 'ष', 31: 'स', 32: 'ह', 33: 'क्ष', 34: 'त्र', 35: 'ज्ञ'
}

# Class mapping for vowels (starting from 0)
vowels_mapping_nepali = {
    0: 'अ', 1: 'आ', 2: 'इ', 3: 'ई', 4: 'उ', 5: 'ऊ',
    6: 'ऋ', 7: 'ए', 8: 'ऐ', 9: 'ओ', 10: 'औ', 11: 'अं',
    12: 'अः'
}

# Initialize the hand tracking module
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.)

# Function to calculate hand bounding box coordinates from landmarks
def get_hand_bounding_box(frame, hand_landmarks, padding=20):
    x_min, x_max, y_min, y_max = frame.shape[1], 0, frame.shape[0], 0
    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
        x_min = min(x_min, x)
        x_max = max(x_max, x)
        y_min = min(y_min, y)
        y_max = max(y_max, y)

    # Add padding to the bounding box coordinates
    x_min = max(0, x_min - padding)
    x_max = min(frame.shape[1], x_max + padding)
    y_min = max(0, y_min - padding)
    y_max = min(frame.shape[0], y_max + padding)

    return x_min, x_max, y_min, y_max

def preprocess_image(frame, hand_landmarks):
    # Access the hand landmarks (keypoints) to get the bounding box coordinates
    x_min, x_max, y_min, y_max = get_hand_bounding_box(frame, hand_landmarks)

    # Crop the hand region
    hand_image = frame[y_min:y_max, x_min:x_max]

    # Resize to match model input size
    hand_image = cv2.resize(hand_image, (128, 128))

    # Convert to tensor
    transform = transforms.Compose([
        transforms.Grayscale(),  # Convert to black and white
        transforms.ToTensor(),
    ])
    input_tensor = transform(hand_image).unsqueeze(0)

    return input_tensor


def predict_hand_sign(frame, model):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            input_tensor = preprocess_image(frame, hand_landmarks)
            model.eval()
            with torch.no_grad():
                output = model(input_tensor)
                
            _, predicted_class = torch.max(output, 1)
            
            return predicted_class.item()

    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if 'image' in request.files and 'model' in request.form:
        file = request.files['image']
        model_type = request.form['model']

        if model_type == 'c':
            model = model_c
            class_mapping = consonants_mapping_nepali
        elif model_type == 'v':
            model = model_v
            class_mapping = vowels_mapping_nepali
        else:
            return jsonify({'error': 'Invalid model type.'})

        predicted_class = None
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        predicted_class = predict_hand_sign(image, model)

        if predicted_class is not None:
            # Get the class label from the mapping
            class_label = class_mapping[predicted_class]
            print("Predicted Class:", class_label)
            return jsonify({'class': class_label})
        else:
            print("No hands detected in the image.")
            return jsonify({'error': 'No hands detected in the image.'})

    print("Invalid request.")
    return jsonify({'error': 'Invalid request.'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

