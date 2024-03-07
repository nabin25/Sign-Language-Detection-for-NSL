import cv2
import torch
import mediapipe as mp
import torch
import torch.nn as nn
from torchvision import transforms
# from pytorch_model import HandSignClassifier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your custom model
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

model = HandSignClassifier(num_classes=36).to(device)
model.load_state_dict(torch.load('hand_sign_classifier.pth'))
model.eval()
print("Model loaded")
# Initialize the hand tracking module
mp_hands = mp.solutions.hands

# Create a hands detector
hands_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.1)

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
    transform = transforms.ToTensor()
    input_tensor = transform(hand_image).unsqueeze(0).to(device)

    return input_tensor

def predict_hand_sign(frame, model, hands_detector):
    # Convert the frame from BGR to RGB
    print("predicting")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands in the frame
    results = hands_detector.process(frame_rgb)

    # Check if any hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Preprocess the image
            input_tensor = preprocess_image(frame, hand_landmarks)

            # Set the model to evaluation mode
            model.eval()

            # Make the prediction
            with torch.no_grad():
                output = model(input_tensor)

            # Get the predicted class index
            _, predicted_class = torch.max(output, 1)
            # Print the predicted class index
            print(f"Predicted class index: {predicted_class.item()}")
    else:
        print("No hands detected in the frame.")

# Specify the path of the image you want to predict
image_path_to_predict = "E:/Downloads/20240221_095053.jpg"

# Read the image from the file
image = cv2.imread(image_path_to_predict)

# Convert the image from BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect hands and make predictions
predict_hand_sign(image_rgb, model, hands_detector)
