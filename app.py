import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


model = torch.load('mobilenet_gender_fullmodel.pth', map_location=device, weights_only=False)
model = model.to(device)
model.eval()

batch_size = 32
image_size = 224

gender_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class_names = ['men', 'women']

faceXML = cv2.CascadeClassifier('files/haarcascade_frontalface_default.xml')
smileXML = cv2.CascadeClassifier('files/haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = faceXML.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 86), 2)

        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        try:
            face_img_rgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
            
            input_tensor = gender_transform(face_img_rgb)
            input_batch = input_tensor.unsqueeze(0).to(device)
        
            with torch.no_grad():
                output = model(input_batch)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                _, predicted_class_idx = torch.max(probabilities, 1)
                predicted_gender = class_names[predicted_class_idx.item()]
                confidence = probabilities[0][predicted_class_idx].item() * 100

            gender_text = f"{predicted_gender} ({confidence:.2f}%)"
            cv2.putText(frame, gender_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 86), 2)

        except Exception as e:
            cv2.putText(frame, "Error in Gender", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            print(f"Error processing gender: {e}")

        
        smiles = smileXML.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25))
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (52, 166, 83), 2)

    cv2.imshow('Face and Gender Detection', frame)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()