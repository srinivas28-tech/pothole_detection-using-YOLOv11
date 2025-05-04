import cv2
import math
import cvzone
from ultralytics import YOLO


# Load YOLO model with custom weights
yolo_model = YOLO("Weights/best.pt")

# Define class names
class_labels = ['Pothole']

# Load the image
image_path = "Media/pothole_3.jpeg"
img = cv2.imread(image_path)

# Perform object detection
results = yolo_model(img)

# Loop through the detections and draw bounding boxes
for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        w, h = x2 - x1, y2 - y1
        
        conf = math.ceil((box.conf[0] * 100)) / 100
        cls = int(box.cls[0])

        if conf > 0.3:
            cvzone.cornerRect(img, (x1, y1, w, h), t=2)
            cvzone.putTextRect(img, f'{class_labels[cls]} {conf}', (x1, y1 - 10), scale=0.8, thickness=1, colorR=(255, 0, 0))

# Display the image with detections
cv2.imshow("Image", img)

# Close window when 'q' button is pressed
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cv2.waitKey(1)
'''
import matplotlib.pyplot as plt
import cv2
import math
import cvzone
from ultralytics import YOLO

# Load YOLO model with custom weights
yolo_model = YOLO("Weights/best.pt")

# Define class names
class_labels = ['Pothole']

# Load the image
image_path = "Media/pothole_3.jpeg"
img = cv2.imread(image_path)

# Perform object detection
results = yolo_model(img)

# Loop through the detections and draw bounding boxes
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers
        w, h = x2 - x1, y2 - y1
        conf = round(box.conf[0].item(), 2)  # Convert to float and round

        cls = int(box.cls[0])
        if conf > 0.3:
            cvzone.cornerRect(img, (x1, y1, w, h), t=2)
            cvzone.putTextRect(img, f'{class_labels[cls]} {conf}', 
                               (x1, y1 - 10), scale=0.8, thickness=1, colorR=(255, 0, 0))

# Convert the modified image to RGB for display
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the detected image
plt.imshow(img)
plt.axis("off")  # Hide axes
plt.title("Pothole Detection")
plt.show()

cv2.imwrite("output.jpg", img)  # Save output
print("Image saved as output.jpg")  # Open manually

'''