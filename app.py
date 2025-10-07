# from flask import Flask, render_template, request, jsonify
# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import base64
# import io
# from PIL import Image
# import time

# app = Flask(__name__)

# class MaskDetector:
#     def __init__(self, model_path='face_mask.h5'):
#         """Initialize the mask detector with optimizations"""
#         print("Loading model...")
        
#         # Optimize TensorFlow for speed
#         tf.config.optimizer.set_jit(True)
        
#         self.model = load_model(model_path)
#         print("Model loaded successfully!")
        
#         # Initialize OpenCV face detection with optimized parameters
#         self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
#         # Class labels
#         self.class_labels = ['with_mask', 'without_mask', 'incorrect_mask']
#         self.class_colors = {
#             'with_mask': (0, 255, 0),      # Green
#             'without_mask': (0, 0, 255),   # Red
#             'incorrect_mask': (0, 165, 255) # Orange
#         }
        
#         # Performance tracking
#         self.last_detection_time = 0
#         self.detection_count = 0
        
#     def preprocess_face(self, face_image):
#         """Preprocess face image for model prediction"""
#         face_resized = cv2.resize(face_image, (224, 224))
#         face_normalized = face_resized.astype('float32') / 255.0
#         face_batch = np.expand_dims(face_normalized, axis=0)
#         return face_batch
    
#     def predict_mask(self, face_image):
#         """Predict mask status for a face image"""
#         processed_face = self.preprocess_face(face_image)
#         predictions = self.model.predict(processed_face, verbose=0)
#         predicted_class_idx = np.argmax(predictions[0])
#         confidence = float(predictions[0][predicted_class_idx])
#         predicted_class = self.class_labels[predicted_class_idx]
#         return predicted_class, confidence
    
#     def process_frame(self, frame):
#         """Process frame and detect masks with optimizations"""
#         start_time = time.time()
        
#         # Resize frame for faster processing (maintain aspect ratio)
#         height, width = frame.shape[:2]
#         if width > 640:
#             scale = 640 / width
#             new_width = 640
#             new_height = int(height * scale)
#             frame = cv2.resize(frame, (new_width, new_height))
        
#         # Convert to grayscale for face detection
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         # Optimized face detection parameters for speed
#         faces = self.face_cascade.detectMultiScale(
#             gray, 
#             scaleFactor=1.2,  # Slightly faster than 1.1
#             minNeighbors=3,   # Reduced from 4 for speed
#             minSize=(50, 50), # Minimum face size
#             maxSize=(300, 300) # Maximum face size
#         )
        
#         detections = []
        
#         # Process only the largest face for speed (or up to 2 faces)
#         if len(faces) > 0:
#             # Sort faces by area (largest first)
#             faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            
#             # Process only the first 2 faces for speed
#             for (x, y, w, h) in faces[:2]:
#                 # Extract face region
#                 face_region = frame[y:y+h, x:x+w]
                
#                 if face_region.size > 0:
#                     # Predict mask status
#                     predicted_class, confidence = self.predict_mask(face_region)
                    
#                     # Scale coordinates back to original size if needed
#                     if width > 640:
#                         scale_factor = width / 640
#                         x = int(x * scale_factor)
#                         y = int(y * scale_factor)
#                         w = int(w * scale_factor)
#                         h = int(h * scale_factor)
                    
#                     detections.append({
#                         'x': int(x),
#                         'y': int(y),
#                         'w': int(w),
#                         'h': int(h),
#                         'class': predicted_class,
#                         'confidence': confidence,
#                         'color': self.class_colors[predicted_class]
#                     })
        
#         # Performance tracking
#         detection_time = time.time() - start_time
#         self.detection_count += 1
#         self.last_detection_time = detection_time
        
#         if self.detection_count % 10 == 0:
#             print(f"Detection speed: {detection_time:.3f}s, FPS: {1/detection_time:.1f}")
        
#         return detections

# # Global detector instance
# detector = MaskDetector()

# @app.route('/')
# def index():
#     """Main page"""
#     return render_template('index.html')

# @app.route('/detect', methods=['POST'])
# def detect():
#     """Detect masks in uploaded frame"""
#     try:
#         # Get image data from request
#         data = request.json
#         image_data = data['image']
        
#         # Decode base64 image
#         image_bytes = base64.b64decode(image_data.split(',')[1])
#         nparr = np.frombuffer(image_bytes, np.uint8)
#         frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
#         if frame is None:
#             return jsonify({'status': 'error', 'message': 'Invalid image'})
        
#         # Process frame
#         detections = detector.process_frame(frame)
        
#         return jsonify({
#             'status': 'success',
#             'detections': detections
#         })
        
#     except Exception as e:
#         print(f"Error: {e}")
#         return jsonify({'status': 'error', 'message': str(e)})

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64
import io
from PIL import Image
import time

app = Flask(__name__)

class MaskDetector:
    def __init__(self, model_path='TM80keras_model.h5', labels_path='labels.txt'):
        """Initialize the mask detector with Teachable Machine model"""
        print("Loading Teachable Machine model...")
        
        # Optimize TensorFlow for speed
        tf.config.optimizer.set_jit(True)
        
        # Load Teachable Machine model (no compile needed)
        self.model = load_model(model_path, compile=False)
        print("Model loaded successfully!")
        
        # Load labels from labels.txt file
        try:
            with open(labels_path, 'r') as f:
                self.class_labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]
            print(f"Loaded labels: {self.class_labels}")
        except FileNotFoundError:
            print("Warning: labels.txt not found, using default labels")
            self.class_labels = ['with_mask', 'without_mask', 'incorrect_mask']
        
        # Initialize OpenCV face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Define colors for each class
        self.class_colors = {
            'with_mask': (0, 255, 0),      # Green
            'without_mask': (0, 0, 255),   # Red
            'incorrect_mask': (0, 165, 255) # Orange
        }
        
        # Performance tracking
        self.last_detection_time = 0
        self.detection_count = 0
        
    def preprocess_face(self, face_image):
        """Preprocess face image for Teachable Machine model prediction"""
        # Teachable Machine uses 224x224 input size
        face_resized = cv2.resize(face_image, (224, 224))
        
        # Convert BGR to RGB (OpenCV uses BGR, model expects RGB)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [-1, 1] range (Teachable Machine normalization)
        face_normalized = (face_rgb.astype('float32') / 127.5) - 1.0
        
        # Add batch dimension
        face_batch = np.expand_dims(face_normalized, axis=0)
        return face_batch
    
    def predict_mask(self, face_image):
        """Predict mask status for a face image"""
        processed_face = self.preprocess_face(face_image)
        predictions = self.model.predict(processed_face, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = self.class_labels[predicted_class_idx]
        return predicted_class, confidence
    
    def process_frame(self, frame):
        """Process frame and detect masks"""
        start_time = time.time()
        
        # Resize frame for faster processing
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = 640
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.2,
            minNeighbors=3,
            minSize=(50, 50),
            maxSize=(300, 300)
        )
        
        detections = []
        
        # Process detected faces
        if len(faces) > 0:
            # Sort faces by area (largest first)
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            
            # Process up to 2 faces for speed
            for (x, y, w, h) in faces[:2]:
                # Extract face region
                face_region = frame[y:y+h, x:x+w]
                
                if face_region.size > 0:
                    # Predict mask status
                    predicted_class, confidence = self.predict_mask(face_region)
                    
                    # Scale coordinates back if frame was resized
                    if width > 640:
                        scale_factor = width / 640
                        x = int(x * scale_factor)
                        y = int(y * scale_factor)
                        w = int(w * scale_factor)
                        h = int(h * scale_factor)
                    
                    # Get color for this class (with fallback)
                    color = self.class_colors.get(predicted_class, (255, 255, 255))
                    
                    detections.append({
                        'x': int(x),
                        'y': int(y),
                        'w': int(w),
                        'h': int(h),
                        'class': predicted_class,
                        'confidence': confidence,
                        'color': color
                    })
        
        # Performance tracking
        detection_time = time.time() - start_time
        self.detection_count += 1
        self.last_detection_time = detection_time
        
        if self.detection_count % 10 == 0:
            print(f"Detection speed: {detection_time:.3f}s, FPS: {1/detection_time:.1f}")
        
        return detections

# Global detector instance
detector = MaskDetector()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Detect masks in uploaded frame"""
    try:
        # Get image data from request
        data = request.json
        image_data = data['image']
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'status': 'error', 'message': 'Invalid image'})
        
        # Process frame
        detections = detector.process_frame(frame)
        
        return jsonify({
            'status': 'success',
            'detections': detections
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)