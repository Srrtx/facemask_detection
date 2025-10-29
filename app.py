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
    def __init__(self, model_path='moblienet.h5'):
        """Initialize the mask detector with optimizations"""
        print("Loading model...")
        
        # Optimize TensorFlow for speed
        tf.config.optimizer.set_jit(True)
        
        # Load model with custom objects if needed
        try:
            self.model = load_model(model_path, compile=False)
            # Recompile model with proper metrics
            self.model.compile(optimizer='adam', 
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])
            print("Model loaded and compiled successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Initialize OpenCV face detection with optimized parameters
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.face_cascade.empty():
            raise ValueError("Error loading face cascade classifier")
        
        # Class labels and confidence thresholds
        self.class_labels = ['with_mask', 'without_mask', 'incorrect_mask']
        self.class_colors = {
            'with_mask': (0, 255, 0),      # Green
            'without_mask': (0, 0, 255),   # Red
            'incorrect_mask': (0, 165, 255) # Orange
        }
        self.confidence_threshold = 0.7  # Minimum confidence for valid detection
        
        # Performance tracking
        self.last_detection_time = 0
        self.detection_count = 0
        
        # Image preprocessing parameters
        self.target_size = (224, 224)  # MobileNetV2 input size
        self.face_padding = 0.2  # Add 20% padding around detected faces
        
    def add_padding_to_face(self, image, x, y, w, h, padding_percent=0.2):
        """Add padding around the face region"""
        height, width = image.shape[:2]
        pad_w = int(w * padding_percent)
        pad_h = int(h * padding_percent)
        
        # Calculate new coordinates with padding
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(width, x + w + pad_w)
        y2 = min(height, y + h + pad_h)
        
        return x1, y1, x2-x1, y2-y1

    def preprocess_face(self, face_image):
        """Preprocess face image for model prediction"""
        if face_image is None or face_image.size == 0:
            raise ValueError("Invalid face image")
            
        # Convert to RGB if needed
        if len(face_image.shape) == 2:  # Grayscale
            face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
        elif face_image.shape[2] == 4:  # RGBA
            face_image = cv2.cvtColor(face_image, cv2.COLOR_RGBA2RGB)
            
        # Apply preprocessing
        face_resized = cv2.resize(face_image, self.target_size)
        face_normalized = face_resized.astype('float32') / 255.0
        face_batch = np.expand_dims(face_normalized, axis=0)
        
        return face_batch
    
    def predict_mask(self, face_image):
        """Predict mask status for a face image"""
        try:
            processed_face = self.preprocess_face(face_image)
            predictions = self.model.predict(processed_face, verbose=0)
            
            # Get probabilities for each class
            class_probs = predictions[0]
            predicted_class_idx = np.argmax(class_probs)
            confidence = float(class_probs[predicted_class_idx])
            
            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                return 'unknown', confidence
                
            predicted_class = self.class_labels[predicted_class_idx]
            
            # Additional validation for "with_mask" prediction
            if predicted_class == 'with_mask':
                # Check if "without_mask" probability is too close
                without_mask_prob = float(class_probs[self.class_labels.index('without_mask')])
                if confidence - without_mask_prob < 0.15:  # If difference is less than 15%
                    return 'unknown', confidence
                    
            return predicted_class, confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return 'unknown', 0.0
    
    def enhance_image(self, frame):
        """Enhance image quality for better detection"""
        # Apply mild brightness and contrast normalization
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        return enhanced

    def process_frame(self, frame):
        """Process frame and detect masks with optimizations"""
        try:
            start_time = time.time()
            
            if frame is None or frame.size == 0:
                raise ValueError("Invalid frame")
            
            # Enhance image quality
            frame = self.enhance_image(frame)
            
            # Resize frame for faster processing (maintain aspect ratio)
            original_height, original_width = frame.shape[:2]
            if original_width > 640:
                scale = 640 / original_width
                new_width = 640
                new_height = int(original_height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Convert to grayscale and apply histogram equalization for better face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            # Optimized face detection parameters
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1,      # More accurate than 1.2
                minNeighbors=5,       # Increased for better accuracy
                minSize=(30, 30),     # Smaller minimum face size
                maxSize=(300, 300),   # Maximum face size
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            detections = []
            
            if len(faces) > 0:
                # Sort faces by area (largest first)
                faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
                
                # Process faces (up to 4 faces for better multi-person detection)
                for (x, y, w, h) in faces[:4]:
                    # Add padding around face
                    x_pad, y_pad, w_pad, h_pad = self.add_padding_to_face(frame, x, y, w, h, self.face_padding)
                    
                    # Extract face region with padding
                    face_region = frame[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
                    
                    if face_region.size > 0:
                        # Predict mask status
                        predicted_class, confidence = self.predict_mask(face_region)
                        
                        # Scale coordinates back to original size if needed
                        if original_width > 640:
                            scale_factor = original_width / 640
                            x_pad = int(x_pad * scale_factor)
                            y_pad = int(y_pad * scale_factor)
                            w_pad = int(w_pad * scale_factor)
                            h_pad = int(h_pad * scale_factor)
                        
                        # Only include detection if confidence meets threshold
                        if predicted_class != 'unknown' and confidence >= self.confidence_threshold:
                            detections.append({
                                'x': int(x_pad),
                                'y': int(y_pad),
                                'w': int(w_pad),
                                'h': int(h_pad),
                                'class': predicted_class,
                                'confidence': float(confidence),  # Ensure JSON serializable
                                'color': self.class_colors.get(predicted_class, (128, 128, 128))  # Default gray for unknown
                            })
            
            # Performance tracking
            detection_time = time.time() - start_time
            self.detection_count += 1
            self.last_detection_time = detection_time
            
            if self.detection_count % 10 == 0:
                fps = 1/detection_time if detection_time > 0 else 0
                print(f"Detection speed: {detection_time:.3f}s, FPS: {fps:.1f}")
            
            return detections
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            return []

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
        if not data or 'image' not in data:
            return jsonify({
                'status': 'error',
                'message': 'No image data provided'
            }), 400
            
        image_data = data['image']
        
        # Validate base64 image
        try:
            # Check if image data is properly formatted
            if ',' in image_data:
                # Remove data URL prefix if present
                image_data = image_data.split(',')[1]
            
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None or frame.size == 0:
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid image format or empty image'
                }), 400
                
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Image decoding error: {str(e)}'
            }), 400
        
        # Check image dimensions
        height, width = frame.shape[:2]
        if width < 32 or height < 32:
            return jsonify({
                'status': 'error',
                'message': 'Image is too small. Minimum size is 32x32 pixels.'
            }), 400
        
        # Process frame
        detections = detector.process_frame(frame)
        
        # Prepare response with additional metadata
        response = {
            'status': 'success',
            'detections': detections,
            'metadata': {
                'processed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'image_size': {
                    'width': width,
                    'height': height
                },
                'detection_count': len(detections),
                'processing_time': detector.last_detection_time
            }
        }
        
        # Add confidence scores for debugging if needed
        if 'debug' in data and data['debug']:
            response['debug'] = {
                'detector_count': detector.detection_count,
                'avg_processing_time': detector.last_detection_time
            }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in detect route: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Internal server error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
