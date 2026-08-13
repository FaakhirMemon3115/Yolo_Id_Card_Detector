import os
import uuid
import cv2
import numpy as np
import time 
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, Response
from ultralytics import YOLO
import pytesseract 
from datetime import datetime
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc
from twilio.rest import Client
from io import BytesIO

# ---------- (Windows only) set Tesseract path if needed ----------
# If Windows & Tesseract not in PATH, uncomment and set the path:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)
app.secret_key = "3UiDUzFW4fXZUqTmzIxAhaOdyA9rr6cF"

# ---------- Database Configuration ----------
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///id_detection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ---------- Twilio Configuration (for SMS alerts) ----------
# Uncomment and add your Twilio credentials if you want SMS alerts
# app.config['TWILIO_ACCOUNT_SID'] = 'your_account_sid'
# app.config['TWILIO_AUTH_TOKEN'] = 'your_auth_token'
# app.config['TWILIO_PHONE_NUMBER'] = 'your_twilio_phone_number'
# app.config['ADMIN_PHONE_NUMBER'] = 'your_admin_phone_number'

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/results'
CROPPED_IDS_FOLDER = 'static/cropped_ids'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['CROPPED_IDS_FOLDER'] = CROPPED_IDS_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CROPPED_IDS_FOLDER, exist_ok=True)

# ---------- Load YOLO model ----------
MODEL_PATH = "best.pt"   # apna trained model ka path
model = YOLO(MODEL_PATH)

# ---------- Database Models ----------
class DetectionRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    image_path = db.Column(db.String(200))
    result_image_path = db.Column(db.String(200))
    cropped_id_path = db.Column(db.String(200))
    id_detected = db.Column(db.Boolean)
    detection_confidence = db.Column(db.Float)
    extracted_text = db.Column(db.Text)
    reason = db.Column(db.String(100))

class DetectionEvent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    count = db.Column(db.Integer)
    
# Create tables
with app.app_context():
    db.create_all()

# Helper: file type check
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper: OCR pre-processing
def preprocess_for_ocr(bgr_img):
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    scale = 2 if max(h, w) < 800 else 1
    if scale > 1:
        gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 5, 55, 55)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

# Helper: Run Tesseract OCR
def run_ocr_on_crop(bgr_crop):
    proc = preprocess_for_ocr(bgr_crop)
    config = "--oem 3 --psm 6"
    text = pytesseract.image_to_string(proc, lang="eng", config=config)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)

# Helper: Send SMS alert
def send_sms_alert(extracted_text):
    try:
        # Uncomment and configure if you want SMS alerts
        # client = Client(app.config['TWILIO_ACCOUNT_SID'], app.config['TWILIO_AUTH_TOKEN'])
        # message = client.messages.create(
        #     body=f"New ID card detected: {extracted_text[:50]}...",
        #     from_=app.config['TWILIO_PHONE_NUMBER'],
        #     to=app.config['ADMIN_PHONE_NUMBER']
        # )
        print(f"SMS would be sent for: {extracted_text[:50]}...")
        return True
    except Exception as e:
        print(f"SMS error: {str(e)}")
        return False

# ---------- Webcam Error Frame Functions ----------
def generate_error_frame():
    """Create an error frame when camera is not available"""
    # Create an error image
    img = np.zeros((300, 500, 3), dtype=np.uint8)
    cv2.putText(img, "Camera Not Available", (50, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    ret, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()

def generate_error_frame_bytes():
    """Create a complete frame with error message"""
    error_frame = generate_error_frame()
    return b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n'

# ---------- Webcam Routes ----------
@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/video_feed')
def video_feed():
    # Webcam URL - replace with your IP webcam URL
    # webcam_url = "http://192.168.1.63:4747/video"
    webcam_url = "http://10.128.207.215:4747/video"
    try:
        return Response(generate_frames(webcam_url), 
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error in video feed: {str(e)}")
        # Return a static error image
        return Response(generate_error_frame_bytes(), 
                       mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames(webcam_url):
    cap = cv2.VideoCapture(webcam_url)
    
    # Set timeout for camera connection (5 seconds)
    start_time = time.time()
    while not cap.isOpened():
        if time.time() - start_time > 5:
            print("Camera connection timeout")
            yield generate_error_frame_bytes()
            return
        time.sleep(0.1)
    
    print("Camera connected successfully")
    
    while True:
        try:
            success, frame = cap.read()
            if not success:
                print("Failed to read frame from webcam")
                # Try to reopen the camera
                cap.release()
                cap = cv2.VideoCapture(webcam_url)
                time.sleep(0.1)
                continue
            
            # Process frame with YOLO model (lightweight processing)
            results = model.predict(frame, conf=0.3, verbose=False, imgsz=320)
            
            for result in results:
                annotated_frame = result.plot()
            
            # Encode the frame
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                print("Failed to encode frame")
                continue
                
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
        except Exception as e:
            print(f"Error in frame generation: {str(e)}")
            yield generate_error_frame_bytes()
            break
    
    cap.release()

@app.route('/capture', methods=['POST'])
def capture():
    try:
        # webcam_url = "http://192.168.1.63:4747/video"
        webcam_url = "http://10.128.207.215:4747/video"
        cap = cv2.VideoCapture(webcam_url)
        
        # Wait a moment for camera to initialize
        time.sleep(0.5)
        
        if not cap.isOpened():
            flash('Failed to connect to webcam. Please check the URL and connection.', 'error')
            return redirect(url_for('webcam'))
        
        success, frame = cap.read()
        cap.release()
        
        if not success or frame is None:
            flash('Failed to capture image from webcam. Please try again.', 'error')
            return redirect(url_for('webcam'))
        
        # Save the captured image
        unique_id = str(uuid.uuid4())[:8]
        filename = f"webcam_capture_{unique_id}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(filepath, frame)
        
        # Process the image with your existing detection code
        # Create result image
        result_filename = f"webcam_result_{unique_id}.jpg"
        result_path = os.path.join(app.config['OUTPUT_FOLDER'], result_filename)
        
        # Run YOLO detection with higher confidence for capture
        detection_results = model.predict(frame, conf=0.5, verbose=False)
        
        for result in detection_results:
            im_array = result.plot()
            im = Image.fromarray(im_array[..., ::-1])
            im.save(result_path)
            break
        
        # Extract information from detection
        id_detected = False
        detection_confidence = 0.0
        cropped_id_paths = []
        class_names = model.names
        
        for result in detection_results:
            for box in result.boxes:
                cls_id = int(box.cls)
                class_name = class_names[cls_id]
                conf = float(box.conf)
                
                if class_name.lower() in ["with_id", "id_card", "card", "withid"]:
                    id_detected = id_detected or (conf > 0.3)
                    detection_confidence = max(detection_confidence, conf)
                    
                    # Crop and save the ID card
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy.tolist()
                    crop = frame[y1:y2, x1:x2]
                    
                    if crop.size > 0:
                        cropped_id_filename = f"cropped_id_{unique_id}_{len(cropped_id_paths)}.jpg"
                        cropped_id_path = os.path.join(app.config['CROPPED_IDS_FOLDER'], cropped_id_filename)
                        cv2.imwrite(cropped_id_path, crop)
                        cropped_id_paths.append(cropped_id_path)
        
        # Save to database
        record = DetectionRecord(
            image_path=filepath,
            result_image_path=result_path,
            cropped_id_path=",".join(cropped_id_paths) if cropped_id_paths else None,
            id_detected=id_detected,
            detection_confidence=detection_confidence,
            extracted_text="Extracted text would go here"  # Add your OCR code here
        )
        db.session.add(record)
        db.session.commit()
        
        # Redirect to results page
        return redirect(url_for('show_result', record_id=record.id))
            
    except Exception as e:
        flash(f'Error capturing image: {str(e)}', 'error')
        return redirect(url_for('webcam'))

@app.route('/result/<int:record_id>')
def show_result(record_id):
    record = DetectionRecord.query.get_or_404(record_id)
    cropped_images = record.cropped_id_path.split(',') if record.cropped_id_path else []
    
    return render_template(
        'result.html',
        id_detected=record.id_detected,
        detection_confidence=record.detection_confidence,
        original_image=record.image_path.replace('\\', '/'),
        result_image=record.result_image_path.replace('\\', '/'),
        cropped_images=[path.replace('\\', '/') for path in cropped_images],
        extracted_text=record.extracted_text,
        record_id=record.id
    )

# ---------- Main Application Routes ----------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)

        files = request.files.getlist('file')
        if not files or all(file.filename == '' for file in files):
            flash('No files selected', 'error')
            return redirect(request.url)

        results = []
        for file in files:
            if file.filename == '':
                continue
                
            if not allowed_file(file.filename):
                flash(f'Invalid file type: {file.filename}. Only JPG, JPEG, PNG allowed', 'error')
                continue

            try:
                file_bytes = file.read()
                np_img = np.frombuffer(file_bytes, np.uint8)
                img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

                # Save original uploaded image
                unique_id = str(uuid.uuid4())[:8]
                upload_filename = f"upload_{unique_id}.jpg"
                upload_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_filename)
                cv2.imwrite(upload_path, img)

                # Processed result image path
                result_filename = f"result_{unique_id}.jpg"
                result_path = os.path.join(app.config['OUTPUT_FOLDER'], result_filename)

                # Run YOLO detection
                detection_results = model.predict(img, conf=0.3, verbose=False)
                
                # For saving cropped ID cards
                cropped_id_paths = []
                
                for result in detection_results:
                    im_array = result.plot()
                    im = Image.fromarray(im_array[..., ::-1])  # BGR->RGB fix
                    im.save(result_path)
                    break

                id_detected = False
                detection_confidence = 0.0
                extracted_text = ""

                id_card_boxes = []
                id_text_boxes = []
                class_names = model.names

                for result in detection_results:
                    for box in result.boxes:
                        cls_id = int(box.cls)
                        class_name = class_names.get(cls_id, str(cls_id)) if isinstance(class_names, dict) else class_names[cls_id]
                        conf = float(box.conf)
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = xyxy.tolist()

                        if class_name.lower() in ["with_id", "id_card", "card", "withid"]:
                            id_detected = id_detected or (conf > 0.3)
                            detection_confidence = max(detection_confidence, conf)
                            id_card_boxes.append((conf, (x1, y1, x2, y2)))
                            
                            # Crop and save the ID card
                            crop = img[y1:y2, x1:x2]
                            if crop.size > 0:
                                cropped_id_filename = f"cropped_id_{unique_id}_{len(cropped_id_paths)}.jpg"
                                cropped_id_path = os.path.join(app.config['CROPPED_IDS_FOLDER'], cropped_id_filename)
                                cv2.imwrite(cropped_id_path, crop)
                                cropped_id_paths.append(cropped_id_path)

                        if class_name.lower() in ["id_text", "text_area", "text", "card_text"]:
                            id_text_boxes.append((conf, (x1, y1, x2, y2)))

                texts = []
                if id_text_boxes:
                    id_text_boxes.sort(key=lambda t: t[0], reverse=True)
                    for conf, (x1, y1, x2, y2) in id_text_boxes[:5]:
                        pad = 4
                        x1p = max(0, x1 - pad)
                        y1p = max(0, y1 - pad)
                        x2p = min(img.shape[1], x2 + pad)
                        y2p = min(img.shape[0], y2 + pad)
                        crop = img[y1p:y2p, x1p:x2p]
                        if crop.size > 0:
                            txt = run_ocr_on_crop(crop)
                            if txt:
                                texts.append(txt)
                elif id_card_boxes:
                    id_card_boxes.sort(key=lambda t: t[0], reverse=True)
                    _, (x1, y1, x2, y2) = id_card_boxes[0]
                    pad = 6
                    x1p = max(0, x1 - pad)
                    y1p = max(0, y1 - pad)
                    x2p = min(img.shape[1], x2 + pad)
                    y2p = min(img.shape[0], y2 + pad)
                    crop = img[y1p:y2p, x1p:x2p]
                    if crop.size > 0:
                        txt = run_ocr_on_crop(crop)
                        if txt:
                            texts.append(txt)

                if texts:
                    merged = []
                    seen = set()
                    for block in texts:
                        for line in block.splitlines():
                            if line not in seen:
                                seen.add(line)
                                merged.append(line)
                    extracted_text = "\n".join(merged)
                    
                    # Send SMS alert if ID detected with text
                    if id_detected and extracted_text:
                        send_sms_alert(extracted_text)

                # Save to database
                record = DetectionRecord(
                    image_path=upload_path,
                    result_image_path=result_path,
                    cropped_id_path=",".join(cropped_id_paths) if cropped_id_paths else None,
                    id_detected=id_detected,
                    detection_confidence=detection_confidence,
                    extracted_text=extracted_text
                )
                db.session.add(record)
                db.session.commit()
                
                # Update detection events count
                today = datetime.utcnow().date()
                event = DetectionEvent.query.filter(
                    db.func.date(DetectionEvent.timestamp) == today
                ).first()
                
                if event:
                    event.count += 1
                else:
                    event = DetectionEvent(count=1)
                    db.session.add(event)
                
                db.session.commit()

                results.append({
                    'id_detected': id_detected,
                    'detection_confidence': detection_confidence,
                    'original_image': upload_path.replace('\\', '/'),
                    'result_image': result_path.replace('\\', '/'),
                    'cropped_images': [path.replace('\\', '/') for path in cropped_id_paths],
                    'extracted_text': extracted_text,
                    'record_id': record.id
                })

            except Exception as e:
                flash(f'Error processing image {file.filename}: {str(e)}', 'error')
                continue

        if not results:
            flash('No images were processed successfully', 'error')
            return redirect(request.url)
            
        # If only one image, show detailed view
        if len(results) == 1:
            result = results[0]
            return render_template(
                'index.html',
                id_detected=result['id_detected'],
                detection_confidence=result['detection_confidence'],
                original_image=result['original_image'],
                result_image=result['result_image'],
                cropped_images=result['cropped_images'],
                extracted_text=result['extracted_text'],
                record_id=result['record_id'],
                multiple_results=None
            )
        else:
            # For multiple images, show summary
            return render_template(
                'index.html',
                multiple_results=results,
                id_detected=None
            )

    # Get stats for dashboard
    total_detections = DetectionRecord.query.count()
    today_detections = DetectionRecord.query.filter(
        db.func.date(DetectionRecord.timestamp) == datetime.utcnow().date()
    ).count()
    
    recent_detections = DetectionRecord.query.order_by(desc(DetectionRecord.timestamp)).limit(5).all()
    
    return render_template(
        'index.html', 
        total_detections=total_detections,
        today_detections=today_detections,
        recent_detections=recent_detections
    )

@app.route('/submit_reason', methods=['POST'])
def submit_reason():
    try:
        reason = request.form.get('reason')
        record_id = request.form.get('record_id')
        
        if not reason:
            flash('Please select a reason', 'error')
            return redirect(url_for('index'))
            
        if record_id:
            record = DetectionRecord.query.get(record_id)
            if record:
                record.reason = reason
                db.session.commit()
                
        flash(f'Reason submitted: {reason}. Thank you!', 'success')
        return redirect(url_for('index'))

    except Exception as e:
        flash(f'Error submitting reason: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    # Get detection statistics
    total_detections = DetectionRecord.query.count()
    today_detections = DetectionRecord.query.filter(
        db.func.date(DetectionRecord.timestamp) == datetime.utcnow().date()
    ).count()
    
    # Get detections with IDs
    id_detections = DetectionRecord.query.filter_by(id_detected=True).count()
    
    # Get daily counts for chart
    daily_counts = db.session.query(
        db.func.date(DetectionRecord.timestamp).label('date'),
        db.func.count(DetectionRecord.id).label('count')
    ).group_by(db.func.date(DetectionRecord.timestamp)).order_by(db.func.date(DetectionRecord.timestamp)).all()
    
    # Get recent detections
    recent_detections = DetectionRecord.query.order_by(desc(DetectionRecord.timestamp)).limit(10).all()
    
    return render_template(
        'dashboard.html',
        total_detections=total_detections,
        today_detections=today_detections,
        id_detections=id_detections,
        daily_counts=daily_counts,
        recent_detections=recent_detections
    )

@app.route('/search')
def search():
    query = request.args.get('q', '')
    date_filter = request.args.get('date', '')
    
    # Build query
    search_query = DetectionRecord.query
    
    if query:
        search_query = search_query.filter(DetectionRecord.extracted_text.ilike(f'%{query}%'))
    
    if date_filter:
        search_query = search_query.filter(db.func.date(DetectionRecord.timestamp) == date_filter)
    
    results = search_query.order_by(desc(DetectionRecord.timestamp)).all()
    
    return render_template('search.html', results=results, query=query, date_filter=date_filter)

@app.route('/export')
def export_data():
    # Get filter parameters
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    
    # Build query
    query = DetectionRecord.query
    
    if start_date:
        query = query.filter(DetectionRecord.timestamp >= start_date)
    
    if end_date:
        query = query.filter(DetectionRecord.timestamp <= end_date)
    
    records = query.all()
    
    # Convert to DataFrame
    data = []
    for record in records:
        data.append({
            'ID': record.id,
            'Timestamp': record.timestamp,
            'ID Detected': 'Yes' if record.id_detected else 'No',
            'Confidence': f"{record.detection_confidence * 100:.1f}%" if record.detection_confidence else 'N/A',
            'Extracted Text': record.extracted_text,
            'Reason': record.reason or 'N/A'
        })
    
    df = pd.DataFrame(data)
    
    # Create Excel file in memory
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='ID Detections', index=False)
    
    output.seek(0)
    
    # Send file
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'id_detections_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
    )

if __name__ == '__main__':
    app.run(debug=True)