"""
Flask web server for Yu-Gi-Oh! Card Recognition
Provides REST API and web interface
"""
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
import cv2
import numpy as np
import base64
from card_detector import CardDetector, CardDatabase, CardRecognizer
from card_recognizer_cnn import CNNCardRecognizer
import json
from datetime import datetime
import atexit
import gc
import time
import os
import urllib.request
import urllib.parse
import json

app = Flask(__name__)
CORS(app)

# Configure for Apache proxy with /ygo-vision base path
# This middleware helps Flask understand it's behind a proxy
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Set APPLICATION_ROOT if running behind proxy
# This can be overridden by environment variable
app.config['APPLICATION_ROOT'] = os.environ.get('FLASK_APPLICATION_ROOT', '/')

# Initialize components
detector = CardDetector()
database = CardDatabase()
recognizer = CardRecognizer(database)

# Initialize CNN card recognizer
print("Initializing CNN Card Recognizer...")
try:
    cnn_recognizer = CNNCardRecognizer(
        model_path='models/card_recognition_subset_v2.pth',
        data_dir='data/augmented_subset_new'
    )
    print("CNN Card Recognizer ready!")
except Exception as e:
    print(f"CNN Card Recognizer failed to initialize: {e}")
    cnn_recognizer = None

# Global camera and lock
camera = None
camera_lock = None

def get_camera_lock():
    """Get or initialize camera lock"""
    global camera_lock
    if camera_lock is None:
        import threading
        camera_lock = threading.Lock()
    return camera_lock

def cleanup_resources():
    """Clean up camera and other resources"""
    global camera, cnn_recognizer, camera_lock
    print("\nCleaning up resources...")
    
    lock = get_camera_lock()
    with lock:
        if camera is not None:
            try:
                camera.release()
                print("Camera released")
            except Exception as e:
                print(f"Error releasing camera: {e}")
            finally:
                camera = None
    
    # Clean up CNN model
    if cnn_recognizer is not None:
        try:
            cnn_recognizer.cleanup()
        except Exception as e:
            print(f"✗ Error cleaning up CNN model: {e}")
    
    # Force garbage collection to clean up any lingering objects
    gc.collect()
    print("Garbage collection completed")
    print("Shutdown complete!")


# Register cleanup handler
# Note: Flask debug mode handles Ctrl+C internally, atexit will be called on normal shutdown
atexit.register(cleanup_resources)

def get_camera():
    """Get or initialize camera with thread safety"""
    global camera
    lock = get_camera_lock()
    
    with lock:
        if camera is None or not camera.isOpened():
            if camera is not None:
                try:
                    camera.release()
                except:
                    pass
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to prevent lag
        return camera

def generate_frames():
    """Generate video frames for streaming with real-time card detection"""
    cam = get_camera()
    last_card_info = None
    stable_frames = 0
    required_stable_frames = 10  # Card must be stable for 10 frames
    
    while True:
        with get_camera_lock():
            success, frame = cam.read()
        if not success:
            break
        
        # Keep original frame for detection (not flipped)
        # Create a copy for drawing
        display_frame = frame.copy()
        
        # Detect card on ORIGINAL frame (not flipped)
        contour = detector.find_card_contour(frame)
        
        if contour is not None:
            # Draw contour
            cv2.drawContours(display_frame, [contour], -1, (0, 255, 0), 3)
            
            # Extract card from ORIGINAL frame
            card_image = detector.extract_card(frame, contour)
            
            if card_image is not None:
                stable_frames += 1
                
                # If card is stable, try to recognize it
                if stable_frames >= required_stable_frames:
                    # Try to recognize card using CNN model
                    card_name = "Unknown Card"
                    card_type = ""
                    confidence = 0.0
                    
                    if cnn_recognizer is not None:
                        try:
                            card_name, confidence = cnn_recognizer.get_best_match(card_image)
                            
                            # Get card info from database
                            card_info = database.search_by_name(card_name)
                            if card_info:
                                card_type = card_info.get('type', '')
                            
                            print(f"Recognized: {card_name} ({confidence:.2%})")
                        except Exception as e:
                            import traceback
                            print(f"CNN Recognizer error: {e}")
                            traceback.print_exc()
                    
                    # Draw semi-transparent overlay
                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, (10, 10), (500, 220), (30, 30, 30), -1)
                    cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
                    
                    # Draw card info
                    y_offset = 40
                    cv2.putText(display_frame, "CARD RECOGNIZED!", (20, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    y_offset += 40
                    # Truncate long names
                    display_name = card_name[:35] + "..." if len(card_name) > 35 else card_name
                    cv2.putText(display_frame, display_name, (20, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    y_offset += 35
                    cv2.putText(display_frame, f"Type: {card_type}", (20, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    
                    y_offset += 30
                    cv2.putText(display_frame, f"Confidence: {confidence:.1%}", (20, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    
                    # Draw small preview of extracted card (NOT flipped)
                    card_preview = cv2.resize(card_image, (100, 145))
                    h, w = card_preview.shape[:2]
                    display_frame[10:10+h, 420:420+w] = card_preview
                else:
                    # Show stabilization progress
                    progress = int((stable_frames / required_stable_frames) * 100)
                    cv2.putText(display_frame, f"Stabilizing: {progress}%", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            stable_frames = 0
            # Show instruction
            cv2.putText(display_frame, "Place Yu-Gi-Oh! card in view", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw frame counter
        cv2.putText(display_frame, f"FPS: 30", (display_frame.shape[1] - 120, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


# Cache for current_card to prevent excessive processing
last_card_result = None
last_card_time = 0
CARD_CACHE_DURATION = 0.1  # Cache for 100ms

# Price cache to avoid API rate limits
price_cache = {}
PRICE_CACHE_DURATION = 3600  # Cache prices for 1 hour

def get_card_price(card_name):
    """Fetch card price from YGOProDeck API"""
    global price_cache
    
    current_time = time.time()
    
    # Check cache first
    if card_name in price_cache:
        cached_data, timestamp = price_cache[card_name]
        if current_time - timestamp < PRICE_CACHE_DURATION:
            return cached_data
    
    try:
        # URL encode the card name
        encoded_name = urllib.parse.quote(card_name)
        url = f"https://db.ygoprodeck.com/api/v7/cardinfo.php?name={encoded_name}"
        
        # Create request with user agent (good practice)
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            
            if data and 'data' in data and len(data['data']) > 0:
                card_data = data['data'][0]
                if 'card_prices' in card_data:
                    prices = card_data['card_prices'][0]
                    sets = card_data.get('card_sets', [])
                    
                    price_data = {
                        'market_prices': prices,
                        'sets': sets
                    }
                    
                    # Cache the result
                    price_cache[card_name] = (price_data, current_time)
                    return price_data
                    
    except Exception as e:
        print(f"Error fetching price for {card_name}: {e}")
        
    return None

@app.route('/current_card', methods=['GET'])
def current_card():
    """Get currently detected card info in real-time"""
    global last_card_result, last_card_time
    
    try:
        # Return cached result if recent enough (rate limiting)
        current_time = time.time()
        if last_card_result is not None and (current_time - last_card_time) < CARD_CACHE_DURATION:
            return jsonify(last_card_result)
        
        cam = get_camera()
        if cam is None or not cam.isOpened():
            return jsonify({'detected': False, 'error': 'Camera not available'}), 500
        
        with get_camera_lock():
            success, frame = cam.read()
        
        if not success or frame is None:
            return jsonify({'detected': False, 'error': 'Failed to capture frame'}), 500
        
        # Don't flip - keep card content readable
        # Detect card
        contour = detector.find_card_contour(frame)
        
        if contour is None:
            return jsonify({'detected': False})
        
        # Extract card
        card_image = detector.extract_card(frame, contour)
        
        if card_image is None:
            return jsonify({'detected': False})
        
        # Convert to base64 for response
        _, buffer = cv2.imencode('.jpg', card_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Recognize card using CNN
        card_name = 'Unknown Card'
        confidence = 0.0
        card_info = None
        
        if cnn_recognizer is not None:
            try:
                card_id_str, confidence = cnn_recognizer.get_best_match(card_image)
                
                # Convert card ID string to integer
                try:
                    card_id = int(card_id_str)
                except ValueError:
                    card_id = None
                
                # Get full card info from database using ID
                if card_id:
                    card_info = database.get_card_info(card_id)
                
                if card_info:
                    # Convert numpy types to native Python types for JSON
                    # Also convert NaN to None (null in JSON)
                    import math
                    cleaned_info = {}
                    for k, v in card_info.items():
                        # Convert numpy types
                        if hasattr(v, 'item'):
                            v = v.item()
                        # Convert NaN to None
                        if isinstance(v, float) and math.isnan(v):
                            v = None
                        cleaned_info[k] = v
                    
                    card_info = cleaned_info
                    card_info['confidence'] = confidence
                    
                    # Fetch price info
                    price_info = get_card_price(card_info['name'])
                    if price_info:
                        card_info['prices'] = price_info
                else:
                    # Card ID recognized but not in database
                    price_info = None
                    # Try to get price if we have a name from somewhere else (unlikely here but for safety)
                    
                    card_info = {
                        'name': f'Card ID: {card_id_str}',
                        'confidence': confidence,
                        'type': 'Unknown',
                        'desc': 'Card information not available in database'
                    }
            except Exception as e:
                print(f"Error recognizing card: {e}")
                import traceback
                traceback.print_exc()
                card_info = {
                    'name': 'Recognition Error',
                    'confidence': 0.0,
                    'desc': str(e)
                }
        else:
            card_info = {
                'name': 'CNN Model Not Loaded',
                'confidence': 0.0,
                'desc': 'Please check server logs'
            }
        
        # Cache the result
        result = {
            'detected': True,
            'image': img_base64,
            'card_info': card_info
        }
        last_card_result = result
        last_card_time = current_time
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in current_card: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'detected': False, 'error': str(e)}), 500



@app.route('/search', methods=['GET'])
def search():
    """Search for card by name"""
    query = request.args.get('q', '')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    result = database.search_by_name(query)
    
    if result is None:
        return jsonify({'error': 'Card not found'}), 404
    
    # Convert numpy types to native Python types for JSON serialization
    # Also convert NaN to None
    import math
    cleaned_result = {}
    for k, v in result.items():
        # Convert numpy types
        if hasattr(v, 'item'):
            v = v.item()
        # Convert NaN to None
        if isinstance(v, float) and math.isnan(v):
            v = None
        cleaned_result[k] = v
    
    # Add price info
    price_info = get_card_price(cleaned_result['name'])
    if price_info:
        cleaned_result['prices'] = price_info
    
    return jsonify(cleaned_result)

@app.route('/cards', methods=['GET'])
def get_cards():
    """Get all cards or paginated results"""
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    
    all_cards = database.get_all_cards()
    
    start = (page - 1) * per_page
    end = start + per_page
    
    cards = all_cards[start:end]
    
    # Convert numpy types
    cards = [
        {k: (v.item() if hasattr(v, 'item') else v) for k, v in card.items()}
        for card in cards
    ]
    
    return jsonify({
        'cards': cards,
        'total': len(all_cards),
        'page': page,
        'per_page': per_page,
        'total_pages': (len(all_cards) + per_page - 1) // per_page
    })

@app.route('/stats', methods=['GET'])
def stats():
    """Get database statistics"""
    all_cards = database.get_all_cards()
    
    return jsonify({
        'total_cards': len(all_cards),
        'database_loaded': database.cards_df is not None,
        'columns': list(database.cards_df.columns) if database.cards_df is not None else []
    })

@app.route('/random_card_images', methods=['GET'])
def random_card_images():
    """Get random card images for background animation"""
    import os
    import random
    
    card_images_dir = 'data/card_images'
    count = int(request.args.get('count', 20))
    
    # Get all card images
    all_images = [f for f in os.listdir(card_images_dir) if f.endswith('.jpg')]
    
    # Select random images
    selected_images = random.sample(all_images, min(count, len(all_images)))
    
    # Return image IDs (without extension)
    image_ids = [os.path.splitext(img)[0] for img in selected_images]
    
    return jsonify({
        'images': image_ids
    })

@app.route('/card_image/<image_id>')
def card_image(image_id):
    """Serve card image by ID"""
    import os
    return send_from_directory('data/card_images', f'{image_id}.jpg')

if __name__ == '__main__':
    print("Starting Yu-Gi-Oh! Card Recognition Server...")
    print("Open http://localhost:5000 in your browser")
    # Disable reloader to prevent semaphore leaks from multiple processes
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
