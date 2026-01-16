"""
Yu-Gi-Oh! Card Recognition System
Main card detector and recognizer using computer vision
"""
import cv2
import numpy as np
import pandas as pd
import os
from typing import Tuple, Optional, Dict, List
import json

class CardDetector:
    """Detect and extract card from camera feed"""
    
    def __init__(self):
        self.min_card_area = 3000  # Giảm thêm để phát hiện thẻ xa hơn
        self.aspect_ratio_range = (0.5, 1.0)  # Mở rộng hơn nữa
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for card detection - optimized for hand-held cards"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding for better edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use both Canny and adaptive threshold
        edges1 = cv2.Canny(blurred, 30, 100)
        
        # Adaptive threshold for better detection in varying lighting
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        edges2 = cv2.Canny(thresh, 30, 100)
        
        # Combine both edge detection methods
        edges = cv2.bitwise_or(edges1, edges2)
        
        # Morphological operations to close gaps
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        return edges
    
    def find_card_contour(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Find the largest rectangular contour (card) - optimized for hand-held"""
        edges = self.preprocess_frame(frame)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest contour that matches card characteristics
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_card_area:
                continue
            
            # Approximate contour to polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Accept both 4-sided (perfect) and 5-6 sided (slightly imperfect)
            if 4 <= len(approx) <= 6:
                # Check aspect ratio
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                
                # More lenient aspect ratio check
                if self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]:
                    # If 5-6 sides, use the 4 corners from bounding rect
                    if len(approx) > 4:
                        approx = np.array([
                            [[x, y]],
                            [[x + w, y]],
                            [[x + w, y + h]],
                            [[x, y + h]]
                        ], dtype=np.int32)
                    
                    valid_contours.append((area, approx))
        
        if not valid_contours:
            return None
        
        # Return largest valid contour
        valid_contours.sort(key=lambda x: x[0], reverse=True)
        return valid_contours[0][1]
    
    def extract_card(self, frame: np.ndarray, contour: np.ndarray) -> Optional[np.ndarray]:
        """Extract and warp card to standard size"""
        if contour is None:
            return None
        
        # Get corner points
        pts = contour.reshape(4, 2)
        rect = self.order_points(pts)
        
        # Standard Yu-Gi-Oh card size (maintaining aspect ratio)
        width, height = 421, 614  # Approximate pixel dimensions
        
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        # Perspective transform
        M = cv2.getPerspectiveTransform(rect.astype(np.float32), dst)
        warped = cv2.warpPerspective(frame, M, (width, height))
        
        return warped
    
    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        """Order points in top-left, top-right, bottom-right, bottom-left order"""
        rect = np.zeros((4, 2), dtype=np.float32)
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect


class CardDatabase:
    """Manage Yu-Gi-Oh! card database"""
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        self.cards_df = None
        self.load_database()
    
    def load_database(self):
        """Load card database from CSV/JSON files"""
        # Try to find any CSV or JSON file in data directory
        import glob
        
        csv_files = glob.glob(os.path.join(self.data_dir, '*.csv'))
        json_files = glob.glob(os.path.join(self.data_dir, '*.json'))
        
        all_files = csv_files + json_files
        
        if all_files:
            filepath = all_files[0]  # Use first file found
            if filepath.endswith('.csv'):
                self.cards_df = pd.read_csv(filepath)
            elif filepath.endswith('.json'):
                self.cards_df = pd.read_json(filepath)
            print(f"Loaded database from {filepath}")
            print(f"Total cards: {len(self.cards_df)}")
            print(f"Columns: {list(self.cards_df.columns)}")
        else:
            print(f"Warning: No card database found in {self.data_dir}")
            print(f"Looking for CSV or JSON files")
    
    def search_by_name(self, name: str) -> Optional[Dict]:
        """Search card by name"""
        if self.cards_df is None:
            return None
        
        # Try exact match first
        result = self.cards_df[self.cards_df['name'].str.lower() == name.lower()]
        
        if result.empty:
            # Try partial match
            result = self.cards_df[self.cards_df['name'].str.lower().str.contains(name.lower(), na=False)]
        
        if not result.empty:
            return result.iloc[0].to_dict()
        
        return None
    
    def get_card_info(self, card_id: int) -> Optional[Dict]:
        """Get card information by ID"""
        if self.cards_df is None:
            return None
        
        result = self.cards_df[self.cards_df['id'] == card_id]
        if not result.empty:
            return result.iloc[0].to_dict()
        
        return None
    
    def get_all_cards(self) -> List[Dict]:
        """Get all cards"""
        if self.cards_df is None:
            return []
        
        return self.cards_df.to_dict('records')


class CardRecognizer:
    """Recognize Yu-Gi-Oh! cards using image matching"""
    
    def __init__(self, database: CardDatabase):
        self.database = database
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
    def extract_features(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract ORB features from image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def recognize_card(self, card_image: np.ndarray) -> Optional[Dict]:
        """Recognize card from extracted image"""
        # For now, this is a placeholder
        # In a full implementation, you would:
        # 1. Use OCR to extract card name from the image
        # 2. Or use a trained CNN model to classify the card
        # 3. Or match against a database of card images
        
        # Placeholder: Return None for now
        # This will be enhanced with actual recognition logic
        return None


def main():
    """Test the card detection system"""
    detector = CardDetector()
    database = CardDatabase()
    recognizer = CardRecognizer(database)
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit, 's' to save detected card")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect card
        contour = detector.find_card_contour(frame)
        
        # Draw contour if found
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            
            # Extract card
            card_image = detector.extract_card(frame, contour)
            if card_image is not None:
                cv2.imshow('Detected Card', card_image)
        
        cv2.imshow('Yu-Gi-Oh! Card Detector', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and contour is not None:
            card_image = detector.extract_card(frame, contour)
            if card_image is not None:
                cv2.imwrite('detected_card.jpg', card_image)
                print("Card saved as detected_card.jpg")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
