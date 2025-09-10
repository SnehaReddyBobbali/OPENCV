"""
🎨 Virtual Painter (Air Canvas) + AI Cartoon Generator
Enhanced version with AI-powered cartoon generation
"""

import cv2
import numpy as np
import time
import os
from datetime import datetime
import requests
import base64
import json

# Try importing mediapipe for hand tracking
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("⚠️  MediaPipe not available - using color tracking mode only")

# Try importing PIL for image processing
try:
    from PIL import Image
    import io
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  PIL not available - AI features will be limited")

# ----------------------------
# Setup
# ----------------------------
ART_DIR = 'artworks'
AI_DIR = 'ai_cartoons'
os.makedirs(ART_DIR, exist_ok=True)
os.makedirs(AI_DIR, exist_ok=True)

# ----------------------------
# 🤖 AI Cartoon Generation Functions
# ----------------------------

def generate_cartoon_prompt(drawing_analysis):
    """Generate appropriate prompts based on drawing content"""
    prompts = {
        'character': "transform this sketch into a colorful cartoon character, Disney animation style, vibrant colors, smooth lines, professional 2D animation",
        'animal': "convert this drawing into a cute cartoon animal, Pixar style, adorable, colorful, animated movie character",
        'scene': "transform this sketch into a cartoon scene, animated movie background, colorful, detailed environment",
        'object': "convert this drawing into a cartoon-style object, animated movie prop, vibrant colors, smooth shading",
        'generic': "transform this simple drawing into a professional cartoon illustration, colorful, animated style, Disney/Pixar quality"
    }
    return prompts.get(drawing_analysis, prompts['generic'])

def analyze_drawing_content(image_path):
    """Simple analysis to determine drawing type - you can enhance this"""
    # This is a placeholder - in a real implementation, you might use:
    # - Object detection models
    # - Edge analysis to determine if it looks like a character, animal, etc.
    # - Color analysis
    # For now, we'll return 'generic'
    return 'generic'

def create_enhanced_cartoon_cv2(img):
    """Create a more cartoon-like version using advanced CV2 techniques"""
    # Step 1: Strong edge detection and enhancement
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create strong, clean edges
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    
    # Step 2: Color quantization for flat cartoon colors
    data = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    # Make colors more vibrant
    centers = np.clip(centers * 1.4, 0, 255).astype(np.uint8)
    quantized = centers[labels.flatten()].reshape(img.shape)
    
    # Step 3: Strong bilateral filtering for smooth regions
    smooth = quantized.copy()
    for _ in range(3):
        smooth = cv2.bilateralFilter(smooth, 20, 100, 100)
    
    # Step 4: Combine with edges
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Create mask for edges
    edge_mask = edges / 255.0
    edge_mask = np.stack([edge_mask] * 3, axis=2)
    
    # Blend smooth colors with black edges
    cartoon = smooth * (1 - edge_mask * 0.8) + edges_colored * edge_mask * 0.8
    cartoon = np.clip(cartoon, 0, 255).astype(np.uint8)
    
    # Step 5: Final enhancement
    cartoon = cv2.convertScaleAbs(cartoon, alpha=1.1, beta=10)
    
    return cartoon


def generate_ai_cartoon_local(image_path):
    """Generate cartoon using local AI model (placeholder for actual implementation)"""
    # This is where you would integrate with local models like:
    # - Stable Diffusion
    # - Custom trained sketch-to-cartoon models
    # - Other local AI services
    
    print("🤖 Generating cartoon with local AI...")
    
    # For now, use enhanced CV2 method as fallback
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Apply enhanced cartoon transformation
    cartoon = create_enhanced_cartoon_cv2(img)
    
    # Save result
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    ai_filename = f'ai_cartoon_{timestamp}.png'
    ai_filepath = os.path.join(AI_DIR, ai_filename)
    cv2.imwrite(ai_filepath, cartoon)
    
    return ai_filepath

def generate_ai_cartoon_api(image_path, api_key=None):
    """Generate cartoon using cloud API (OpenAI DALL-E style)"""
    if not api_key:
        print("⚠️  No API key provided - using local generation")
        return generate_ai_cartoon_local(image_path)
    
    # Placeholder for API integration
    # You would implement calls to services like:
    # - OpenAI DALL-E
    # - Stability AI
    # - Other image generation APIs
    
    print("🌐 API cartoon generation not implemented - using local method")
    return generate_ai_cartoon_local(image_path)

def create_character_variations(base_cartoon_path):
    """Create multiple variations of the same character"""
    base_img = cv2.imread(base_cartoon_path)
    if base_img is None:
        return []
    
    variations = []
    variation_names = ['happy', 'surprised', 'cool', 'kawaii']
    
    for i, name in enumerate(variation_names):
        # Apply different color schemes and effects
        variation = base_img.copy()
        
        if name == 'happy':
            # Brighter, warmer colors
            hsv = cv2.cvtColor(variation, cv2.COLOR_BGR2HSV)
            hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.3, 0, 255)  # More saturation
            hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.1, 0, 255)  # Brighter
            variation = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
        elif name == 'surprised':
            # Higher contrast, cooler colors
            variation = cv2.convertScaleAbs(variation, alpha=1.2, beta=0)
            
        elif name == 'cool':
            # Add blue tint
            variation[:,:,0] = np.clip(variation[:,:,0] * 1.2, 0, 255)
            
        elif name == 'kawaii':
            # Softer, pastel-like colors
            variation = cv2.addWeighted(variation, 0.8, np.full_like(variation, 255), 0.2, 0)
        
        # Save variation
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        var_filename = f'cartoon_{name}_{timestamp}.png'
        var_filepath = os.path.join(AI_DIR, var_filename)
        cv2.imwrite(var_filepath, variation)
        variations.append((name, var_filepath))
    
    return variations

# ----------------------------
# Original Art Style Functions (keeping existing ones)
# ----------------------------

def cartoonize(img):
    """Apply cartoon effect using bilateral filter + edge detection"""
    # Smooth the image while preserving edges
    smooth = cv2.bilateralFilter(img, d=15, sigmaColor=50, sigmaSpace=50)
    smooth2 = cv2.bilateralFilter(smooth, d=15, sigmaColor=50, sigmaSpace=50)
    
    # Create edge mask
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY, 7, 7)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Combine smooth image with edges
    cartoon = cv2.bitwise_and(smooth2, edges)
    return cartoon


def pencil_sketch(img):
    """Convert to pencil sketch effect"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Invert the grayscale image
    gray_inv = 255 - gray
    
    # Apply Gaussian blur to the inverted image
    blur = cv2.GaussianBlur(gray_inv, (21, 21), 0)
    
    # Create the pencil sketch by dividing the grayscale image by the inverted blurred image
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    
    # Enhance contrast
    sketch = cv2.convertScaleAbs(sketch, alpha=1.2, beta=10)
    
    # Convert back to BGR for consistency
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

def oil_paint_effect(img):
    """Apply oil painting effect"""
    try:
        # Try using xphoto oil painting
        return cv2.xphoto.oilPainting(img, 7, 1)
    except AttributeError:
        print("ℹ️  cv2.xphoto not available - using alternative oil effect")
        # Alternative oil paint effect using multiple bilateral filters
        oil = img.copy()
        for _ in range(3):
            oil = cv2.bilateralFilter(oil, 20, 80, 80)
        
        # Add some color enhancement
        oil = cv2.convertScaleAbs(oil, alpha=1.1, beta=10)
        return oil

def pop_art_style(img, k=6):
    """Create pop art style using K-means color quantization"""
    # Reshape image to be a list of pixels
    data = img.reshape((-1, 3)).astype(np.float32)
    
    # Apply K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    # Convert centers to uint8
    centers = np.uint8(centers)
    
    # Enhance colors for pop art effect (make them more vibrant)
    centers = np.clip(centers * 1.3, 0, 255).astype(np.uint8)
    
    # Map each pixel to its cluster center
    quantized = centers[labels.flatten()]
    result = quantized.reshape(img.shape)
    
    # Apply slight edge enhancement
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Combine with original for subtle edge effect
    result = cv2.addWeighted(result, 0.9, edges, 0.1, 0)
    
    return result

def watercolor_effect(img):
    """Create watercolor painting effect"""
    # Apply edge-preserving filter
    smooth = cv2.edgePreservingFilter(img, flags=2, sigma_s=50, sigma_r=0.4)
    
    # Add bilateral filtering for smoothness
    smooth = cv2.bilateralFilter(smooth, 15, 80, 80)
    
    # Create a slightly blurred version
    blur = cv2.GaussianBlur(smooth, (7, 7), 0)
    
    # Combine for watercolor effect
    watercolor = cv2.addWeighted(smooth, 0.7, blur, 0.3, 0)
    
    # Enhance saturation
    hsv = cv2.cvtColor(watercolor, cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = hsv[:,:,1] * 1.2  # Increase saturation
    hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
    watercolor = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return watercolor

# ----------------------------
# Drawing utilities
# ----------------------------

def smooth_brush_stroke(paint, pt1, pt2, color, size, erasing=False):
    """Draw smooth brush strokes between two points"""
    if pt1 is None or pt2 is None:
        return
    
    distance = int(np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))
    
    if distance == 0:
        cv2.circle(paint, pt2, size, (0,0,0) if erasing else color, -1)
        return
    
    # Draw smooth line with multiple points
    for i in range(0, distance, 2):
        t = i / distance if distance > 0 else 0
        x = int(pt1[0] + (pt2[0] - pt1[0]) * t)
        y = int(pt1[1] + (pt2[1] - pt1[1]) * t)
        brush_color = (0,0,0) if erasing else color
        brush_radius = size * 2 if erasing else size
        cv2.circle(paint, (x, y), brush_radius, brush_color, -1)


def overlay_paint_on_canvas(canvas, paint):
    """Overlay paint on canvas with transparency"""
    mask = cv2.cvtColor(paint, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    
    canvas_bg = cv2.bitwise_and(canvas, canvas, mask=mask_inv)
    paint_fg = cv2.bitwise_and(paint, paint, mask=mask)
    
    return cv2.add(canvas_bg, paint_fg)

# ----------------------------
# UI Components
# ----------------------------

def draw_enhanced_toolbar(frame, brush_color, brush_size, erasing, mode, last_saved):
    """Draw an enhanced toolbar with better visuals"""
    h, w = frame.shape[:2]
    toolbar_height = 100
    
    # Create toolbar background with gradient
    toolbar = np.zeros((toolbar_height, w, 3), dtype=np.uint8)
    for i in range(toolbar_height):
        intensity = int(60 - (i * 30 / toolbar_height))
        toolbar[i, :] = [intensity, intensity, intensity]
    
    # Color palette
    colors = [
        ((20, 20), (70, 70), (0, 0, 255), "RED", "1"),      # Red
        ((80, 20), (130, 70), (0, 255, 0), "GREEN", "2"),   # Green
        ((140, 20), (190, 70), (255, 0, 0), "BLUE", "3"),   # Blue
        ((200, 20), (250, 70), (0, 255, 255), "YELLOW", "4"), # Yellow
        ((260, 20), (310, 70), (255, 255, 255), "WHITE", "5"), # White
        ((320, 20), (370, 70), (128, 0, 128), "PURPLE", "6"), # Purple
    ]
    
    # Draw color buttons
    for (pt1, pt2, color, name, key) in colors:
        selected = (color == brush_color and not erasing)
        thickness = -1 if selected else 3
        cv2.rectangle(toolbar, pt1, pt2, color, thickness)
        if not selected:
            cv2.rectangle(toolbar, pt1, pt2, (200, 200, 200), 2)
        
        # Add key label
        text_color = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
        cv2.putText(toolbar, key, (pt1[0] + 15, pt2[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    
    # Eraser button
    eraser_rect = ((380, 20), (450, 70))
    eraser_color = (255, 100, 100) if erasing else (150, 150, 150)
    cv2.rectangle(toolbar, eraser_rect[0], eraser_rect[1], eraser_color, -1 if erasing else 3)
    cv2.rectangle(toolbar, eraser_rect[0], eraser_rect[1], (200, 200, 200), 2)
    cv2.putText(toolbar, "ERASE", (385, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # AI Button
    ai_rect = ((460, 20), (520, 70))
    ai_color = (100, 255, 100)
    cv2.rectangle(toolbar, ai_rect[0], ai_rect[1], ai_color, 3)
    cv2.rectangle(toolbar, ai_rect[0], ai_rect[1], (200, 200, 200), 2)
    cv2.putText(toolbar, "AI", (475, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Info panel
    info_x = 530
    cv2.putText(toolbar, f"Brush: {brush_size}px", (info_x, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(toolbar, f"Mode: {mode.upper()}", (info_x, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(toolbar, f"Saved: {'Yes' if last_saved else 'No'}", (info_x, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if last_saved else (255, 100, 100), 2)
    
    return toolbar, colors, eraser_rect, ai_rect


def draw_help_panel(frame):
    """Draw help instructions"""
    help_text = [
        "🎨 VIRTUAL PAINTER CONTROLS:",
        "s - Save drawing",
        "r - Reset canvas", 
        "c - Cartoonify",
        "p - Pencil sketch",
        "o - Oil painting",
        "k - Pop art style",
        "w - Watercolor",
        "a - AI Cartoon Generation",
        "v - Character Variations",
        "g - Gallery view",
        "m - Switch mode",
        "[ ] - Brush size",
        "e - Toggle eraser",
        "q - Quit"
    ]
    
    h, w = frame.shape[:2]
    panel_width = 280
    panel_height = len(help_text) * 25 + 20
    
    # Create semi-transparent panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - panel_width - 10, 10), 
                 (w - 10, panel_height + 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Add text
    for i, text in enumerate(help_text):
        y_pos = 35 + i * 25
        color = (0, 255, 255) if i == 0 else (255, 255, 255)
        font_scale = 0.6 if i == 0 else 0.5
        thickness = 2 if i == 0 else 1
        cv2.putText(frame, text, (w - panel_width, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def handle_toolbar_click(x, y, colors, eraser_rect, ai_rect):
    """Handle toolbar button clicks"""
    if y > 100:  # Below toolbar
        return None, None
    
    # Check color buttons
    for (pt1, pt2, color, name, key) in colors:
        if pt1[0] <= x <= pt2[0] and pt1[1] <= y <= pt2[1]:
            return 'color', color
    
    # Check eraser button
    if (eraser_rect[0][0] <= x <= eraser_rect[1][0] and 
        eraser_rect[0][1] <= y <= eraser_rect[1][1]):
        return 'eraser', None
    
    # Check AI button
    if (ai_rect[0][0] <= x <= ai_rect[1][0] and 
        ai_rect[0][1] <= y <= ai_rect[1][1]):
        return 'ai', None
    
    return None, None

# ----------------------------
# Tracking Classes
# ----------------------------
def draw_canvas_pointer(frame, position, brush_size, brush_color, is_drawing=False):
    """Draw a pointer/cursor on the canvas to show current drawing position"""
    if position is None:
        return
    
    x, y = position
    
    # Skip if in toolbar area
    if y <= 100:
        return
    
    # Main cursor circle
    if is_drawing:
        # Drawing state - solid colored circle
        cv2.circle(frame, (x, y), 8, brush_color, -1)
        cv2.circle(frame, (x, y), 8, (255, 255, 255), 2)  # White border
        
        # Show brush size preview
        cv2.circle(frame, (x, y), brush_size, brush_color, 2)
    else:
        # Not drawing - hollow circle
        cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)
        cv2.circle(frame, (x, y), 8, (0, 0, 0), 1)
        
        # Show brush size preview (faded)
        cv2.circle(frame, (x, y), brush_size, (150, 150, 150), 1)
    
    # Optional: Add crosshair for precision
    cv2.line(frame, (x-15, y), (x-5, y), (255, 255, 255), 1)
    cv2.line(frame, (x+5, y), (x+15, y), (255, 255, 255), 1)
    cv2.line(frame, (x, y-15), (x, y-5), (255, 255, 255), 1)
    cv2.line(frame, (x, y+5), (x, y+15), (255, 255, 255), 1)


class HandTracker:
    """Hand tracking using MediaPipe"""
    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.7):
        if not MP_AVAILABLE:
            raise ImportError("MediaPipe not available")
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
    
    def get_finger_position(self, frame):
        """Get index finger position and drawing state"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return None
        
        h, w = frame.shape[:2]
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Get landmark positions
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append([int(lm.x * w), int(lm.y * h)])
        
        # Index finger tip (landmark 8)
        index_tip = landmarks[8]
        
        # Check if index finger is up and middle finger is down
        index_up = landmarks[8][1] < landmarks[6][1]  # Index tip above index PIP
        middle_up = landmarks[12][1] < landmarks[10][1]  # Middle tip above middle PIP
        
        # Drawing when index up and middle down
        is_drawing = index_up and not middle_up
        
        return (index_tip[0], index_tip[1], is_drawing)

class ColorTracker:
    """Track colored objects (e.g., green marker)"""
    def __init__(self, lower_bound=(35, 80, 80), upper_bound=(85, 255, 255)):
        self.lower = np.array(lower_bound)
        self.upper = np.array(upper_bound)
    
    def get_position(self, frame):
        """Get position of colored object"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        mask = cv2.medianBlur(mask, 9)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) < 500:
            return None
        
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        return (int(x), int(y))

# ----------------------------
# 🎯 Main Application
# ----------------------------


def main():
    print("🎨 Starting Enhanced Virtual Painter with AI...")
    print("📹 Initializing camera...")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera!")
        return
    
    # Get first frame to setup canvas
    ret, frame = cap.read()
    if not ret:
        print("❌ Could not read from camera!")
        return
    
    h, w = frame.shape[:2]
    
    # Initialize canvas and UI
    paint_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Drawing state
    brush_color = (0, 0, 255)  # Red
    brush_size = 8
    erasing = False
    
    # Tracking
    mode = 'hand' if MP_AVAILABLE else 'color'
    hand_tracker = HandTracker() if MP_AVAILABLE else None
    color_tracker = ColorTracker()
    
    prev_point = None
    last_saved_path = None
    last_ai_cartoon = None
    show_help = True
    
    print(f"✅ Ready! Mode: {mode.upper()}")
    print("💡 Press 'h' to toggle help panel")
    print("🤖 Press 'a' to generate AI cartoon from your drawing!")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # Mirror effect
        
        # Create white canvas for final output
        white_canvas = np.full_like(frame, 255)
        
        # Get drawing position
        current_point = None
        is_drawing = False
        
        if mode == 'hand' and hand_tracker:
            result = hand_tracker.get_finger_position(frame)
            if result:
                x, y, is_drawing = result
                current_point = (x, y)
                
                # Draw finger tracking indicator
                color = (0, 255, 0) if is_drawing else (255, 255, 255)
                cv2.circle(frame, (x, y), 8, color, -1)
                cv2.circle(frame, (x, y), brush_size + 5, color, 2)
        
        else:  # Color tracking mode
            result = color_tracker.get_position(frame)
            if result:
                current_point = result
                is_drawing = True  # Always drawing in color mode
                
                # Draw tracking indicator
                cv2.circle(frame, current_point, 10, (0, 255, 0), -1)
                cv2.circle(frame, current_point, brush_size + 5, (255, 255, 255), 2)
        
        # Handle drawing
        if current_point and is_drawing:
            x, y = current_point
            
            # Check for toolbar interaction
            if y <= 100:  # In toolbar area
                prev_point = None
            else:
                # Draw on canvas
                if prev_point and prev_point[1] > 100:  # Previous point also in drawing area
                    smooth_brush_stroke(paint_canvas, prev_point, current_point, 
                                      brush_color, brush_size, erasing)
                prev_point = current_point
        else:
            prev_point = None
        
        # Create final frame
        output_frame = overlay_paint_on_canvas(white_canvas, paint_canvas)
        
        # Add toolbar
        toolbar, color_buttons, eraser_rect, ai_rect = draw_enhanced_toolbar(
            output_frame, brush_color, brush_size, erasing, mode, last_saved_path)
        output_frame[:100] = toolbar
        
        # Add camera preview (small)
        preview_size = (200, 150)
        preview = cv2.resize(frame, preview_size)
        ph, pw = output_frame.shape[:2]
        output_frame[ph-preview_size[1]-10:ph-10, pw-preview_size[0]-10:pw-10] = preview
        
        # Show help panel
        if show_help:
            draw_help_panel(output_frame)
        # Add this after the drawing logic and before cv2.imshow
# Draw pointer on canvas
        draw_canvas_pointer(output_frame, current_point, brush_size, brush_color, is_drawing)


        # Display
        cv2.imshow("🎨 AI-Enhanced Virtual Painter", output_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("👋 Goodbye!")
            break
        elif key == ord('h'):
            show_help = not show_help
        elif key == ord('r'):
            paint_canvas[:] = 0
            last_saved_path = None
            last_ai_cartoon = None
            print("🧹 Canvas cleared!")
        elif key == ord('s'):
            # Save current drawing
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'drawing_{timestamp}.png'
            filepath = os.path.join(ART_DIR, filename)
            
            # Save on white background
            save_canvas = overlay_paint_on_canvas(white_canvas, paint_canvas)
            cv2.imwrite(filepath, save_canvas)
            last_saved_path = filepath
            print(f"💾 Drawing saved: {filename}")
        
        elif key == ord('m'):
            # Switch tracking mode
            if MP_AVAILABLE:
                mode = 'color' if mode == 'hand' else 'hand'
                prev_point = None
                print(f"🔄 Switched to {mode.upper()} mode")
        
        elif key == ord('['):
            brush_size = max(1, brush_size - 1)
        elif key == ord(']'):
            brush_size = min(50, brush_size + 1)
        
        elif key == ord('e'):
            erasing = not erasing
            print("🧹 Eraser ON" if erasing else "✏️ Eraser OFF")
            
        elif key == ord('+') or key == ord('='):
            brush_size = min(100, brush_size + 2)  # increase brush size
            print("🔺 Brush size:", brush_size)
        elif key == ord('-') or key == ord('_'):
            brush_size = max(2, brush_size - 2)    # decrease brush size
            print("🔻 Brush size:", brush_size)

            
        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')]:
            # Color selection
            color_map = {
                ord('1'): (0, 0, 255),      # Red
                ord('2'): (0, 255, 0),      # Green  
                ord('3'): (255, 0, 0),      # Blue
                ord('4'): (0, 255, 255),    # Yellow
                ord('5'): (255, 255, 255),  # White
                ord('6'): (128, 0, 128),    # Purple
            }
            brush_color = color_map[key]
            erasing = False
        
        # 🤖 AI Cartoon Generation
        elif key == ord('a'):
            if last_saved_path is None:
                print("⚠️  Save a drawing first with 's' to generate AI cartoon!")
            else:
                print("🤖 Generating AI cartoon from your drawing...")
                ai_cartoon_path = generate_ai_cartoon_local(last_saved_path)
                if ai_cartoon_path:
                    last_ai_cartoon = ai_cartoon_path
                    ai_cartoon = cv2.imread(ai_cartoon_path)
                    cv2.imshow("🤖 AI Generated Cartoon", ai_cartoon)
                    print(f"✨ AI cartoon generated and saved: {ai_cartoon_path}")
                else:
                    print("❌ Failed to generate AI cartoon")
        
        # Character Variations
        elif key == ord('v'):
            if last_ai_cartoon is None:
                print("⚠️  Generate an AI cartoon first with 'a' to create variations!")
            else:
                print("🎭 Creating character variations...")
                variations = create_character_variations(last_ai_cartoon)
                if variations:
                    # Show all variations in a grid
                    base_img = cv2.imread(last_ai_cartoon)
                    if base_img is not None:
                        img_h, img_w = base_img.shape[:2]
                        # Create 2x3 grid for variations
                        grid = np.ones((2 * img_h + 60, 3 * img_w + 40, 3), dtype=np.uint8) * 255
                        
                        # Add original in center

                        grid[30:30+img_h, img_w+20:img_w+20+img_w] = base_img
                        cv2.putText(grid, "Original", (img_w+30, 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                        
                        # Add variations around it
                        positions = [(0, 0), (0, 2*img_w+40), (img_h+60, 0), (img_h+60, 2*img_w+40)]
                        for i, (name, var_path) in enumerate(variations[:4]):
                            if i < len(positions):
                                var_img = cv2.imread(var_path)
                                if var_img is not None:
                                    y, x = positions[i]
                                    grid[y:y+img_h, x:x+img_w] = var_img
                                    cv2.putText(grid, name.title(), (x+10, y-10 if y > 0 else y+img_h+20),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        
                        cv2.imshow("🎭 Character Variations", grid)
                        print(f"🎭 Created {len(variations)} character variations!")
                else:
                    print("❌ Failed to create variations")
        
        # Original Art style transformations
        elif key in [ord('c'), ord('p'), ord('o'), ord('k'), ord('w')]:
            if last_saved_path is None:
                print("⚠️  Save a drawing first with 's' to apply styles!")
            else:
                print("🎨 Applying art style...")
                original = cv2.imread(last_saved_path)
                if original is not None:
                    if key == ord('c'):
                        styled = cartoonize(original)
                        style_name = "cartoon"
                    elif key == ord('p'):
                        styled = pencil_sketch(original)  
                        style_name = "sketch"
                    elif key == ord('o'):
                        styled = oil_paint_effect(original)
                        style_name = "oil"
                    elif key == ord('k'):
                        styled = pop_art_style(original, k=6)
                        style_name = "popart"
                    elif key == ord('w'):
                        styled = watercolor_effect(original)
                        style_name = "watercolor"
                    
                    # Save styled version
                    style_path = last_saved_path.replace('drawing_', f'{style_name}_')
                    cv2.imwrite(style_path, styled)
                    
                    # Show result
                    cv2.imshow(f"✨ {style_name.upper()} Style", styled)
                    print(f"✨ {style_name.upper()} style applied and saved!")
        
        elif key == ord('g'):
            # Enhanced Gallery mode - show all styles including AI
            if last_saved_path is None:
                print("⚠️  Save a drawing first to create gallery!")
            else:
                print("🖼  Creating enhanced art gallery...")
                original = cv2.imread(last_saved_path)
                if original is not None:
                    styles = {
                        'Original': original,
                        'AI Cartoon': cv2.imread(last_ai_cartoon) if last_ai_cartoon else create_enhanced_cartoon_cv2(original),
                        'Classic Cartoon': cartoonize(original),
                        'Sketch': pencil_sketch(original),
                        'Oil Paint': oil_paint_effect(original),
                        'Pop Art': pop_art_style(original),
                        'Watercolor': watercolor_effect(original)
                    }
                    
                    # Create gallery layout (2x4 grid)
                    img_h, img_w = original.shape[:2]
                    gallery = np.ones((2 * img_h + 100, 4 * img_w + 60, 3), dtype=np.uint8) * 255
                    
                    positions = [
                        (0, 0), (0, img_w + 20), (0, 2*img_w + 40), (0, 3*img_w + 60),

                        (img_h + 50, 0), (img_h + 50, img_w + 20), (img_h + 50, 2*img_w + 40)
                    ]
                    
                    for i, (style_name, styled_img) in enumerate(styles.items()):
                        if i < len(positions) and styled_img is not None:
                            y, x = positions[i]
                            gallery[y:y+img_h, x:x+img_w] = styled_img
                            
                            # Add label
                            text_color = (255, 0, 0) if 'AI' in style_name else (0, 0, 0)
                            cv2.putText(gallery, style_name, (x + 10, y + img_h + 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                    
                    cv2.imshow("🖼 Enhanced Art Gallery", gallery)
                    print("🖼  Enhanced gallery created with AI styles!")
    
    cap.release()
    cv2.destroyAllWindows()
    print("🎨 AI-Enhanced Virtual Painter closed successfully!")

if __name__ == '__main__':
    main()
