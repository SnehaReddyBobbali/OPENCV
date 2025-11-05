"""
🎨 Virtual Painter (Air Canvas) + AI Cartoon Generator
Enhanced version with AI-powered cartoon generation
"""

import cv2
import numpy as np
import time
import colorsys
import os
import random
from datetime import datetime
import requests
import base64
import json

# Quick diagnostic: ensure the imported cv2 provides VideoCapture (helps surface installer or shadowing problems)
try:
    _has_vc = hasattr(cv2, 'VideoCapture')
except Exception:
    _has_vc = False
if not _has_vc:
    import sys
    print(f"ERROR: cv2 module loaded from {getattr(cv2, '__file__', None)} does not provide 'VideoCapture'.")
    print(f"Run this script with the Python interpreter you installed OpenCV into (sys.executable={sys.executable}), and make sure 'opencv-python' is installed.")
    print("Also ensure there is no local file named 'cv2.py' or a folder named 'cv2' that would shadow the real package.")
    raise ImportError("cv2 missing VideoCapture - install 'opencv-python' or fix import shadowing")

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

# Runtime color/palette settings (user-configurable)
PALETTE = [
    # Vibrant BGR colors with friendly names and key bindings
    ((20, 20, 255), "Scarlet", "1"),      # bright red
    ((50, 205, 50), "Lime", "2"),        # lime green
    ((200, 50, 30), "Royal Blue", "3"),  # warmer blue-ish
    ((0, 215, 255), "Gold", "4"),        # gold (BGR approximated)
    ((255, 0, 255), "Magenta", "5"),     # magenta
    ((0, 140, 255), "Tangerine", "6"),   # orange
    ((180, 0, 180), "Violet", "7"),      # violet
    ((255, 200, 0), "Cyan", "8"),        # cyan-like (bright)
    ((90, 165, 200), "Teal", "9"),       # teal
    ((0, 0, 0), "Black", "0"),           # black
    ((255, 255, 255), "Eraser", "E")     # eraser / white
]

# Brush alpha (0.0 - 1.0) and rainbow toggle
brush_alpha = 1.0
rainbow_mode = False
hue_base = 0
# Mouse click position storage for toolbar clicks
mouse_click_pos = None
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
# ----------------- CUSTOM BRUSHES -----------------
def square_brush(canvas, pt1, pt2, color, size):
    x1,y1 = pt1; x2,y2 = pt2
    steps = int(max(abs(x2-x1), abs(y2-y1))/size)+1
    for i in range(steps):
        x = int(x1 + (x2-x1)*i/steps)
        y = int(y1 + (y2-y1)*i/steps)
        cv2.rectangle(canvas, (x-size//2,y-size//2), (x+size//2,y+size//2), color, -1)

def spray_brush(canvas, point, color, size):
    x0,y0 = point
    for _ in range(size*3):  # more = denser
        x = x0 + random.randint(-size, size)
        y = y0 + random.randint(-size, size)
        if 0<=x<canvas.shape[1] and 0<=y<canvas.shape[0]:
            canvas[y,x] = color

def calligraphy_brush(canvas, point, color, size, angle=45):
    axes = (size, size//3)
    cv2.ellipse(canvas, point, axes, angle, 0, 360, color, -1)
# ---------------------------------------------------

def smooth_brush_stroke(paint, pt1, pt2, color, size, erasing=False, alpha=1.0, rainbow=False):
    """Draw smooth brush strokes between two points with optional alpha and rainbow mode.

    alpha: float 0.0-1.0 blending of stroke onto paint (only for drawing, not erasing).
    rainbow: bool to cycle hue along the stroke.
    """
    if pt1 is None or pt2 is None:
        return

    distance = int(np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))
    hgt, wid = paint.shape[:2]

    if distance == 0:
        if erasing:
            cv2.circle(paint, pt2, size, (0,0,0), -1)
        else:
            # blend single circle
            x, y = pt2
            y0, y1 = max(0, y-size), min(hgt, y+size)
            x0, x1 = max(0, x-size), min(wid, x+size)
            layer = paint[y0:y1, x0:x1].copy()
            overlay = np.zeros_like(layer)
            center = (min(size, x-x0), min(size, y-y0))
            cv2.circle(overlay, center, size, color, -1)
            mask = overlay.any(axis=2)
            layer[mask] = (overlay[mask].astype(np.float32) * alpha + layer[mask].astype(np.float32) * (1-alpha)).astype(np.uint8)
            paint[y0:y1, x0:x1] = layer
        return

    # Draw smooth line with multiple blended points
    global hue_base
    for i in range(0, distance, 2):
        t = i / distance if distance > 0 else 0
        x = int(pt1[0] + (pt2[0] - pt1[0]) * t)
        y = int(pt1[1] + (pt2[1] - pt1[1]) * t)
        if erasing:
            brush_color = (0,0,0)
            brush_radius = size * 2
            cv2.circle(paint, (x, y), brush_radius, brush_color, -1)
            continue

        brush_radius = size
        if rainbow:
            # compute hue based on position along stroke
            hue = (hue_base + i * 2) % 360
            r, g, b = colorsys.hsv_to_rgb(hue/360.0, 1.0, 1.0)
            brush_color = (int(b*255), int(g*255), int(r*255))
        else:
            brush_color = color

        # ROI for blending
        y0, y1 = max(0, y-brush_radius), min(hgt, y+brush_radius)
        x0, x1 = max(0, x-brush_radius), min(wid, x+brush_radius)
        if y0 >= y1 or x0 >= x1:
            continue
        layer = paint[y0:y1, x0:x1].copy()
        overlay = np.zeros_like(layer)
        cx = x - x0
        cy = y - y0
        cv2.circle(overlay, (cx, cy), brush_radius, brush_color, -1)
        mask = overlay.any(axis=2)
        if mask.any():
            # blend only where overlay has paint
            layer[mask] = (overlay[mask].astype(np.float32) * alpha + layer[mask].astype(np.float32) * (1-alpha)).astype(np.uint8)
            paint[y0:y1, x0:x1] = layer


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
    """Draw an enhanced toolbar with modern visuals requested by the user.

    Features:
    - Gradient background (dark blue-gray)
    - Bigger glossy color buttons (55px) with spacing
    - White glow around selected color
    - Shadowed number labels
    - Glass-morphism eraser & clear buttons with visual feedback
    - Live brush size preview (actual circle)
    - Arrow icons for Undo/Redo and color-coded status
    """
    h, w = frame.shape[:2]
    toolbar_height = 160

    # Gradient background (dark blue-gray -> darker)
    top_color = np.array([50, 70, 90], dtype=np.uint8)
    bottom_color = np.array([18, 24, 36], dtype=np.uint8)
    toolbar = np.zeros((toolbar_height, w, 3), dtype=np.uint8)
    for i in range(toolbar_height):
        t = i / max(1, toolbar_height - 1)
        col = (top_color * (1 - t) + bottom_color * t).astype(np.uint8)
        toolbar[i, :] = col

    palette = PALETTE

    # Button sizing and spacing
    btn_size = 55
    gap = 15
    x_offset = 12
    y_offset = 18
    buttons_per_row = 5

    colors = []
    for i, (color, name, key) in enumerate(palette):
        row = i // buttons_per_row
        col = i % buttons_per_row
        x1 = x_offset + col * (btn_size + gap)
        y1 = y_offset + row * (btn_size + gap)
        x2 = x1 + btn_size
        y2 = y1 + btn_size
        colors.append(((x1, y1), (x2, y2), color, name, key))

    # Helper to draw glossy button with subtle highlight
    def _draw_glossy(rect_from, rect_to, base_color, selected=False):
        x1, y1 = rect_from
        x2, y2 = rect_to
        # Base filled rectangle
        cv2.rectangle(toolbar, (x1, y1), (x2, y2), base_color, -1, cv2.LINE_AA)

        # Glossy overlay: a translucent white at the top
        overlay = toolbar.copy()
        gloss_h = max(6, int((y2 - y1) * 0.45))
        cv2.rectangle(overlay, (x1+2, y1+2), (x2-2, y1+gloss_h), (255, 255, 255), -1)
        alpha = 0.12 if not selected else 0.22
        cv2.addWeighted(overlay, alpha, toolbar, 1 - alpha, 0, toolbar)

        # Border and subtle inner shadow
        border_col = (220, 220, 220) if selected else (50, 50, 50)
        cv2.rectangle(toolbar, (x1, y1), (x2, y2), border_col, 2, cv2.LINE_AA)

        # If selected, add white glow
        if selected:
            glow = np.zeros_like(toolbar)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            radius = int(btn_size * 0.9)
            cv2.circle(glow, (cx, cy), radius, (255, 255, 255), -1)
            cv2.addWeighted(glow, 0.10, toolbar, 0.90, 0, toolbar)

    # Draw color buttons with glossy effect
    for (pt1, pt2, color, name, key) in colors:
        selected = (color == brush_color and not erasing)
        _draw_glossy(pt1, pt2, color, selected)

        # Shadowed number label (shadow offset + foreground)
        tx = pt1[0] + btn_size // 2 - 6
        ty = pt2[1] - 12
        # Shadow
        cv2.putText(toolbar, key, (tx+2, ty+2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (10, 10, 10), 3, cv2.LINE_AA)
        # Foreground with slight outline
        text_col = (255, 255, 255) if sum(color) < 380 else (0, 0, 0)
        cv2.putText(toolbar, key, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_col, 2, cv2.LINE_AA)

    # Glass-morphism Eraser and Clear buttons
    eraser_w = btn_size + 10
    eraser_h = btn_size
    eraser_x1 = x_offset + (buttons_per_row) * (btn_size + gap)
    eraser_y1 = y_offset
    eraser_x2 = eraser_x1 + eraser_w
    eraser_y2 = eraser_y1 + eraser_h
    eraser_rect = ((eraser_x1, eraser_y1), (eraser_x2, eraser_y2))
    
    clear_x1 = eraser_x1
    clear_y1 = eraser_y1 + eraser_h + gap
    clear_x2 = clear_x1 + eraser_w
    clear_y2 = clear_y1 + eraser_h
    clear_rect = ((clear_x1, clear_y1), (clear_x2, clear_y2))

    # Draw semi-transparent glass background
    glass = toolbar.copy()
    alpha_glass = 0.16
    cv2.rectangle(glass, eraser_rect[0], eraser_rect[1], (255, 255, 255), -1)
    cv2.rectangle(glass, clear_rect[0], clear_rect[1], (255, 255, 255), -1)
    cv2.addWeighted(glass, alpha_glass, toolbar, 1 - alpha_glass, 0, toolbar)

    # Borders
    cv2.rectangle(toolbar, eraser_rect[0], eraser_rect[1], (200, 200, 200), 2, cv2.LINE_AA)
    cv2.rectangle(toolbar, clear_rect[0], clear_rect[1], (200, 200, 200), 2, cv2.LINE_AA)

    # Eraser icon (animated-style simplified: slanted rectangle + spark)
    ex1, ey1 = eraser_rect[0]
    ex2, ey2 = eraser_rect[1]
    icon_center = ((ex1 + ex2) // 2, (ey1 + ey2) // 2)
    cv2.rectangle(toolbar, (icon_center[0]-14, icon_center[1]-8), (icon_center[0]+14, icon_center[1]+8), (150, 150, 150), -1, cv2.LINE_AA)
    cv2.line(toolbar, (icon_center[0]-12, icon_center[1]+10), (icon_center[0]+12, icon_center[1]-10), (255,255,255), 2)
    if erasing:
        # Active feedback: tint and bolder border
        cv2.rectangle(toolbar, eraser_rect[0], eraser_rect[1], (60,200,100), 3, cv2.LINE_AA)
        cv2.putText(toolbar, "ERASE", (ex1+8, ey2-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60,200,100), 2, cv2.LINE_AA)
    else:
        cv2.putText(toolbar, "ERASE", (ex1+8, ey2-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 2, cv2.LINE_AA)

    # Clear icon (trash can-ish)
    cx1, cy1 = clear_rect[0]
    cx2, cy2 = clear_rect[1]
    cv2.rectangle(toolbar, (cx1+10, cy1+8), (cx2-10, cy2-12), (230,230,230), -1, cv2.LINE_AA)
    cv2.rectangle(toolbar, (cx1+10, cy1+8), (cx2-10, cy2-12), (180,180,180), 2, cv2.LINE_AA)
    cv2.putText(toolbar, "CLR", (cx1+12, cy2-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80,80,80), 2, cv2.LINE_AA)

    # AI Button (compact)
    ai_x1 = clear_x2 + gap
    ai_y1 = y_offset
    ai_x2 = ai_x1 + 70
    ai_y2 = ai_y1 + 55
    ai_rect = ((ai_x1, ai_y1), (ai_x2, ai_y2))
    cv2.rectangle(toolbar, ai_rect[0], ai_rect[1], (100,255,150), 2, cv2.LINE_AA)
    cv2.putText(toolbar, "AI", (ai_x1+18, ai_y1+36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100,255,150), 3, cv2.LINE_AA)

    # Compact info area near AI button (reduced — removed large right-side panel as requested)
    info_x = ai_x2 + 12
    # Live brush size preview (small, next to AI button)
    preview_cx = info_x + 44
    preview_cy = y_offset + 24
    cv2.circle(toolbar, (preview_cx, preview_cy), max(4, brush_size), brush_color, -1)
    cv2.circle(toolbar, (preview_cx, preview_cy), max(4, brush_size), (255,255,255), 1)
    cv2.putText(toolbar, f"{brush_size}px", (info_x + 88, preview_cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1)

    # Small undo/redo indicators (no large panel)
    arrow_x = info_x + 12
    arrow_y = preview_cy + 36
    cv2.putText(toolbar, "Z:⬅︎  Y:➡︎", (info_x + 70, arrow_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

    # Compact shortcuts line
    cv2.putText(toolbar, "S=Save  A=AI  F=Shapes  M=Mode  Q=Quit", (info_x + 12, arrow_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170,170,170), 1, cv2.LINE_AA)

    return toolbar, colors, eraser_rect, ai_rect



def draw_help_panel(frame):
    help_text = [
        "🎨 VIRTUAL PAINTER CONTROLS",
        "S  - Save       R  - Reset/Clear",
        "C  - Cartoonify  P  - Pencil",
        "O  - Oil paint   K  - Pop art",
        "W  - Watercolor  A  - AI Cartoon",
        "V  - Character   G  - Gallery",
        "M  - Switch mode  [ ] - Brush size",
        "E  - Eraser      Q  - Quit",
        "F  - Shapes      H  - Toggle help",
        "I  - Eyedropper  T  - Rainbow",
        ",/. - Opacity    +/- - Brush size",
        "Z  - Undo        Y  - Redo",
        "",
        "Pointers (Accessibility):",
        "- Finger",
        "- Nose",
        "- Elbow",
        "- Head-mounted pointer",
        "- Eye-tracking pointer",
    ]
    
    h, w = frame.shape[:2]
    panel_width = 350
    panel_height = len(help_text) * 25 + 20
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - panel_width - 10, 10), 
                  (w - 10, panel_height + 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    for i, text in enumerate(help_text):
        y_pos = 35 + i * 25
        color = (0, 255, 255) if i == 0 else (255, 255, 255)
        font_scale = 0.6 if i == 0 else 0.5
        thickness = 2 if i == 0 else 1
        cv2.putText(frame, text, (w - panel_width + 10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def handle_toolbar_click(x, y, colors, eraser_rect, ai_rect):
    """Handle toolbar button clicks.

    Always return a 3-tuple: (action, data, name)
    action: 'color'|'eraser'|'ai'|None
    data: color tuple or None
    name: human-readable name or None
    """
    if y > 120:  # Below toolbar region (consider toolbar height change)
        return None, None, None

    # Check color buttons
    for (pt1, pt2, color, name, key) in colors:
        if pt1[0] <= x <= pt2[0] and pt1[1] <= y <= pt2[1]:
            return 'color', color, name

    # Check eraser button
    if (eraser_rect[0][0] <= x <= eraser_rect[1][0] and 
        eraser_rect[0][1] <= y <= eraser_rect[1][1]):
        return 'eraser', None, 'Eraser'

    # Check AI button
    if (ai_rect[0][0] <= x <= ai_rect[1][0] and 
        ai_rect[0][1] <= y <= ai_rect[1][1]):
        return 'ai', None, 'AI'

    return None, None, None

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
    """Hand tracking using MediaPipe with improved gesture detection"""
    def __init__(self, max_hands=2, detection_confidence=0.7, tracking_confidence=0.7):
        if not MP_AVAILABLE:
            raise ImportError("MediaPipe not available")
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
    
    def get_finger_positions(self, frame):
        """Get index finger positions and drawing state for up to max_hands.
        Returns list of tuples: (x, y, is_drawing) for each detected hand (ordered by detection).
        Drawing is active when index finger is up and middle finger is down.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return []

        h, w = frame.shape[:2]
        positions = []
        
        for hand_landmarks in results.multi_hand_landmarks:
            # Get landmark positions
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([int(lm.x * w), int(lm.y * h)])

            # Index finger tip (landmark 8) and base joints
            index_tip = landmarks[8]      # Fingertip
            index_pip = landmarks[6]      # Middle joint
            index_mcp = landmarks[5]      # Base joint
            
            # Middle finger landmarks
            middle_tip = landmarks[12]
            middle_pip = landmarks[10]
            middle_mcp = landmarks[9]
            
            # Wrist for reference
            wrist = landmarks[0]
            
            # FIXED: Check if fingers are UP (tip should be ABOVE base joints)
            # For y-coordinates, smaller values are higher on screen
            index_up = index_tip[1] < index_pip[1] - 10  # Index tip above PIP with threshold
            middle_down = middle_tip[1] > middle_pip[1] + 10  # Middle tip below PIP
            
            # Additional check: index should be significantly extended
            index_extended = index_tip[1] < index_mcp[1] - 20
            
            # Drawing gesture: index up AND middle down
            is_drawing = index_up and middle_down and index_extended
            
            # Optional: Draw hand landmarks on frame for debugging
            # Uncomment this to see hand skeleton overlay
            # self.mp_drawing.draw_landmarks(
            #     frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            positions.append((index_tip[0], index_tip[1], is_drawing))

        return positions

    def draw_hand_debug(self, frame, hand_landmarks):
        """Draw hand landmarks for debugging (optional utility method)"""
        self.mp_drawing.draw_landmarks(
            frame, 
            hand_landmarks, 
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

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
def draw_shape_overlay(canvas, shape="circle", alpha=0.3):
    """Overlay a transparent shape as a tracing guide."""
    overlay = canvas.copy()
    h, w = canvas.shape[:2]
    center = (w // 2, h // 2 + 50)  # keep below toolbar
    size = min(h, w) // 4

    color = (0, 0, 0)  # black (overlay will be blended by alpha)

    if shape == "circle":
        cv2.circle(overlay, center, size, color, thickness=3)

    elif shape == "square":
        top_left = (center[0] - size, center[1] - size)
        bottom_right = (center[0] + size, center[1] + size)
        cv2.rectangle(overlay, top_left, bottom_right, color, thickness=3)

    elif shape == "triangle":
        pts = np.array([
            [center[0], center[1] - size],
            [center[0] - size, center[1] + size],
            [center[0] + size, center[1] + size]
        ], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=3)

    elif shape == "line":
        cv2.line(overlay, (center[0] - size, center[1]),
                 (center[0] + size, center[1]), color, thickness=3)

    # Blend overlay with transparency
    return cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)


# ===========================
# ADVANCED SHAPE STAMPS (from user-supplied fixed painter)
# ===========================

def draw_cat(canvas, center, size, color):
    """Draw a cute cat face"""
    x, y = center
    cv2.circle(canvas, (x, y), size, color, -1, cv2.LINE_AA)
    cv2.circle(canvas, (x, y), size, (0, 0, 0), 2, cv2.LINE_AA)
    ear_offset = int(size * 0.7)
    ear_size = int(size * 0.4)
    left_ear = np.array([
        [x - ear_offset, y - size//2],
        [x - ear_offset - ear_size, y - size - ear_size],
        [x - ear_offset + ear_size//2, y - size]
    ], np.int32)
    cv2.fillPoly(canvas, [left_ear], color, cv2.LINE_AA)
    cv2.polylines(canvas, [left_ear], True, (0, 0, 0), 2, cv2.LINE_AA)
    right_ear = np.array([
        [x + ear_offset, y - size//2],
        [x + ear_offset + ear_size, y - size - ear_size],
        [x + ear_offset - ear_size//2, y - size]
    ], np.int32)
    cv2.fillPoly(canvas, [right_ear], color, cv2.LINE_AA)
    cv2.polylines(canvas, [right_ear], True, (0, 0, 0), 2, cv2.LINE_AA)
    eye_y = y - size//4
    eye_offset = size//3
    cv2.ellipse(canvas, (x - eye_offset, eye_y), (size//6, size//4), 0, 0, 360, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (x + eye_offset, eye_y), (size//6, size//4), 0, 0, 360, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(canvas, (x - eye_offset + 3, eye_y - 5), 3, (255, 255, 255), -1)
    cv2.circle(canvas, (x + eye_offset + 3, eye_y - 5), 3, (255, 255, 255), -1)
    nose_pts = np.array([
        [x, y + size//8],
        [x - size//8, y],
        [x + size//8, y]
    ], np.int32)
    cv2.fillPoly(canvas, [nose_pts], (255, 150, 150), cv2.LINE_AA)
    cv2.ellipse(canvas, (x - size//4, y + size//4), (size//6, size//8), 0, 0, 180, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.ellipse(canvas, (x + size//4, y + size//4), (size//6, size//8), 0, 0, 180, (0, 0, 0), 2, cv2.LINE_AA)
    whisker_len = int(size * 0.8)
    cv2.line(canvas, (x - size//2, y), (x - size//2 - whisker_len, y - size//4), (0, 0, 0), 2, cv2.LINE_AA)
    cv2.line(canvas, (x - size//2, y + size//8), (x - size//2 - whisker_len, y + size//8), (0, 0, 0), 2, cv2.LINE_AA)
    cv2.line(canvas, (x + size//2, y), (x + size//2 + whisker_len, y - size//4), (0, 0, 0), 2, cv2.LINE_AA)
    cv2.line(canvas, (x + size//2, y + size//8), (x + size//2 + whisker_len, y + size//8), (0, 0, 0), 2, cv2.LINE_AA)


def draw_dog(canvas, center, size, color):
    """Draw a cute dog face"""
    x, y = center
    cv2.circle(canvas, (x, y), size, color, -1, cv2.LINE_AA)
    cv2.circle(canvas, (x, y), size, (0, 0, 0), 2, cv2.LINE_AA)
    ear_offset = int(size * 0.8)
    cv2.ellipse(canvas, (x - ear_offset, y), (size//3, size//2), 30, 0, 360, color, -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (x - ear_offset, y), (size//3, size//2), 30, 0, 360, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.ellipse(canvas, (x + ear_offset, y), (size//3, size//2), -30, 0, 360, color, -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (x + ear_offset, y), (size//3, size//2), -30, 0, 360, (0, 0, 0), 2, cv2.LINE_AA)
    eye_y = y - size//4
    eye_offset = size//3
    cv2.circle(canvas, (x - eye_offset, eye_y), size//6, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(canvas, (x + eye_offset, eye_y), size//6, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(canvas, (x - eye_offset + 4, eye_y - 4), 4, (255, 255, 255), -1)
    cv2.circle(canvas, (x + eye_offset + 4, eye_y - 4), 4, (255, 255, 255), -1)
    cv2.ellipse(canvas, (x, y + size//3), (size//2, size//3), 0, 0, 360, (240, 240, 240), -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (x, y + size//3), (size//2, size//3), 0, 0, 360, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.ellipse(canvas, (x, y + size//6), (size//6, size//5), 0, 0, 360, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.line(canvas, (x, y + size//6), (x, y + size//2), (0, 0, 0), 2, cv2.LINE_AA)
    cv2.ellipse(canvas, (x - size//4, y + size//2), (size//5, size//6), 0, 0, 180, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.ellipse(canvas, (x + size//4, y + size//2), (size//5, size//6), 0, 0, 180, (0, 0, 0), 2, cv2.LINE_AA)
    tongue = np.array([
        [x - size//8, y + size//2],
        [x + size//8, y + size//2],
        [x, y + size//2 + size//4]
    ], np.int32)
    cv2.fillPoly(canvas, [tongue], (100, 100, 255), cv2.LINE_AA)
    cv2.polylines(canvas, [tongue], True, (0, 0, 0), 2, cv2.LINE_AA)


def draw_heart(canvas, center, size, color):
    """Draw a heart shape"""
    x, y = center
    heart_pts = []
    for angle in range(0, 360):
        t = np.radians(angle)
        hx = size * 16 * np.sin(t)**3
        hy = -size * (13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))
        heart_pts.append([int(x + hx/16), int(y + hy/16)])
    heart_pts = np.array(heart_pts, np.int32)
    cv2.fillPoly(canvas, [heart_pts], color, cv2.LINE_AA)
    cv2.polylines(canvas, [heart_pts], True, (0, 0, 0), 2, cv2.LINE_AA)


def draw_star(canvas, center, size, color):
    """Draw a 5-pointed star"""
    x, y = center
    points = []
    for i in range(10):
        angle = np.pi * 2 * i / 10 - np.pi / 2
        r = size if i % 2 == 0 else size // 2
        px = int(x + r * np.cos(angle))
        py = int(y + r * np.sin(angle))
        points.append([px, py])
    points = np.array(points, np.int32)
    cv2.fillPoly(canvas, [points], color, cv2.LINE_AA)
    cv2.polylines(canvas, [points], True, (0, 0, 0), 2, cv2.LINE_AA)


def draw_flower(canvas, center, size, color):
    """Draw a simple flower"""
    x, y = center
    petal_size = int(size * 0.4)
    petal_offset = int(size * 0.5)
    for angle in [0, 60, 120, 180, 240, 300]:
        rad = np.radians(angle)
        px = int(x + petal_offset * np.cos(rad))
        py = int(y + petal_offset * np.sin(rad))
        cv2.circle(canvas, (px, py), petal_size, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), petal_size, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.circle(canvas, (x, y), petal_size, (255, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(canvas, (x, y), petal_size, (0, 0, 0), 2, cv2.LINE_AA)


def draw_sun(canvas, center, size, color):
    """Draw a sun with rays"""
    x, y = center
    for angle in range(0, 360, 30):
        rad = np.radians(angle)
        x1 = int(x + size * 0.7 * np.cos(rad))
        y1 = int(y + size * 0.7 * np.sin(rad))
        x2 = int(x + size * 1.2 * np.cos(rad))
        y2 = int(y + size * 1.2 * np.sin(rad))
        cv2.line(canvas, (x1, y1), (x2, y2), color, 4, cv2.LINE_AA)
    cv2.circle(canvas, (x, y), int(size * 0.6), color, -1, cv2.LINE_AA)
    cv2.circle(canvas, (x, y), int(size * 0.6), (0, 0, 0), 2, cv2.LINE_AA)
    eye_offset = int(size * 0.25)
    eye_y = y - int(size * 0.15)
    cv2.circle(canvas, (x - eye_offset, eye_y), size//10, (0, 0, 0), -1)
    cv2.circle(canvas, (x + eye_offset, eye_y), size//10, (0, 0, 0), -1)
    cv2.ellipse(canvas, (x, y + size//8), (size//3, size//4), 0, 0, 180, (0, 0, 0), 2, cv2.LINE_AA)


def draw_cloud(canvas, center, size, color):
    """Draw a fluffy cloud"""
    x, y = center
    circles = [
        (x - size//2, y, int(size * 0.4)),
        (x - size//4, y - size//3, int(size * 0.5)),
        (x, y, int(size * 0.6)),
        (x + size//4, y - size//3, int(size * 0.5)),
        (x + size//2, y, int(size * 0.4)),
    ]
    for cx, cy, r in circles:
        cv2.circle(canvas, (cx, cy), r, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), r, (0, 0, 0), 2, cv2.LINE_AA)


def draw_butterfly(canvas, center, size, color):
    """Draw a butterfly"""
    x, y = center
    cv2.ellipse(canvas, (x, y), (size//8, size//2), 0, 0, 360, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.line(canvas, (x, y - size//2), (x - size//4, y - size), (0, 0, 0), 2, cv2.LINE_AA)
    cv2.line(canvas, (x, y - size//2), (x + size//4, y - size), (0, 0, 0), 2, cv2.LINE_AA)
    cv2.circle(canvas, (x - size//4, y - size), 4, (0, 0, 0), -1)
    cv2.circle(canvas, (x + size//4, y - size), 4, (0, 0, 0), -1)
    cv2.ellipse(canvas, (x - size//2, y - size//4), (size//2, size//2), 45, 0, 360, color, -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (x - size//2, y + size//4), (size//3, size//3), 45, 0, 360, color, -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (x + size//2, y - size//4), (size//2, size//2), -45, 0, 360, color, -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (x + size//2, y + size//4), (size//3, size//3), -45, 0, 360, color, -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (x - size//2, y - size//4), (size//2, size//2), 45, 0, 360, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.ellipse(canvas, (x - size//2, y + size//4), (size//3, size//3), 45, 0, 360, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.ellipse(canvas, (x + size//2, y - size//4), (size//2, size//2), -45, 0, 360, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.ellipse(canvas, (x + size//2, y + size//4), (size//3, size//3), -45, 0, 360, (0, 0, 0), 2, cv2.LINE_AA)


def draw_fish(canvas, center, size, color):
    """Draw a cute fish"""
    x, y = center
    cv2.ellipse(canvas, (x, y), (size, int(size * 0.6)), 0, 0, 360, color, -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (x, y), (size, int(size * 0.6)), 0, 0, 360, (0, 0, 0), 2, cv2.LINE_AA)
    tail_pts = np.array([
        [x + size, y - size//2],
        [x + size, y + size//2],
        [x + int(size * 1.5), y]
    ], np.int32)
    cv2.fillPoly(canvas, [tail_pts], color, cv2.LINE_AA)
    cv2.polylines(canvas, [tail_pts], True, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.circle(canvas, (x - size//2, y - size//4), size//6, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, (x - size//2, y - size//4), size//8, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (x - size//3, y + size//4), (size//6, size//8), 0, 0, 180, (0, 0, 0), 2, cv2.LINE_AA)


# Mapping name -> function for advanced shapes
SHAPE_FUNCTIONS = {
    'cat': draw_cat,
    'dog': draw_dog,
    'heart': draw_heart,
    'star': draw_star,
    'flower': draw_flower,
    'sun': draw_sun,
    'cloud': draw_cloud,
    'butterfly': draw_butterfly,
    'fish': draw_fish,
}


def stamp_shape(canvas, shape, center, size, color, erasing=False, thickness=-1):
    """Stamp a shape onto the canvas at center with given size and color.
    thickness=-1 fills the shape; set to >0 for outline.
    """
    # If shape is one of the complex shape stamps, dispatch to its function
    try:
        SHAPE_FUNCTIONS
    except NameError:
        # no-op if mapping not yet defined
        pass
    else:
        if shape in SHAPE_FUNCTIONS:
            draw_fn = SHAPE_FUNCTIONS[shape]
            # KEY FIX: Use the actual brush color (not white). Black only when erasing.
            draw_color = (0, 0, 0) if erasing else color
            # Debug: print stamping info to console so user can confirm color used
            try:
                print(f"Stamping shape='{shape}' at={center} size={size} color={draw_color} erasing={erasing}")
            except Exception:
                pass
            # Render advanced shapes as outlines so users can trace them (therapy-friendly).
            try:
                temp = np.zeros_like(canvas)
                # draw the shape filled in white on the temporary layer
                draw_fn(temp, center, size, (255, 255, 255))
                gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                outline_thickness = max(3, int(size * 0.06))
                outline_color = (255, 255, 255) if erasing else draw_color

                # If the chosen color is very light, draw a thin dark rim first for contrast
                if not erasing and isinstance(draw_color, tuple) and len(draw_color) == 3:
                    brightness = sum(int(x) for x in draw_color)
                    if brightness > 600:
                        for cnt in contours:
                            cv2.drawContours(canvas, [cnt], -1, (0, 0, 0), outline_thickness + 2, cv2.LINE_AA)

                # Draw the actual outline (contours) in the selected color (or white when erasing)
                for cnt in contours:
                    cv2.drawContours(canvas, [cnt], -1, outline_color, outline_thickness, cv2.LINE_AA)

                # Draw a visible color swatch at bottom-left so the user can confirm stamp color
                h, w = canvas.shape[:2]
                sw_x1, sw_y1 = 10, h - 60
                sw_x2, sw_y2 = 60, h - 20
                sw_color = (255, 255, 255) if erasing else draw_color
                cv2.rectangle(canvas, (sw_x1, sw_y1), (sw_x2, sw_y2), sw_color, -1, cv2.LINE_AA)
                cv2.rectangle(canvas, (sw_x1, sw_y1), (sw_x2, sw_y2), (0, 0, 0), 2, cv2.LINE_AA)
                try:
                    txt = f"{sw_color[0]},{sw_color[1]},{sw_color[2]}"
                    cv2.putText(canvas, txt, (sw_x2 + 8, sw_y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220), 1, cv2.LINE_AA)
                except Exception:
                    pass
            except Exception:
                # Fallback: draw filled shape if outline rendering fails
                try:
                    draw_fn(canvas, center, size, draw_color)
                except Exception:
                    pass
            return
    # Use black fill/outline for shapes; when not erasing, use semi-transparent black for filled shapes
    black = (0, 0, 0)
    # For simple shapes: draw outline-only when stamping (thickness < 0) so users can trace them.
    if thickness < 0 and not erasing:
        outline_thickness = max(2, int(size * 0.06))
        outline_color = color
        if shape == "circle":
            cv2.circle(canvas, center, size, outline_color, outline_thickness)
        elif shape == "square":
            top_left = (center[0] - size, center[1] - size)
            bottom_right = (center[0] + size, center[1] + size)
            cv2.rectangle(canvas, top_left, bottom_right, outline_color, outline_thickness)
        elif shape == "triangle":
            pts = np.array([
                [center[0], center[1] - size],
                [center[0] - size, center[1] + size],
                [center[0] + size, center[1] + size]
            ], np.int32)
            cv2.polylines(canvas, [pts], isClosed=True, color=outline_color, thickness=outline_thickness)
        elif shape == "line":
            pt1 = (center[0] - size, center[1])
            pt2 = (center[0] + size, center[1])
            cv2.line(canvas, pt1, pt2, outline_color, max(2, outline_thickness))
    else:
        # Solid fill or erasing behavior (when erasing, we still fill with white to remove)
        draw_color = black if not erasing else (255, 255, 255)
        if shape == "circle":
            if thickness < 0:
                cv2.circle(canvas, center, size, draw_color, -1)
            else:
                cv2.circle(canvas, center, size, draw_color, thickness)
        elif shape == "square":
            top_left = (center[0] - size, center[1] - size)
            bottom_right = (center[0] + size, center[1] + size)
            if thickness < 0:
                cv2.rectangle(canvas, top_left, bottom_right, draw_color, -1)
            else:
                cv2.rectangle(canvas, top_left, bottom_right, draw_color, thickness)
        elif shape == "triangle":
            pts = np.array([
                [center[0], center[1] - size],
                [center[0] - size, center[1] + size],
                [center[0] + size, center[1] + size]
            ], np.int32)
            if thickness < 0:
                cv2.fillPoly(canvas, [pts], draw_color)
            else:
                cv2.polylines(canvas, [pts], isClosed=True, color=draw_color, thickness=thickness)
        elif shape == "line":
            pt1 = (center[0] - size, center[1])
            pt2 = (center[0] + size, center[1])
            cv2.line(canvas, pt1, pt2, draw_color, max(2, thickness if thickness>0 else 2))


def main():
    print("🎨 Starting Enhanced Virtual Painter with AI...")
    print("📹 Initializing camera...")
    # Use global runtime color settings so keyboard handlers modify them
    global brush_alpha, rainbow_mode, hue_base, PALETTE
    global mouse_click_pos
    shapes = [
        "circle", "square", "triangle", "line",
        "cat", "dog", "heart", "star", "flower", "sun", "cloud", "butterfly", "fish"
    ]
    current_shape_idx = 0
    current_mode = "canvas"   # start in free-draw mode
    show_shape = False
    brush_size = 5            # default brush size

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
    brush_name = "Red"
    erasing = False
    brush_type = "round"  # default brush

    # Tracking
    mode = 'hand' if MP_AVAILABLE else 'color'
    hand_tracker = HandTracker() if MP_AVAILABLE else None
    color_tracker = ColorTracker()
    
    # Support up to two pointers for multi-user (per-hand) drawing
    prev_points = [None, None]
    last_saved_path = None
    last_ai_cartoon = None
    show_help = True
    
    print(f"✅ Ready! Mode: {mode.upper()}")
    print("💡 Press 'h' to toggle help panel")
    print("🤖 Press 'a' to generate AI cartoon from your drawing!")
    
    # Create named window and register mouse callback for color picking
    wnd_name = "🎨 AI-Enhanced Virtual Painter"
    cv2.namedWindow(wnd_name, cv2.WINDOW_NORMAL)

    def _mouse_cb(event, x, y, flags, param):
        global mouse_click_pos
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_click_pos = (x, y)

    cv2.setMouseCallback(wnd_name, _mouse_cb)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # Mirror effect
        
        # Create white canvas for final output
        white_canvas = np.full_like(frame, 255)
        
        # Get drawing positions (support two pointers in hand mode)
        current_point = None
        is_drawing = False
        pointers = [None, None]  # up to two (x,y,is_drawing)

        if mode == 'hand' and hand_tracker:
            hand_results = hand_tracker.get_finger_positions(frame)
            # Define per-hand display colors (hand 0 uses brush_color, hand 1 a secondary color)
            hand_display_colors = [brush_color, (0, 200, 0)]

            for i in range(min(2, len(hand_results))):  # Process up to 2 hands
                x, y, drawing = hand_results[i]
                pointers[i] = (x, y, drawing)
                
                # Draw finger tracking indicator for each hand
                disp_color = hand_display_colors[i] if drawing else (100, 100, 100)
                
                # Draw filled circle when drawing, hollow when not
                if drawing:
                    cv2.circle(frame, (x, y), 10, disp_color, -1)
                    cv2.circle(frame, (x, y), 12, (255, 255, 255), 2)
                    # Show brush preview
                    cv2.circle(frame, (x, y), brush_size + 5, disp_color, 2)
                    # Add "DRAWING" label
                    label = f"Hand {i+1} DRAW"
                    cv2.putText(frame, label, (x + 15, y - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, disp_color, 2)
                else:
                    cv2.circle(frame, (x, y), 8, disp_color, 2)
                    # Add "IDLE" label
                    label = f"Hand {i+1}"
                    cv2.putText(frame, label, (x + 15, y - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            # Clear unused pointer slots
            for i in range(len(hand_results), 2):
                pointers[i] = None
        else:  # Color tracking mode
            result = color_tracker.get_position(frame)
            if result:
                current_point = result
                is_drawing = True  # Always drawing in color mode

                # Draw tracking indicator
                cv2.circle(frame, current_point, 10, (0, 255, 0), -1)
                cv2.circle(frame, current_point, brush_size + 5, (255, 255, 255), 2)
        
        # Handle drawing for either color tracking (single pointer) or hand multi-pointer
        if mode == 'hand' and hand_tracker:
            # Use pointers list for up to 2 hands
            for i, p in enumerate(pointers):
                if p and p[2]:
                    x, y, _ = p
                    # Check for toolbar interaction
                    if y <= 100:
                        prev_points[i] = None
                        continue

                    # Per-hand brush color (hand 0 uses current brush_color)
                    hand_color = brush_color if i == 0 else (0, 200, 0)

                    if prev_points[i] and prev_points[i][1] > 100:
                        # Always allow freehand drawing even when a shape overlay is shown.
                        smooth_brush_stroke(paint_canvas, prev_points[i], (x, y), hand_color, brush_size, erasing, alpha=brush_alpha, rainbow=rainbow_mode)
                    prev_points[i] = (x, y)
                else:
                    prev_points[i] = None
        else:
            # Color tracking / single-pointer mode
            if current_point and is_drawing:
                x, y = current_point
                if y <= 100:
                    prev_points = [None, None]
                else:
                    if prev_points[0] and prev_points[0][1] > 100:
                        # Allow freehand tracing when shape overlay is active instead of auto-stamping
                        smooth_brush_stroke(paint_canvas, prev_points[0], current_point, brush_color, brush_size, erasing, alpha=brush_alpha, rainbow=rainbow_mode)
                    prev_points[0] = current_point
            else:
                prev_points = [None, None]
        
        # Create final frame
        output_frame = overlay_paint_on_canvas(white_canvas, paint_canvas)
        
        # Add toolbar
        toolbar, color_buttons, eraser_rect, ai_rect = draw_enhanced_toolbar(
            output_frame, brush_color, brush_size, erasing, mode, last_saved_path)
        toolbar_height = toolbar.shape[0]
        output_frame[:toolbar_height] = toolbar

        # Handle mouse clicks (from mouse callback) and hand-pointer clicks on toolbar
        # Mouse click handler sets mouse_click_pos as (x,y) when user clicks
        if mouse_click_pos is not None:
            mx, my = mouse_click_pos
            res = handle_toolbar_click(mx, my, color_buttons, eraser_rect, ai_rect)
            mouse_click_pos = None
            if res:
                action, col, name = res
                if action == 'color':
                    brush_color = tuple(int(c) for c in col)
                    brush_name = name
                    erasing = True if name.lower() == 'eraser' or brush_color == (255,255,255) else False
                elif action == 'eraser':
                    erasing = True

        # Hand-pointer selection: tap while pointer is in toolbar region
        if mode == 'hand' and hand_tracker:
            for p in pointers:
                if p:
                    px, py, pd = p
                    if py <= toolbar_height and pd:
                        res = handle_toolbar_click(px, py, color_buttons, eraser_rect, ai_rect)
                        if res:
                            action, col, name = res
                            if action == 'color':
                                brush_color = tuple(int(c) for c in col)
                                brush_name = name
                                erasing = True if name.lower() == 'eraser' or brush_color == (255,255,255) else False
                            elif action == 'eraser':
                                erasing = True

        
        # Add camera preview (small)
        preview_size = (200, 150)
        preview = cv2.resize(frame, preview_size)
        ph, pw = output_frame.shape[:2]
        output_frame[ph-preview_size[1]-10:ph-10, pw-preview_size[0]-10:pw-10] = preview
        
        # Show help panel
        if show_help:
            draw_help_panel(output_frame)
        # Add this after the drawing logic and before cv2.imshow
        # Draw pointer(s) on canvas
        if mode == 'hand' and hand_tracker:
            for i, p in enumerate(pointers):
                if p:
                    px, py, pd = p
                    hand_color = brush_color if i == 0 else (0, 200, 0)
                    draw_canvas_pointer(output_frame, (px, py), brush_size, hand_color, pd)
        else:
            draw_canvas_pointer(output_frame, current_point, brush_size, brush_color, is_drawing)
        if show_shape:
            # Only show overlay guide for simple shapes (circle, square, triangle, line)
            simple_shapes = ["circle", "square", "triangle", "line"]
            cur_shape = shapes[current_shape_idx]
            if cur_shape in simple_shapes:
                output_frame = draw_shape_overlay(output_frame, cur_shape, alpha=0.4)
            else:
                # For advanced shapes (cat, dog, heart, ...), render a traceable outline preview
                try:
                    h_out, w_out = output_frame.shape[:2]
                    size_preview = min(h_out, w_out) // 4
                    # Place preview so the top of the shape is below the toolbar
                    center_x = w_out // 2
                    preferred_y = toolbar_height + size_preview + 10
                    center_y = min(max(preferred_y, toolbar_height + 10), h_out - size_preview - 10)
                    center = (center_x, center_y)

                    # Render the advanced shape onto a temporary layer and extract contours
                    temp = np.zeros_like(output_frame)
                    draw_fn = SHAPE_FUNCTIONS.get(cur_shape)
                    if draw_fn is not None:
                        # draw filled white on temp so contours are detectable
                        draw_fn(temp, center, size_preview, (255, 255, 255))
                        gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
                        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
                        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        # Choose a preview color visible over the toolbar/background
                        try:
                            preview_col = brush_color
                            if isinstance(preview_col, tuple) and sum(int(x) for x in preview_col) > 600:
                                # very light selected color -> use cyan for visibility
                                preview_col = (0, 255, 255)
                        except Exception:
                            preview_col = (0, 255, 255)

                        overlay = output_frame.copy()
                        th = max(2, int(size_preview * 0.04))
                        for cnt in contours:
                            cv2.drawContours(overlay, [cnt], -1, preview_col, th, cv2.LINE_AA)

                        # Blend the outline preview onto the output frame with low opacity
                        output_frame = cv2.addWeighted(overlay, 0.25, output_frame, 0.75, 0)
                except Exception:
                    # If preview fails, silently continue (no preview shown)
                    pass

            # Draw current shape label under the toolbar for clear feedback
            try:
                label = f"SHAPE: {shapes[current_shape_idx].upper()} (press SPACE to stamp)"
                cv2.putText(output_frame, label, (20, toolbar_height + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            except Exception:
                pass


        # Display
        cv2.imshow(wnd_name, output_frame)

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

        # Eyedropper: sample color under current pointer
        elif key == ord('i'):
            sample_pt = None
            if current_point and current_point[1] > 100:
                sample_pt = current_point
            elif pointers and pointers[0]:
                px, py, _ = pointers[0]
                if py > 100:
                    sample_pt = (px, py)
            if sample_pt:
                sx, sy = sample_pt
                if 0 <= sy < paint_canvas.shape[0] and 0 <= sx < paint_canvas.shape[1]:
                    brush_color = tuple(int(c) for c in paint_canvas[sy, sx])
                    erasing = False
                    print(f"🎯 Eyedropper picked color: {brush_color} at {sample_pt}")
                else:
                    print("⚠️ Eyedropper: sample point out of canvas")
            else:
                print("⚠️ Eyedropper: no pointer detected")

        # Brush opacity control
        elif key == ord(','):
            brush_alpha = max(0.05, round(brush_alpha - 0.1, 2))
            print(f"Opacity: {brush_alpha:.2f}")
        elif key == ord('.'):
            brush_alpha = min(1.0, round(brush_alpha + 0.1, 2))
            print(f"Opacity: {brush_alpha:.2f}")

        # Rainbow brush toggle
        elif key == ord('t'):
            rainbow_mode = not rainbow_mode
            print("🌈 Rainbow mode:" , "ON" if rainbow_mode else "OFF")

        # Palette save/load (shift+S / shift+L)
        elif key == ord('S'):
            try:
                with open('palette.json', 'w') as f:
                    json.dump([[(int(c[0][0]), int(c[0][1]), int(c[0][2])), c[1], c[2]] for c in PALETTE], f)
                print("💾 Palette saved to palette.json")
            except Exception as e:
                print("❌ Failed to save palette:", e)
        elif key == ord('L'):
            try:
                with open('palette.json', 'r') as f:
                    data = json.load(f)
                PALETTE.clear()
                for item in data:
                    PALETTE.append((tuple(item[0]), item[1], item[2]))
                print("📂 Palette loaded from palette.json")
            except Exception as e:
                print("❌ Failed to load palette:", e)

        elif key in (ord('n'), ord('N')):  # Next shape
            current_shape_idx = (current_shape_idx + 1) % len(shapes)
            print(f"✏ Shape: {shapes[current_shape_idx]}")

        elif key in (ord('b'), ord('B')):  # Previous shape
            current_shape_idx = (current_shape_idx - 1) % len(shapes)
            print(f"✏ Shape: {shapes[current_shape_idx]}")

        elif key in (ord('f'), ord('F')):  # Toggle shape ON/OFF
            if current_mode == "canvas":
                current_mode = shapes[current_shape_idx]  # enter shape mode
                show_shape = True
                print(f"👁 Shape Mode: {current_mode}")
            else:
                current_mode = "canvas"  # return to free drawing
                show_shape = False
                print("🙈 Back to Canvas")

        # Stamp shape manually with SPACE when shape overlay is visible
        elif key == ord(' '):
            if show_shape:
                # Prefer stamping at current pointer if available and in drawing area
                if current_point and current_point[1] > 100:
                    center_pt = current_point
                else:
                    center_pt = (w // 2, h // 2 + 50)

                size = min(h, w) // 4
                stamp_shape(paint_canvas, shapes[current_shape_idx], center_pt, size, brush_color, erasing)
                print(f"🖼 Stamped shape: {shapes[current_shape_idx]} at {center_pt}")

        elif key == 27:  # ESC → always return to canvas
            current_mode = "canvas"
            show_shape = False
            print("🔙 Reset to Canvas")



        elif key in [ord(str(i)) for i in range(10)] or key in [ord('e'), ord('E')]:
            # Color selection driven by runtime PALETTE keys (allows reordering and persistence)
            pressed = chr(key).upper()
            selected = None
            for col, name, k in PALETTE:
                if k.upper() == pressed:
                    selected = (col, name, k)
                    break

            if selected:
                col, name, k = selected
                brush_color = tuple(int(c) for c in col)
                # If selected palette entry is named 'Eraser' or color is white, enable erasing
                erasing = True if name.lower() == 'eraser' or brush_color == (255,255,255) else False
                print(f"🎨 Selected: {name} -> {brush_color} | Eraser:{erasing}")
            else:
                print(f"⚠️ No palette entry for key '{pressed}'")
        
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
