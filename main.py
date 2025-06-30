import sys
import time
import pyautogui
import pytesseract
import torch
import ctypes
from mss import mss
from PIL import Image, ImageFilter
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QColor, QFont, QFontMetrics, QImage
from transformers import pipeline

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
# pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'  # macOS
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Linux

class KeywordDetector:
    def __init__(self):
        self.keyword_pipe = pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16
        )
        self.prompt = """Extract 1-3 most important keywords from the text. 
        Keywords should be: nouns, verbs, or named entities. 
        Output as comma-separated list.
        
        Text: "{text}"
        Keywords:"""
    
    def extract_keywords(self, text, max_length=50):
        """Keyword extraction"""
        
        truncated = text[:1000] if len(text) > 1000 else text
        prompt = self.prompt.format(text=truncated)
        
        results = self.keyword_pipe(
            prompt,
            max_length=max_length,
            num_beams=3,
            early_stopping=True
        )
        
        keywords = [kw.strip() for kw in results[0]['generated_text'].split(",")]
        return keywords[:3]

def find_keyword_positions(paragraph_text, keywords):
    """Locate keywords within paragraph text"""
    positions = []
    for keyword in keywords:
        start = 0
        while True:
            idx = paragraph_text.lower().find(keyword.lower(), start)
            if idx == -1:
                break
            positions.append((idx, idx + len(keyword)))
            start = idx + len(keyword)
    return positions

class TextOverlay(QWidget):
    def __init__(self):
        super().__init__()
        
        # PyQt6 specific setup
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool |
            Qt.WindowType.BypassWindowManagerHint |
            Qt.WindowType.X11BypassWindowManagerHint |
            Qt.WindowType.WindowDoesNotAcceptFocus |
            Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background:transparent;")
        
        # Get screen dimensions
        screen = QApplication.primaryScreen().availableGeometry()
        self.setGeometry(screen)
        
        # Initialize components
        self.keyword_detector = KeywordDetector()
        
        # Text detection state
        self.words = []
        self.paragraphs = []
        
        # Performance tracking
        self.last_process_time = 0
        self.last_ocr_time = 0
        self.last_llm_time = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Fonts for rendering
        self.paragraph_font = QFont("Arial", 5)
        self.stats_font = QFont("Consolas", 7)
        self.stats_font.setBold(True)
        
        # Setup update timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_text)
        self.timer.start(1000)  # Update every 1.0 seconds

        # Factor for scaling down image for speed at the cost of accuracy
        self.scaling_factor = 0.75

        # Word detection confidence threshold
        self.confidence_threshold = 0
        
    def capture_screen(self):
        """Capture screen region using MSS"""
        with mss() as sct:
            # Prevent overlay from being captured
            hwnd = int(overlay.winId())
            user32 = ctypes.windll.user32
            user32.SetWindowDisplayAffinity(hwnd, 0x11)

            # Capture primary screen
            monitor = sct.monitors[1]
            screenshot = sct.grab(monitor)

            # Make overlay capurable again, for user screenshots
            user32.SetWindowDisplayAffinity(hwnd, 0x00)

            # Scale down image for performance
            return Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX").resize((int(screenshot.width * self.scaling_factor), int(screenshot.height  * self.scaling_factor)))
    
    def get_text(self, image):
        """Detect paragraphs and extract keywords"""
        start_time = time.time()
        
        # Preprocess image for better OCR
        image = image.convert('L')  # Grayscale
        image = image.filter(ImageFilter.SHARPEN)
        
        # OCR processing
        data = pytesseract.image_to_data(
            image, 
            output_type=pytesseract.Output.DICT,
            config='--psm 3 --oem 3'  # Fully automatic page segmentation, LSTM OCR
        )
        ocr_time = time.time() - start_time
        
        # Save words of text & Group text into paragraphs
        self.words.clear()
        paragraphs = {}
        for i in range(len(data['text'])):
            conf = int(float(data['conf'][i]))
            text = data['text'][i].strip()
            level = data['level'][i]
            # If confidence is high enough and text is just a random character or empty
            if conf > self.confidence_threshold and len(text.strip()) > 1:
                # Word saving
                self.words.append({list(data.keys())[j]: v[i] for j, v in enumerate(data.values())})
                # Apply correction factor to position data (OCR error as well as scaling down)
                self.words[-1]["left"] = int(self.words[-1]["left"] * 0.8 / self.scaling_factor)
                self.words[-1]["top"] = int(self.words[-1]["top"] * 0.8 / self.scaling_factor)
                self.words[-1]["width"] = int(self.words[-1]["width"] * 0.8 / self.scaling_factor)
                self.words[-1]["height"] = int(self.words[-1]["height"] * 0.8 / self.scaling_factor)
                
                # Paragraph grouping
                if level == 5:  # Word level
                    block = self.words[-1]['block_num']
                    para = self.words[-1]['par_num']
                    key = (block, para)
                    
                    if key not in paragraphs:
                        paragraphs[key] = {'text': [], 'bboxes': []}
                    
                    x, y, w, h = self.words[-1]['left'], self.words[-1]['top'], self.words[-1]['width'], self.words[-1]['height']
                    paragraphs[key]['bboxes'].append((x, y, w, h))
                    paragraphs[key]['text'].append(text)
        print(self.words)
        
        # Paragraph processing
        processed_paragraphs = []
        for key, para in paragraphs.items():
            bboxes = para['bboxes']
            x_min = min(word[0] for word in bboxes)
            y_min = min(word[1] for word in bboxes)
            x_max = max(word[0] + word[2] for word in bboxes)
            y_max = max(word[1] + word[3] for word in bboxes)
            
            full_text = ' '.join(para['text'])
            processed_paragraphs.append({
                'word_count': len(para['text']),
                'text': full_text,
                'position': (x_min, y_min, x_max - x_min, y_max - y_min),
                'keywords': [],
                'keyword_positions': []
            })
        
        # Keyword detection
        keyword_start = time.time()
        for para in processed_paragraphs:
            if para['word_count'] > 10:  # Only process meaningful paragraphs
                try:
                    keywords = self.keyword_detector.extract_keywords(para['text'])
                    para['keywords'] = keywords
                    para['keyword_positions'] = find_keyword_positions(para['text'], keywords)
                except Exception as e:
                    print(f"Keyword extraction failed: {e}")
                    para['keywords'] = []
                    para['keyword_positions'] = []
        llm_time = time.time() - keyword_start

        return processed_paragraphs, ocr_time, llm_time

    def update_text(self):
        self.frame_count += 1
        update_start = time.time()
        
        try:
            
            # Process screen
            img = self.capture_screen()
            paragraphs, ocr_time, llm_time = self.get_text(img)
            self.paragraphs = paragraphs
            
            # Update performance metrics
            self.last_ocr_time = ocr_time
            self.last_llm_time = llm_time
            self.last_process_time = time.time() - update_start
            
            self.update()
        except Exception as e:
            print(f"Detection error: {e}")
            self.last_process_time = time.time() - update_start

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.setFont(self.paragraph_font)

        # Draw words
        for word in self.words:
            x, y, w, h = word['left'], word['top'], word['width'], word['height']

            # Draw word bounding box
            painter.setPen(QColor(255, 80, 80, 100))
            painter.drawRect(x, y, w, h)

            # Draw text label
            text = word['text'][:50] + "..." if len(word['text']) > 50 else word['text']
            painter.drawText(x, y - 5, text)

            # Draw confidence
            painter.setPen(QColor(0, 200, 0, 90))
            painter.drawText(x + w - 10, y - 5, f"{word['conf']}%")
        
        # Draw paragraphs
        for para in self.paragraphs:
            x, y, w, h = para['position']
            
            # Draw paragraph bounding box
            painter.setPen(QColor(80, 80, 255, 220))
            painter.drawRect(x, y, w, h)
            
            # Draw text label
            print(para)
            text = ", ".join(para['keywords'])
            painter.drawText(x, y - 15, text)
            
            # # Draw keywords if available
            # if 'keyword_positions' in para:
            #     self.draw_keywords(painter, para, x, y + h + 5)
        
        # Draw performance stats last (so it stays on top)
        self.draw_performance_stats(painter)
    
    def draw_keywords(self, painter, para, start_x, start_y):
        """Render paragraph text with underlined keywords"""
        text = para['text']
        positions = para.get('keyword_positions', [])
        
        # Set up text rendering
        fm = painter.fontMetrics()
        current_x = start_x
        
        # Split text into segments
        last_pos = 0
        segments = []
        
        for start, end in positions:
            # Add normal text before keyword
            if start > last_pos:
                segments.append({
                    'text': text[last_pos:start],
                    'underline': False
                })
            
            # Add keyword
            segments.append({
                'text': text[start:end],
                'underline': True
            })
            
            last_pos = end
        
        # Add remaining text
        if last_pos < len(text):
            segments.append({
                'text': text[last_pos:],
                'underline': False
            })
        
        # Render segments
        for segment in segments:
            text_segment = segment['text']
            width = fm.horizontalAdvance(text_segment)  # PyQt6 method
            
            # Draw text
            painter.setPen(QColor(255, 255, 255, 220))
            painter.drawText(current_x, start_y, text_segment)
            
            # Draw underline for keywords
            if segment['underline']:
                painter.setPen(QColor(255, 100, 100, 255))
                underline_y = start_y + 3
                painter.drawLine(
                    current_x, 
                    underline_y,
                    current_x + width,
                    underline_y
                )
            
            current_x += width

    def draw_performance_stats(self, painter):
        """Render performance metrics at bottom left corner"""
        stats_y = self.height() - 20
        stats_x = 10
        
        # Set font and colors
        painter.setFont(self.stats_font)
        painter.setPen(QColor(255, 255, 255, 220))
        painter.setBrush(QColor(0, 0, 0, 180))
        
        # Calculate metrics
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        # Create stats text
        stats_text = [
            f"Frame: {self.frame_count}",
            f"FPS: {fps:.1f}",
            f"Process: {self.last_process_time*1000:.1f}ms",
            f"OCR: {self.last_ocr_time*1000:.1f}ms",
            f"LLM: {self.last_llm_time*1000:.1f}ms"
        ]
        text_block = "  |  ".join(stats_text)
        
        # Get text width
        fm = QFontMetrics(self.stats_font)
        text_width = fm.horizontalAdvance(text_block) + 20
        
        # Draw background
        painter.drawRect(stats_x - 5, stats_y - 20, text_width, 25)
        
        # Draw text
        painter.drawText(stats_x, stats_y, text_block)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    overlay = TextOverlay()
    overlay.show()
    sys.exit(app.exec())