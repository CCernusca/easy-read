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
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
# pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'  # macOS
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Linux

class KeywordDetector:
    def __init__(self, model_path="./models/flan-t5-small"):
        # Load tokenizer from local directory
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model from local directory
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Create pipeline with local model and tokenizer
        self.keyword_pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16
        )

        self.prompt = """
        You are a expert linguist, with decades of experience in extracting key information and keywords from text. 
        Extract the 1-3 most important keywords from the following text. Keywords should be: nouns, verbs, or named entities. 
        Output the keywords in the language of the original text as a space-separated list.
        
        "{text}"
        """
    
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
        
        # Clean up keywords
        keywords = [kw.strip(" .,_´`-")  for kw in results[0]['generated_text'].split()]
        # Basic plural is added manually, as the model seems to struggle with it
        for keyword in keywords.copy():
            keywords.append(f"{keyword}s")
        return keywords

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

        # Minimum word count for a paragraph to be considered meaningful
        self.text_paragraph_word_count = 10
        
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
        paragraphs = {}
        for i in range(len(data['text'])):
            conf = int(float(data['conf'][i]))
            text = data['text'][i].strip()
            level = data['level'][i]
            # If confidence is high enough and text is just a random character or empty
            if conf > self.confidence_threshold and len(text.strip()) > 1:
                # Word saving
                word = {list(data.keys())[j]: v[i] for j, v in enumerate(data.values())}
                # Apply correction factor to position data (OCR error as well as scaling down)
                word["left"] = int(word["left"] * 0.8 / self.scaling_factor)
                word["top"] = int(word["top"] * 0.8 / self.scaling_factor)
                word["width"] = int(word["width"] * 0.8 / self.scaling_factor)
                word["height"] = int(word["height"] * 0.8 / self.scaling_factor)
                
                # Paragraph grouping
                if level == 5:  # Word level
                    block = word['block_num']
                    para = word['par_num']
                    key = (block, para)
                    
                    if key not in paragraphs:
                        paragraphs[key] = {'text': [], 'bboxes': [], 'words': []}
                    
                    x, y, w, h = word['left'], word['top'], word['width'], word['height']
                    paragraphs[key]['bboxes'].append((x, y, w, h))
                    paragraphs[key]['text'].append(text)
                    paragraphs[key]['words'].append(word)
        
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
                'words': para['words'],
                'text': full_text,
                'position': (x_min, y_min, x_max - x_min, y_max - y_min),
                'keywords': [],
                'text_paragraph': len(para['text']) > self.text_paragraph_word_count  # Whether the paragraph is meaningful
            })
        
        # Keyword detection
        keyword_start = time.time()
        for para in processed_paragraphs:
            if para['text_paragraph']:  # Only process meaningful paragraphs
                try:
                    keywords = self.keyword_detector.extract_keywords(para['text'])
                    para['keywords'] = keywords
                except Exception as e:
                    raise Exception(f"Keyword extraction failed: {e}")
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
            raise Exception(f"Detection error: {e}")
            self.last_process_time = time.time() - update_start

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.setFont(self.paragraph_font)

        # Save words to underline
        highlights = []

        # Draw paragraphs, their keywords and words:
        for para in self.paragraphs:
            px, py, pw, ph = para['position']

            # Draw words
            for word in para['words']:
                wx, wy, ww, wh = word['left'], word['top'], word['width'], word['height']

                # Draw word bounding box, color is based on if it is a keyword
                if word['text'] in para['keywords']:
                    painter.setPen(QColor(255, 0, 0, 100))  # Keywords are red
                    painter.drawRect(wx, wy, ww, wh)

                    # Draw text label
                    text = word['text'][:50] + "..." if len(word['text']) > 50 else word['text']
                    painter.drawText(wx, wy - 5, text)

                    # Draw confidence
                    painter.drawText(wx + ww - 10, wy - 5, f"{word['conf']}%")

                    # Save word to underline
                    highlights.append((wx, wy, ww, wh))

                elif not word['text'] in para['keywords']:
                    painter.setPen(QColor(0, 0, 255, 100))  # Other words are blue
                    painter.drawRect(wx, wy, ww, wh)

                    # Draw text label
                    text = word['text'][:50] + "..." if len(word['text']) > 50 else word['text']
                    painter.drawText(wx, wy - 5, text)

                    # Draw confidence
                    painter.drawText(wx + ww - 10, wy - 5, f"{word['conf']}%")

            # Draw paragraph bounding box, color is based on if it is a meaningful paragraph
            if para['text_paragraph']:
                painter.setPen(QColor(0, 255, 0, 220))  # Meaningful paragraphs are green
            else:
                painter.setPen(QColor(0, 100, 0, 220))  # Meaningless paragraphs are dark green
            painter.drawRect(px, py, pw, ph)
            
            # Draw text label
            text = ", ".join(para['keywords'])
            painter.drawText(px, py - 10, text)
        
        # Underline highlighted words
        for highlight in highlights:
            painter.setPen(QColor(255, 0, 0, 255))
            painter.drawLine(highlight[0] - highlight[2] // 10, highlight[1] + highlight[3], highlight[0] + highlight[2] + highlight[2] // 10, highlight[1] + highlight[3])
        
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