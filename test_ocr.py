import pytesseract
from PIL import Image, ImageDraw

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
# pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'  # macOS
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Linux

# Create test image
img = Image.new('RGB', (300, 100), color=(255, 255, 255))
draw = ImageDraw.Draw(img)
draw.text((10, 40), "OCR Test Successful!", fill=(0, 0, 0))
img.save("test.png")

# Run OCR
text = pytesseract.image_to_string(img)
print(f"Recognized text: {text}")