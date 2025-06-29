# EasyRead
EasyRead is a tool made with Python which automatically detects text on screens and highlights relevant keywords to enhance the casual reading experience. It is made for the [BWKI Österreich](https://bwki.asai.ac.at/) competition.
## Workflow
 1. A screenshot is taken using **mss**.
 2. Text is detected on the screen using **Pytesseract**, which is a Python interface for **Tesseract OCR**, as well as finding text groups like paragraphs.
 3. Contextually relevant keywords of the text are found through the usage of an LLM with **transformers**.
 4. The keywords are underlined on the screen through an overlay using **PyQt**.
## AI Usage
 - Text detection and recognition with **Tesseract**.
 - Keyword extraction with **transformers**-based LLMs.
## Languages
- Python 3.13