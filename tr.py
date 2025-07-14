"""import os

try:
    os.startfile(r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE")
    print("✅ Excel launched successfully")
except Exception as e:
    print("❌ Error:", e)
"""

import pytesseract
from PIL import Image

# Update path to tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract\tesseract.exe'#"D:\Tesseract\tesseract.exe"

img = Image.open('hello_tesseract.png')
text = pytesseract.image_to_string(img)
print(text)
