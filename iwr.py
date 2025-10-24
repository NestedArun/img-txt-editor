import cv2
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Path to tesseract executable (if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def get_text_boxes(image_path):
    # Load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect text using pytesseract
    boxes = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    return image, boxes

def replace_text(image_path, search_word, replace_word, output_path):
    image, boxes = get_text_boxes(image_path)
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    for i, word in enumerate(boxes['text']):
        if word.strip() == search_word:
            # Get bounding box
            x, y, w, h = boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i]

            # Sample background color from the box region
            roi = image[y:y+h, x:x+w]
            mean_color = tuple(map(int, cv2.mean(roi)[:3]))

            # Cover the old text with background color
            draw.rectangle([x, y, x+w, y+h], fill=mean_color)

            # Approximate font size
            font_size = int(h * 0.8)
            font = ImageFont.truetype("arial.ttf", font_size)

            # Write new text
            draw.text((x, y), replace_word, fill=(0,0,0), font=font)  # default black text

    pil_image.save(output_path)
    print(f"Saved edited image as {output_path}")

# Example usage
replace_text("example.png", "Yashwanth.", "arun", "output.png")
