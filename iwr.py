import cv2
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def get_text_boxes(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    return image, boxes

def get_text_color_and_bold(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(mask) == 0:
        text_color = (0,0,0)
    else:
        text_color = tuple(map(int, cv2.mean(roi, mask=mask)[:3]))
   
    density = np.count_nonzero(mask) / (mask.shape[0]*mask.shape[1])
    is_bold = density > 0.5
    return text_color, is_bold

def find_phrase_positions(boxes, phrase):
    words = phrase.split()
    positions = []
    i = 0
    while i < len(boxes['text']):
        match = True
        for j, w in enumerate(words):
            if i+j >= len(boxes['text']) or boxes['text'][i+j].strip() != w:
                match = False
                break
        if match:
            positions.append((i, i+len(words)-1))
            i += len(words)
        else:
            i += 1
    return positions

def replace_phrase(image_path, search_phrase, replace_phrase, output_path):
    image, boxes = get_text_boxes(image_path)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    positions = find_phrase_positions(boxes, search_phrase)
    colors_bold = []

    for start, end in positions:
        x1 = boxes['left'][start]
        y1 = boxes['top'][start]
        x2 = boxes['left'][end] + boxes['width'][end]
        y2 = boxes['top'][end] + boxes['height'][end]

        h = y2 - y1
        y1 = max(0, y1 - int(h*0.15))
        y2 = min(image.shape[0], y2 + int(h*0.15))

        roi = image[y1:y2, x1:x2]
        color, is_bold = get_text_color_and_bold(roi)
        colors_bold.append((color, is_bold))

        mask[y1:y2, x1:x2] = 255

    
    inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    pil_image = Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

   
    for idx, (start, end) in enumerate(positions):
        x1 = boxes['left'][start]
        y1 = boxes['top'][start]
        x2 = boxes['left'][end] + boxes['width'][end]
        y2 = boxes['top'][end] + boxes['height'][end]

        h = y2 - y1
        y1 = max(0, y1 - int(h*0.15))
        y2 = min(image.shape[0], y2 + int(h*0.15))
        font_size = int((y2 - y1) * 0.8)

        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

        color, is_bold = colors_bold[idx]

        stroke = 1 if is_bold else 0
        draw.text((x1, y1), replace_phrase, fill=color, font=font, stroke_width=stroke, stroke_fill=color)

    pil_image.save(output_path)
    print(f"Saved edited image as {output_path}")

replace_phrase("example.png", "Yashwanth.", "Arun", "output.png")
