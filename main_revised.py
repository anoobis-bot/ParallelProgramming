import os
import time
import cv2
import csv
import numpy as np
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pytesseract
from ultralytics import YOLO

# Set Tesseract path if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def detect_plate_yolo(image, model):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(img_rgb, verbose=False)
    try:
        boxes_tensor = results[0].boxes.xyxy
        confs_tensor = results[0].boxes.conf 
        if boxes_tensor is not None and boxes_tensor.shape[0] > 0:
            boxes = boxes_tensor.cpu().numpy()
            confs = confs_tensor.cpu().numpy()
            idx = np.argmax(confs)  
            return boxes[idx]  
        else:
            return None
    except Exception as e:
        print("YOLO detection error:", e)
        return None

def refine_plate_with_contours(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(filtered, 30, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            return crop[y:y+h, x:x+w]
    return crop

def process_image(image_path, model, debug_folder):
    image = cv2.imread(image_path)
    if image is None:
        return os.path.basename(image_path), "Image not loaded"
    H, W = image.shape[:2]
    plate_text = "Plate not detected"


    yolo_box = detect_plate_yolo(image, model)
    if yolo_box is not None:
        x1, y1, x2, y2 = map(int, yolo_box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 > x1 and y2 > y1:
            yolo_crop = image[y1:y2, x1:x2]

            refined_crop = refine_plate_with_contours(yolo_crop)
            crop_to_use = refined_crop if refined_crop is not None else yolo_crop

            upscaled = cv2.resize(crop_to_use, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            kernel_sharpen = np.array([[-1, -1, -1],
                                       [-1, 9, -1],
                                       [-1, -1, -1]])
            sharpened = cv2.filter2D(upscaled, -1, kernel_sharpen)
            
            gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            debug_path = os.path.join(debug_folder, os.path.basename(image_path).split('.')[0] + "_preproc.png")
            cv2.imwrite(debug_path, thresh)
            
            custom_config = r'--psm 7'
            text = pytesseract.image_to_string(thresh, config=custom_config)
            if text and text.strip():
                plate_text = text.strip()
            else:
                plate_text = "OCR returned empty"
    else:
        plate_text = "Plate not detected"
    return os.path.basename(image_path), plate_text

def main():
    image_directory = "archive/images"
    output_csv = "plate_numbers_revised.csv"
    output_stats = "processing_stats_revised.txt"
    num_workers = 4
    debug_folder = "debug"
    os.makedirs(debug_folder, exist_ok=True)
    
    weights_dir = os.path.join(os.getcwd(), "weights")
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, "yolov10b.pt")
    yolov10b_url = "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10b.pt"
    if not os.path.exists(weights_path):
        print("Downloading YOLOv10b weights...")
        urllib.request.urlretrieve(yolov10b_url, weights_path)
        print(f"Downloaded YOLOv10b weights to: {weights_path}")
    model = YOLO(weights_path)
    print("Hybrid model loaded.")
    
    image_files = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_image, image_path, model, debug_folder) for image_path in image_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            results.append(future.result())
    
    total_time = time.time() - start_time
    
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Plate Number"])
        writer.writerows(results)
    
    num_processed = len(image_files)
    num_identified = sum(1 for _, plate in results if plate not in ["Plate not detected", "OCR returned empty"])
    with open(output_stats, 'w') as f:
        f.write(f"Number of images processed: {num_processed}\n")
        f.write(f"Total number of plates identified: {num_identified}\n")
        f.write(f"Total processing time (seconds): {total_time:.2f}\n")
    
    print(f"Processed {num_processed} images, identified {num_identified} plates in {total_time:.2f} seconds.")
    print(f"Results saved to '{output_csv}' and stats saved to '{output_stats}'.")

if __name__ == "__main__":
    main()
