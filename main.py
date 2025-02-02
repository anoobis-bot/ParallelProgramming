import os
import cv2
import easyocr
import csv
import threading
import time

# Set the path to the directory containing images
image_directory = "data"
output_csv = "plate_numbers.csv"
output_stats = "processing_stats.txt"
num_threads = 4  # Define the number of threads for parallel processing

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Lock for thread-safe operations
lock = threading.Lock()


def extract_plate_number(image_path, results):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edged = cv2.Canny(gray, 100, 200)

    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:  # Assuming plate is a rectangle
            x, y, w, h = cv2.boundingRect(approx)
            plate_region = gray[y:y + h, x:x + w]
            plate_text = reader.readtext(plate_region, detail=0)

            lock.acquire()
            try:
                results.append((image_path, plate_text[0] if plate_text else "Plate not detected"))
            finally:
                lock.release()
            return

    lock.acquire()
    try:
        results.append((image_path, "Plate not detected"))
    finally:
        lock.release()


def main():
    # Get a list of image files
    image_files = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if
                   f.lower().endswith(('png', 'jpg', 'jpeg'))]

    # Start processing time
    start_time = time.time()

    # Process images in parallel using threading
    results = []
    threads = []
    for image_path in image_files:
        if len(threads) >= num_threads:
            for thread in threads:
                thread.join()
            threads.clear()

        thread = threading.Thread(target=extract_plate_number, args=(image_path, results))
        threads.append(thread)
        thread.start()

    # Ensure all threads finish
    for thread in threads:
        thread.join()

    # End processing time
    end_time = time.time()
    total_time = end_time - start_time

    # Write results to CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Plate Number"])
        for image_path, plate_number in results:
            writer.writerow([os.path.basename(image_path), plate_number])
            print(f"Image: {os.path.basename(image_path)} -> Plate Number: {plate_number}")

    # Compute statistics
    num_processed = len(image_files)
    num_identified = sum(1 for _, plate in results if plate != "Plate not detected")

    # Write statistics to text file
    with open(output_stats, "w") as stats_file:
        stats_file.write(f"Number of images processed: {num_processed}\n")
        stats_file.write(f"Total number of plates identified: {num_identified}\n")
        stats_file.write(f"Total processing time (seconds): {total_time:.2f}\n")

    print(f"CSV file '{output_csv}' and statistics file '{output_stats}' have been created.")


if __name__ == "__main__":
    main()
