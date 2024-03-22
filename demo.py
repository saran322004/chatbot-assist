import cv2
import pytesseract

def perform_ocr(image):
    try:
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Perform adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Perform OCR using Tesseract
        text = pytesseract.image_to_string(thresh)
        return True, text.strip()  # Strip leading/trailing whitespace
    except Exception as e:
        return False, str(e)

def main():
    # Open the camera
    camera = cv2.VideoCapture(0)
    
    # Check if the camera is opened successfully
    if not camera.isOpened():
        print("Error: Unable to open camera.")
        return

    print("Camera opened successfully.")

    # Capture a frame from the camera
    ret, frame = camera.read()

    # Check if the frame is captured successfully
    if not ret:
        print("Error: Unable to capture frame.")
        camera.release()
        return

    print("Frame captured successfully.")

    # Perform OCR on the captured frame
    success, text = perform_ocr(frame)

    # Check if OCR is successful
    if success:
        print("OCR performed successfully.")
        # Print the recognized text
        print("Recognized text:")
        print(text)
        
        # Draw the recognized text on the frame
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Save the frame with recognized text to a file
        cv2.imwrite("frame_with_text.jpg", frame)
        print("Frame with recognized text saved as 'frame_with_text.jpg'")
    else:
        print("Error occurred during OCR:", text)

    # Release the camera
    camera.release()
    print("Camera released.")

if __name__ == "__main__":
    main()
