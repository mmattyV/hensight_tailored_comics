from constants import *

def detect_vertices(image_path, draw=False):
    # Predefined colors for drawing
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
        (255, 127, 127), (127, 255, 127), (127, 127, 255),
        (255, 127, 0), (127, 255, 0), (0, 255, 127)
    ]
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Optionally apply Gaussian Blur to smooth the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Estimate the background color by finding the mode of the borders
    top_row = blurred[0, :]
    bottom_row = blurred[-1, :]
    left_column = blurred[:, 0]
    right_column = blurred[:, -1]
    border_pixels = np.hstack((top_row, bottom_row, left_column, right_column))
    background_color = mode(border_pixels)[0]

    # Dynamically set adaptive threshold
    if background_color > 128:  # Light background
        binary_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    else:  # Dark background
        binary_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Edge detection using Canny
    edges = cv2.Canny(binary_image, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare the output image if drawing is enabled
    if draw:
        output = image.copy()

    # List to hold vertices of each frame
    frames_vertices = []
    color_index = 0  # Index to keep track of which color to use

    for contour in contours:
        # Approximate contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)  # Adjust epsilon to control the approximation accuracy
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Draw and store vertices if it's a quadrilateral or a significant polygon
        if len(approx) == 4 and cv2.contourArea(approx) > 100:  # Check for quadrilaterals
            frame_vertices = [tuple(vertex[0]) for vertex in approx]
            frames_vertices.append(frame_vertices)
            if draw:
                current_color = colors[color_index % len(colors)]  # Cycle through colors
                cv2.drawContours(output, [approx], -1, current_color, 3)
                for vertex in frame_vertices:
                    cv2.circle(output, vertex, 4, current_color, -1)  # Draw vertices using the current color
                color_index += 1  # Increment color index

    # If drawing is enabled, display the output image
    if draw:
        cv2.imshow('Detected Vertices', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return frames_vertices

detect_vertices('../../comic_images/5080.jpg', True)