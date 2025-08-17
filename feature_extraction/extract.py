import cv2
import numpy as np

# Load the binary mask
image_path = "D:/College-MRE/Term 10/Image Processing/3- Feature Extraction/Picture1.png"
binary_mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Ensure it's binary
_, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)

# Convert to color for visualization
output_image = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

# Define scaling factor (estimated): 1 pixel = 0.03 cm
scaling_factor = 0.03  # cm per pixel

# Find contours
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    for cnt in contours:
        # Area & Perimeter in pixels
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        # Area & Perimeter in cm
        area_cm2 = area * (scaling_factor ** 2)
        perimeter_cm = perimeter * scaling_factor

        # Bounding Box
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        width_cm = w * scaling_factor
        height_cm = h * scaling_factor

        # Convex Hull
        hull = cv2.convexHull(cnt)

        # Minimum Enclosing Circle
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        radius_cm = radius * scaling_factor

        # Centroid
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
        else:
            centroid_x, centroid_y = 0, 0

        centroid_x_cm = centroid_x * scaling_factor
        centroid_y_cm = centroid_y * scaling_factor

        # Orientation (angle of the fitted ellipse)
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            orientation = ellipse[2]  # Angle of the major axis
        else:
            orientation = None

        # Eccentricity calculation
        if "mu20" in M and "mu02" in M and M["mu02"] + M["mu20"] != 0:
            eccentricity = np.sqrt(1 - (M["mu20"] / (M["mu02"] + 1e-5)))
        else:
            eccentricity = 0

        # Draw contours and features
        cv2.drawContours(output_image, [cnt], -1, (0, 255, 0), 2)  # Green contour
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red bounding box
        cv2.polylines(output_image, [hull], True, (0, 255, 255), 2)  # Yellow convex hull
        cv2.circle(output_image, (centroid_x, centroid_y), 4, (255, 0, 0), -1)  # Blue centroid

        # Draw orientation arrow
        if orientation is not None:
            length = 50  # Arrow length (in pixels)
            angle_rad = np.deg2rad(orientation)
            end_x = int(centroid_x + length * np.cos(angle_rad))
            end_y = int(centroid_y + length * np.sin(angle_rad))
            cv2.arrowedLine(output_image, (centroid_x, centroid_y), (end_x, end_y), (0, 0, 255), 2)

        # Print real-world values in terminal
        print(f"--- Wound Feature Summary ---")
        print(f"Area: {area_cm2:.2f} cm^2, Perimeter: {perimeter_cm:.2f} cm")
        print(f"Bounding Box: {width_cm:.2f} x {height_cm:.2f} cm")
        print(f"Centroid: ({centroid_x_cm:.2f}, {centroid_y_cm:.2f}) cm")
        print(f"Enclosing Circle Radius: {radius_cm:.2f} cm")
        print(f"Orientation: {orientation:.1f} deg" if orientation else "Orientation: N/A")
        print(f"Eccentricity: {eccentricity:.2f}")
        print("--------------------------------\n")

else:
    print("No contours found!")

# Show final image
cv2.imshow("Wound Features (Scaled)", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
