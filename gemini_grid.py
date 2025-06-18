# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 15:46:43 2025

@author: dekel
"""

import cv2
import matplotlib.pyplot as plt

def overlay_grid_with_labels(
    image_path,
    num_rows,
    num_cols,
    start_row=0,
    start_col_char='a',
    font_scale=1.5,
    font_thickness=2,
    line_color=(0, 0, 0),       # Black in BGR format
    text_color=(255, 255, 255), # White in BGR format
    line_thickness=1
):
    """
    Overlays a grid with alphanumeric labels (e.g., a0, a1, b0...) onto an image.

    Args:
        image_path (str): Path to the input image.
        num_rows (int): The number of rows in the grid.
        num_cols (int): The number of columns in the grid.
        start_row (int, optional): The starting number for row labels. Defaults to 0.
        start_col_char (str, optional): The starting character for column labels. Defaults to 'a'.
        font_scale (float, optional): The scale of the label font. Defaults to 1.5.
        font_thickness (int, optional): The thickness of the label font. Defaults to 2.
        line_color (tuple, optional): The BGR color for grid lines. Defaults to black.
        text_color (tuple, optional): The BGR color for label text. Defaults to white.
        line_thickness (int, optional): The thickness of grid lines. Defaults to 1.

    Returns:
        numpy.ndarray: The image with the grid and labels overlayed, in BGR format.
                       Returns None if the image cannot be loaded.
    """
    # 1. Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at path: {image_path}")
        return None

    # Create a copy to draw on, to keep the original intact
    overlay = image.copy()

    # 2. Get image dimensions and calculate cell size
    height, width, _ = image.shape
    cell_height = height / num_rows
    cell_width = width / num_cols

    # Get the starting column index (e.g., 'a' -> 0, 'b' -> 1)
    start_col_index = ord(start_col_char.lower()) - ord('a')

    # 3. Iterate over rows and columns to draw the grid and labels
    for r in range(num_rows):
        for c in range(num_cols):
            # --- Draw grid lines (by drawing a rectangle for each cell) ---
            start_point = (int(c * cell_width), int(r * cell_height))
            end_point = (int((c + 1) * cell_width), int((r + 1) * cell_height))
            cv2.rectangle(overlay, start_point, end_point, line_color, line_thickness)

            # --- Prepare and draw labels ---
            # Generate the label string (e.g., "b0", "b1", "c0")
            col_label = chr(ord('a') + start_col_index + c)
            row_label = start_row + r
            label = f"{col_label}{row_label}"

            # Calculate text position to be near the top-left of the cell.
            # These proportional offsets make it robust to different image/cell sizes.
            text_x = int(start_point[0] + cell_width * 0.1)
            text_y = int(start_point[1] + cell_height * 0.4) # Adjusted for better vertical centering

            # Put the text on the image
            cv2.putText(
                overlay,
                label,
                (text_x, text_y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                color=text_color,
                thickness=font_thickness,
                lineType=cv2.LINE_AA # For smoother text rendering
            )
            
    return overlay

# --- Main execution block to run the script ---
if __name__ == '__main__':
    # Use the image you provided.
    # IMPORTANT: Save your image as 'original_image.png' in the same folder as this script.
    input_image_path = r"D:\Docs\test6\Projects\Robotics\Samples\Moley_Robot_Kitchen.jpg"
    # Save the final image to a file
    output_path = './outputs/img_grid.png'
    
    # --- Configuration based on your example image ---
    # The grid has rows 0-5 and columns b-g
    NUM_ROWS = 6         # Corresponds to labels 0, 1, 2, 3, 4, 5
    NUM_COLS = 6         # Corresponds to labels b, c, d, e, f, g
    START_ROW = 0
    START_COL_CHAR = 'b' # The first column is labeled 'b'

    # Generate the overlay by calling our function
    gridded_image = overlay_grid_with_labels(
        image_path=input_image_path,
        num_rows=NUM_ROWS,
        num_cols=NUM_COLS,
        start_row=START_ROW,
        start_col_char=START_COL_CHAR,
        # Adjust font/line properties to better match your example
        font_scale=1.2,
        font_thickness=2,
        line_color=(40, 40, 40),      # Dark gray, not pure black
        text_color=(240, 240, 240),  # Off-white for a softer look
        line_thickness=2
    )

    # --- Save and Display the Result ---
    if gridded_image is not None:        
        cv2.imwrite(output_path, gridded_image)
        print(f"Successfully saved the gridded image to: {output_path}")

        # Display the original and the new image side-by-side for comparison
        # Convert from BGR (OpenCV's default) to RGB for Matplotlib
        original_image_rgb = cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2RGB)
        gridded_image_rgb = cv2.cvtColor(gridded_image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(14, 7))
        
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(original_image_rgb)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Image with Grid Overlay')
        plt.imshow(gridded_image_rgb)
        plt.axis('off')

        plt.tight_layout()
        plt.show()