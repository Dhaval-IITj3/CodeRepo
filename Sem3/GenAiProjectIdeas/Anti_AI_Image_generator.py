import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random


def generate_question_paper(questions, output_path="question_paper.png"):
    # Image dimensions
    width, height = 800, 600
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    # Load a stylized font (replace with a CAPTCHA-like or handwritten font)
    try:
        font = ImageFont.truetype("arial.ttf", 24)  # Replace with a custom font
    except:
        font = ImageFont.load_default()

    # Add questions with distortions
    y_offset = 50
    for i, question in enumerate(questions, 1):
        # Randomly adjust text color (low contrast)
        text_color = (random.randint(100, 150), random.randint(100, 150), random.randint(100, 150))
        text = f"Q{i}: {question}"

        # Draw text
        draw.text((50, y_offset), text, fill=text_color, font=font)
        y_offset += 50

    # Convert to OpenCV format for additional distortions
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Apply geometric distortion (e.g., wave effect)
    rows, cols = image_np.shape[:2]
    map_x = np.zeros((rows, cols), np.float32)
    map_y = np.zeros((rows, cols), np.float32)

    for i in range(rows):
        for j in range(cols):
            map_x[i, j] = j + 10 * np.sin(i / 30.0)
            map_y[i, j] = i

    distorted = cv2.remap(image_np, map_x, map_y, cv2.INTER_LINEAR)

    # Add subtle noise
    noise = np.random.normal(0, 10, distorted.shape).astype(np.uint8)
    distorted = cv2.add(distorted, noise)

    # Save the final image
    cv2.imwrite(output_path, distorted)
    print(f"Question paper saved as {output_path}")


# Example usage
questions = [
    "What is the capital of France?",
    "Explain the theory of relativity.",
    "Solve: 2x + 3 = 7"
]
generate_question_paper(questions)