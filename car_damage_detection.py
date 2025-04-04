import cv2
from ultralytics import YOLO  # For YOLOv8
import gradio as gr
import numpy as np
import torch
import os
import json
import time
from datetime import datetime
from groq import Groq

# Initialize Groq client - Get API key from environment or provide it here
client = Groq(api_key="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")
# You should set your Groq API key as an environment variable GROQ_API_KEY
# or replace the line above with: client = Groq(api_key="your-groq-api-key")

# Load YOLO models only once for efficiency
damage_model = YOLO("best2.pt")  # Custom-trained YOLO model for damage detection
car_model = YOLO("yolov8m.pt")   # Pretrained YOLO model for car detection

# Define severity points and repair recommendations for each type of damage
damage_info = {
    "scratch": {
        "severity_points": 1,
        "description": "Surface level damage affecting only the paint layer",
        "repair_methods": {
            "light": "Light buffing and touch-up paint",
            "medium": "Sanding, filling, and repainting the affected area",
            "severe": "Panel replacement or extensive bodywork may be required"
        },
        "estimated_costs": {
            "light": "$100-$300",
            "medium": "$300-$800",
            "severe": "$800-$1500"
        }
    },
    "dent": {
        "severity_points": 2,
        "description": "Deformation of the body panel that may or may not have paint damage",
        "repair_methods": {
            "light": "Paintless dent repair (PDR) technique",
            "medium": "Traditional body repair with filler and repainting",
            "severe": "Panel replacement and painting"
        },
        "estimated_costs": {
            "light": "$150-$400",
            "medium": "$400-$1200",
            "severe": "$1200-$3000"
        }
    },
    "rust": {
        "severity_points": 2,
        "description": "Oxidation of metal components leading to structural deterioration",
        "repair_methods": {
            "light": "Sanding, rust converter application, and repainting",
            "medium": "Cutting out rusted sections and welding new metal",
            "severe": "Complete panel replacement or structural repair"
        },
        "estimated_costs": {
            "light": "$200-$500",
            "medium": "$500-$1500",
            "severe": "$1500-$5000"
        }
    },
    "paint-damage": {
        "severity_points": 2,
        "description": "Deterioration of paint layer exposing the primer or metal beneath",
        "repair_methods": {
            "light": "Spot repair and blending",
            "medium": "Panel repainting",
            "severe": "Multiple panel repainting and clear coat restoration"
        },
        "estimated_costs": {
            "light": "$150-$400",
            "medium": "$400-$1000",
            "severe": "$1000-$3000"
        }
    },
    "crack": {
        "severity_points": 2,
        "description": "Structural split in body panels, plastic components, or glass",
        "repair_methods": {
            "light": "Plastic welding or bonding (for plastic parts)",
            "medium": "Fiberglass repair or partial panel replacement",
            "severe": "Complete component replacement"
        },
        "estimated_costs": {
            "light": "$200-$500",
            "medium": "$500-$1500",
            "severe": "$1500-$4000"
        }
    }
}

# Function to detect damage and return class counts and locations
# Enhanced function to detect damage and provide better analysis
def detect_damage(image):
    results = damage_model(image, verbose=False)

    dic = results[0].names
    classes = results[0].boxes.cls.cpu().numpy()
    probability = results[0].boxes.conf.cpu().numpy()
    boxes = results[0].boxes.xyxy.cpu().numpy()

    # Create detailed damage report including enhanced location and confidence
    detailed_damages = []
    for i, cls in enumerate(classes):
        class_name = dic[int(cls)]
        conf = probability[i]
        box = boxes[i]

        # Skip low confidence detections
        if conf < 0.3:
            continue

        # Calculate relative position on car
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        width = results[0].orig_shape[1]
        height = results[0].orig_shape[0]

        # Determine position with more granularity
        position = ""
        if x_center < width * 0.33:
            position += "left "
        elif x_center > width * 0.66:
            position += "right "
        else:
            position += "center "

        if y_center < height * 0.33:
            position += "front"
        elif y_center > height * 0.66:
            position += "rear"
        else:
            position += "middle"

        # Calculate damage size based on bounding box
        damage_area = (box[2] - box[0]) * (box[3] - box[1])
        image_area = width * height
        relative_size = damage_area / image_area

        # Better severity determination based on size, confidence and damage type
        severity = "light"

        # Specific rules for dents
        if class_name == "dent":
            if relative_size > 0.04 or conf > 0.9:
                severity = "severe"
            elif relative_size > 0.02 or conf > 0.7:
                severity = "medium"
        # Rules for other damage types
        else:
            if relative_size > 0.05 or conf > 0.85:
                severity = "severe"
            elif relative_size > 0.02 or conf > 0.6:
                severity = "medium"

        # Calculate depth estimate for dents (based on relative size and confidence)
        depth_estimate = None
        if class_name == "dent":
            # Simple depth estimation heuristic based on confidence and size
            depth_estimate = round((relative_size * 30 + conf * 0.5), 2)

        detailed_damages.append({
            "type": class_name,
            "position": position,
            "confidence": float(conf),
            "severity": severity,
            "relative_size": float(relative_size),
            "depth_estimate": depth_estimate,
            "box": [float(b) for b in box]
        })

    # Also calculate simple class count
    class_count = {}
    for damage in detailed_damages:
        class_name = damage["type"]
        class_count[class_name] = class_count.get(class_name, 0) + 1

    return class_count, detailed_damages, results

# Function to detect car and crop the image
def car_detection_and_cropping(image_path):
    # Check if image_path is a string (file path) or already a numpy array
    if isinstance(image_path, str):
        # Make sure the image file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load image first to verify it's valid
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image file: {image_path}")

        # Now run detection on the verified path
        r = car_model(image_path, verbose=False)
    else:
        # If it's already an image array, use it directly
        image = image_path
        r = car_model(image, verbose=False)

    names = r[0].names
    boxes = r[0].boxes.xyxy.cpu().numpy().astype(int)
    classes = set(r[0].boxes.cls.cpu().numpy())
    detected_classes = [names[int(i)] for i in classes]

    # If image was already loaded, don't load it again
    if isinstance(image_path, str) and image is None:
        image = cv2.imread(image_path)

    if boxes.size != 0 and 2 in classes:  # 2 is typically the class ID for car in COCO
        # Find the largest car detected (in case of multiple)
        area = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
        max_index = np.argmax(area)  # Get index of largest car

        # Crop to the largest detected car
        x1, y1, x2, y2 = boxes[max_index]
        cropped_image = image[y1:y2, x1:x2]

        # Detect damage on cropped car image
        class_count, detailed_damages, results = detect_damage(cropped_image)

        # Adjust bounding box coordinates to match original image
        for damage in detailed_damages:
            damage["box"][0] += x1  # Adjust x1
            damage["box"][1] += y1  # Adjust y1
            damage["box"][2] += x1  # Adjust x2
            damage["box"][3] += y1  # Adjust y2

        # Store cropping information
        crop_info = {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
    else:
        # If no car detected, process the full image
        class_count, detailed_damages, results = detect_damage(image_path)
        crop_info = None

    return class_count, detailed_damages, results, crop_info, image
# Function to calculate damage condition score
def calculate_condition_score(detailed_damages):
    total_score = 0

    # Enhanced severity multipliers
    severity_multipliers = {
        "light": 1.0,    # Increased from 0.7
        "medium": 1.5,   # Increased from 1.0
        "severe": 2.5    # Increased from 1.5
    }

    # Special damage type multipliers
    damage_type_multipliers = {
        "dent": 1.5,     # Dents should impact score more
        "crack": 1.8,    # Cracks are serious structural issues
        "rust": 1.7,     # Rust indicates long-term issues
        "scratch": 0.8,  # Scratches are generally cosmetic
        "paint-damage": 1.0
    }

    # Position importance multipliers (some areas are more critical)
    position_multipliers = {
        "center": 1.2,
        "front": 1.3,
        "rear": 1.2,
        "side": 1.0
    }

    # Apply size thresholds for better severity classification
    for damage in detailed_damages:
        damage_type = damage["type"]
        severity = damage["severity"]
        confidence = damage["confidence"]
        relative_size = damage.get("relative_size", 0.01)
        position = damage["position"]

        # Recalculate severity based on better thresholds
        if damage_type == "dent" and confidence > 0.8:
            if relative_size > 0.03:
                severity = "severe"
            elif relative_size > 0.01:
                severity = "medium"

        # Only count damages with reasonable confidence
        if confidence > 0.4 and damage_type in damage_type_multipliers:
            base_points = 2.0  # Base damage points
            type_multiplier = damage_type_multipliers.get(damage_type, 1.0)
            severity_multiplier = severity_multipliers.get(severity, 1.0)

            # Apply position multiplier
            position_multiplier = 1.0
            for pos_key, pos_value in position_multipliers.items():
                if pos_key in position.lower():
                    position_multiplier = pos_value
                    break

            # Apply confidence and size factor
            confidence_factor = confidence * (1 + relative_size * 10)

            # Calculate score for this damage instance
            damage_score = base_points * type_multiplier * severity_multiplier * position_multiplier * confidence_factor
            total_score += damage_score

    return total_score

def normalize_score(score, max_score=12):
    # Using a logarithmic scale to better represent damage severity
    import math
    if score <= 0:
        return 10  # Perfect score if no damage

    # Log scale helps distinguish between minor and major damages
    # Now inverted: higher damage score = lower condition rating
    normalized = 10 - min(10 * math.log(1 + score) / math.log(1 + max_score), 10)
    return round(normalized, 1)  # Round to one decimal place

# Function to estimate the condition of the car
def estimate_condition(detailed_damages):
    score = calculate_condition_score(detailed_damages)
    normalized_score = normalize_score(score)

    # Inverted thresholds where higher score = better condition
    if normalized_score >= 9:
        return "Perfect", normalized_score
    elif 7 <= normalized_score < 9:
        return "Very Good", normalized_score
    elif 5 <= normalized_score < 7:
        return "Good", normalized_score
    elif 3 <= normalized_score < 5:
        return "Fair", normalized_score
    elif 1 <= normalized_score < 3:
        return "Poor", normalized_score
    else:
        return "Very Poor", normalized_score

# Enhanced function to generate a more accurate and detailed repair report
def generate_repair_report(detailed_damages, condition, normalized_score):
    # Prepare the data for the LLM prompt
    damage_summary = {}
    for damage in detailed_damages:
        damage_type = damage["type"]
        if damage_type not in damage_summary:
            damage_summary[damage_type] = []
        damage_summary[damage_type].append(damage)

    # Build a detailed damage report with enhanced insights
    damage_details = []
    repair_steps = []
    severity_counts = {"light": 0, "medium": 0, "severe": 0}

    for damage_type, instances in damage_summary.items():
        if damage_type in damage_info:
            for i, instance in enumerate(instances):
                severity = instance["severity"]
                position = instance["position"]
                severity_counts[severity] += 1

                # Add depth information for dents
                depth_info = ""
                if damage_type == "dent" and "depth_estimate" in instance and instance["depth_estimate"]:
                    depth_info = f" with estimated depth factor of {instance['depth_estimate']}"

                # Get repair method and cost based on severity
                repair_method = damage_info[damage_type]["repair_methods"].get(severity, "Requires professional assessment")
                estimated_cost = damage_info[damage_type]["estimated_costs"].get(severity, "Varies")

                damage_details.append(f"{damage_type.title()} ({severity}) at {position}{depth_info}")
                repair_steps.append({
                    "damage_type": damage_type,
                    "location": position,
                    "severity": severity,
                    "repair_method": repair_method,
                    "estimated_cost": estimated_cost,
                    "confidence": instance["confidence"],
                    "relative_size": instance["relative_size"]
                })

    # Calculate better safety assessment based on damage types and severity
    safety_concerns = []
    if severity_counts["severe"] > 0:
        safety_concerns.append("Some damages may affect vehicle safety and should be addressed promptly")

    # Enhanced prompt for the LLM with better context
    prompt = f"""
    Generate a detailed car damage repair report based on the following damage assessment:

    Overall Condition: {condition} (Score: {normalized_score:.1f}/10)

    Severity summary:
    - Severe damages: {severity_counts["severe"]}
    - Medium damages: {severity_counts["medium"]}
    - Light damages: {severity_counts["light"]}

    Detected damages:
    {json.dumps(damage_details, indent=2)}

    Suggested repair approach for each damage:
    {json.dumps(repair_steps, indent=2)}

    Safety concerns:
    {json.dumps(safety_concerns, indent=2)}

    Please provide:
    1. A summary of the damage assessment that accurately reflects the condition
    2. Detailed repair recommendations for each damage type
    3. Priority order for repairs (which damages should be fixed first)
    4. Estimated total repair cost range
    5. Whether the car is safe to drive in its current condition
    6. Any diminished value considerations

    Format the report in markdown with clear sections.
    """

    try:
        # Call Groq API for detailed analysis
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are an expert automotive damage assessor and repair technician with 20+ years of experience. Be direct and honest about vehicle condition."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )

        # Extract the generated report
        report = response.choices[0].message.content
        return report

    except Exception as e:
        # Fallback if API call fails with enhanced basic report
        print(f"Error calling Groq API: {e}")

        # Generate a better basic report without the API
        basic_report = f"""
# Car Damage Assessment Report

## Overview
- **Condition Rating:** {condition} ({normalized_score:.1f}/10)
- **Number of Damages Detected:** {len(detailed_damages)}
- **Severity Distribution:** Severe: {severity_counts["severe"]}, Medium: {severity_counts["medium"]}, Light: {severity_counts["light"]}
- **Primary Issues:** {', '.join(damage_summary.keys())}

## Damage Details
"""
        total_cost_min = 0
        total_cost_max = 0

        for damage_type, instances in damage_summary.items():
            if damage_type in damage_info:
                basic_report += f"\n### {damage_type.title()} ({len(instances)} instance{'s' if len(instances) > 1 else ''})\n"
                basic_report += f"- **Description:** {damage_info[damage_type]['description']}\n"

                for i, instance in enumerate(instances):
                    severity = instance["severity"]
                    position = instance["position"]

                    # Extract cost range and calculate totals
                    cost_range = damage_info[damage_type]['estimated_costs'].get(severity, "$0-$0")
                    if "-" in cost_range:
                        try:
                            min_cost = int(cost_range.split("-")[0].replace("$", "").replace(",", ""))
                            max_cost = int(cost_range.split("-")[1].replace("$", "").replace(",", ""))
                            total_cost_min += min_cost
                            total_cost_max += max_cost
                        except:
                            pass

                    # Add depth information for dents
                    depth_info = ""
                    if damage_type == "dent" and "depth_estimate" in instance and instance["depth_estimate"]:
                        depth_info = f" (Depth factor: {instance['depth_estimate']})"

                    basic_report += f"- **Location {i+1}:** {position} (Severity: {severity}){depth_info}\n"
                    basic_report += f"  - **Recommended Repair:** {damage_info[damage_type]['repair_methods'].get(severity, 'Professional assessment needed')}\n"
                    basic_report += f"  - **Estimated Cost:** {damage_info[damage_type]['estimated_costs'].get(severity, 'Varies')}\n"

        basic_report += f"""
## Estimated Total Cost
- **Cost Range:** ${total_cost_min:,} - ${total_cost_max:,}

## Repair Priority
1. Structural damages (cracks) should be addressed first for safety
2. Rust issues to prevent further deterioration
3. Dents and paint damage
4. Scratches can be addressed last

## Safety Assessment
"""
        if severity_counts["severe"] > 0:
            basic_report += "- **Caution:** Vehicle has severe damage(s) that may affect safety. Professional inspection recommended before regular use.\n"
        elif severity_counts["medium"] > 0:
            basic_report += "- **Note:** Vehicle has moderate damage(s) that should be repaired but likely don't affect basic safety.\n"
        else:
            basic_report += "- **Safe:** Vehicle damage appears to be cosmetic and shouldn't affect safety.\n"

        basic_report += """
## General Recommendations
- Obtain quotes from at least 3 different repair shops
- Consider insurance coverage before proceeding with repairs
- Some minor cosmetic damages may be deferred if budget is limited
"""
        return basic_report

# Main function to process uploaded images
def process_data(files):
    print("Processing files:", files)
    # Handle different types of input (string, list, tuple)
    if isinstance(files, (str, tuple, list)):
        if isinstance(files, str):
            file_names = [files]
        else:
            file_names = list(files)
    else:
        # If it's not a recognizable format, return an error
        return "Error: Invalid file format provided", None, "Please upload valid image files."

    processed_images = []
    all_detailed_damages = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create directory for saving reports if it doesn't exist
    report_dir = "damage_reports"
    os.makedirs(report_dir, exist_ok=True)

    # Check if there are any valid files to process
    if not file_names or not any(os.path.exists(f[0]) if isinstance(f, tuple) else os.path.exists(f) for f in file_names):
        return "Error: No valid image files found", None, "Please upload valid image files."

    try:
        for file_item in file_names:
            # Handle tuples from Gradio (file_path, file_name)
            if isinstance(file_item, tuple):
                file_path = file_item[0]
            else:
                file_path = file_item

            print("Processing Image:", file_path)

            # Make sure file exists
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue

            try:
                _, detailed_damages, results, crop_info, original_image = car_detection_and_cropping(file_path)

                # Add damages from this image to overall list
                all_detailed_damages.extend(detailed_damages)

                for r in results:
                    im_array = r.plot(pil=True)  # Get PIL image with bounding boxes
                    im_array = np.array(im_array)  # Convert PIL to NumPy array
                    rgb_image = im_array[..., ::-1]  # Convert BGR to RGB
                    processed_images.append(rgb_image)
            except Exception as e:
                print(f"Error processing image {file_path}: {str(e)}")
                continue

        # If no images were processed successfully
        if not all_detailed_damages:
            return "Error: No damages detected in any images", None, "Please upload clear images of vehicles with visible damage."

        # Calculate overall condition
        condition, score = estimate_condition(all_detailed_damages)

        # Generate repair report using Groq
        repair_report = generate_repair_report(all_detailed_damages, condition, score)

        # Save the report to a file
        report_filename = f"{report_dir}/car_damage_report_{timestamp}.md"
        with open(report_filename, "w") as f:
            f.write(repair_report)

        # Create a summary for the Gradio interface
        summary = f"""
        **Overall Condition: {condition}** (Score: {score:.1f}/10)

        **Damages Detected:**
        """

        damage_counts = {}
        for damage in all_detailed_damages:
            damage_type = damage["type"]
            damage_counts[damage_type] = damage_counts.get(damage_type, 0) + 1

        for damage_type, count in damage_counts.items():
            summary += f"- {damage_type.title()}: {count}\n"

        summary += f"\nDetailed report saved to: {report_filename}"

        return summary, processed_images, repair_report

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in process_data: {str(e)}\n{error_details}")
        return f"Error processing images: {str(e)}", None, "An error occurred during processing. Please check the console for details."
# Gradio interface with fixed inputs
interface = gr.Interface(
    fn=process_data,
    inputs=gr.Gallery(label="Upload Images of Car", type="filepath"),  # Removed file_count parameter
    outputs=[
        gr.Textbox(label="Car Condition Summary"),
        gr.Gallery(label="Detected Damage", type="pil"),
        gr.Markdown(label="Detailed Repair Report")
    ],
    title="🚘 Advanced Car Damage Assessment",
    description="Upload images of a car, and the system will detect damages, assess severity, and provide repair recommendations.",
)

if __name__ == "__main__":
    interface.launch(debug=True)
