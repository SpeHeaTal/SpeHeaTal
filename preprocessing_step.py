import os
import cv2
from PIL import Image
import numpy as np
from scipy.ndimage import label
from skimage import color, filters, morphology
from scipy.ndimage import binary_dilation
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def batch_resize_images(start_index, end_index, width, height, input_folder, output_folder):
    for i in range(start_index, end_index + 1):
        input_file_name = f"{i:03}.jpg"
        output_file_name = f"{i:03}.jpg"
        input_path = os.path.join(input_folder, input_file_name)
        output_path = os.path.join(output_folder, output_file_name)
        if os.path.exists(input_path):
            img = Image.open(input_path)
            img_resized = img.resize((width, height))
            img_resized.save(output_path)
            print(f"Resized {input_file_name} to {width}x{height} and saved as {output_file_name}")
        else:
            print(f"File {input_file_name} does not exist in the input folder.")

def adjust_brightness(image_path, target_brightness):
    with Image.open(image_path) as img:
        hsv_image = img.convert('HSV')
        h, s, v = hsv_image.split()
        v_array = np.array(v, dtype=np.float64)
        current_brightness = np.mean(v_array)
        if current_brightness == 0 or current_brightness == target_brightness:
            return img
        v_array *= target_brightness / current_brightness
        v_array[v_array > 255] = 255
        v_array = v_array.astype(np.uint8)
        adjusted_hsv_image = Image.merge('HSV', (h, s, Image.fromarray(v_array)))
        return adjusted_hsv_image.convert('RGB')

def adjust_brightness_process(directory_path):
    target_brightness = 220
    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.jpg'):
            file_path = os.path.join(directory_path, filename)
            adjusted_image = adjust_brightness(file_path, target_brightness)
            adjusted_image.save(file_path)
            print(f"Adjusted brightness for {filename}")
    print("Brightness adjustment completed.")

def adjust_rgb_channels(image_path, target_r, target_g, target_b):
    with Image.open(image_path) as img:
        r, g, b = img.split()
        r_avg, g_avg, b_avg = map(np.mean, (r, g, b))
        r_scale = target_r / r_avg if r_avg > 0 else 0
        g_scale = target_g / g_avg if g_avg > 0 else 0
        b_scale = target_b / b_avg if b_avg > 0 else 0
        r = (np.array(r) * r_scale).clip(0, 255).astype(np.uint8)
        g = (np.array(g) * g_scale).clip(0, 255).astype(np.uint8)
        b = (np.array(b) * b_scale).clip(0, 255).astype(np.uint8)
        adjusted_image = Image.merge('RGB', (Image.fromarray(r), Image.fromarray(g), Image.fromarray(b)))
        return adjusted_image

def adjust_rgb_process(directory_path):
    target_r = 216.5
    target_g = 212.5
    target_b = 219.5
    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.jpg'):
            file_path = os.path.join(directory_path, filename)
            adjusted_image = adjust_rgb_channels(file_path, target_r, target_g, target_b)
            adjusted_image.save(file_path)
            print(f"Adjusted RGB channels for {filename}")
    print("RGB channel adjustment completed.")

def rescale(width, height, start_index, end_index):    
    input_folder = "2 synchronized brightness sperm" 
    output_folder = "2 synchronized brightness sperm" 
    batch_resize_images(start_index, end_index, width, height, input_folder, output_folder)
    
def dehaze_image(img, clip_limit=150, tile_size=(2, 2)):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

def sharpen_image(img, factor):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel * factor)

def adjust_contrast(img, factor):
    mean = np.mean(img)
    return np.clip((1 + factor) * (img - mean) + mean, 0, 255).astype(np.uint8)

def process_image_skimage(image_opencv):
    image_rgb = cv2.cvtColor(image_opencv, cv2.COLOR_BGR2RGB)
    gray_sk = color.rgb2gray(image_rgb)
    threshold_value = filters.threshold_otsu(gray_sk)
    binary = gray_sk < threshold_value
    cleaned = morphology.remove_small_objects(binary, min_size=2000)
    mask = np.where(cleaned == 0, 1, 0).astype(bool)
    image_rgb[mask] = [255, 255, 255]
    whitening_mask_sk = np.all(image_rgb > [200, 200, 200], axis=-1)
    image_rgb[whitening_mask_sk] = [255, 255, 255]
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

def pre_process(folder_path):
    file_list = os.listdir(folder_path)
    for filename in file_list:
        if filename.endswith('.jpg'): 
            full_path = os.path.join(folder_path, filename)
            image_opencv = cv2.imread(full_path)
            if image_opencv is None:
                print(f"Unable to read image:{filename}")
                continue
            img_original_rgb = cv2.cvtColor(image_opencv, cv2.COLOR_BGR2RGB)
            img_new_adjustment = img_original_rgb.copy()
            img_new_adjustment = adjust_contrast(img_new_adjustment, 1.0)
            img_new_adjustment = sharpen_image(img_new_adjustment, 1.235)
            img_new_adjustment = dehaze_image(img_new_adjustment)
            img_noise_reduced = cv2.fastNlMeansDenoisingColored(img_new_adjustment, None, 20, 20, 7, 21)
            img_final = process_image_skimage(img_noise_reduced)
            cv2.imwrite(full_path, img_final)
            print(f"Processed and saved image:{filename}")

def preprocessing(start_index, end_index, directory_path):
    rescale(1440, 1080, start_index, end_index+1)
    adjust_brightness_process(directory_path)
    adjust_rgb_process(directory_path)
    pre_process(directory_path)
    rescale(720, 540, start_index, end_index+1)

def SAM_seg(sam_checkpoint, model_type, device, start_index, end_index):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    for i in range(start_index, end_index+1): 
        image_name = f"{i:03}.jpg"  
        image_path = f"2 synchronized brightness sperm/{image_name}"
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_generator = SamAutomaticMaskGenerator(sam)
        masks = mask_generator.generate(image)
        masks = [np.array(seg['segmentation'], dtype=np.uint8) for seg in masks]
        masks = np.stack(masks)
        masks_file = f"2 synchronized brightness sperm/masks_{i:03}.npy" 
        np.save(masks_file, masks)
        print(f"Processed and saved masks for {image_name}")

def layer_filter(base_path, start_index, end_index):
    for i in range(start_index, end_index+1):
        file_name = f'masks_{i:03}.npy' 
        file_path = os.path.join(base_path, file_name)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        masks = np.load(file_path)
        if masks.ndim == 2:
            masks = masks[np.newaxis, ...]
        half_area = masks.shape[1] * masks.shape[2] / 2
        filtered_masks = [mask for mask in masks if np.sum(mask) <= half_area]
        if len(filtered_masks) != len(masks):
            np.save(file_path, np.array(filtered_masks))
            print(f"Updated file: {file_path} with {len(filtered_masks)} layers")

def max_theoretical_area(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_dist = 0
    if contours:
        all_points = np.concatenate(contours, axis=0)
        for point in all_points:
            distances = np.linalg.norm(all_points - point, axis=2)
            max_dist = max(max_dist, np.max(distances))
    return max_dist ** 2

def calculate_q_value(mask):
    S = max_theoretical_area(mask)
    s1 = np.sum(mask)
    return s1 / S if S != 0 else 0

def is_foreground_excessive(mask):
    total_pixels = mask.size
    foreground_pixels = np.sum(mask)
    return foreground_pixels > total_pixels * 0.5

def split_connected_components(mask):
    labeled_array, num_features = label(mask)
    return [labeled_array == i for i in range(1, num_features + 1)]

def visualize_and_save_masks(masks, save_path_q, save_path_full):
    q_masks = []
    full_masks = []
    for i, mask in enumerate(masks):
        if is_foreground_excessive(mask):
            full_masks.append(mask)
        else:
            q_value = calculate_q_value(mask)
            if q_value > 0.20:
                q_masks.append(mask)
            else:
                full_masks.append(mask)
    if q_masks:
        np.save(save_path_q, np.array(q_masks))
        np.save(save_path_full, np.array(full_masks))

def q_value_filter(start_index, end_index):
    for i in range(start_index, end_index+1): 
        img_number = f"{i:03d}" 
        masks_path = f"2 synchronized brightness sperm\\masks_{img_number}.npy"  
        masks = np.load(masks_path)  
        processed_masks = []
        for mask in masks:
            if np.any(mask):
                components = split_connected_components(mask)
                processed_masks.extend(components)
        save_path_q = f"2 synchronized brightness sperm\\masks_q_{img_number}.npy" 
        save_path_full = f"2 synchronized brightness sperm\\masks_full_{img_number}.npy" 
        visualize_and_save_masks(processed_masks, save_path_q, save_path_full)

def Pick_header_files(base_image_path, base_mask_path, start_index, end_index):
    lower_purple = np.array([100, 35, 20])
    upper_purple = np.array([180, 255, 255])
    lower_green = np.array([70, 0, 20])
    upper_green = np.array([120, 50, 255])
    for i in range(start_index, end_index+1): 
        image_number = f'{i:03}' 
        image_path = os.path.join(base_image_path, image_number + '.jpg')
        mask_path = os.path.join(base_mask_path, 'masks_q_' + image_number + '.npy')
        image = cv2.imread(image_path)
        if image is None:
            continue 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if not os.path.exists(mask_path):
            continue 
        masks = np.load(mask_path)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        filtered_masks = []
        for current_mask in masks:
            masked_purple = purple_mask[current_mask == 1]
            masked_green = green_mask[current_mask == 1]
            total_area = np.sum(current_mask)
            purple_area = np.sum(masked_purple > 0)
            green_area = np.sum(masked_green > 0)
            purple_percentage = (purple_area / total_area) * 100 if total_area > 0 else 0
            green_percentage = (green_area / total_area) * 100 if total_area > 0 else 0
            if purple_percentage > 40 and green_percentage < 20 and 600 <= total_area <= 3500:
                filtered_masks.append(current_mask)
        if filtered_masks: 
            filtered_masks = np.array(filtered_masks)
            new_mask_path = os.path.join(base_mask_path, f'mask_head_{image_number}.npy')
            np.save(new_mask_path, filtered_masks)
            print(f'Image {image_number}: Filtered masks count = {len(filtered_masks)}')
        else:
            print(f'Image {image_number}: No masks saved due to size constraints')

def Screening_intact_sperm(start_index, end_index, base_image_path, base_mask_path):
    lower_purple = np.array([100, 35, 20])
    upper_purple = np.array([180, 255, 255])
    lower_green = np.array([70, 0, 20])
    upper_green = np.array([120, 50, 255])
    for i in range(start_index, end_index+1): 
        image_number = f'{i:03}'  
        image_path = os.path.join(base_image_path, image_number + '.jpg')
        mask_path = os.path.join(base_mask_path, 'masks_full_' + image_number + '.npy')
        image = cv2.imread(image_path)
        if image is None:
            continue 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if not os.path.exists(mask_path):
            continue 
        masks = np.load(mask_path)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        filtered_masks = []
        for current_mask in masks:
            masked_purple = purple_mask[current_mask == 1]
            masked_green = green_mask[current_mask == 1]
            total_area = np.sum(current_mask)
            purple_area = np.sum(masked_purple > 0)
            green_area = np.sum(masked_green > 0)
            purple_percentage = (purple_area / total_area) * 100 if total_area > 0 else 0
            green_percentage = (green_area / total_area) * 100 if total_area > 0 else 0
            labeled_array, num_features = label(masked_purple > 0)
            large_areas_count = np.sum([np.sum(labeled_array == i) > 100 for i in range(1, num_features + 1)])
            q_value = calculate_q_value(current_mask)  
            if 20 <= purple_percentage <= 75 and large_areas_count <= 1 and green_percentage >= 16 and q_value < 0.1:
                filtered_masks.append(current_mask)
        if filtered_masks:
            filtered_masks = np.array(filtered_masks)
            new_mask_path = os.path.join(base_mask_path, f'masks_full_full_{image_number}.npy')
            np.save(new_mask_path, filtered_masks)
            print(f'Image {image_number}: Filtered masks count = {len(filtered_masks)}')
        else:
            print(f'Image {image_number}: No masks saved due to filter criteria')

def process_mask_mappings(image_number, base_mask_path):
    full_mask_path = os.path.join(base_mask_path, f'masks_full_full{image_number}.npy')
    head_mask_path = os.path.join(base_mask_path, f'mask_head_{image_number}.npy')
    if not os.path.exists(full_mask_path) or not os.path.exists(head_mask_path):
        print(f"One of the mask files for image {image_number} does not exist.")
        return
    full_masks = np.load(full_mask_path)
    head_masks = np.load(head_mask_path)
    mapping_counts = np.zeros(len(full_masks), dtype=int)
    for head_mask in head_masks:
        head_area = np.sum(head_mask)
        for i, full_mask in enumerate(full_masks):
            overlap_area = np.sum(head_mask & full_mask) 
            overlap_percentage = (overlap_area / head_area) * 100
            if overlap_percentage >= 80:
                mapping_counts[i] += 1
                break 
    to_delete = np.where(mapping_counts > 1)[0]
    if len(to_delete) > 0:
        filtered_masks = np.delete(full_masks, to_delete, axis=0)
        np.save(full_mask_path, filtered_masks)
        print(f"Image {image_number}: Removed {len(to_delete)} layers due to multiple mappings.")
    else:
        print(f"Image {image_number}: No layers removed, all mappings are unique.")

def delete_empty_files(base_path):
    files = [f for f in os.listdir(base_path) if f.startswith('masks_full_') and f.endswith('.npy')]
    for file in files:
        file_path = os.path.join(base_path, file)
        try:
            data = np.load(file_path)
            if data.size == 0:
                os.remove(file_path)
                print(f"Deleted empty file: {file}")
        except Exception as e:
            print(f"Error processing file {file}: {e}")

def process_image_with_contour_method(img, area_threshold=300):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    small_impurities = [c for c in contours if cv2.contourArea(c) < area_threshold]
    cv2.drawContours(img, small_impurities, -1, (255, 255, 255), thickness=cv2.FILLED)
    return img

def process_image_skimage(image_opencv, min_size=300, color_threshold=[200, 200, 200]):
    image_rgb = cv2.cvtColor(image_opencv, cv2.COLOR_BGR2RGB)
    gray_sk = color.rgb2gray(image_rgb)
    threshold_value = filters.threshold_otsu(gray_sk)
    binary = gray_sk < threshold_value
    cleaned = morphology.remove_small_objects(binary, min_size=min_size)
    mask = np.where(cleaned == 0, 1, 0).astype(bool)
    image_rgb[mask] = [255, 255, 255]
    whitening_mask_sk = np.all(image_rgb > color_threshold, axis=-1)
    image_rgb[whitening_mask_sk] = [255, 255, 255]
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

def generate_tail_image(start_index, end_index):
    for i in range(start_index, end_index+1):  
        img_number = f"{i:03d}"
        image_path = f'2 synchronized brightness sperm\\{img_number}.jpg'  
        mask_path = f'2 synchronized brightness sperm\\masks_q_{img_number}.npy' 
        image = cv2.imread(image_path)
        masks = np.load(mask_path)
        if image is None:
            raise ValueError(f"Can not load: {image_path}")
        for mask in masks:
            dilated_mask = binary_dilation(mask, iterations=1)
            for c in range(3):  
                image[:, :, c][dilated_mask] = 255
        output_path = f'2 synchronized brightness sperm\\{img_number}_pro_1.jpg'
        cv2.imwrite(output_path, image)
    for i in range(start_index, end_index+1):
        index = f"{i:03}"
        img_path = f'2 synchronized brightness sperm\\{index}_pro_1.jpg' 
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_no_residual = process_image_with_contour_method(img)
        img_whitened = process_image_skimage(img_no_residual)
        output_path = f'2 synchronized brightness sperm\\{index}_pro_2.jpg' 
        cv2.imwrite(output_path, img_whitened)
    print("Processing completed.")

def segmentation_and_masking(directory_path, base_image_path, sam_checkpoint, model_type, start_index, end_index):
    SAM_seg(sam_checkpoint, model_type, "cuda", start_index, end_index)
    layer_filter(directory_path, start_index, end_index)
    q_value_filter(start_index, end_index)
    Pick_header_files(base_image_path, directory_path, start_index, end_index)
    Screening_intact_sperm(start_index, end_index, base_image_path, directory_path)
    for i in range(start_index, end_index+1): 
        image_number = f'{i:03}'
        process_mask_mappings(image_number, directory_path)
    delete_empty_files(directory_path)

def main():
    start_index = 1      
    end_index = 25  
    directory_path = '2 synchronized brightness sperm' # Folder to place super-resolution images.
    base_image_path = 'original_image' # Folder to place original images.

    preprocessing(start_index, end_index, directory_path)

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "default"
    
    # sam_checkpoint = "sam_vit_b_01ec64.pth"
    # model_type = "vit_b"    

    # sam_checkpoint = "sam_vit_l_0b3195.pth"
    # model_type = "vit_l"  
    
    segmentation_and_masking(directory_path, base_image_path, sam_checkpoint, model_type, start_index, end_index)

    generate_tail_image(start_index, end_index)

if __name__ == "__main__":
    main()