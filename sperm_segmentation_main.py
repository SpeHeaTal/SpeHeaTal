import os
import cv2
import csv
import glob
import math
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from collections import defaultdict
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import pdist
from skimage.measure import label, regionprops
from scipy.spatial.distance import euclidean
from skimage import morphology, io, color, measure
from itertools import combinations, product
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter, binary_dilation, convolve

def Clustering(j, run_folder_path):
    if os.path.exists(run_folder_path) and os.path.isdir(run_folder_path):
        files_and_folders = os.listdir(run_folder_path)
        if files_and_folders:
            for item in files_and_folders:
                item_path = os.path.join(run_folder_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path) 
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
    image_path = f'2 synchronized brightness sperm/{j}_pro_2.jpg'
    original_image = Image.open(image_path)
    gray_image_array = np.array(original_image.convert('L'))
    binary_threshold = 128
    binary_image = gray_image_array < binary_threshold
    selem = morphology.disk(0)  
    cleaned_image = morphology.opening(binary_image, selem)
    cleaned_image_array = np.where(cleaned_image, 0, 255).astype(np.uint8) 
    cleaned_image = Image.fromarray(cleaned_image_array)
    cleaned_image_skimage = np.array(cleaned_image)
    if cleaned_image_skimage.shape[-1] == 4:  
        cleaned_image_skimage = color.rgba2rgb(cleaned_image_skimage)
    elif cleaned_image_skimage.shape[-1] == 3: 
        cleaned_image_skimage = color.rgb2gray(cleaned_image_skimage)
    binary_image = cleaned_image_skimage < 0.1
    label_image = measure.label(binary_image, connectivity=2)
    area_threshold = 25 
    final_image = morphology.remove_small_objects(label_image, min_size=area_threshold)
    final_image = (final_image > 0) * 255
    final_image = final_image.astype(np.uint8)
    inverted_image = 255 - final_image
    inverted_image_pil = Image.fromarray(inverted_image)
    inverted_image_pil.save('Run\\inverted_image.jpg')
    colors = ['orange', 'red', 'gold', 'skyblue', 'green', 'purple', 'pink', 'brown']
    image_path = 'Run\\inverted_image.jpg' 
    image = io.imread(image_path)
    if image.shape[-1] == 3: 
        image = color.rgb2gray(image)
    binary_image = image < 1
    black_pixels = np.column_stack(np.where(binary_image))
    df_black_pixels = pd.DataFrame(black_pixels, columns=['y', 'x'])
    dbscan = DBSCAN(eps=4, min_samples=1)
    clusters = dbscan.fit_predict(df_black_pixels[['x', 'y']])
    df_black_pixels['cluster'] = clusters
    num_clusters = len(set(clusters))
    if num_clusters > len(colors):
        colors += ['C{}'.format(i) for i in range(num_clusters - len(colors))]
    csv_path = image_path.replace('.jpg', '.csv')
    df_black_pixels.to_csv(csv_path, index=False)
    csv_path = 'Run\\inverted_image.csv' 
    df = pd.read_csv(csv_path)
    q_values = {}
    areas = {}
    for cluster_id in df['cluster'].unique():
        if cluster_id != -1:  
            cluster_points = df[df['cluster'] == cluster_id][['x', 'y']].values
            if len(cluster_points) < 2:
                continue
            distances = pdist(cluster_points, 'sqeuclidean')
            S = np.max(distances)
            s = len(cluster_points)
            q_values[cluster_id] = s / S
            areas[cluster_id] = s
    clusters_to_keep = [cluster_id for cluster_id, q in q_values.items() if q <= 0.08 and areas[cluster_id] >= 100]
    df_filtered = df[df['cluster'].isin(clusters_to_keep)].copy()
    unique_clusters = sorted(df_filtered['cluster'].unique())
    cluster_mapping = {old: new for new, old in enumerate(unique_clusters)}
    df_filtered.loc[:, 'cluster'] = df_filtered['cluster'].map(cluster_mapping)
    updated_csv_path = 'Run\\inverted_image_filtered.csv'  
    df_filtered.to_csv(updated_csv_path, index=False)

def calculate_average_coordinates(layer):
    y_coords, x_coords = np.where(layer)
    if len(x_coords) > 0 and len(y_coords) > 0:
        return np.mean(x_coords), np.mean(y_coords)
    return None, None

def Skeletonization_labeling():
    csv_file_path = 'Run\\inverted_image_filtered.csv' 
    data = pd.read_csv(csv_file_path)
    clusters = data['cluster'].unique()
    final_image_shape = (540, 720)
    all_layers = []
    for a in clusters:
        filtered_data = data[data['cluster'] == a]
        coordinates = filtered_data[['y', 'x']].values.astype(int)
        mask = np.zeros(final_image_shape, dtype=np.uint8)
        for y, x in coordinates:
            if 0 <= y < final_image_shape[0] and 0 <= x < final_image_shape[1]:
                mask[y, x] = 1
        radius_for_smoothing = 5
        smoothed_mask = morphology.binary_closing(mask, morphology.disk(radius_for_smoothing))
        skeleton = morphology.skeletonize(smoothed_mask)
        if np.sum(skeleton) >= 15:
            all_layers.append(skeleton)
    if not all_layers:
        print("No layers with more than 15 pixels were found.")
    else:
        multi_layer_skeleton = np.stack(all_layers, axis=0)
        npy_file_path = 'Run\\multi_layer_skeletonized_image.npy'
        np.save(npy_file_path, multi_layer_skeleton)
        print(f"Saved multi-layer skeletonized image as numpy array to {npy_file_path}")

def get_endpoints(layer):
    pad_layer = np.pad(layer, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    binary_layer = (pad_layer > 0).astype(np.uint8)
    kernels = [
        np.array([[-1, -1, 0], [-1, 1, 1], [-1, -1, 0]], dtype=np.int8),
        np.array([[-1, -1, -1], [-1, 1, -1], [0, 1, 0]], dtype=np.int8),
        np.array([[0, -1, -1], [1, 1, -1], [0, -1, -1]], dtype=np.int8),
        np.array([[0, 1, 0], [-1, 1, -1], [-1, -1, -1]], dtype=np.int8),
        np.array([[-1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.int8),
        np.array([[-1, -1, -1], [-1, 1, -1], [1, -1, -1]], dtype=np.int8),
        np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, -1]], dtype=np.int8),
        np.array([[-1, -1, 1], [-1, 1, -1], [-1, -1, -1]], dtype=np.int8)
    ]
    endpoint_img = np.zeros_like(binary_layer, dtype=np.uint8)
    for kernel in kernels:
        hitmiss = cv2.morphologyEx(binary_layer, cv2.MORPH_HITMISS, kernel)
        endpoint_img = cv2.bitwise_or(endpoint_img, hitmiss)
    y_coords, x_coords = np.where(endpoint_img > 0)
    endpoints = [(x - 1, y - 1) for x, y in zip(x_coords, y_coords)]
    return endpoints

def get_layer_endpoints(file_path):
    data = np.load(file_path)
    layer_endpoints = {}
    for layer_index in range(data.shape[0]):
        layer = data[layer_index, :, :]
        endpoints = get_endpoints(layer)
        layer_endpoints[layer_index] = endpoints
    return layer_endpoints

def find_branch_points(skeleton):
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    convolved = convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
    return np.logical_and(skeleton, convolved >= 13)

def find_endpoints(skeleton):
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    convolved = convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
    return np.logical_and(skeleton, np.isin(convolved, [11, 21]))

def is_fully_connected(skeleton):
    labeled_skeleton, num_components = label(skeleton, return_num=True)
    return num_components == 1

def is_adjacent_to_branch(endpoint, branch_points):
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            neighbor = (endpoint[0] + dx, endpoint[1] + dy)
            if neighbor in branch_points:
                return True
    return False

def process_skeletons(skeleton):
    branch_points = find_branch_points(skeleton)
    branch_point_coordinates = np.argwhere(branch_points)
    skeleton_without_branches = skeleton.copy()
    for coord in branch_point_coordinates:
        skeleton_without_branches[tuple(coord)] = 0
    labeled_skeleton, _ = label(skeleton_without_branches, return_num=True)
    small_components = [region for region in regionprops(labeled_skeleton) if region.area < 8]
    for component in small_components:
        component_mask = np.zeros_like(skeleton_without_branches, dtype=bool)
        for coord in component.coords:
            component_mask[tuple(coord)] = True
        endpoints = find_endpoints(component_mask)
        endpoints_coordinates = np.argwhere(endpoints)
        if len(endpoints_coordinates) != 2:
            continue
        branch_points_list = [tuple(coord) for coord in branch_point_coordinates]
        if not all(is_adjacent_to_branch(endpoint, branch_points_list) for endpoint in endpoints_coordinates):
            for coord in component.coords:
                skeleton_without_branches[tuple(coord)] = 0
    for point in branch_point_coordinates:
        skeleton_without_branches[tuple(point)] = 1
    return skeleton_without_branches

def Endpoints_intersections():
    skeletonized_image_path = 'Run\\multi_layer_skeletonized_image.npy'
    skeletonized_segmentations = np.load(skeletonized_image_path)
    processed_layers = []
    average_coordinates = [] 
    for layer in skeletonized_segmentations:
        processed_layer = process_skeletons(layer)
        processed_layers.append(processed_layer)
        avg_x, avg_y = calculate_average_coordinates(processed_layer)
        average_coordinates.append((avg_x, avg_y))
    processed_multi_layer_skeleton = np.stack(processed_layers, axis=0)
    npy_file_path = 'Run\\processed_multi_layer_skeleton.npy'
    np.save(npy_file_path, processed_multi_layer_skeleton)

def check_endpoints_in_new_file(old_endpoints, new_file_path):
    new_data = np.load(new_file_path)
    endpoint_checks = {}
    for layer_index, endpoints in old_endpoints.items():
        new_layer = new_data[layer_index, :, :]
        checks = []
        for endpoint in endpoints:
            x, y = endpoint
            is_foreground = new_layer[y, x] > 0
            checks.append(is_foreground)
        endpoint_checks[layer_index] = checks
    return endpoint_checks

def Saved_layers(new_file_path, num_endpoints_per_layer, num_not_foreground_per_layer, endpoints_per_layer_two):
    new_data = np.load(new_file_path)
    differences = {}
    for layer_index in num_endpoints_per_layer.keys():
        diff = num_endpoints_per_layer[layer_index] - num_not_foreground_per_layer.get(layer_index, 0)
        differences[layer_index] = diff
    layers_to_save = [layer for layer, diff in differences.items() if diff >= 1]
    for layer_index, layer in enumerate(new_data):
        if len(endpoints_per_layer_two[layer_index]) > 6:                    
            layers_to_save.append(layer_index)
    layers_to_save = list(set(layers_to_save)) 
    layers_to_save_data = new_data[layers_to_save, :, :]
    save_file_path = 'Run\\layers_with_difference_gt_4.npy' 
    np.save(save_file_path, layers_to_save_data)
    print(f"Saved layers with differences greater than 4 or endpoints greater than 6 to '{save_file_path}'")

def combined_skeletonized_show():
    file_path = 'Run\\layers_with_difference_gt_4.npy' 
    saved_layers = np.load(file_path)
    if saved_layers.size == 0:
        print("No layers to process.")

def get_junction_points(layer):
    pad_layer = np.pad(layer, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    binary_layer = (pad_layer > 0).astype(np.uint8)
    kernels = [
        np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=np.uint8),
        np.array([[0, 1, 0], [0, 1, 1], [1, 0, 0]], dtype=np.uint8),
        np.array([[0, 0, 1], [1, 1, 0], [0, 0, 1]], dtype=np.uint8),
        np.array([[1, 0, 0], [0, 1, 1], [0, 1, 0]], dtype=np.uint8),
        np.array([[0, 1, 0], [0, 1, 0], [1, 0, 1]], dtype=np.uint8),
        np.array([[0, 0, 1], [1, 1, 0], [0, 1, 0]], dtype=np.uint8),
        np.array([[1, 0, 0], [0, 1, 1], [1, 0, 0]], dtype=np.uint8),
        np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.uint8),
        np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1]], dtype=np.uint8),
        np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8),
        np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]], dtype=np.uint8),
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.uint8)
    ]
    junction_img = np.zeros_like(binary_layer, dtype=np.uint8)
    for kernel in kernels:
        hitmiss = cv2.morphologyEx(binary_layer, cv2.MORPH_HITMISS, kernel)
        junction_img = cv2.bitwise_or(junction_img, hitmiss)
    y_coords, x_coords = np.where(junction_img > 0)
    junctions = [(x - 1, y - 1) for x, y in zip(x_coords, y_coords)]
    return junctions

def get_all_layers_junctions(file_path):
    data = np.load(file_path)
    all_layers_junctions = {}
    for layer_index in range(data.shape[0]):
        layer = data[layer_index, :, :]
        junctions = get_junction_points(layer)
        all_layers_junctions[layer_index] = junctions
    return all_layers_junctions

def remove_points_within_radius(layer, points, radius):
    """Remove points within a given radius around each point in the layer."""
    for x, y in points:
        mask = np.zeros_like(layer, dtype=np.uint8)
        cv2.circle(mask, (x, y), radius, color=1, thickness=-1)
        layer[mask == 1] = 0
    return layer

def process_layers(file_path, junctions_per_layer, radius=3):
    data = np.load(file_path)
    if data.size == 0 or data.shape[0] == 0:
        print("No layers to process.")
        return data
    for layer_index in range(data.shape[0]):
        layer = data[layer_index, :, :]
        if layer_index in junctions_per_layer:
            junctions = junctions_per_layer[layer_index]
            data[layer_index, :, :] = remove_points_within_radius(layer, junctions, radius)
    np.save(file_path, data)

def split_layers_into_connected_components(file_path, area_threshold=10):
    """Split each layer of the npy file into connected components and return a new array."""
    data = np.load(file_path)
    if data.size == 0 or data.shape[0] == 0:
        print("No layers to process.")
        return np.array([])
    new_layers = []
    for layer in data:
        labeled_layer, num_features = measure.label(layer, return_num=True)
        for i in range(1, num_features + 1):
            component = (labeled_layer == i)
            area = np.sum(component)
            if area >= area_threshold:
                new_layers.append(component)
    return np.array(new_layers, dtype=np.uint8)

def save_into_csv(endpoints_per_layer, endpoint_checks_per_layer):
    file_path = 'Run\\processed_multi_layer_skeleton.npy'
    multi_layer_skeleton = np.load(file_path)
    num_endpoints_per_layer = {layer: len(endpoints) for layer, endpoints in endpoints_per_layer.items()}
    num_not_foreground_per_layer = {layer: checks.count(False) for layer, checks in endpoint_checks_per_layer.items()}
    list1 = []  
    list2 = []
    list3 = []
    list4 = []
    for layer_index in num_endpoints_per_layer.keys():
        diff = num_endpoints_per_layer[layer_index] - num_not_foreground_per_layer.get(layer_index, 0)
        print(f"Layer {layer_index}: Difference in numbers = {diff}")
        if diff > 2 :
            list1.append(layer_index)
        elif (diff in [0, 1]):
            list4.append(layer_index)
    np.save('Run\\pass.npy', multi_layer_skeleton[list1])
    np.save('Run\\3.npy', multi_layer_skeleton[list3])
    all_layers = set(range(multi_layer_skeleton.shape[0]))
    excluded_layers = all_layers - set(list1) - set(list2) - set(list3) - set(list4)
    np.save('Run\\0.npy', multi_layer_skeleton[list(excluded_layers)])
    zero_layers = np.load('Run\\0.npy')
    save_dir = 'Run'
    os.makedirs(save_dir, exist_ok=True)
    combined_df = pd.DataFrame(columns=['X', 'Y', 'Label'])
    for i, layer in enumerate(zero_layers):
        y_coords, x_coords = np.where(layer)
        layer_df = pd.DataFrame({'X': x_coords, 'Y': y_coords, 'Label': i + 1})
        combined_df = pd.concat([combined_df, layer_df], ignore_index=True)
    csv_file_name = '1_1.csv'
    csv_file_path = os.path.join(save_dir, csv_file_name)
    combined_df.to_csv(csv_file_path, index=False)
    print(f'Saved combined data to {csv_file_path}')

def combined_df(run_folder_path):
    csv_files = glob.glob(os.path.join(run_folder_path, '*_1.csv'))
    combined_df = pd.DataFrame(columns=['X', 'Y', 'Label'])
    label_offset = 0
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if df.empty:
            continue
        if '1_1.csv' not in csv_file:
            df['Label'] += label_offset
        if not df.empty:
            label_offset = df['Label'].max()
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    if combined_df.empty:
        combined_df = pd.DataFrame({'X': [0], 'Y': [0], 'Label': [1]})
    combined_df.to_csv('Run/total.csv', index=False)
    print('Saved combined data to Run/total.csv')

def load_csv_data(csv_file_path):
    return pd.read_csv(csv_file_path)

def load_npy_data(npy_file_path):
    return np.load(npy_file_path)

def add_layer_to_csv(layer, label, csv_data):
    y_indices, x_indices = np.nonzero(layer) 
    layer_df = pd.DataFrame({'X': x_indices, 'Y': y_indices, 'Label': label})
    return pd.concat([csv_data, layer_df], ignore_index=True)

def update_csv_with_layers(csv_file_path, npy_file_path):
    csv_data = load_csv_data(csv_file_path)
    npy_data = load_npy_data(npy_file_path)
    max_label = csv_data['Label'].max()
    for i, layer in enumerate(npy_data):
        new_label = max_label + 1 + i
        csv_data = add_layer_to_csv(layer, new_label, csv_data)
    return csv_data

def updated_csv_data():
    csv_file_path = 'Run/total.csv'
    npy_file_path = 'Run\\layers_with_difference_gt_4.npy'
    updated_csv_data = update_csv_with_layers(csv_file_path, npy_file_path)
    print(updated_csv_data.tail())
    updated_csv_data.to_csv('Run/total.csv', index=False)
    file_path = 'Run/total.csv'  
    new_file_path = 'Run/total.csv' 
    data = pd.read_csv(file_path)
    data_no_duplicates = data.drop_duplicates(subset=['X', 'Y'], keep='first')
    data_no_duplicates.to_csv(new_file_path, index=False)
    print(f"Duplicates removed. Data saved to {new_file_path}.")

def Reassign_Label():
    data = pd.read_csv('Run/total.csv')
    label_counts = data['Label'].value_counts()
    single_entry_labels = label_counts[label_counts == 1].index
    data = data[~data['Label'].isin(single_entry_labels)]
    unique_labels = data['Label'].unique()
    new_label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels, start=1)}
    data['Label'] = data['Label'].map(new_label_mapping)
    data.to_csv('Run/total.csv', index=False)

def calculate_slope1(x, y):
    if len(x) <= 1:
        return np.nan  
    slope, _ = np.polyfit(x, y, 1)
    return slope

def calculate_vector_angle(vector1, vector2):
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector1, unit_vector2)
    angle = np.arccos(dot_product)
    return np.degrees(angle)

def find_farthest_points(group):
    max_distance = 0
    farthest_pair = None
    for point1, point2 in combinations(group.itertuples(), 2):
        distance = euclidean((point1.X, point1.Y), (point2.X, point2.Y))
        if distance > max_distance:
            max_distance = distance
            farthest_pair = ((point1.X, point1.Y), (point2.X, point2.Y))
    return farthest_pair

def find_closest_distance_between_groups(group1_endpoints, group2_endpoints):
    min_distance = float('inf')
    closest_points = None
    for point1, point2 in product(group1_endpoints, group2_endpoints):
        distance = euclidean(point1, point2)
        if distance < min_distance:
            min_distance = distance
            closest_points = (point1, point2)
    return closest_points, min_distance

def calculate_acute_angle(m1, m2):
    tan_theta = abs((m2 - m1) / (1 + m1 * m2))
    theta_radians = math.atan(tan_theta)
    if theta_radians > math.pi / 2:
        theta_radians = math.pi - theta_radians
    return math.degrees(theta_radians)

def BLS(distance_threshold, N):
    file_path = 'Run/total.csv' 
    data = pd.read_csv(file_path)
    average_slopes = {}
    for label in data['Label'].unique():
        label_data = data[data['Label'] == label]
        average_slopes[label] = calculate_slope1(label_data['X'], label_data['Y'])
    farthest_points = {}
    for label in data['Label'].unique():
        label_data = data[data['Label'] == label]
        farthest_points[label] = find_farthest_points(label_data)
    all_pairs_closest_distances = []
    for label1, label2 in combinations(data['Label'].unique(), 2):
        distance = find_closest_distance_between_groups(farthest_points[label1], farthest_points[label2])
        all_pairs_closest_distances.append((label1, label2, distance))
    eligible_pairs = []
    for label1, label2, _ in all_pairs_closest_distances:
        farthest_points1 = farthest_points[label1]
        farthest_points2 = farthest_points[label2]
        closest_pair_of_points, distance = find_closest_distance_between_groups(farthest_points1, farthest_points2)
        if closest_pair_of_points[0] == farthest_points1[0]:
            vector_layer1 = np.array([farthest_points1[1][0] - farthest_points1[0][0], farthest_points1[1][1] - farthest_points1[0][1]])
        else:
            vector_layer1 = np.array([farthest_points1[0][0] - farthest_points1[1][0], farthest_points1[0][1] - farthest_points1[1][1]])
        if closest_pair_of_points[1] == farthest_points2[0]:
            vector_layer2 = np.array([farthest_points2[1][0] - farthest_points2[0][0], farthest_points2[1][1] - farthest_points2[0][1]])
        else:
            vector_layer2 = np.array([farthest_points2[0][0] - farthest_points2[1][0], farthest_points2[0][1] - farthest_points2[1][1]])
        angle_between_vectors = calculate_vector_angle(vector_layer1, vector_layer2)
        if calculate_acute_angle(average_slopes[label1], average_slopes[label2]) < N and distance < distance_threshold:
            if angle_between_vectors >= 90:
                eligible_pairs.append((label1, label2, distance))
    selected_pairs = []
    used_labels = set()
    for label1, label2, _ in eligible_pairs:
        if label1 not in used_labels and label2 not in used_labels:
            selected_pairs.append((label1, label2))
            used_labels.add(label1)
            used_labels.add(label2)
    for label1, label2 in selected_pairs:
        data.loc[data['Label'] == label2, 'Label'] = label1
    unique_labels = sorted(data['Label'].unique())
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels, start=1)}
    data['Label'] = data['Label'].map(label_mapping)
    data.to_csv(file_path, index=False)
    print("Label merging complete. Results have been saved to the original file.")

def clean_and_relabel(csv_file_path, n):
    data = pd.read_csv(csv_file_path)
    label_counts = data['Label'].value_counts()
    labels_to_remove = label_counts[label_counts < n].index
    data_cleaned = data[~data['Label'].isin(labels_to_remove)]
    unique_labels = sorted(data_cleaned['Label'].unique())
    new_label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels, start=1)}
    data_cleaned['Label'] = data_cleaned['Label'].map(new_label_mapping)
    print(f"Removed labels: {list(labels_to_remove)}")
    print(f"Original label count: {len(label_counts)}")
    print(f"New label count: {len(unique_labels)}")
    new_file_path = csv_file_path.replace('.csv', '.csv')
    data_cleaned.to_csv(new_file_path, index=False)
    print(f"Cleaned data saved to {new_file_path}")

def find_nearest_outside_15px(point, others):
    distances = np.sqrt((others['X'] - point[0]) ** 2 + (others['Y'] - point[1]) ** 2)
    outside_15px = distances > 15 
    if outside_15px.any():
        nearest_index = distances[outside_15px].idxmin()
        return others.loc[nearest_index, ['X', 'Y']].values
    return [None, None]  

def calculate_slope(x1, y1, x2, y2):
    if x1 == x2:
        return np.inf
    elif y1 == y2:
        return 0
    else:
        return (y2 - y1) / (x2 - x1)

def calculate_distance(x1, y1, x2, y2):
    return euclidean((x1, y1), (x2, y2))

def calculate_angle(slope1, slope2):
    if slope1 in [np.inf, -np.inf] or slope2 in [np.inf, -np.inf]:
        return 90.0
    elif slope1 == 0 and slope2 == 0:
        return 0.0
    else:
        try:
            tan_theta = abs((slope2 - slope1) / (1 + slope1 * slope2))
            angle = np.degrees(np.arctan(tan_theta))
            return angle if angle <= 90 else 180 - angle
        except ZeroDivisionError:
            return 90.0

def calculate_angle2(x1, y1, x2, y2):
    if x2 == x1: 
        angle = 90
    else:
        slope = (y2 - y1) / (x2 - x1)
        angle = math.degrees(math.atan(slope))
    return abs(angle) 

def deduplicate_pairs(df):
    sorted_pairs = pd.DataFrame(np.sort(df[['Label1', 'Label2']], axis=1), columns=['LabelA', 'LabelB']).drop_duplicates()
    return sorted_pairs

def merge_labels(total_df, pairs_df):
    merge_map = {row['LabelB']: row['LabelA'] for _, row in pairs_df.iterrows()}
    total_df['Label'] = total_df['Label'].replace(merge_map)
    return total_df

def reindex_labels(df):
    unique_labels = sorted(df['Label'].unique())
    label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels, start=1)}
    df['Label'] = df['Label'].map(label_map)
    return df

def process_and_save_data(optimal_pairs_df, total_csv_path, output_csv_path):
    total_df = pd.read_csv(total_csv_path)
    deduplicated_pairs_df = deduplicate_pairs(optimal_pairs_df)
    merged_total_df = merge_labels(total_df, deduplicated_pairs_df)
    reindexed_total_df = reindex_labels(merged_total_df)
    reindexed_total_df.to_csv(output_csv_path, index=False)

def tangent_merge():
    data = pd.read_csv('Run/total.csv')
    points_info = []
    for label, group in data.groupby('Label'):
        distances = squareform(pdist(group[['X', 'Y']]))
        np.fill_diagonal(distances, 0) 
        farthest_pair_idx = np.unravel_index(np.argmax(distances, axis=None), distances.shape)
        point_a = group.iloc[farthest_pair_idx[0]][['X', 'Y']].values
        point_b = group.iloc[farthest_pair_idx[1]][['X', 'Y']].values
        point_a1 = find_nearest_outside_15px(point_a, group)
        point_b1 = find_nearest_outside_15px(point_b, group)
        points_info.append([label, *point_a, *point_b, *point_a1, *point_b1])
    points_df = pd.DataFrame(points_info, columns=['Label', 'Ax', 'Ay', 'Bx', 'By', 'A1x', 'A1y', 'B1x', 'B1y'])
    points_df.to_csv('Run/farthest_points.csv', index=False)
    file_path = 'Run/farthest_points.csv' 
    data = pd.read_csv(file_path)       
    data['Slope_AA1'] = data.apply(lambda row: calculate_slope(row['Ax'], row['Ay'], row['A1x'], row['A1y']), axis=1)
    data['Slope_BB1'] = data.apply(lambda row: calculate_slope(row['Bx'], row['By'], row['B1x'], row['B1y']), axis=1)
    output_file_path = 'Run/farthest_points.csv' 
    data.to_csv(output_file_path, index=False)
    print(f"Data with slopes saved to {output_file_path}.")
    data = pd.read_csv('Run/farthest_points.csv')
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    data.to_csv('Run/farthest_points.csv', index=False)
    data = pd.read_csv('Run/farthest_points.csv')
    potential_pairs = []
    for i, row1 in data.iterrows():
        for j, row2 in data.iterrows():
            if i == j:
                continue
            for end_point1 in ['A', 'B']:
                for suffix1 in ['', '1']:
                    for end_point2 in ['A', 'B']:
                        for suffix2 in ['', '1']:
                            x1, y1 = row1[f'{end_point1}{suffix1}x'], row1[f'{end_point1}{suffix1}y']
                            x2, y2 = row2[f'{end_point2}{suffix2}x'], row2[f'{end_point2}{suffix2}y']
                            distance = calculate_distance(x1, y1, x2, y2)
                            slope1 = calculate_slope(x1, y1, row1[f'{end_point1}1x'], row1[f'{end_point1}1y'])
                            slope2 = calculate_slope(x2, y2, row2[f'{end_point2}1x'], row2[f'{end_point2}1y'])
                            angle = calculate_angle(slope1, slope2)
                            if distance < 10 and angle < 30:
                                potential_pairs.append((row1['Label'], row2['Label'], distance, angle))
    potential_pairs_df = pd.DataFrame(potential_pairs, columns=['Label1', 'Label2', 'Distance', 'Angle'])
    sorted_pairs = potential_pairs_df.sort_values(by=['Distance', 'Angle'])
    optimal_pairs_df = sorted_pairs.drop_duplicates(subset=['Label1', 'Label2'], keep='first')
    print("Optimized matched pairs have been saved to optimized_matched_pairs.csv.")
    return optimal_pairs_df

def apply_dbscan_and_rename_clusters(data, eps=20, min_samples=5):
    updated_data = []
    for label in data['Label'].unique():
        label_indices = data[data['Label'] == label].index
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(data.loc[label_indices, ['X', 'Y']])
        if len(set(clusters)) - (1 if -1 in clusters else 0) > 1:
            new_labels = [f"{label}_{cluster}" if cluster != -1 else label for cluster in clusters]
            data.loc[label_indices, 'Label'] = new_labels
        updated_data.append(data.loc[label_indices])
    return pd.concat(updated_data, ignore_index=True)

def read_original_csv(csv_path):
    return pd.read_csv(csv_path)

def process_data(data):
    data['Label'] = data['Label'].astype(str)

    connected_labels = data[data['Label'].str.contains('_')]
    if not connected_labels.empty:
        split_labels = connected_labels['Label'].str.split('_', expand=True)
        connected_labels = connected_labels.assign(MajorLabel=split_labels[0], MinorLabel=split_labels[1])
    else:
        return data
    results = []
    for major_label, group in connected_labels.groupby('MajorLabel'):
        subgroups = list(group.groupby('Label'))
        distances = []
        for i in range(len(subgroups)):
            for j in range(i + 1, len(subgroups)):
                label1, data1 = subgroups[i]
                label2, data2 = subgroups[j]
                dist = cdist(data1[['X', 'Y']], data2[['X', 'Y']])
                idx1, idx2 = np.unravel_index(np.argmin(dist), dist.shape)
                point1, point2 = data1.iloc[idx1][['X', 'Y']], data2.iloc[idx2][['X', 'Y']]
                distances.append({
                    'Label 1': label1,
                    'Label 2': label2,
                    'Distance': np.min(dist),
                    'Point 1 (X, Y)': (point1['X'], point1['Y']),
                    'Point 2 (X, Y)': (point2['X'], point2['Y'])
                })
        sorted_distances = sorted(distances, key=lambda x: x['Distance'])
        results.extend(sorted_distances[:len(subgroups) - 1])
    result_df = pd.DataFrame(results)
    processed_data = data.copy()
    for _, row in result_df.iterrows():
        line_x = np.linspace(row['Point 1 (X, Y)'][0], row['Point 2 (X, Y)'][0], num=10)
        line_y = np.linspace(row['Point 1 (X, Y)'][1], row['Point 2 (X, Y)'][1], num=10)
        line_points = pd.DataFrame({'X': line_x, 'Y': line_y, 'Label': row['Label 1'].split('_')[0]})
        processed_data = pd.concat([processed_data, line_points], ignore_index=True)
    processed_data['Label'] = processed_data['Label'].str.split('_').str[0]
    return processed_data

def save_processed_data(processed_data, save_path):
    processed_data.to_csv(save_path, index=False)

def breakpoint_completion(original_csv_path, processed_csv_path):
    file_path = 'Run/total.csv' 
    data = pd.read_csv(file_path)
    updated_data = apply_dbscan_and_rename_clusters(data)
    updated_file_path = 'Run/total_2.csv'
    updated_data.to_csv(updated_file_path, index=False, float_format='%g')
    print(f"Updated data saved to {updated_file_path}")
    original_data = read_original_csv(original_csv_path)
    processed_data = process_data(original_data)
    save_processed_data(processed_data, processed_csv_path)

def visualize_and_save_filtered_data(file_path):
    data = pd.read_csv(file_path)
    label_counts = data['Label'].value_counts()
    labels_to_keep = label_counts[label_counts >= 30].index
    filtered_data = data[data['Label'].isin(labels_to_keep)]
    filtered_data.to_csv(file_path, index=False)

def find_farthest_points_npy(layer):
    y, x = np.where(layer > 0)
    points = np.array(list(zip(x, y)))
    if len(points) < 2:
        return None, None
    distances = cdist(points, points)
    farthest_pair_indices = np.unravel_index(distances.argmax(), distances.shape)
    return points[farthest_pair_indices[0]], points[farthest_pair_indices[1]]

def find_farthest_points_in_npy_layer(layer):
    y, x = np.where(layer > 0)
    points = np.c_[x, y]
    if points.shape[0] < 2:
        return None
    distances = cdist(points, points)
    farthest_pair_indices = np.unravel_index(np.argmax(distances), distances.shape)
    return points[farthest_pair_indices[0]], points[farthest_pair_indices[1]]

def find_farthest_points_in_group(group):
    points = group[['X', 'Y']].values
    if len(points) < 2:
        return None, None
    dist_matrix = cdist(points, points)
    np.fill_diagonal(dist_matrix, np.nan)
    farthest_pair_indices = np.unravel_index(np.nanargmax(dist_matrix), dist_matrix.shape)
    return points[farthest_pair_indices[0]], points[farthest_pair_indices[1]]

def find_nearest_point_outside_radius(points, reference_point, radius=20):
    distances = np.linalg.norm(points - reference_point, axis=1)
    outside_points = points[distances > radius]
    if len(outside_points) == 0:
        return None
    outside_distances = np.linalg.norm(outside_points - reference_point, axis=1)
    nearest_point_index = np.argmin(outside_distances)
    return outside_points[nearest_point_index]

def generate_mapping_table(npy_file_path, csv_file_path, output_csv_path):
    npy_data = np.load(npy_file_path)
    csv_data = pd.read_csv(csv_file_path)
    output_df = pd.DataFrame(columns=[
        "NPY Layer", "CSV Label",
        "Point A (X)", "Point A (Y)",
        "Point B (X)", "Point B (Y)",
        "Point C (X)", "Point C (Y)",
        "Point D (X)", "Point D (Y)"
    ])
    for npy_layer in range(npy_data.shape[0]):
        point_A, point_B = find_farthest_points_npy(npy_data[npy_layer])
        if point_A is None or point_B is None:
            continue
        for csv_label, group in csv_data.groupby('Label'):
            group_points = group[['X', 'Y']].values
            if group_points.shape[0] < 2:
                continue
            point_C1, point_C2 = find_farthest_points_in_group(group)
            if point_C1 is None or point_C2 is None:
                continue
            point_D1 = find_nearest_point_outside_radius(group_points, point_C1, radius=20)    
            point_D2 = find_nearest_point_outside_radius(group_points, point_C2, radius=20)
            if point_D1 is None or point_D2 is None:
                continue
            new_row_1 = pd.DataFrame({
                "NPY Layer": [npy_layer + 1],
                "CSV Label": [csv_label],
                "Point A (X)": [point_A[0]],
                "Point A (Y)": [point_A[1]],
                "Point B (X)": [point_B[0]],
                "Point B (Y)": [point_B[1]],
                "Point C (X)": [point_C1[0]],
                "Point C (Y)": [point_C1[1]],
                "Point D (X)": [point_D1[0]],
                "Point D (Y)": [point_D1[1]]
            })
            output_df = pd.concat([output_df, new_row_1], ignore_index=True)
            new_row_2 = pd.DataFrame({
                "NPY Layer": [npy_layer + 1],
                "CSV Label": [csv_label],
                "Point A (X)": [point_A[0]],
                "Point A (Y)": [point_A[1]],
                "Point B (X)": [point_B[0]],
                "Point B (Y)": [point_B[1]],
                "Point C (X)": [point_C2[0]],
                "Point C (Y)": [point_C2[1]],
                "Point D (X)": [point_D2[0]],
                "Point D (Y)": [point_D2[1]]
            })
            output_df = pd.concat([output_df, new_row_2], ignore_index=True)
    output_df.to_csv(output_csv_path, index=False)

def calculate_min_distance_and_closest_pair(row):
    point_A = (row['Point A (X)'], row['Point A (Y)'])
    point_B = (row['Point B (X)'], row['Point B (Y)'])
    point_C = (row['Point C (X)'], row['Point C (Y)'])
    point_D = (row['Point D (X)'], row['Point D (Y)'])
    distance_AC = np.linalg.norm(np.array(point_A) - np.array(point_C))
    distance_AD = np.linalg.norm(np.array(point_A) - np.array(point_D))
    distance_BC = np.linalg.norm(np.array(point_B) - np.array(point_C))
    distance_BD = np.linalg.norm(np.array(point_B) - np.array(point_D))
    distances = {'AC': distance_AC, 'AD': distance_AD, 'BC': distance_BC, 'BD': distance_BD}
    closest_pair = min(distances, key=distances.get)
    min_distance = distances[closest_pair]
    return min_distance, closest_pair

def determine_vector_direction(row):
    closest_pair = row['Closest Pair']
    vector_direction_npy = 'AB' if 'A' in closest_pair else 'BA'
    vector_direction_csv = 'CD' if 'C' in closest_pair else 'DC'
    return vector_direction_npy, vector_direction_csv

def gen_mapping_table_df():
    mapping_table_path = 'Run/mapping_table.csv'
    mapping_table_df = pd.read_csv(mapping_table_path)
    mapping_table_df[['Min Distance', 'Closest Pair']] = mapping_table_df.apply(
        calculate_min_distance_and_closest_pair, axis=1, result_type='expand')
    mapping_table_df[['Vector Direction (NPY)', 'Vector Direction (CSV)']] = mapping_table_df.apply(
        determine_vector_direction, axis=1, result_type='expand')
    updated_mapping_table_path = 'Run/mapping_table.csv'
    mapping_table_df.to_csv(updated_mapping_table_path, index=False)
    mapping_table_df = pd.read_csv('Run/mapping_table.csv')
    mapping_table_df[['Angle Between Vectors (Degrees)', 'Supplementary Angle (Degrees)']] = mapping_table_df.apply(
        calculate_angle_between_vectors, axis=1, result_type='expand')
    updated_csv_path_with_angles = 'Run/mapping_table.csv'
    mapping_table_df.to_csv(updated_csv_path_with_angles, index=False)

def calculate_angle_between_vectors(row):
    vectors = {
        'AB': (row['Point B (X)'] - row['Point A (X)'], row['Point B (Y)'] - row['Point A (Y)']),
        'BA': (row['Point A (X)'] - row['Point B (X)'], row['Point A (Y)'] - row['Point B (Y)']),
        'CD': (row['Point D (X)'] - row['Point C (X)'], row['Point D (Y)'] - row['Point C (Y)']),
        'DC': (row['Point C (X)'] - row['Point D (X)'], row['Point C (Y)'] - row['Point D (Y)'])
    }
    vector_npy = vectors[row['Vector Direction (NPY)']]
    vector_csv = vectors[row['Vector Direction (CSV)']]
    unit_vector_npy = vector_npy / np.linalg.norm(vector_npy)
    unit_vector_csv = vector_csv / np.linalg.norm(vector_csv)
    dot_product = np.clip(np.dot(unit_vector_npy, unit_vector_csv), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    supplementary_angle_deg = 180 - angle_deg
    return angle_deg, supplementary_angle_deg

def read_csv_file(csv_path):
    return pd.read_csv(csv_path)

def get_coordinates(row):
    pair = row['Closest Pair']
    points = {
        'AB': ((row['Point A (X)'], row['Point A (Y)']), (row['Point B (X)'], row['Point B (Y)'])),
        'AC': ((row['Point A (X)'], row['Point A (Y)']), (row['Point C (X)'], row['Point C (Y)'])),
        'AD': ((row['Point A (X)'], row['Point A (Y)']), (row['Point D (X)'], row['Point D (Y)'])),
        'BC': ((row['Point B (X)'], row['Point B (Y)']), (row['Point C (X)'], row['Point C (Y)'])),
        'BD': ((row['Point B (X)'], row['Point B (Y)']), (row['Point D (X)'], row['Point D (Y)'])),
        'CD': ((row['Point C (X)'], row['Point C (Y)']), (row['Point D (X)'], row['Point D (Y)'])),
    }
    return points[pair]

def create_line_points(row):
    coordinates = get_coordinates(row)
    line_x = np.linspace(coordinates[0][0], coordinates[1][0], num=25)
    line_y = np.linspace(coordinates[0][1], coordinates[1][1], num=25)
    label = row['CSV Label']
    return pd.DataFrame({'X': line_x, 'Y': line_y, 'Label': label})

def csv_sorting(j):
    mapping_table_df = pd.read_csv('Run/mapping_table.csv')
    filtered_by_distance_df = mapping_table_df[mapping_table_df['Min Distance'] <= 20]
    final_filtered_df = filtered_by_distance_df[filtered_by_distance_df['Supplementary Angle (Degrees)'] <= 25]
    pro_file_path = 'Run/mapping_table_pro.csv'
    final_filtered_df.to_csv(pro_file_path, index=False)
    source_file_path = 'Run/mapping_table.csv'
    output_file_path = 'Run/mapping_table_pro.csv'

    with open(pro_file_path, 'r') as pro_file:
        reader = csv.reader(pro_file)
        rows = list(reader)
        num_rows = len(rows)

    if num_rows == 1:
        with open(source_file_path, 'r') as source_file:
            reader = csv.reader(source_file)
            next(reader)
            second_row = next(reader)
        with open(output_file_path, 'w', newline='') as output_file:
            writer = csv.writer(output_file)
            writer.writerow(rows[0])
            writer.writerow(second_row)
        file_path = 'Run/no_layer.txt' 
        with open(file_path, 'a') as file:
            file.write(f"'{j}.jpg',\n")
    df = pd.read_csv('Run/mapping_table_pro.csv')
    deduplicated_df = df.sort_values(by='Min Distance').drop_duplicates(subset='NPY Layer', keep='first')
    deduplicated_csv_path = 'Run/mapping_table_pro.csv'
    deduplicated_df.to_csv(deduplicated_csv_path, index=False)
    df = pd.read_csv('Run/mapping_table_pro.csv') 
    df_deduplicated_min_distance = df.sort_values('Min Distance').drop_duplicates('CSV Label', keep='first')
    deduplicated_min_distance_csv_path = 'Run/mapping_table_pro_distance.csv' 
    df_deduplicated_min_distance.to_csv(deduplicated_min_distance_csv_path, index=False)
    df_deduplicated_min_distance.head(8)
    csv_file_path = 'Run/mapping_table_pro_distance.csv' 
    data = pd.read_csv(csv_file_path)
    for index, row in data.iterrows():
        angle_ab = calculate_angle2(row['Point A (X)'], row['Point A (Y)'],
                                row['Point B (X)'], row['Point B (Y)'])
        data.at[index, 'AB Direction'] = 1 if angle_ab < 45 else 0
    new_csv_file_path = 'Run/mapping_table_pro_distance.csv'
    data.to_csv(new_csv_file_path, index=False)
    mapping_table_path = 'Run/mapping_table_pro_distance.csv' 
    total_csv_path = 'Run//total_3.csv'
    mapping_table_data = read_csv_file(mapping_table_path)
    total_data = read_csv_file(total_csv_path)
    selected_labels = mapping_table_data['CSV Label'].unique().tolist()
    selected_pairs = mapping_table_data[mapping_table_data['CSV Label'].isin(selected_labels)]
    line_points_data = pd.concat([create_line_points(row) for _, row in selected_pairs.iterrows()], ignore_index=True)
    merged_data = pd.concat([total_data, line_points_data], ignore_index=True)
    merged_csv_path = 'Run/total_4.csv'
    merged_data.to_csv(merged_csv_path, index=False)
    csv_data_file_path = 'Run/total_4.csv'  
    data = pd.read_csv(csv_data_file_path)

def process_and_visualize(mapping_table_path, total_4_path):
    mapping_table = pd.read_csv(mapping_table_path)
    total_4 = pd.read_csv(total_4_path)
    csv_labels = mapping_table['CSV Label'].unique()
    filtered_total_4 = total_4[total_4['Label'].isin(csv_labels)]
    filtered_csv_path = 'Run/filtered_total_4.csv'
    filtered_total_4.to_csv(filtered_csv_path, index=False)

def tail_file_integration(j):
    image_path = f'2 synchronized brightness sperm/{j}_pro_2.jpg'
    original_image = Image.open(image_path)
    gray_image_array = np.array(original_image.convert('L'))
    binary_threshold = 128
    binary_image = gray_image_array < binary_threshold
    selem = morphology.disk(0)
    cleaned_image = morphology.opening(binary_image, selem)
    cleaned_image_array = np.where(cleaned_image, 0, 255).astype(np.uint8) 
    cleaned_image = Image.fromarray(cleaned_image_array)
    cleaned_image_skimage = np.array(cleaned_image)
    if cleaned_image_skimage.shape[-1] == 4: 
        cleaned_image_skimage = color.rgba2rgb(cleaned_image_skimage)
    elif cleaned_image_skimage.shape[-1] == 3: 
        cleaned_image_skimage = color.rgb2gray(cleaned_image_skimage)
    binary_image = cleaned_image_skimage < 0.1
    label_image = measure.label(binary_image, connectivity=2)
    area_threshold = 25 
    final_image = morphology.remove_small_objects(label_image, min_size=area_threshold)
    final_image = (final_image > 0) * 255
    final_image = final_image.astype(np.uint8)
    inverted_image = 255 - final_image
    inverted_image_pil = Image.fromarray(inverted_image)
    inverted_image_pil.save('Run/inverted_image.jpg')
    image_path = 'Run/inverted_image.jpg' 
    csv_path = 'Run/DBCSAN.csv'
    image = io.imread(image_path)
    if image.shape[-1] == 3:
        image = color.rgb2gray(image)
    binary_image = image < 1
    black_pixels = np.column_stack(np.where(binary_image))
    df_black_pixels = pd.DataFrame(black_pixels, columns=['Y', 'X'])
    dbscan = DBSCAN(eps=15, min_samples=1) 
    clusters = dbscan.fit_predict(df_black_pixels[['X', 'Y']])
    df_black_pixels['Label'] = clusters + 1 
    df_black_pixels = df_black_pixels.groupby('Label').filter(lambda x: len(x) >= 20)
    df_black_pixels.to_csv(csv_path, index=False)

def create_point_sets(df, point_columns, group_column):
    point_sets = defaultdict(set)
    for _, row in df.iterrows():
        point = tuple(row[point_columns])
        group = row[group_column]
        point_sets[group].add(point)
    return point_sets

def generate_label_pairs(csv_path1, csv_path2):
    data1 = pd.read_csv(csv_path1)
    data2 = pd.read_csv(csv_path2)
    point_sets1 = create_point_sets(data1, ['X', 'Y'], 'Label')
    point_sets2 = create_point_sets(data2, ['X', 'Y'], 'Label')
    overlap_pairs = {}
    for label1, points1 in point_sets1.items():
        max_overlap = 0
        max_label2 = None
        for label2, points2 in point_sets2.items():
            overlap = len(points1 & points2)
            if overlap > max_overlap:
                max_overlap = overlap
                max_label2 = label2
        overlap_pairs[label1] = max_label2
    return overlap_pairs

def create_mask_for_pair(csv_path1, label1, csv_path2, label2, expansion_px=8):
    data1 = pd.read_csv(csv_path1)
    data2 = pd.read_csv(csv_path2)
    data1_label = data1[data1['Label'] == label1]
    data2_label = data2[data2['Label'] == label2]
    x_min, y_min = min(data2['X']), min(data2['Y'])
    x_max, y_max = max(data2['X']), max(data2['Y'])
    mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=bool)
    for _, row1 in data1_label.iterrows():
        x1, y1 = int(row1['X']), int(row1['Y'])
        for _, row2 in data2_label.iterrows():
            x2, y2 = int(row2['X']), int(row2['Y'])
            if (x1 - expansion_px <= x2 <= x1 + expansion_px) and (y1 - expansion_px <= y2 <= y1 + expansion_px):
                mask[y2 - y_min, x2 - x_min] = True
    return mask

def all_pairs_and_save(csv_path1, csv_path2, pairs, expansion_px=5):
    data2 = pd.read_csv(csv_path2)
    x_min_data2, y_min_data2 = min(data2['X']), min(data2['Y'])
    for i, (label1, label2) in enumerate(pairs.items()):
        mask = create_mask_for_pair(csv_path1, label1, csv_path2, label2, expansion_px)
        rows = []
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if mask[y, x]:
                    x_coord = x + x_min_data2
                    y_coord = y + y_min_data2
                    rows.append({'X': x_coord, 'Y': y_coord, 'Label': label1})
        layer_df = pd.DataFrame(rows)
        layer_df.to_csv(f'Run/{label1}.0_final.csv', index=False)

def generate_csv_from_npy(npy_file_path, csv_output_path):
    mask = np.load(npy_file_path)
    data = []
    for layer_index, layer in enumerate(mask, start=1):
        y_indices, x_indices = np.nonzero(layer)
        for (x, y) in zip(x_indices, y_indices):
            data.append([x, y, layer_index])
    df = pd.DataFrame(data, columns=['X', 'Y', 'Label'])
    df.to_csv(csv_output_path, index=False)

def generate_new_npy(j):
    source_dir = '2 synchronized brightness sperm'
    destination_dir = 'Run'
    filename = f'mask_head_{j}.npy'
    source_file = os.path.join(source_dir, filename)
    destination_file = os.path.join(destination_dir, filename)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    if os.path.exists(source_file):
        shutil.copy2(source_file, destination_file)
    mapping_table_path = 'Run/mapping_table_pro_distance.csv'
    npy_file_path = f'Run/mask_head_{j}.npy'
    output_npy_file_path = f'Run/mask_head_{j}.npy'
    mapping_table = pd.read_csv(mapping_table_path)
    npy_data = np.load(npy_file_path)
    layers_indices = mapping_table['NPY Layer'] - 1
    selected_layers = npy_data[layers_indices]
    np.save(output_npy_file_path, selected_layers)

def add_points_from_head(csv_file_path, head_csv_path, label_to_add, label_in_csv):
    try:
        target_data = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        target_data = pd.DataFrame(columns=['X', 'Y', 'Label'])
    head_data = pd.read_csv(head_csv_path)
    filtered_head_data = head_data[head_data['Label'] == label_to_add]
    filtered_head_data['Label'] = label_in_csv
    updated_data = pd.concat([target_data, filtered_head_data], ignore_index=True)
    updated_data.to_csv(csv_file_path, index=False)

def merge_csv_files(source_dir, output_file):
    file_pattern = os.path.join(source_dir, '*.0_final.csv')
    file_list = glob.glob(file_pattern)
    all_data = []
    for file in file_list:
        df = pd.read_csv(file)
        all_data.append(df)
    merged_df = pd.concat(all_data, ignore_index=True)
    merged_df.to_csv(os.path.join(source_dir, output_file), index=False)

def create_and_visualize_mask(csv_file_path, npy_file_path):
    df = pd.read_csv(csv_file_path)
    height, width = 540, 720
    unique_labels = sorted(df['Label'].unique())
    mask = np.zeros((len(unique_labels), height, width), dtype=np.uint8)
    for _, row in df.iterrows():
        layer_index = unique_labels.index(row['Label'])
        x, y = int(row['X']), int(row['Y'])
        mask[layer_index, y, x] = 1
    np.save(npy_file_path, mask)

def process_and_visualize_mask(npy_file_path):
    mask = np.load(npy_file_path)
    filled_mask = np.zeros_like(mask)
    for i, layer in enumerate(mask):
        closed_mask = binary_closing(layer, structure=np.ones((4, 4)))
        filled_mask[i] = binary_fill_holes(closed_mask)
    np.save(npy_file_path, filled_mask)

def expand_and_smooth_mask(npy_file_path, dilation_iter=1, sigma=0.1):
    mask = np.load(npy_file_path)
    expanded_smoothed_mask = np.zeros_like(mask)
    for i, layer in enumerate(mask):
        dilated_layer = binary_dilation(layer, iterations=dilation_iter)
        expanded_smoothed_mask[i] = gaussian_filter(dilated_layer, sigma=sigma)
    np.save(npy_file_path, expanded_smoothed_mask)

def check_overlap_and_merge(source_npy, target_npy, iou_threshold=0.35, j=0):
    if not os.path.exists(source_npy):
        return None
    source_masks = np.load(source_npy)
    target_masks = np.load(target_npy)
    keep_layers = []
    for t_layer in target_masks:
        max_iou_with_source = 0
        for s_layer in source_masks:
            intersection = np.logical_and(t_layer, s_layer).sum()
            union = np.logical_or(t_layer, s_layer).sum()
            iou = intersection / union if union > 0 else 0
            max_iou_with_source = max(max_iou_with_source, iou)
        if max_iou_with_source < iou_threshold:
            keep_layers.append(t_layer)
    updated_target_masks = np.array(keep_layers)
    merged_masks = np.concatenate((updated_target_masks, source_masks), axis=0)
    save_dir = 'Run'
    os.makedirs(save_dir, exist_ok=True)
    new_file_path = os.path.join(save_dir, 'final.npy')
    np.save(new_file_path, merged_masks)

def calculate_iou(layer1, layer2):
    return jaccard_score(layer1.flatten(), layer2.flatten())

def match_layers(data1, data2):
    iou_matrix = np.zeros((data1.shape[0], data2.shape[0]))
    for i in range(data1.shape[0]):
        for j in range(data2.shape[0]):
            iou_matrix[i, j] = calculate_iou(data1[i], data2[j] > 0)
    max_indices = np.argmax(iou_matrix, axis=0)  
    return max_indices

def npy_integration(j):
    data_mask_head = np.load(f'Run/mask_head_{j}.npy')
    data_final = np.load('Run/final.npy' )
    best_matches = match_layers(data_mask_head, data_final)
    reordered_data = data_mask_head[best_matches]
    reordered_file_path = f'Run/mask_head_{j}.npy' 
    np.save(reordered_file_path, reordered_data)

def visualize_colored_masks(image_path, npy_path, colors):
    original_image = plt.imread(image_path)
    processed_layers = np.load(npy_path)
    cmap = ListedColormap([(plt.cm.colors.to_rgba(c, alpha=0.5)) for c in colors])
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(original_image)
    for j, layer in enumerate(processed_layers):
        color = colors[j % len(colors)]
        rgba_color = plt.cm.colors.to_rgba(color, alpha=0.35) 
        overlay = np.zeros((*layer.shape, 4)) 
        overlay[layer > 0] = rgba_color
        ax.imshow(overlay, extent=(0, original_image.shape[1], original_image.shape[0], 0))
    ax.axis('off')
    plt.show()

def clustering_segmentation_process(j, run_folder_path):
    Clustering(j, run_folder_path)
    Skeletonization_labeling()
    file_path = 'Run\\multi_layer_skeletonized_image.npy' 
    endpoints_per_layer = get_layer_endpoints(file_path)
    Endpoints_intersections()
    file_path = 'Run\\processed_multi_layer_skeleton.npy' 
    endpoints_per_layer_two = get_layer_endpoints(file_path)
    endpoint_checks_per_layer = check_endpoints_in_new_file(endpoints_per_layer, file_path) 
    num_endpoints_per_layer = {layer: len(endpoints) for layer, endpoints in endpoints_per_layer.items()}
    num_not_foreground_per_layer = {layer: checks.count(False) for layer, checks in endpoint_checks_per_layer.items()}
    Saved_layers(file_path, num_endpoints_per_layer, num_not_foreground_per_layer, endpoints_per_layer_two)
    combined_skeletonized_show()
    file_path = 'Run\\layers_with_difference_gt_4.npy'  
    all_junctions = get_all_layers_junctions(file_path)  
    process_layers(file_path, all_junctions)
    processed_layers = split_layers_into_connected_components(file_path)
    if processed_layers.size > 0:
        np.save(file_path, processed_layers)  
    else:
        print("No processed layers to save or visualize.")
    save_into_csv(endpoints_per_layer, endpoint_checks_per_layer)
    combined_df(run_folder_path)
    updated_csv_data()
    Reassign_Label()
    BLS(15,15)
    csv_file_path = 'Run/total.csv' 
    clean_and_relabel(csv_file_path, 16)
    optimal_pairs_df = tangent_merge()
    process_and_save_data(optimal_pairs_df, 'Run/total.csv', 'Run/total.csv')
    clean_and_relabel(csv_file_path, 20)
    BLS(20,20)
    clean_and_relabel(csv_file_path, 45)
    BLS(20,25)
    breakpoint_completion('Run/total_2.csv', 'Run/total_3.csv')
    file_path = 'Run/total_3.csv'
    visualize_and_save_filtered_data(file_path)

def tail_recovery_process(j, run_folder_path, image_path):
    npy_file_path = f'2 synchronized brightness sperm/mask_head_{j}.npy' 
    generate_mapping_table(npy_file_path, 'Run/total_3.csv', 'Run/mapping_table.csv' )
    gen_mapping_table_df()
    csv_sorting(j)
    process_and_visualize('Run/mapping_table_pro_distance.csv', 'Run//total_4.csv')
    tail_file_integration(j)
    pairs = generate_label_pairs('Run/filtered_total_4.csv', 'Run/DBCSAN.csv')
    print(pairs)    
    all_pairs_and_save('Run/filtered_total_4.csv', 'Run/DBCSAN.csv', pairs)
    generate_csv_from_npy(f"2 synchronized brightness sperm/mask_head_{j}.npy", 'Run/head.csv')
    generate_new_npy(j)
    mapping_table = pd.read_csv('Run/mapping_table_pro_distance.csv')
    head_csv_path = 'Run/head.csv'
    for index, row in mapping_table.iterrows():
        npy_layer = int(row['NPY Layer'])
        csv_label = int(row['CSV Label'])
        csv_file_name = f"{csv_label}.0_final.csv"
        csv_file_path = f"Run/{csv_file_name}"
        add_points_from_head(csv_file_path, head_csv_path, npy_layer, csv_label)
    merge_csv_files(run_folder_path, 'final.csv')
    csv_file_path = 'Run/final.csv' 
    npy_file_path = 'Run/final.npy' 
    create_and_visualize_mask(csv_file_path, npy_file_path)
    process_and_visualize_mask(npy_file_path)
    expand_and_smooth_mask(npy_file_path, dilation_iter=2, sigma=0.2)
    expand_and_smooth_mask(npy_file_path, dilation_iter=1, sigma=0.1)
    source_path = f'2 synchronized brightness sperm/masks_full_full_{j}.npy'
    target_path = 'Run/final.npy'
    check_overlap_and_merge(source_path, target_path, 0.35, j)
    npy_integration(j)
    npy_path = 'Run/final.npy'
    colors = ['orange', 'red', 'gold', 'skyblue', 'green', 'purple', 'pink', 'brown']
    visualize_colored_masks(image_path, npy_path, colors)




def main():
    # Specify the name of the image
    j = k = '024'
    # Specify the run folder
    run_folder_path = 'Run'
    image_path = f'original_image/{j}.jpg'

    clustering_segmentation_process(j, run_folder_path)
    tail_recovery_process(k, run_folder_path, image_path)
    
if __name__ == "__main__":
    main()
