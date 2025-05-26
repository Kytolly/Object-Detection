import mmcv
from mmdet.apis import init_detector, inference_detector
import os

def detect_objects_in_image(image_path, config_file, checkpoint_file, device='cpu'):
    """
    Detects objects in a given image using MMDetection.

    Args:
        image_path (str): Path to the input image.
        config_file (str): Path to the model configuration file.
        checkpoint_file (str): Path to the model checkpoint file.
        device (str): Device to use for inference ('cpu' or 'cuda:0').

    Returns:
        list: Detection results.
    """
    model = init_detector(config_file, checkpoint_file, device=device)
    img = mmcv.imread(image_path, channel_order='rgb')
    result = inference_detector(model, img)
    return result

if __name__ == '__main__':
    # Define paths
    data_dir = 'data/'
    mmdet_config_dir = 'data/mmdet/' # Assuming mmdet configs and checkpoints are here
    
    config_file = os.path.join(mmdet_config_dir, 'rtmdet_tiny_8xb32-300e_coco.py')
    checkpoint_file = os.path.join(mmdet_config_dir, 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth')

    # List of images to process
    images = ['枫丹.jpg', '璃月.jpg']

    for image_name in images:
        image_path = os.path.join(data_dir, image_name)
        print(f"Processing {image_path}...")
        
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            continue
        
        if not os.path.exists(config_file):
            print(f"Error: Config file not found at {config_file}")
            print("Please ensure 'rtmdet_tiny_8xb32-300e_coco.py' is in 'data/mmdet/'")
            continue

        if not os.path.exists(checkpoint_file):
            print(f"Error: Checkpoint file not found at {checkpoint_file}")
            print("Please ensure 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth' is in 'data/mmdet/'")
            continue

        try:
            detections = detect_objects_in_image(image_path, config_file, checkpoint_file, device='cpu')
            print(f"Detections for {image_name}:")
            print(detections)
            # You can add code here to visualize the detections or save them
        except Exception as e:
            print(f"An error occurred while processing {image_name}: {e}")
