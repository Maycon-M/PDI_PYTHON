import take_images_from_web
import extract_rois
import argparse
import taken_background_ia
import taken_background
import resize_images
import compose_dataset
from dotenv import load_dotenv

def get_args():
    parser = argparse.ArgumentParser(description="Process some images and extract ROIs.")
    parser.add_argument('--method', '-m', type=str, choices=['web', 'extract', 'ia_background', 'background', 'resize', 'compose'], required=True, help="Choose 'web' to take images from the web, 'extract' to extract ROIs, 'ia_background' to use AI backgrounds, 'background' to fetch backgrounds, 'resize' to resize images, or 'compose' to generate composites and YOLO labels.")
    return parser.parse_args()

def main():
    args = get_args()
    if args.method == 'web':
        take_images_from_web.take_images_from_web()
    elif args.method == 'extract':
        extract_rois.run()
    elif args.method == 'ia_background':
        taken_background_ia.take_background()
    elif args.method == 'background':
        taken_background.take_background()
    elif args.method == 'resize':
        # padrão: redimensiona a pasta 'backgrounds' para 'backgrounds_640' em 640x640 com corte central (sem barras)
        resize_images.resize_images(in_dir='backgrounds', out_dir='backgrounds_640', size=(640, 640), mode='crop')
    elif args.method == 'compose':
        # padrão: usa 'backgrounds_640' e 'rois', gera 200 composições com 3 objetos
        compose_dataset.make_composites(
            backgrounds_dir='backgrounds_640',
            rois_dir='rois',
            out_images_dir='dataset/images',
            out_labels_dir='dataset/labels',
            num_images=200,
            objects_per_image=3,
        )

if __name__ == "__main__":
    load_dotenv()
    main()
