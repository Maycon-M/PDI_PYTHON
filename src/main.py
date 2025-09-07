import take_images_from_web
import extract_rois
import argparse
import taken_background_ia
import taken_background
from dotenv import load_dotenv

def get_args():
    parser = argparse.ArgumentParser(description="Process some images and extract ROIs.")
    parser.add_argument('--method', '-m', type=str, choices=['web', 'extract', 'ia_background', 'background'], required=True, help="Choose 'web' to take images from the web or 'extract' to extract ROIs from images.")
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

if __name__ == "__main__":
    load_dotenv()
    main()
