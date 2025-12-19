import os
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np 

input_folder = r"C:\Users\riley\OneDrive\Documents\University\Year_3\Semester_2\Applied_science_project_2\Image_data\Streak_like_testing_images\fit_images"     
output_folder = r"C:\Users\riley\OneDrive\Documents\University\Year_3\Semester_2\Applied_science_project_2\Image_data\Streak_like_testing_images\png_images"

#Source: computer generated using GPT5 LLM
# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate over all files in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith('.fit'):
        fits_path = os.path.join(input_folder, filename)
        
        # Open FITS file
        hdul = fits.open(fits_path)
        image_data = hdul[0].data
        hdul.close()

        # Normalize image contrast to match the contrast of ROO images.
        vmin = np.percentile(image_data, 5)
        vmax = np.percentile(image_data, 99)

        # Generate output filename by replacing .fits with .png
        png_filename = filename[:-4] + '.png'
        png_path = os.path.join(output_folder, png_filename)

        # Plot and save as PNG
        plt.imshow(image_data, cmap='gray', vmin = vmin, vmax = vmax )
        plt.axis('off')
        plt.savefig(png_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f'Converted {filename} -> {png_filename}')


