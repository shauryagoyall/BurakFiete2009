"""
Create animated GIF from sequential frame images.
Converts frame_XXXXXX.png files to an MP4 or GIF.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def create_gif_from_frames(module_num, output_format='gif', framerate=20, step=1):
    """
    Create a GIF or MP4 from sequential frame images.
    
    Parameters:
    -----------
    module_num : int
        Module number (e.g., 1 for module_1)
    output_format : str
        Output format: 'gif' or 'mp4' (default: 'gif')
    framerate : int
        Framerate for output (default: 20 fps)
    step : int
        Keep every Nth frame (default: 1 = use all frames)
        
    Returns:
    --------
    None (creates file in plots/simulation/module_N/)
    """
    
    frame_dir = Path(f'plots/simulation/module_{module_num}')
    
    if not frame_dir.exists():
        print(f"Error: Directory {frame_dir} does not exist!")
        return
    
    # List frame files
    frames = sorted(frame_dir.glob('frame_*.png'))
    if not frames:
        print(f"Error: No frame_*.png files found in {frame_dir}")
        return
    
    # Subsample frames if requested
    if step < 1:
        print("Error: step must be >= 1")
        return
    frames = frames[::step]
    
    print(f"Found {len(frames)} frames in {frame_dir}")
    
    
    if output_format.lower() == 'gif':
        create_gif(frame_dir, module_num, frames, framerate)
    elif output_format.lower() == 'mp4':
        create_mp4(frame_dir, module_num, frames, framerate)
    else:
        print(f"Unsupported format: {output_format}. Use 'gif' or 'mp4'.")


def create_gif(frame_dir, module_num, frames, framerate=20):
    """
    Create GIF using ImageMagick (convert command).
    
    Requires: brew install imagemagick
    """
    output_path = frame_dir / f'module_{module_num}_animation.gif'
    
    delay = int(100 / framerate)  # Convert fps to centiseconds
    
    cmd = [
        'magick',
        '-delay', str(delay),
        '-loop', '0',
        *[str(f) for f in frames],
        str(output_path)
    ]
    
    print(f"\nCreating GIF: {output_path}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, shell=False, check=True)
        print(f"✓ GIF created successfully: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error creating GIF: {e}")
        print("\nMake sure ImageMagick is installed:")
        print("  brew install imagemagick")


def create_mp4(frame_dir, module_num, frames, framerate=20):
    """
    Create MP4 using ffmpeg.
    
    Requires: brew install ffmpeg
    """
    output_path = frame_dir / f'module_{module_num}_animation.mp4'
    temp_dir = None

    try:
        # If frames are non-contiguous (sampling), create a temp sequential set
        if frames and frames != sorted(frame_dir.glob('frame_*.png')):
            temp_dir = Path(tempfile.mkdtemp(prefix='mp4_frames_'))
            for idx, src in enumerate(frames):
                dest = temp_dir / f"frame_{idx:06d}.png"
                shutil.copy2(src, dest)
            input_pattern_dir = temp_dir
        else:
            input_pattern_dir = frame_dir

        cmd = [
            'ffmpeg',
            '-framerate', str(framerate),
            '-i', str(input_pattern_dir / 'frame_%06d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-y',
            str(output_path)
        ]

        print(f"\nCreating MP4: {output_path}")
        print(f"Framerate: {framerate} fps")

        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✓ MP4 created successfully: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error creating MP4: {e}")
        print("\nMake sure ffmpeg is installed:")
        print("  brew install ffmpeg")
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    import sys
    
    # Usage: python create_gif.py [module_num] [format] [framerate] [step]
    # Example: python create_gif.py 1 gif 20 5   # keep every 5th frame
    module_num = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    output_format = sys.argv[2] if len(sys.argv) > 2 else 'gif'
    framerate = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    step = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    
    create_gif_from_frames(module_num, output_format, framerate, step)
