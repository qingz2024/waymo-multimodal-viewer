# script: multimodal_view.py
import io
import numpy as np
import imageio
import matplotlib.pyplot as plt
from PIL import Image

from frame_loader import frame_index, load_frame

# camera ID
CAMERA_IDS = [1, 2, 3, 4, 5]

# lidar_box
X_COL = "[LiDARBoxComponent].box.center.x"
Y_COL = "[LiDARBoxComponent].box.center.y"
L_COL = "[LiDARBoxComponent].box.size.x"
W_COL = "[LiDARBoxComponent].box.size.y"
HEADING_COL = "[LiDARBoxComponent].box.heading"
TYPE_COL = "[LiDARBoxComponent].type"

# BEV range (meters)
X_MIN, X_MAX = -60, 60   # front and back
Y_MIN, Y_MAX = -40, 40   # left and right

# max frames to process
MAX_FRAMES = None # right now is all frames


def extract_camera_image(cam_df, camera_id):

    rows = cam_df[cam_df["key.camera_name"] == camera_id]
    if len(rows) == 0:
        return None

    img_data = rows.iloc[0]["[CameraImageComponent].image"]

    if isinstance(img_data, (bytes, bytearray)):
        img = Image.open(io.BytesIO(img_data))
        img = np.array(img)
    else:
        img = np.array(img_data)

    return img


def make_mosaic_frame(cam_df):

    imgs = []
    for cid in CAMERA_IDS:
        img = extract_camera_image(cam_df, cid)
        if img is None:
            continue
        imgs.append(img)

    if len(imgs) == 0:
        return None

    # rearrange to the same height and width
    base_h, base_w = imgs[0].shape[:2]
    resized = []
    for img in imgs:
        h, w = img.shape[:2]
        if (h, w) != (base_h, base_w):
            pil = Image.fromarray(img)
            pil = pil.resize((base_w, base_h))
            img = np.array(pil)
        resized.append(img)

    mosaic = np.concatenate(resized, axis=1)
    return mosaic


# BEV functions

def get_color_for_type(t):

    try:
        ti = int(t)
    except Exception:
        ti = -1

    if ti == 1:
        return "tab:blue"    # car
    elif ti == 2:
        return "tab:orange"  # pedestrian
    elif ti == 3:
        return "tab:green"   # riders
    else:
        return "gray"        # others


def get_box_corners_xy(cx, cy, l, w, heading):

    x = l / 2.0
    y = w / 2.0
    corners = np.array([
        [ x,  y],
        [ x, -y],
        [-x, -y],
        [-x,  y],
    ])

    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, -s],
                    [s,  c]])
    rotated = corners @ rot.T
    rotated[:, 0] += cx
    rotated[:, 1] += cy
    return rotated


def draw_bev_to_image(box_df, frame_idx=None):
    # plot boxes in BEV
    fig, ax = plt.subplots(figsize=(6, 6))

    for _, box in box_df.iterrows():
        cx = float(box[X_COL])
        cy = float(box[Y_COL])
        l  = float(box[L_COL])
        w  = float(box[W_COL])
        hd = float(box[HEADING_COL])
        t  = box[TYPE_COL]

        corners = get_box_corners_xy(cx, cy, l, w, hd)
        color = get_color_for_type(t)

        xs = list(corners[:, 0]) + [corners[0, 0]]
        ys = list(corners[:, 1]) + [corners[0, 1]]
        ax.plot(xs, ys, color=color, linewidth=1.5)

    # ego vehicle at origin
    ax.scatter([0], [0], c="red", s=30)

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel("X (forward, m)")
    ax.set_ylabel("Y (left, m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3)

    if frame_idx is not None:
        ax.set_title(f"BEV | frame {frame_idx}")

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    # use tostring_argb to get RGBA image
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf = buf.reshape((h, w, 4))
    img = buf[:, :, 1:4].copy()  # ARGB -> RGB

    plt.close(fig)
    return img


# resize and stack functions

def resize_to_width(img, width):

    h, w = img.shape[:2]
    if w == width:
        return img
    scale = width / w
    new_size = (width, int(h * scale))
    pil = Image.fromarray(img)
    pil = pil.resize(new_size)
    return np.array(pil)


def stack_vertical(top_img, bottom_img):

    h1, w1, _ = top_img.shape
    h2, w2, _ = bottom_img.shape
    width = min(w1, w2)
    top_resized = resize_to_width(top_img, width)
    bottom_resized = resize_to_width(bottom_img, width)
    return np.concatenate([top_resized, bottom_resized], axis=0)


def main():
    frames = []

    total_frames = len(frame_index)
    if MAX_FRAMES is not None:
        total_frames = min(total_frames, MAX_FRAMES)

    print("Total frames to process:", total_frames)

    for i in range(total_frames):
        row = frame_index.iloc[i]
        seg = row["key.segment_context_name"]
        ts  = row["key.frame_timestamp_micros"]

        cam_df, _, box_df = load_frame(seg, ts)

        mosaic = make_mosaic_frame(cam_df)
        if mosaic is None:
            print(f"Frame {i}: no camera images, skip.")
            continue

        if len(box_df) == 0:
            print(f"Frame {i}: no boxes, skip.")
            continue
        bev_img = draw_bev_to_image(box_df, frame_idx=i)

        combined = stack_vertical(mosaic, bev_img)
        frames.append(combined)

        if i % 20 == 0:
            print(f"Processed frame {i}/{total_frames}")

    print("Collected", len(frames), "combined frames.")
    if len(frames) == 0:
        print("No frames to save.")
        return

    out_name = "waymo_multimodal_bev_cam.gif"
    imageio.mimsave(out_name, frames, fps=10)
    print("Saved:", out_name)


if __name__ == "__main__":
    main()
