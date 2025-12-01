import pandas as pd


# put your file paths here
CAMERA_PATH = r"C:\Users\abc\Desktop\Autonomous Driving\data\raw\training_camera_image_10017090168044687777_6380_000_6400_000.parquet"
LIDAR_PATH = r"C:\Users\abc\Desktop\Autonomous Driving\data\raw\training_lidar_10017090168044687777_6380_000_6400_000.parquet"
LIDAR_BOX_PATH = r"C:\Users\abc\Desktop\Autonomous Driving\data\raw\training_lidar_box_10017090168044687777_6380_000_6400_000.parquet"

camera_df = pd.read_parquet(CAMERA_PATH)
lidar_df  = pd.read_parquet(LIDAR_PATH)
lidar_box_df = pd.read_parquet(LIDAR_BOX_PATH)

print("camera rows:", len(camera_df))
print("lidar rows:", len(lidar_df))
print("lidar_box rows:", len(lidar_box_df))

# find all frame's key（segment + timestamp）
FRAME_KEYS = ["key.segment_context_name", "key.frame_timestamp_micros"]

# make a index table of unique frames
frame_index = (
    camera_df[FRAME_KEYS]
    .drop_duplicates()
    .sort_values(FRAME_KEYS)
    .reset_index(drop=True)
)

print("unique frames:", len(frame_index))


# write a loader function
def load_frame(segment_name: str, frame_ts: int):
    cam = camera_df[
        (camera_df["key.segment_context_name"] == segment_name)
        & (camera_df["key.frame_timestamp_micros"] == frame_ts)
    ]

    lidar = lidar_df[
        (lidar_df["key.segment_context_name"] == segment_name)
        & (lidar_df["key.frame_timestamp_micros"] == frame_ts)
    ]

    box = lidar_box_df[
        (lidar_box_df["key.segment_context_name"] == segment_name)
        & (lidar_box_df["key.frame_timestamp_micros"] == frame_ts)
    ]

    return cam, lidar, box


# # demo：testing the loader
# first = frame_index.iloc[160]  # 160 a random frame
# seg = first["key.segment_context_name"]
# ts  = first["key.frame_timestamp_micros"]

# print("demo frame:")
# print("  segment:", seg)
# print("  ts:", ts)

# cam, lidar, box = load_frame(seg, ts)

# print("  cam rows:", len(cam))
# print("  lidar rows:", len(lidar))
# print("  box rows:", len(box))


# merged_box_lidar = box.merge(
#     lidar,
#     on=FRAME_KEYS,
#     how="inner",
#     suffixes=("_box", "_lidar"),
# )

# print("  merged_box_lidar rows:", len(merged_box_lidar))

# merged_box_lidar.to_parquet("one_frame_merged_box_lidar.parquet")
# print("saved one_frame_merged_box_lidar.parquet")