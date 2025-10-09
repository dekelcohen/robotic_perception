#!/usr/bin/env python3
r"""
* First time
conda create -n dl_310 -y python=3.10

pip install lerobot torch pillow requests huggingface_hub[hf_xet] ipython
# moondream bbox and pointing
pip install moondream

* Problem: crash related to .mp4 decoding using torchcodec and ffmpeg 
  Workaround: pip unintall torchcodec
  Solution: If on linux - try to install the required ffmpeg 
  
* Every time:
cd /d D:\Docs\test6\Projects\Robotics\FeatureEng_Markers
# desktop home pc     
conda activate dl_310
# laptop
conda activate robot_markers

set MOONDREAM_API_KEY=<>
set GEMINI_API_KEY=<>
 
huggingface-cli login <search in auto_topics_tags.txt for complete with token>

Usage:
    python detect_object_bbox.py --dataset <repo_or_local_path> --camera <camera_key> [--object-prompt "red mug"]
    Optional: --out <annotated_image_path> --out-dataset <modified_dataset_dir>
    
    Example (WSL):
       python convert_lerobot_dataset_to_bbox.py --repo_id Shani123/pickup_cup_1 --camera observation.images.table --object-prompt "red cup" --out_repo_id Shani123/pickup_cup_1_bbox_no_cam
       
    Example (Windows):
    
        cd /d  D:\NLP\Robotics\robotic_perception\features_markers # laptop
        # Full example with bbox, tracker, annotated images, and upload to Hugging Face
        python -m convert_lerobot_dataset_to_bbox --repo_id  lerobot/svla_so100_pickplace --camera observation.images.top --object-prompt "orange cube" --bbox-detector moondream --tracker csrt --annotate-image true --out-repo-id svla_so_100_pickplace_bbox_test --upload-out-repo overwrite
        # For testing, remove --upload-out-repo overwrite and add --max-episodes 1
        # test moondream object detector on first frame (do not convert the whole dataset)
        python -m convert_lerobot_dataset_to_bbox --repo_id  Shani123/pickup_toothpicks_2_plus_recovery --camera observation.images.table --object-prompt "jar with blue cap" --bbox-detector moondream
        # Add --out-repo-id Shani123/pickup_toothpicks_2_plus_recovery_bbox_no_cam to convert the dataset
Behavior (added/changed):
  - After obtaining a primary bounding box from Gemini, saves the annotated image (same as before).
  - Adds a new column 'observation.environment_state' to the dataset containing only {'bbox_pixels': [x1,y1,x2,y2]}
    repeated for every frame (static object).
  - Removes camera columns/features from the modified dataset.
  - Saves modified dataset to disk with the path specified by --out-dataset.
"""

import os
from pathlib import Path
import sys
import argparse
import json
import re
from io import BytesIO
import shutil

import requests
from PIL import Image, ImageDraw
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES
from bbox_providers.bbox_provider_factory import get_bbox_provider
import torch



# --- Gemini REST configuration ---
GEMINI_MODEL = "gemini-2.5-flash" # Changed to match bbox_providers.py
GEMINI_REST_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"


def gemini_rest_call_text(prompt_text: str, api_key: str, timeout: int = 30):
    payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    resp = requests.post(GEMINI_REST_URL, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    try:
        cands = data.get("candidates") or []
        if cands:
            cand0 = cands[0]
            content = cand0.get("content") or cand0.get("message") or {}
            parts = content.get("parts") if isinstance(content, dict) else None
            if parts:
                texts = []
                for p in parts:
                    if isinstance(p, dict):
                        if "text" in p and p["text"]:
                            texts.append(p["text"])
                        elif "html" in p and p["html"]:
                            texts.append(p["html"])
                if texts:
                    return "\n".join(texts)
    except Exception:
        pass

    def find_texts(obj):
        found = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ("text", "html") and isinstance(v, str):
                    found.append(v)
                else:
                    found += find_texts(v)
        elif isinstance(obj, list):
            for item in obj:
                found += find_texts(item)
        return found
    
    found_texts = find_texts(data)
    if found_texts:
        return "\n".join(found_texts)
    return json.dumps(data)


def extract_first_json_like(text: str):
    text = text.strip()
    if text.startswith("{") or text.startswith("["):
        try:
            return json.loads(text)
        except Exception:
            pass
    patterns = [r"(\{.*\})", r"(\[.*\])"]
    for pat in patterns:
        m = re.search(pat, text, flags=re.DOTALL)
        if m:
            candidate = m.group(1)
            try:
                return json.loads(candidate)
            except Exception:
                for end in range(len(candidate), 0, -1):
                    try:
                        return json.loads(candidate[:end])
                    except Exception:
                        continue
    raise ValueError("No JSON found in text.")


def tensor_to_pil(img_tensor):
    try:
        import torch as _torch
        import numpy as _np
        if isinstance(img_tensor, _torch.Tensor):
            arr = img_tensor.detach().cpu().numpy()
        else:
            arr = _np.asarray(img_tensor)
    except Exception:
        arr = np.asarray(img_tensor)

    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    elif arr.dtype == np.int64:
        arr = arr.astype(np.uint8)
    return Image.fromarray(arr)

def draw_bbox_on_image(image, bbox):
    # Annotate a PIL image with the bbox and return the annotated image (no save).
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated, "RGBA")
    draw.rectangle(bbox, outline=(255, 0, 0, 255), width=4)
    return annotated


def draw_bbox_and_save(image, bbox, out_path):
    """
    Draws a bounding box and label onto a PIL image and saves it.
    bbox: (x1, y1, x2, y2) but may not be ordered correctly.
    """
    annotated = draw_bbox_on_image(image, bbox)    
    annotated.save(out_path)
    print(f"Saved annotated image to {out_path}")


def process_frame(frame, object_prompt, bbox=None, bbox_provider=None, tracker=None, annotate=False):
    """
    Processes a single frame to detect or track an object and optionally annotate the frame.

    Args:
        frame (PIL.Image): The input frame.
        object_prompt (str): The prompt for the object to detect.
        bbox (list, optional): The previously detected bounding box. If None, detection is performed.
        bbox_provider: The bounding box provider for initial detection.
        tracker: The tracker to update the bounding box.
        annotate (bool): Whether to draw the bounding box on the frame.

    Returns:
        tuple: A tuple containing the annotated frame (or original if annotate=False) and the bounding box.
    """
    new_bbox = None
    if bbox is None:
        # Detect
        if bbox_provider:
            detections, _ = bbox_provider.detect(frame, object_prompt)
            if detections:
                new_bbox = detections[0]["box_pixels"]
    else:
        # Track
        if tracker:
            new_bbox = tracker.update(frame)

    annotated_frame = frame
    if annotate and new_bbox:
        annotated_frame = draw_bbox_on_image(frame, tuple(new_bbox))

    return annotated_frame, new_bbox


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in ("true", "t", "1", "yes", "y"):
        return True
    if v in ("false", "f", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false).")


def add_bbox_and_remove_camera_features(
    repo_id: str,
    new_repo_id: str,
    bbox_pixels: list[int],
    root: str | Path | None = None,
    keep_image_keys: list[str] | None = None,
    camera_key_for_tracking: str | None = None,
    tracker_name: str = "none",
    annotate_images: bool = False,
    max_episodes: int | None = None,  # Added parameter to limit episodes for testing
):
    """
    Load a LeRobotDataset, remove its camera features and save it as a new LeRobotDataset to disk.

    Args:
        repo_id (str): The repo id of the source dataset.
        new_repo_id (str): The repo id of the destination dataset.
        bbox_pixels (list[int]): A list of four integers representing the bounding box pixels [x1, y1, x2, y2].
        root (str | Path | None, optional): The root directory where the datasets are stored.
            Defaults to None.
        keep_image_keys (list[str] | None, optional): A list of image/video keys to preserve.
            If None, all image/video features are removed. Defaults to None.
        max_episodes (int, optional): Maximum number of episodes to convert (for testing). If None, converts all episodes.
    """
    # Load the source dataset
    src_dataset = LeRobotDataset(repo_id, root=root)
    keep_image_keys = keep_image_keys or []

    # Build sets of media keys
    image_keys = {k for k, v in src_dataset.features.items() if v.get("dtype") == "image"}
    video_keys = {k for k, v in src_dataset.features.items() if v.get("dtype") == "video"}
    media_keys = image_keys | video_keys

    # Determine which (single) video key should be re-encoded (only the annotated camera if it is a video)
    annotated_video_keys = set()
    if annotate_images and camera_key_for_tracking and camera_key_for_tracking in video_keys:
        annotated_video_keys = set([camera_key_for_tracking])        
        # If annotate_images is requested, keep the camera that is annotated, even if it wasn't specified in keep_image_keys
        keep_image_keys = sorted(annotated_video_keys | set(keep_image_keys))
        print(f"Annotate images requested: will keep and annotate: {camera_key_for_tracking}")
   
    # Cameras to keep but not annotate - skip writing frames (encode is slow) --> but copy their videos (much faster)
    fast_copy_keys = set(keep_image_keys) - annotated_video_keys
    # Create a new dataset, preserving camera keys of annotated cams. The rest of cams to keep will be added during video copy
    new_features = {
        key: value
        for key, value in src_dataset.features.items()
        if (value["dtype"] not in ["image", "video"]) or (key in annotated_video_keys)
    }
    new_features["observation.environment_state"] = {
        "dtype": "float32",
        "shape": (4,),
        "name": ["y0", "x0", "y1", "x1"],
    }

    # Enable video writing if we keep any video keys
    use_videos_out = any(k in video_keys for k in keep_image_keys)

    print('LeRobotDataset.create(features) : ', new_features, '\nuse_videos_out=', use_videos_out)
    
    dst_dataset = LeRobotDataset.create(
        new_repo_id,
        fps=src_dataset.fps,
        features=new_features,
        root=root,
        use_videos=use_videos_out,
    )

    # Decide if tracking is enabled
    tracking_enabled = (tracker_name is not None) and (tracker_name.lower() != "none") and bool(camera_key_for_tracking)
    if tracking_enabled:
        from tracking_providers.tracking_provider_factory import get_tracking_provider as _get_tracking_provider

    def _as_int(x):
        try:
            return int(x.item())
        except Exception:
            pass
        if isinstance(x, (list, tuple, np.ndarray)):
            return int(x[0])
        return int(x)

    
    # Precompute episode -> list of frame indices to avoid filtering the whole dataset per episode
    hf = src_dataset.hf_dataset
    episode_to_indices: dict[int, list[int]] = {}
    for i, row in enumerate(hf):
        ep = int(row["episode_index"])
        episode_to_indices.setdefault(ep, []).append(i)

    episode_counter = 0  # new counter for episodes converted
    for episode_index in sorted(episode_to_indices.keys()):
        if max_episodes is not None and episode_counter >= max_episodes:
            break
        # If tracking is enabled, initialize a new tracker per episode
        tracker = None
        last_bbox = bbox_pixels
        if tracking_enabled:
            tracker = _get_tracking_provider(tracker_name, bbox_pixels)

        for idx in episode_to_indices[episode_index]:
            frame = hf[idx]
            new_frame = {}

            # Copy non-media features only
            for key, value in frame.items():
                if key in new_features and key not in DEFAULT_FEATURES:
                    if key in media_keys:
                        continue
                    if isinstance(value, torch.Tensor):
                        new_frame[key] = value.numpy()
                    else:
                        new_frame[key] = value

            # Determine bbox for this frame (track over selected camera, if any)
            per_frame_bbox = bbox_pixels
            pil_tracked = None
            if tracking_enabled and camera_key_for_tracking in src_dataset.features:
                try:
                    pil_tracked = tensor_to_pil(src_dataset[idx][camera_key_for_tracking])
                    _, per_frame_bbox = process_frame(pil_tracked, None, bbox=last_bbox, tracker=tracker, annotate=False)
                    last_bbox = per_frame_bbox
                except Exception:
                    per_frame_bbox = last_bbox

            # Write media frames:
            for mkey in keep_image_keys:
                # Skip re-encoding for video keys unless this is the annotated video camera
                if (mkey in video_keys) and not (mkey in annotated_video_keys):
                    continue
                try:
                    # Reuse the already decoded tracked frame if this is the same camera
                    if pil_tracked is not None and mkey == camera_key_for_tracking:
                        pil_media = pil_tracked
                    else:
                        pil_media = tensor_to_pil(src_dataset[idx][mkey])
                    # Annotate only the specified camera key
                    if annotate_images and camera_key_for_tracking and mkey == camera_key_for_tracking:
                        pil_media = draw_bbox_on_image(pil_media, tuple(per_frame_bbox))
                    new_frame[mkey] = np.array(pil_media)
                except Exception:
                    # If decoding fails, skip this media key for this frame
                    pass

            new_frame["observation.environment_state"] = np.array(per_frame_bbox, dtype=np.float32)

            dst_dataset.add_frame(
                new_frame,
                task=src_dataset.meta.tasks[_as_int(frame["task_index"])],
                timestamp=float(frame["timestamp"][0] if isinstance(frame["timestamp"], (list, tuple, np.ndarray)) else getattr(frame["timestamp"], "item", lambda: frame["timestamp"])()),
            )
        dst_dataset.save_episode()
        episode_counter += 1
        
    # Cleanup: remove intermediate images written for video keys (not needed in final repo)
    try:
        images_root = Path(dst_dataset.root) / "images"
        if images_root.exists():
            for vkey in (set(keep_image_keys) & video_keys):
                v_images_dir = images_root / vkey
                if v_images_dir.exists():
                    shutil.rmtree(v_images_dir, ignore_errors=True)
            # If images root is now empty, remove it
            if images_root.exists() and not any(images_root.iterdir()):
                images_root.rmdir()
    except Exception as _e:
        # Non-fatal: leave any residual images if cleanup fails
        pass

    # TODO: Refactor to another function
    # Fast path: copy original videos for all kept video keys except the annotated (re-encoded) one
    if use_videos_out:
        try:
            src_videos_root = Path(src_dataset.root) / "videos"
            dst_videos_root = Path(dst_dataset.root) / "videos"
            dst_videos_root.mkdir(parents=True, exist_ok=True)
            for vkey in (set(keep_image_keys) & video_keys):
                if vkey in annotated_video_keys:
                    continue  # this one was re-encoded
                src_dir = src_videos_root / vkey
                dst_dir = dst_videos_root / vkey
                if src_dir.exists():
                    if dst_dir.exists():
                        shutil.rmtree(dst_dir, ignore_errors=True)
                    shutil.copytree(src_dir, dst_dir)
            # Path meta/info.json with cams to keep but not annotated (these were already added in above loop)
            try:
                src_meta = json.load(open(Path(src_dataset.root) / "meta" / "info.json"))
                dst_meta_path = Path(dst_dataset.root) / "meta" / "info.json"
                dst_meta = json.load(open(dst_meta_path))
    
                # Copy metadata for cameras we fast-copied
                for vkey in fast_copy_keys:
                    if vkey in src_meta.get("features", {}):
                        dst_meta["features"][vkey] = src_meta["features"][vkey]
    
                # Save updated meta
                json.dump(dst_meta, open(dst_meta_path, "w"), indent=2)
                print(f"Patched meta/info.json with fast-copied camera keys: {sorted(fast_copy_keys)}")
            except Exception as e:
                print(f"Warning: failed to patch meta/info.json for fast-copied cameras: {e}")
            if annotated_video_keys:
                print(f"Re-encoded video for '{annotated_video_keys}', copied originals for: {sorted((set(keep_image_keys) & video_keys) - {annotated_video_keys})}")
            else:
                print(f"Copied original videos for: {sorted(set(keep_image_keys) & video_keys)}")
        except Exception as _e:
            print(f"Warning: failed to copy original videos: {_e}")

    return dst_dataset

def _upload_out_repo_if_requested(local_dir: str | Path, repo_id: str, mode: str):
    """
    Upload the dataset folder to Hugging Face Hub according to mode:
      - 'false': do nothing.
      - 'true': create and upload if repo does not exist; if exists, skip and print message.
      - 'overwrite': create if missing; otherwise upload to existing repo (overwrite matching files).
    """
    mode = (mode or "false").lower()
    if mode not in ("false", "true", "overwrite"):
        print(f"Unknown --upload-out-repo mode '{mode}', skipping upload.")
        return
    if mode == "false":
        print("Upload mode is 'false' (default). Skipping upload to Hugging Face Hub.")
        return

    try:
        from huggingface_hub import HfApi
    except Exception as e:
        print(f"huggingface_hub not available. Cannot upload dataset. Error: {e}")
        return

    api = HfApi()

    # Resolve full repo id (namespace/repo). If no namespace provided, use the authenticated user.
    if "/" in repo_id:
        full_repo_id = repo_id
    else:
        try:
            me = api.whoami()
            namespace = me.get("name") or (me.get("orgs") or [None])[0]
        except Exception:
            namespace = None
        full_repo_id = f"{namespace}/{repo_id}" if namespace else repo_id

    # Detect if dataset repo exists
    exists = False
    try:
        api.dataset_info(full_repo_id)
        exists = True
    except Exception:
        exists = False

    if mode == "true":
        if exists:
            print(f"Dataset repo '{full_repo_id}' already exists. Not uploading (use --upload-out-repo overwrite to overwrite).")
            return
        print(f"Creating dataset repo '{full_repo_id}' and uploading...")
        api.create_repo(repo_id=full_repo_id, repo_type="dataset", exist_ok=False)
        api.upload_folder(
            repo_id=full_repo_id,
            repo_type="dataset",
            folder_path=str(local_dir),
            path_in_repo=".",
            commit_message="Initial upload of bbox-converted dataset",
        )
        print(f"Uploaded dataset to https://huggingface.co/datasets/{full_repo_id}")
        return

    # mode == "overwrite"
    if not exists:
        print(f"Dataset repo '{full_repo_id}' does not exist. Creating and uploading...")
        api.create_repo(repo_id=full_repo_id, repo_type="dataset", exist_ok=True)
    else:
        print(f"Uploading to existing dataset repo '{full_repo_id}' (overwriting matching files).")
    api.upload_folder(
        repo_id=full_repo_id,
        repo_type="dataset",
        folder_path=str(local_dir),
        path_in_repo=".",
        commit_message="Upload (overwrite) bbox-converted dataset",
    )
    print(f"Uploaded dataset to https://huggingface.co/datasets/{full_repo_id}")

def main():
    # TODO: Refactor args parsing to a function 
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", "-d", required=True, help="Hugging Face repo id or local path")
    p.add_argument("--camera", "-c", required=True, help="camera key to read from dataset.meta.camera_keys for extract first frame for bbox detection")
    p.add_argument("--object-prompt", "-o", default=None, help="Optional object prompt / object name")
    p.add_argument("--api-key", help="API key for the selected bbox detector. If not provided, the detector will try to read from its specific environment variable (e.g., GEMINI_API_KEY, MOONDREAM_API_KEY).")
    p.add_argument("--out", default="out_bbox.png", help="Output annotated image path")
    p.add_argument("--bbox-detector", default="gemini", choices=["gemini", "moondream"], help="Bounding box detector to use.")
    p.add_argument("--tracker", default="none", choices=["none", "csrt"], help="Tracker to propagate bbox across frames.")
    p.add_argument("--out-repo-id", help="name of the new repo e.g Shani123/pick_up_glass_bbox. Saved in ")
    p.add_argument("--keep-image-keys", nargs='*', default=[], help="List of image/video keys to preserve in the output dataset.")
    p.add_argument("--annotate-image", type=str2bool, default=False, help="If true, keep images and annotate them with the bbox in the output dataset video.")
    p.add_argument(
        "--upload-out-repo",
        default="false",
        choices=["false", "true", "overwrite"],
        help="Upload the output dataset repo to Hugging Face. 'false' (default): do not upload; 'true': create and upload if not exists, otherwise skip; 'overwrite': upload to existing repo (overwriting matching files).",
    )
    p.add_argument("--max-episodes", type=int, default=None, 
                   help="Max episodes to convert (for testing; if not provided, converts all episodes)")
    args = p.parse_args()

    # API key for object prompt extraction (always uses Gemini)
    # This is explicitly handled here because object prompt extraction is not yet part of a provider.
    gemini_prompt_api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not gemini_prompt_api_key:
        print("Error: GEMINI_API_KEY environment variable or --api-key argument is required for object prompt extraction (which uses Gemini).")
        sys.exit(2)

    repo_or_path = args.repo_id
    use_local = os.path.exists(repo_or_path)
    print(f"Loading dataset from {'local path' if use_local else 'Hugging Face repo id'}: {repo_or_path}")

    try:
        ds = LeRobotDataset(repo_or_path)        
    except Exception as exc:
        print("Error loading LeRobotDataset:", exc)
        raise

    meta = getattr(ds, "meta", None)
    camera_key = args.camera

    # Grab the first frame
    try:
        first_item = ds[0]
    except Exception as exc:
        print("Error accessing dataset[0]:", exc)
        raise

    if camera_key not in first_item:
        print(f"Error: camera key '{camera_key}' not found in dataset[0] contents. Keys available: {list(first_item.keys())}")
        sys.exit(1)

    frame_tensor = first_item[camera_key]
    pil_img = tensor_to_pil(frame_tensor)
    img_buf = BytesIO()
    pil_img.save(img_buf, format="JPEG")
    img_bytes = img_buf.getvalue()
    mime_type = "image/jpeg"

    # TODO: Refactor to a function: Gemini call to extract object prompt from task description
    # If --object-prompt "jar with blue cap" is not provided:
    # Given Task prompt (e.g place red bottle in the drawer) extract the main first object to interact with (red bottle)
    ## Call Object Detector to extract the bbox of the object 
    object_prompt = args.object_prompt
    if object_prompt is None:
        task_text = None
        for candidate in ("tasks", "task", "description", "task_prompt"):
            if hasattr(meta, candidate):
                cval = getattr(meta, candidate)
                if cval:
                    task_text = " ".join(cval) if isinstance(cval, (list, tuple)) else str(cval)
                    break
        if not task_text:
            task_text = f"Dataset id: {repo_or_path}. Please identify the main object to grasp for the robot."

        ask = (
            "From the following task description, extract the single most important object the robot should grasp or interact with. "
            "Return a JSON object {\"object\": \"<object_name>\"} and nothing else.\n\n"
            f"TASK DESCRIPTION: {task_text}\n\n"
            "Respond only with the JSON object and no additional commentary."
        )
        print("Asking Gemini to extract main object from task description...")
        try:
            gem_text = gemini_rest_call_text(ask, gemini_prompt_api_key)
            try:
                parsed = extract_first_json_like(gem_text)
                if isinstance(parsed, dict) and "object" in parsed:
                    object_prompt = parsed["object"]
                else:
                    object_prompt = str(parsed) if isinstance(parsed, str) else str(gem_text).strip().splitlines()[0]
            except Exception:
                object_prompt = gem_text.strip().splitlines()[0]
        except Exception as e:
            print("Error calling Gemini to extract object:", e)
            raise

    object_prompt = (object_prompt or "").strip().strip('"').strip("'")
    print(f"Object to detect: '{object_prompt}'")

    # TODO: Refactor to a function: Instantiate bbox provider and call detect()
    # Instantiate and use the bounding box provider
    try:
        bbox_provider = get_bbox_provider(args.bbox_detector)
        detections, parsed = bbox_provider.detect(pil_img, object_prompt)
    except Exception as e:
        print("Error calling bounding box detector (provider):", e)
        raise

    if not detections:
        print("No detections returned by Gemini.")
        print("Full parsed JSON:")
        print(json.dumps(parsed, indent=2))
        sys.exit(0)

    primary = detections[0]
    print("Primary detection (rescaled to pixels):")
    print(json.dumps(primary, indent=2))

    
    # 1) Save annotated image (requirement for first frame)
    out_path = args.out
    # TODO: Refactor to a function: bbox_pixels normalization
    bbox_pixels = primary["box_pixels"]  # [x1, y1, x2, y2]
    x1_orig, y1_orig, x2_orig, y2_orig = bbox_pixels
    # Ensure coordinates are in the right order - min x, min y, max x, max y (draw requires it)
    x1, x2 = sorted([x1_orig, x2_orig])
    y1, y2 = sorted([y1_orig, y2_orig])
    bbox_pixels = [x1, y1, x2, y2]    
    draw_bbox_and_save(pil_img, tuple(bbox_pixels), out_path)
    print(f"Saved visualization to: {out_path}")

    # Instantiate tracker after bbox is extracted (as requested)
    if args.tracker and args.tracker.lower() != "none":
        try:
            from tracking_providers.tracking_provider_factory import get_tracking_provider as _get_tracking_provider
            _ = _get_tracking_provider(args.tracker, bbox_pixels)
            print(f"Tracker '{args.tracker}' instantiated.")
        except Exception as e:
            print(f"Warning: could not instantiate tracker '{args.tracker}': {e}")

    # 2) Convert the dataset to bboxes (if requested)
    if args.out_repo_id:    
        ds_new = add_bbox_and_remove_camera_features(
            repo_id=ds.root,
            new_repo_id=args.out_repo_id,
            bbox_pixels=bbox_pixels,
            keep_image_keys=args.keep_image_keys,
            camera_key_for_tracking=args.camera,
            tracker_name=args.tracker,
            annotate_images=args.annotate_image,
            max_episodes=args.max_episodes  # pass the new cmdline argument
        )
        print(f"Done. Modified dataset includes 'observation.environment_state' and camera columns removed. Saved in {ds_new.root}")
        # Upload to HF if requested
        _upload_out_repo_if_requested(local_dir=ds_new.root, repo_id=args.out_repo_id, mode=args.upload_out_repo)
    else:
        print("Warning: Did not convert the dataset to bboxes: --out-repo-id <Shani123/new_repo_with_bbox> output repo not provided. Exiting.")

if __name__ == "__main__":    
    main()
