# pixel -> mm measurement using a credit/ID card as the reference object.
# ISO/IEC 7810 ID-1 cards are 85.60 x 53.98 mm  = universal

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

register_heif_opener()


CARD_LONG_MM = 85.60
CARD_SHORT_MM = 53.98
CARD_ASPECT = CARD_LONG_MM / CARD_SHORT_MM   # ~ 1.586

CALIB_WIDTH = 4032   # K was calibrated at this width


def load_model(weights_path, device):
    weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, 2)
    in_feats_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feats_mask, 256, 2)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def read_image(path):
    # Handles JPEG, PNG, and HEIC. Applies EXIF orientation and rotates
    # portrait images to landscape so the calibration K applies as-is.
    img = cv2.imread(str(path))
    if img is None:
        pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def scale_K(K, w):
    s = w / CALIB_WIDTH
    K2 = K.copy()
    K2[0, 0] *= s; K2[1, 1] *= s
    K2[0, 2] *= s; K2[1, 2] *= s
    return K2


def undistort(img, K, dist):
    h, w = img.shape[:2]
    Ks = scale_K(K, w)
    new_K, _ = cv2.getOptimalNewCameraMatrix(Ks, dist, (w, h), alpha=1.0)
    return cv2.undistort(img, Ks, dist, None, new_K)


def detect_card(img, aspect_tol=0.15, min_area_frac=0.003, max_area_frac=0.15):
    # Edge + Otsu candidates, filter by aspect ratio of the min-area rect
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    contour_sets = []
    edges = cv2.Canny(blur, 30, 100)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    contour_sets.append(cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contour_sets.append(cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contour_sets.append(cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])

    best, best_score = None, -1
    for contours in contour_sets:
        for c in contours:
            area = cv2.contourArea(c)
            if area < W * H * min_area_frac or area > W * H * max_area_frac:
                continue
            rect = cv2.minAreaRect(c)
            (_, _), (rw, rh), _ = rect
            if rw < 5 or rh < 5:
                continue
            long_s, short_s = max(rw, rh), min(rw, rh)
            aspect = long_s / short_s
            err = abs(aspect - CARD_ASPECT) / CARD_ASPECT
            if err > aspect_tol:
                continue
            fill = area / (rw * rh)
            score = (1 - err) * fill
            if score > best_score:
                best_score = score
                best = rect
    return best


def click_card_corners(img):
    # User clicks the 4 corners of the card (any order) on a downscaled
    # preview. Returns 4 points in image coords, ordered TL, TR, BR, BL.
    #
    # Keys: LEFT CLICK = add corner (up to 4), R = reset, ENTER = confirm,
    #       ESC = cancel.
    scale = min(1.0, 1100 / max(img.shape[:2]))
    show = cv2.resize(img, None, fx=scale, fy=scale)
    win = "click 4 corners of the card  -  ENTER confirm, R reset, ESC cancel"

    state = {"pts": []}

    def redraw():
        view = show.copy()
        pts = state["pts"]
        for i, p in enumerate(pts):
            cv2.circle(view, p, 7, (0, 255, 255), -1)
            cv2.putText(view, str(i + 1), (p[0] + 10, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if len(pts) >= 2:
            cv2.polylines(view, [np.array(pts)], len(pts) == 4,
                          (0, 255, 0), 2)
        cv2.imshow(win, view)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(state["pts"]) < 4:
            state["pts"].append((x, y))
            redraw()

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    redraw()

    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == 13 and len(state["pts"]) == 4:
            break
        if k == 27:
            cv2.destroyWindow(win)
            return None
        if k in (ord("r"), ord("R")):
            state["pts"] = []
            redraw()

    cv2.destroyWindow(win)
    pts = np.array(state["pts"], dtype=np.float32) / scale
    return order_corners_tl_tr_br_bl(pts)


def order_corners_tl_tr_br_bl(pts):
    # TL has smallest x+y, BR has largest x+y. Of the remaining two,
    # TR has smaller y-x, BL has larger y-x.
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    mask = np.ones(4, dtype=bool)
    mask[np.argmin(s)] = False
    mask[np.argmax(s)] = False
    rest = pts[mask]
    diff = rest[:, 1] - rest[:, 0]
    tr = rest[np.argmin(diff)]
    bl = rest[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def card_homography(corners):
    # compute homography image -> mm in the card plane. Picks the long-side
    # orientation by comparing top edge length vs right edge length.
    top_len = np.linalg.norm(corners[1] - corners[0])
    right_len = np.linalg.norm(corners[2] - corners[1])
    if top_len >= right_len:
        dst = np.array([[0, 0],
                        [CARD_LONG_MM, 0],
                        [CARD_LONG_MM, CARD_SHORT_MM],
                        [0, CARD_SHORT_MM]], dtype=np.float32)
    else:
        dst = np.array([[0, 0],
                        [CARD_SHORT_MM, 0],
                        [CARD_SHORT_MM, CARD_LONG_MM],
                        [0, CARD_LONG_MM]], dtype=np.float32)
    H, _ = cv2.findHomography(corners, dst)
    return H


@torch.no_grad()
def predict_notebook_mask(model, img_bgr, device, score_thresh=0.5):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torchvision.transforms.functional.to_tensor(rgb).to(device)
    out = model([t])[0]
    if len(out["scores"]) == 0:
        return None, 0.0
    best = out["scores"].argmax()
    score = float(out["scores"][best])
    if score < score_thresh:
        return None, score
    mask = (out["masks"][best, 0] > 0.5).cpu().numpy().astype(np.uint8)
    return mask, score


def fit_min_rect(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    return cv2.minAreaRect(c)


def measure(img_bgr, K, dist, model, device, auto_card=False):
    # pipeline: undistort -> click 4 card corners -> compute homography
    # to mm -> segment notebook -> transform mask contour to mm space ->
    # min-area rect gives true on-plane width and height.
    und = undistort(img_bgr, K, dist)

    corners = click_card_corners(und)
    if corners is None:
        raise RuntimeError("could not locate reference card")
    H = card_homography(corners)

    mask, score = predict_notebook_mask(model, und, device)
    if mask is None:
        raise RuntimeError("notebook not detected")

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("no contour from mask")
    contour_px = max(contours, key=cv2.contourArea)
    pts_mm = cv2.perspectiveTransform(
        contour_px.astype(np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)
    rect_mm = cv2.minAreaRect(pts_mm)
    _, (w_mm, h_mm), _ = rect_mm

    return {
        "width_mm":      max(w_mm, h_mm),
        "height_mm":     min(w_mm, h_mm),
        "confidence":    score,
        "card_corners":  corners,
        "card_rect":     cv2.minAreaRect(corners.astype(np.float32)),
        "notebook_rect": cv2.minAreaRect(contour_px),
        "mask":          mask,
        "undistorted":   und,
        "homography":    H,
    }
