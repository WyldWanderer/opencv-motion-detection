import cv2
import numpy as np

def initialize_video_capture():
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Cannot open camera")
        exit()
    return video_capture

def get_video_properties(video_capture):
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    return frame_width, frame_height, fps

def initialize_video_writers(frame_width, frame_height, fps):
    size = (frame_width, frame_height)
    size_quad = (int(2 * frame_width), int(2 * frame_height))
    video_out_alert = cv2.VideoWriter('video_out_alert.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    video_out_quad = cv2.VideoWriter('video_out_quad.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size_quad)
    return video_out_alert, video_out_quad

def draw_banner_text(frame, text, banner_height_percent=0.08, font_scale=0.8, text_color=(0, 255, 0), font_thickness=2):
    banner_height = int(banner_height_percent * frame.shape[0])
    cv2.rectangle(frame, (0, 0), (frame.shape[1], banner_height), (0, 0, 0), thickness=-1)
    left_offset = 20
    location = (left_offset, int(10 + (banner_height_percent * frame.shape[0]) / 2))
    cv2.putText(frame, text, location, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA)

def process_frame(frame, bg_subtractor, ksize, max_contours, frame_count, frame_start):
    frame_erode_c = frame.copy()
    fg_mask = bg_subtractor.apply(frame)
    if frame_count <= frame_start:
        return frame, frame_erode_c, fg_mask, None, None

    motion_area = cv2.findNonZero(fg_mask)
    if motion_area is not None:
        x, y, w, h = cv2.boundingRect(motion_area)
        cv2.rectangle(frame, (x, y), (x + w, y + h), red, thickness=2)
        draw_banner_text(frame, 'Intrusion Alert', text_color=red)

    fg_mask_erode_c = cv2.erode(fg_mask, np.ones(ksize, np.uint8))
    motion_area_erode = cv2.findNonZero(fg_mask_erode_c)
    if motion_area_erode is not None:
        xe, ye, we, he = cv2.boundingRect(motion_area_erode)
        cv2.rectangle(frame_erode_c, (xe, ye), (xe + we, ye + he), red, thickness=2)
        draw_banner_text(frame_erode_c, 'Intrusion Alert', text_color=red)

    frame_fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
    frame_fg_mask_erode_c = cv2.cvtColor(fg_mask_erode_c, cv2.COLOR_GRAY2BGR)

    contours_erode, _ = cv2.findContours(fg_mask_erode_c, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours_erode) > 0:
        cv2.drawContours(frame_fg_mask_erode_c, contours_erode, -1, green, thickness=2)
        contours_sorted = sorted(contours_erode, key=cv2.contourArea, reverse=True)
        for idx in range(min(max_contours, len(contours_sorted))):
            xc, yc, wc, hc = cv2.boundingRect(contours_sorted[idx])
            if idx == 0:
                x1, y1, x2, y2 = xc, yc, xc + wc, yc + hc
            else:
                x1, y1, x2, y2 = min(x1, xc), min(y1, yc), max(x2, xc + wc), max(y2, yc + hc)
        cv2.rectangle(frame_erode_c, (x1, y1), (x2, y2), yellow, thickness=2)
        draw_banner_text(frame_erode_c, 'Intrusion Alert', text_color=red)

    draw_banner_text(frame_fg_mask, 'Foreground Mask')
    draw_banner_text(frame_fg_mask_erode_c, 'Foreground Mask (Eroded + Contours)')

    return frame, frame_erode_c, frame_fg_mask, frame_fg_mask_erode_c, fg_mask_erode_c

def build_quad_view(frame_fg_mask, frame, frame_fg_mask_erode_c, frame_erode_c):
    # Ensure all frames have the same dimensions
    frame_fg_mask = cv2.resize(frame_fg_mask, (frame.shape[1], frame.shape[0]))
    frame_fg_mask_erode_c = cv2.resize(frame_fg_mask_erode_c, (frame.shape[1], frame.shape[0]))
    frame_erode_c = cv2.resize(frame_erode_c, (frame.shape[1], frame.shape[0]))

    frame_top = np.hstack([frame_fg_mask, frame])
    frame_bot = np.hstack([frame_fg_mask_erode_c, frame_erode_c])
    frame_composite = np.vstack([frame_top, frame_bot])
    fc_h, fc_w, _ = frame_composite.shape
    cv2.line(frame_composite, (int(fc_w / 2), 0), (int(fc_w / 2), fc_h), yellow, thickness=3, lineType=cv2.LINE_AA)
    cv2.line(frame_composite, (0, int(fc_h / 2)), (fc_w, int(fc_h / 2)), yellow, thickness=3, lineType=cv2.LINE_AA)
    return frame_composite

def main():
    video_capture = initialize_video_capture()
    frame_width, frame_height, fps = get_video_properties(video_capture)
    video_out_alert, video_out_quad = initialize_video_writers(frame_width, frame_height, fps)

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
    ksize = (5, 5)
    max_contours = 3
    frame_count = 0
    frame_start = 5
    global red, yellow, green
    red, yellow, green = (0, 0, 255), (0, 255, 255), (0, 255, 0)

    while True:
        ret, frame = video_capture.read()
        frame_count += 1
        if frame is None:
            break

        frame, frame_erode_c, frame_fg_mask, frame_fg_mask_erode_c, fg_mask_erode_c = process_frame(
            frame, bg_subtractor, ksize, max_contours, frame_count, frame_start)

        # Ensure all frames are not None before building the quad view
        if frame_fg_mask is not None and frame_fg_mask_erode_c is not None and frame_erode_c is not None:
            frame_composite = build_quad_view(frame_fg_mask, frame, frame_fg_mask_erode_c, frame_erode_c)
            video_out_alert.write(frame_erode_c)
            video_out_quad.write(frame_composite)
            cv2.imshow('Frame Quad View', frame_composite)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    video_out_alert.release()
    video_out_quad.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()