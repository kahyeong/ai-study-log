import os, warnings, time, csv, argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GLOG_minloglevel"] = "2"
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import cv2
import mediapipe as mp
from datetime import datetime

def draw_text(img, text, org=(20,40)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

def open_cam(index=0, w=640, h=480):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    return cap

def run_face(cap):
    with mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as det:
        ok, frame = cap.read()
        if not ok: return None, 0
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = det.process(rgb)
        n = 0
        if res.detections:
            h,w = frame.shape[:2]
            for d in res.detections:
                n += 1
                bb = d.location_data.relative_bounding_box
                x1,y1 = int(bb.xmin*w), int(bb.ymin*h)
                x2,y2 = int((bb.xmin+bb.width)*w), int((bb.ymin+bb.height)*h)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        return frame, n

def run_pose(cap):
    with mp.solutions.pose.Pose(model_complexity=0) as pose:
        ok, frame = cap.read()
        if not ok: return None, 0
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        n = 0
        if res.pose_landmarks:
            n = len(res.pose_landmarks.landmark)
            mp.solutions.drawing_utils.draw_landmarks(frame, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        return frame, n

def run_hands(cap):
    with mp.solutions.hands.Hands(model_complexity=0, max_num_hands=2) as hands:
        ok, frame = cap.read()
        if not ok: return None, 0
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        n = 0
        if res.multi_hand_landmarks:
            for h in res.multi_hand_landmarks:
                n += len(h.landmark)
                mp.solutions.drawing_utils.draw_landmarks(frame, h, mp.solutions.hands.HAND_CONNECTIONS)
        return frame, n

def auto_video_name(mode):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{mode}_{ts}.mp4"

def make_writer(cap, outfile=None):
    # cap에서 fps를 읽되, 실패 시 30fps로
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(outfile, fourcc, fps, (w, h))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["face","pose","hands"], required=True)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--record", action="store_true", help="CSV로 fps/키포인트 로그 저장")
    ap.add_argument("--outfile", default=None, help="CSV 파일 경로(기본: {mode}_log.csv)")
    # ✅ 추가: 영상 저장 전용 옵션
    ap.add_argument("--save-video", action="store_true", help="분석 오버레이가 그려진 영상을 mp4로 저장")
    ap.add_argument("--video-file", default=None, help="영상 파일 경로(기본: {mode}_YYYYmmdd_HHMMSS.mp4)")
    args = ap.parse_args()

    cap = open_cam(args.camera, 640, 480)
    runner = {"face": run_face, "pose": run_pose, "hands": run_hands}[args.mode]

    # CSV 로깅 준비
    csv_path = args.outfile or f"{args.mode}_log.csv"
    writer = None
    f_csv = None

    # ▶ 영상 저장 준비 (옵션 켠 경우만)
    vw = None
    if args.save_video:
        video_path = args.video_file or auto_video_name(args.mode)
        vw = make_writer(cap, video_path)
        if not vw.isOpened():
            print(f"[WARN] 비디오 파일을 열 수 없습니다: {video_path}")

    f_prev = time.time()

    while True:
        t0 = time.time()
        frame, n_kpts = runner(cap)
        if frame is None:
            break

        # FPS
        fps = 1.0 / max(1e-6, t0 - f_prev)
        f_prev = t0
        draw_text(frame, f"{args.mode.upper()}  FPS:{fps:.1f}  KPTS:{n_kpts}")

        # CSV 기록
        if args.record:
            if writer is None:
                f_csv = open(csv_path, "w", newline="", encoding="utf-8")
                writer = csv.writer(f_csv)
                writer.writerow(["ts","fps","n_kpts"])
            writer.writerow([t0, f"{fps:.3f}", n_kpts])

        # ▶ 영상 저장
        if vw is not None and vw.isOpened():
            vw.write(frame)

        cv2.imshow("detect_experiment - ESC to quit", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # 자원 정리
    if f_csv:
        f_csv.close()
        print(f"[INFO] CSV 저장 완료: {csv_path}")
    if vw is not None and vw.isOpened():
        vw.release()
        print("[INFO] 비디오 저장 완료")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
