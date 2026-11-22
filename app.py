import importlib.util
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import av

try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False

st.set_page_config(
    page_title="Real-Time Face Recognition System",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_PATH = os.path.join(BASE_DIR, "src", "utils.py")


@st.cache_resource(show_spinner=False)
def _load_project_utils(path, mtime):
    spec = importlib.util.spec_from_file_location("utils", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["utils"] = module
    spec.loader.exec_module(module)
    return module


if not os.path.exists(UTILS_PATH):
    st.error(f"src/utils.py not found. Expected at:\n{UTILS_PATH}")
    st.stop()

utils = _load_project_utils(UTILS_PATH, os.path.getmtime(UTILS_PATH))

REQUIRED_UTILS = [
    "RESULTS_DIR", "LABEL_MAP_PATH", "METHODS", "METHOD_LABELS", "LABEL_TO_KEY", "RESULT_STEMS",
    "model_path", "load_recognizer", "load_label_map", "id_to_name_map",
    "get_threshold", "threshold_source", "decode_image", "limit_size", "to_gray",
    "recognize_image",
]
_missing = [n for n in REQUIRED_UTILS if not hasattr(utils, n)]
if _missing:
    st.error("The `src/utils.py` on disk is an OLD version and is missing: " + ", ".join(_missing))
    st.stop()

if not hasattr(cv2, "face"):
    st.error("OpenCV Contrib is required: the `cv2.face` module is missing.")
    st.code("python -m pip uninstall opencv-python opencv-contrib-python -y\n"
            "python -m pip install opencv-contrib-python")
    st.stop()


st.markdown("""
<style>
.main-title {font-size: 2.2rem; font-weight: 700; margin-bottom: 0px;}
.sub-title {font-size: 1.05rem; color: #A0AEC0; margin-bottom: 25px;}
footer {visibility: hidden;}
.custom-footer {text-align:center; padding:20px; font-size:0.85rem; color:#718096;
                border-top:1px solid #2D3748; margin-top:50px;}
</style>""", unsafe_allow_html=True)
st.markdown('<p class="main-title">Real-Time Face Recognition System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Computer Vision | Face Recognition Methods Comparison</p>',
            unsafe_allow_html=True)

METRICS_PATH = os.path.join(utils.RESULTS_DIR, "metrics.csv")


@st.cache_resource(show_spinner=False)
def _cached_model(method, model_mtime, labels_mtime):
    return utils.load_recognizer(method)


def get_model(method):
    p = utils.model_path(method)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Model not found:\n{p}\nTrain {utils.METHOD_LABELS[method]} first.")
    lm = os.path.getmtime(utils.LABEL_MAP_PATH) if os.path.exists(utils.LABEL_MAP_PATH) else 0
    return _cached_model(method, os.path.getmtime(p), lm)


def get_id_to_name():
    return utils.id_to_name_map(utils.load_label_map())


st.sidebar.title("System Sections")
section = st.sidebar.radio("Select Module:", [
    "1. Face Recognition Demo",
    "1B. Webcam / Real-Time Recognition",
    "2. Method Comparison",
    "3. Evaluation & Results",
    "4. About Project",
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Project Details**")
st.sidebar.caption("Area: Computer Vision")
st.sidebar.caption("Framework: Python + OpenCV")
st.sidebar.caption("Detector: Haar Cascade")
st.sidebar.caption("Classifiers: LBPH, Eigenfaces, Fisherfaces")


# ============================================================
# 1. STATIC IMAGE DEMO
# ============================================================
if section == "1. Face Recognition Demo":
    st.header("1. Face Recognition Demo")
    st.write("Upload a face image. Detection, preprocessing, model loading, label mapping and "
             "threshold logic are identical to the CLI / webcam system.")

    col_in, col_out = st.columns(2, gap="medium")

    with col_in:
        st.subheader("Input & Settings")
        method_label = st.selectbox("Select Recognition Algorithm:", list(utils.LABEL_TO_KEY))
        method = utils.LABEL_TO_KEY[method_label]
        compare_all = st.checkbox("Also compare all three methods on this image")
        precropped = st.checkbox(
            "Accept tightly-cropped face images when Haar finds no face",
            value=True,
        )
        central_thr = utils.get_threshold(method)
        st.info(f"Unknown threshold for {method_label}: {central_thr:.2f} "
                f"({utils.threshold_source(method)})")
        with st.expander("Advanced: override threshold", expanded=True):
            use_override = st.checkbox("Use custom threshold", value=False)
            override_val = st.number_input("Threshold", value=float(central_thr), step=5.0)
        uploaded = st.file_uploader("Upload Image (JPG / PNG):", type=["jpg", "jpeg", "png"])
        run_button = st.button("Recognize Face", type="primary", use_container_width=True)

    with col_out:
        st.subheader("Recognition Output")
        if uploaded is None:
            if run_button:
                st.warning("Upload an image first.")
        else:
            img = utils.decode_image(uploaded.getvalue())
            if img is None:
                st.error("Could not read the uploaded image.")
            elif not run_button:
                st.image(cv2.cvtColor(utils.limit_size(img), cv2.COLOR_BGR2RGB),
                         caption="Uploaded Image Preview", use_container_width=True)
            else:
                t0 = time.time()
                frame = utils.limit_size(img)
                gray = utils.to_gray(frame)
                try:
                    id_to_name = get_id_to_name()
                    methods = list(utils.METHODS) if compare_all else [method]
                    rows, models = [], {}
                    for m in methods:
                        models[m] = get_model(m)
                    for m in methods:
                        rec, mpath, _ = models[m]
                        thr = utils.get_threshold(
                            m, override_val if (use_override and m == method) else None
                        )
                        found = utils.recognize_image(
                            gray, rec, id_to_name, thr, allow_precropped=precropped
                        )
                        for i, r in enumerate(found, 1):
                            rows.append(dict(face=i, key=m, model_path=mpath, **r))
                except Exception as e:
                    st.error(str(e))
                    st.stop()

                if not rows:
                    st.error("No face detected in the uploaded image.")
                else:
                    faces = [(r["face"], r["face_img"]) for r in rows if r["key"] == method]
                    out = frame.copy()
                    for r in [r for r in rows if r["key"] == method]:
                        x, y, w, h = r["box"]
                        color = (0, 255, 0) if r["known"] else (0, 0, 255)
                        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(out, f"{r['name']} ({r['distance']:.1f})", (x, max(y - 10, 25)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    elapsed = (time.time() - t0) * 1000
                    st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
                             caption=f"{method_label} Recognition Result", use_container_width=True)

                    st.markdown("---")
                    st.subheader("Recognition Details")
                    for r in [r for r in rows if r["key"] == method]:
                        st.markdown(f"**Face {r['face']}**")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Predicted Name", r["name"])
                        c2.metric("Distance", f"{r['distance']:.2f}")
                        c3.metric("Recognition Status", "Recognized" if r["known"] else "Unknown")
                        c4.metric("Recognition Method", method_label)
                        st.caption(f"Closest class in model: **{r['closest']}** (ID {r['label_id']}) | "
                                   f"Threshold: {r['threshold']:.2f}")

                    if compare_all:
                        st.subheader("All Methods")
                        st.dataframe(pd.DataFrame([{
                            "Face": r["face"], "Method": utils.METHOD_LABELS[r["key"]],
                            "Predicted Name": r["name"], "Closest Class": r["closest"],
                            "Distance": round(r["distance"], 2), "Threshold": round(r["threshold"], 2),
                            "Status": "Recognized" if r["known"] else "Unknown"} for r in rows]),
                            use_container_width=True, hide_index=True)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Faces Detected", len(faces))
                    m2.metric("Recognized", sum(r["known"] for r in rows if r["key"] == method))
                    m3.metric("Execution Time", f"{elapsed:.1f} ms")


# ============================================================
# 1B. WEBCAM / REAL-TIME RECOGNITION
# ============================================================
elif section == "1B. Webcam / Real-Time Recognition":
    st.header("1B. Webcam / Real-Time Recognition")
    st.write("Use your browser webcam for real-time face recognition.")

    if not WEBRTC_AVAILABLE:
        st.error("Browser webcam support is not available.")
        st.code("python -m pip install streamlit-webrtc av")
        st.stop()

    c_alg, c_thr = st.columns(2)
    with c_alg:
        webcam_method_label = st.selectbox(
            "Select Recognition Algorithm:",
            list(utils.LABEL_TO_KEY),
            index=0,
            key="webcam_method_label",
        )
        webcam_method = utils.LABEL_TO_KEY[webcam_method_label]
        webcam_precropped = st.checkbox(
            "Accept tightly-cropped faces when Haar finds no face",
            value=False,
            key="webcam_precropped",
        )

    with c_thr:
        default_thr = float(utils.get_threshold(webcam_method))
        max_slider = 200.0 if webcam_method == "lbph" else (10000.0 if webcam_method == "eigen" else 2000.0)
        step_slider = 2.0 if webcam_method == "lbph" else (100.0 if webcam_method == "eigen" else 20.0)
        webcam_threshold = st.slider(
            f"Live Unknown Threshold ({webcam_method_label}):",
            min_value=10.0,
            max_value=max_slider,
            value=min(default_thr, max_slider),
            step=step_slider,
            help="Lower = stricter (more Unknowns). Higher = looser. Watch the distance number on screen to tune this.",
        )

    try:
        webcam_id_to_name = get_id_to_name()
        webcam_recognizer, _, _ = get_model(webcam_method)
    except Exception as exc:
        st.error("Could not load the selected recognition model.")
        st.exception(exc)
        st.stop()

    class FaceRecognitionVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.method = webcam_method
            self.threshold = float(webcam_threshold)
            self.precropped = webcam_precropped
            self.recognizer = webcam_recognizer
            self.id_to_name = webcam_id_to_name
            self.frame_count = 0
            self.prediction_history = []
            self.history_size = 7

        def _stable_prediction(self, result):
            if result is None:
                self.prediction_history.append(("Unknown", False, None))
            else:
                name = result.get("name", "Unknown")
                known = bool(result.get("known", False))
                distance = float(result.get("distance", 0.0))
                self.prediction_history.append((name if known else "Unknown", known, distance))

            if len(self.prediction_history) > self.history_size:
                self.prediction_history.pop(0)

            known_predictions = [p for p in self.prediction_history if p[1]]
            if known_predictions:
                name_counts = {}
                for name, _, _ in known_predictions:
                    name_counts[name] = name_counts.get(name, 0) + 1

                best_name = max(name_counts, key=name_counts.get)
                if name_counts[best_name] >= 3:
                    distances = [p[2] for p in known_predictions if p[0] == best_name and p[2] is not None]
                    avg_dist = sum(distances) / len(distances) if distances else None
                    return best_name, True, avg_dist

            all_dists = [p[2] for p in self.prediction_history if p[2] is not None]
            avg_dist = sum(all_dists) / len(all_dists) if all_dists else None
            return "Unknown", False, avg_dist

        def recv(self, frame):
            raw_image = frame.to_ndarray(format="bgr24")
            self.frame_count += 1

            try:
                image = utils.limit_size(cv2.flip(raw_image, 1))
                gray = utils.to_gray(image)

                found = utils.recognize_image(
                    gray,
                    self.recognizer,
                    self.id_to_name,
                    self.threshold,
                    allow_precropped=self.precropped,
                )

                if not found:
                    self._stable_prediction(None)
                    cv2.putText(
                        image, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
                    )
                else:
                    for idx, result in enumerate(found):
                        x, y, w, h = result["box"]
                        distance = float(result.get("distance", 0.0))
                        closest = result.get("closest", "?")

                        if idx == 0:
                            stable_name, stable_known, stable_dist = self._stable_prediction(result)
                            disp_dist = stable_dist if stable_dist is not None else distance
                        else:
                            stable_known = bool(result["known"])
                            stable_name = result["name"] if stable_known else "Unknown"
                            disp_dist = distance

                        if stable_known:
                            color = (0, 255, 0)
                            label = f"{stable_name} ({disp_dist:.1f})"
                        else:
                            color = (0, 0, 255)
                            label = f"Unknown [near: {closest}] ({disp_dist:.1f})"

                        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(
                            image, label, (x, max(y - 10, 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA
                        )

                cv2.putText(
                    image, f"Method: {utils.METHOD_LABELS[self.method]} | Thr: {self.threshold:.1f}",
                    (20, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
                )

            except Exception as exc:
                image = raw_image
                cv2.putText(
                    image, "Recognition error", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA
                )
                if self.frame_count == 1:
                    print("Webcam recognition error:", repr(exc))

            return av.VideoFrame.from_ndarray(image, format="bgr24")

    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    ctx = webrtc_streamer(
        key=f"face-recognition-webcam-{webcam_method}",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=FaceRecognitionVideoProcessor,
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 15}},
            "audio": False,
        },
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.threshold = float(webcam_threshold)
        ctx.video_processor.precropped = webcam_precropped


# ============================================================
# 2, 3, 4. COMPARISON, EVALUATION & ABOUT
# ============================================================
elif section == "2. Method Comparison":
    st.header("2. Method Performance Comparison")
    st.write("Quantitative comparison across LBPH, Eigenfaces, and Fisherfaces.")
    if os.path.exists(METRICS_PATH):
        st.dataframe(pd.read_csv(METRICS_PATH), use_container_width=True, hide_index=True)
        st.markdown("---")
        st.subheader("Key Findings")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**LBPH (Local Binary Patterns)**")
            st.write("- Uses local texture histograms.")
            st.write("- Most tolerant of lighting changes of the three.")
        with c2:
            st.markdown("**Eigenfaces (PCA)**")
            st.write("- Global appearance in a PCA subspace.")
            st.write("- Sensitive to lighting and pose.")
        with c3:
            st.markdown("**Fisherfaces (LDA)**")
            st.write("- Maximizes between-class separation.")
            st.write("- Only (classes - 1) dimensions; needs varied training data.")
    else:
        st.info("No metrics.csv found in results/. Run option 7 in main.py first.")

elif section == "3. Evaluation & Results":
    st.header("3. Experimental Evaluation & Charts")
    st.subheader("Evaluation Metrics Table")
    if os.path.exists(METRICS_PATH):
        st.dataframe(pd.read_csv(METRICS_PATH), use_container_width=True, hide_index=True)
    else:
        st.warning("`results/metrics.csv` not found. Run evaluation first.")
    st.markdown("---")
    st.subheader("Performance Comparison")
    cmp_path = os.path.join(utils.RESULTS_DIR, "comparison.png")
    if os.path.exists(cmp_path):
        st.image(cmp_path, use_container_width=True)
    else:
        st.info("comparison.png not found.")
    st.markdown("---")
    st.subheader("Confusion Matrices")
    for stem in utils.RESULT_STEMS.values():
        p = os.path.join(utils.RESULTS_DIR, f"confusion_matrix_{stem}.png")
        if os.path.exists(p):
            st.image(p, caption=os.path.basename(p), use_container_width=True)
    st.subheader("Distance Distributions (threshold calibration)")
    for key in utils.METHODS:
        p = os.path.join(utils.RESULTS_DIR, f"distance_hist_{key}.png")
        if os.path.exists(p):
            st.image(p, caption=os.path.basename(p), use_container_width=True)

elif section == "4. About Project":
    st.header("4. Project Documentation & Methodology")
    st.subheader("Pipeline Architecture")
    st.write("""
1. **Face detection**: Haar Cascade finds face boxes (largest first).
2. **Preprocessing** (`utils.normalize_face`, used everywhere): grayscale, face crop,
   resize to 200 x 200, histogram equalization.
3. **Classification**: LBPH, Eigenfaces (PCA) or Fisherfaces (LDA) return the closest known
   person and a distance.
4. **Unknown detection**: the distance is compared with a per-method threshold
   (`models/thresholds.json`, defaults LBPH 85 / Eigenfaces 4000 / Fisherfaces 350 until
   calibrated).
5. **Names**: label IDs are converted to names through `models/label_map.json`.
""")
    st.markdown("---")
    st.subheader("Current Label Mapping")
    try:
        mapping = utils.id_to_name_map(utils.load_label_map())
        st.dataframe(pd.DataFrame([{"Label ID": i, "Person": n} for i, n in sorted(mapping.items())]),
                     use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(str(e))
    st.subheader("Thresholds in use")
    st.dataframe(pd.DataFrame([{"Method": utils.METHOD_LABELS[m],
                                "Threshold": round(utils.get_threshold(m), 2),
                                "Source": utils.threshold_source(m)} for m in utils.METHODS]),
                 use_container_width=True, hide_index=True)

st.markdown(
    '<div class="custom-footer">'
    'Real-Time Face Recognition System | Computer Vision Assignment<br>'
    '<span>Developed by Shirsam Bhattacharjee & Roni Ghosh</span>'
    '</div>',
    unsafe_allow_html=True,
)