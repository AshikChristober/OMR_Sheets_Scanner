import React, { useRef, useState } from "react";
import Webcam from "react-webcam";

export default function CameraCapture({ taskId, mode, onClose }) {
  const webcamRef = useRef(null);
  const [captured, setCaptured] = useState(null);
  const [loading, setLoading] = useState(false);
  const [readyToEvaluate, setReadyToEvaluate] = useState(false);
  const [result, setResult] = useState(null);

  const capture = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    setCaptured(imageSrc);
  };

  const retake = () => {
    setCaptured(null);
    setReadyToEvaluate(false);
    setResult(null);
  };

  const handleUpload = async () => {
    setLoading(true);

    // simulate a 2-second processing delay
    setTimeout(() => {
      setLoading(false);
      setReadyToEvaluate(true);
    }, 2000);
  };

  const handleEvaluate = async () => {
    setLoading(true);
    const blob = await fetch(captured).then((r) => r.blob());
    const formData = new FormData();
    formData.append("file", blob, "camera_capture.jpg");

    const res = await fetch(`http://127.0.0.1:5000/evaluate_task/${taskId}`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    setResult(data.result);
    setLoading(false);
  };

  const generateExcel = async () => {
    const res = await fetch(`http://127.0.0.1:5000/generate_report/${taskId}`);
    const data = await res.json();
    if (data.file_url) {
      window.open(`http://127.0.0.1:5000${data.file_url}`, "_blank");
    }
  };

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.85)",
        color: "white",
        zIndex: 1000,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <h2>{mode === "scan" ? "📷 Scan OMR Sheet" : "🎥 Camera Capture"}</h2>

      {/* show camera */}
      {!captured && (
        <>
          <Webcam
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            style={{ width: "500px", borderRadius: "10px" }}
          />
          <button
            onClick={capture}
            style={{
              marginTop: "15px",
              padding: "10px 25px",
              background: "#28a745",
              color: "white",
              border: "none",
              borderRadius: "8px",
              cursor: "pointer",
            }}
          >
            Capture
          </button>
        </>
      )}

      {/* preview + upload */}
      {captured && !result && !loading && (
        <>
          <img
            src={captured}
            alt="Captured"
            style={{ width: "500px", borderRadius: "10px", marginTop: "15px" }}
          />
          <div style={{ marginTop: "10px" }}>
            <button
              onClick={retake}
              style={{
                padding: "10px 20px",
                background: "#ffc107",
                border: "none",
                borderRadius: "8px",
                marginRight: "10px",
              }}
            >
              Retake
            </button>
            {!readyToEvaluate ? (
              <button
                onClick={handleUpload}
                style={{
                  padding: "10px 20px",
                  background: "#007bff",
                  border: "none",
                  color: "white",
                  borderRadius: "8px",
                }}
              >
                Upload
              </button>
            ) : (
              <button
                onClick={handleEvaluate}
                style={{
                  padding: "10px 20px",
                  background: "#17a2b8",
                  border: "none",
                  color: "white",
                  borderRadius: "8px",
                }}
              >
                Evaluate
              </button>
            )}
          </div>
        </>
      )}

      {/* show loading spinner */}
      {loading && (
        <div style={{ marginTop: "30px", textAlign: "center" }}>
          <div
            className="loader"
            style={{
              border: "6px solid #f3f3f3",
              borderTop: "6px solid #3498db",
              borderRadius: "50%",
              width: "60px",
              height: "60px",
              animation: "spin 1s linear infinite",
              margin: "auto",
            }}
          ></div>
          <p style={{ marginTop: "15px" }}>Processing... Please wait</p>

          <style>
            {`
              @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
              }
            `}
          </style>
        </div>
      )}

      {/* show results */}
      {result && (
        <div style={{ marginTop: "20px", textAlign: "center" }}>
          <h3>✅ Evaluation Complete</h3>
          <p>Score: {result.score_percent}%</p>
          <p>
            Correct Answers: {result.correct_answers} / {result.total_questions}
          </p>
          <img
            src={`http://127.0.0.1:5000/${result.debug_image}`}
            alt="Processed Debug"
            style={{
              width: "400px",
              borderRadius: "10px",
              marginTop: "10px",
              border: "2px solid white",
            }}
          />
          <button
            onClick={generateExcel}
            style={{
              marginTop: "20px",
              padding: "10px 25px",
              background: "#ffc107",
              color: "black",
              border: "none",
              borderRadius: "8px",
              cursor: "pointer",
            }}
          >
            📊 Download Excel Report
          </button>
        </div>
      )}

      <button
        onClick={onClose}
        style={{
          marginTop: "25px",
          padding: "10px 25px",
          background: "#dc3545",
          border: "none",
          borderRadius: "8px",
          color: "white",
          cursor: "pointer",
        }}
      >
        Close
      </button>
    </div>
  );
}
