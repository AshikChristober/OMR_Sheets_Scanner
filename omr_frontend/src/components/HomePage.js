import React, { useEffect, useState } from "react";
import CameraCapture from "./CameraCapture";

export default function HomePage({ onAddTask }) {
  const [tasks, setTasks] = useState([]);
  const [cameraMode, setCameraMode] = useState(null);
  const [activeTaskId, setActiveTaskId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [resultData, setResultData] = useState(null);

  // Fetch existing tasks (answer keys)
  useEffect(() => {
    fetch("http://127.0.0.1:5000/get_tasks")
      .then((res) => res.json())
      .then((data) => setTasks(data))
      .catch((err) => console.error("Error loading tasks:", err));
  }, []);

  // Handle file upload and evaluation
  const handleUpload = async (taskId) => {
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = "image/*";
    fileInput.onchange = async (e) => {
      const file = e.target.files[0];
      if (!file) return;

      setLoading(true);
      setResultData(null);

      const formData = new FormData();
      formData.append("file", file);

      try {
        // Step 1: Evaluate task (upload + analyze)
        const res = await fetch(`http://127.0.0.1:5000/evaluate_task/${taskId}`, {
          method: "POST",
          body: formData,
        });
        const data = await res.json();

        if (!data.result) {
          alert("❌ Evaluation failed!");
          setLoading(false);
          return;
        }

        // Step 2: Generate Excel report
        const reportRes = await fetch(`http://127.0.0.1:5000/generate_report/${taskId}`);
        const reportData = await reportRes.json();

        // Combine results
        setResultData({
          ...data.result,
          report_url: `http://127.0.0.1:5000${reportData.file_url}`,
        });
      } catch (error) {
        console.error("Error during evaluation:", error);
        alert("Something went wrong during evaluation!");
      } finally {
        setLoading(false);
      }
    };
    fileInput.click();
  };

  return (
    <div style={{ textAlign: "center" }}>
      <button
        onClick={onAddTask}
        style={{
          padding: "12px 30px",
          fontSize: "18px",
          backgroundColor: "#007bff",
          color: "white",
          border: "none",
          borderRadius: "8px",
          cursor: "pointer",
          marginBottom: "20px",
        }}
      >
        ➕ Add Task
      </button>

      <h3>📋 Created Tasks</h3>
      {tasks.length === 0 && <p>No tasks created yet.</p>}

      {tasks.map((task) => (
        <div
          key={task.id}
          style={{
            margin: "10px auto",
            padding: "15px",
            background: "#f8f9fa",
            borderRadius: "8px",
            width: "60%",
            boxShadow: "0 2px 6px rgba(0,0,0,0.1)",
          }}
        >
          <p><strong>{task.category}</strong></p>
          <p>Created on: {task.created_at}</p>
          <div style={{ display: "flex", justifyContent: "center", gap: "10px" }}>
            <button
              onClick={() => {
                setActiveTaskId(task.id);
                setCameraMode("scan");
              }}
            >
              📷 Scan
            </button>
            <button
              onClick={() => {
                setActiveTaskId(task.id);
                setCameraMode("camera");
              }}
            >
              🎥 Camera
            </button>
            <button onClick={() => handleUpload(task.id)}>📁 Evaluate OMR</button>
          </div>
        </div>
      ))}

     {loading && (
  <div className="loading-overlay">
    <div className="loader"></div>
    <p style={{ color: "#007bff", fontWeight: "bold" }}>Evaluating... Please wait ⏳</p>
  </div>
)}


      {/* Show evaluation results */}
      {resultData && (
        <div
          style={{
            background: "#ffffff",
            borderRadius: "10px",
            padding: "20px",
            marginTop: "30px",
            width: "50%",
            marginLeft: "auto",
            marginRight: "auto",
            boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
          }}
        >
          <h3>✅ Evaluation Result</h3>
          <p><strong>Total Questions:</strong> {resultData.total_questions}</p>
          <p><strong>Correct Answers:</strong> {resultData.correct_answers}</p>
          <p><strong>Score:</strong> {resultData.score_percent}%</p>

          {/* Debug image preview */}
          {resultData.debug_image && (
            <div>
              <h4>🖼️ Detected Bubbles</h4>
              <img
                src={`http://127.0.0.1:5000${resultData.debug_image}`}
                alt="debug"
                style={{
                  width: "80%",
                  borderRadius: "10px",
                  marginTop: "10px",
                  border: "2px solid #007bff",
                }}
              />
            </div>
          )}

          {/* Download report button */}
          {resultData.report_url && (
            <div style={{ marginTop: "20px" }}>
              <a
                href={resultData.report_url}
                target="_blank"
                rel="noopener noreferrer"
                download
              >
                <button
                  style={{
                    background: "#28a745",
                    color: "white",
                    border: "none",
                    padding: "10px 20px",
                    borderRadius: "6px",
                    cursor: "pointer",
                  }}
                >
                  📊 Download Excel Report
                </button>
              </a>
            </div>
          )}
        </div>
      )}

      {/* Camera capture component (unchanged) */}
      {cameraMode && (
        <CameraCapture
          taskId={activeTaskId}
          mode={cameraMode}
          onClose={() => {
            setCameraMode(null);
            setActiveTaskId(null);
          }}
        />
      )}
    </div>
  );
}
