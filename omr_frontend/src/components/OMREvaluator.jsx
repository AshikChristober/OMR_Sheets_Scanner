import React, { useState } from "react";
import axios from "axios";

const OMREvaluator = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [reportUrl, setReportUrl] = useState("");
  const [taskId, setTaskId] = useState(1); // Example: testing with Task 1

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setMessage("");
    setReportUrl("");
  };

  const handleUpload = async () => {
    if (!file) {
      setMessage("Please select a file first!");
      return;
    }

    setLoading(true);
    setMessage("Uploading and evaluating OMR sheet...");

    const formData = new FormData();
    formData.append("file", file);

    try {
      // Step 1 → Evaluate OMR (upload + process)
      const res = await axios.post(
        `http://127.0.0.1:5000/evaluate_task/${taskId}`,
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      if (res.data.result) {
        setMessage("✅ Evaluation completed! Now generating report...");
        // Step 2 → Generate report
        const reportRes = await axios.get(
          `http://127.0.0.1:5000/generate_report/${taskId}`
        );

        if (reportRes.data.file_url) {
          const url = `http://127.0.0.1:5000${reportRes.data.file_url}`;
          setReportUrl(url);
          setMessage("✅ Report ready for download!");
        } else {
          setMessage("⚠️ Report generation failed!");
        }
      } else {
        setMessage("❌ Evaluation failed.");
      }
    } catch (err) {
      console.error(err);
      setMessage("❌ Something went wrong while evaluating.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        background: "#f5f8fa",
        padding: "30px",
        maxWidth: "500px",
        margin: "50px auto",
        borderRadius: "12px",
        boxShadow: "0 2px 10px rgba(0,0,0,0.1)",
        textAlign: "center",
      }}
    >
      <h2>📄 OMR Evaluation</h2>

      <input
        type="file"
        onChange={handleFileChange}
        accept="image/*"
        style={{ marginBottom: "15px" }}
      />
      <br />

      <button
        onClick={handleUpload}
        disabled={loading}
        style={{
          background: "#0078d7",
          color: "#fff",
          border: "none",
          padding: "10px 20px",
          borderRadius: "6px",
          cursor: "pointer",
        }}
      >
        {loading ? "Processing..." : "Evaluate"}
      </button>

      <p style={{ marginTop: "15px", fontWeight: "bold" }}>{message}</p>

      {reportUrl && (
        <a href={reportUrl} download>
          <button
            style={{
              background: "#28a745",
              color: "#fff",
              border: "none",
              padding: "10px 20px",
              borderRadius: "6px",
              marginTop: "10px",
            }}
          >
            📊 Download Excel Report
          </button>
        </a>
      )}
    </div>
  );
};

export default OMREvaluator;
