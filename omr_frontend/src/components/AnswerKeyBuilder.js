import React, { useState } from "react";

export default function AnswerKeyBuilder({ selectedGroup, onBack }) {
  const [answers, setAnswers] = useState([{ qno: 1, answer: "A" }]);

  const addQuestion = () => {
    setAnswers([...answers, { qno: answers.length + 1, answer: "A" }]);
  };

  const updateAnswer = (index, value) => {
    const newAnswers = [...answers];
    newAnswers[index].answer = value;
    setAnswers(newAnswers);
  };

  const saveAnswers = async () => {
    const formatted = {};
    answers.forEach((item) => {
      formatted[item.qno] = item.answer;
    });

    const response = await fetch("http://127.0.0.1:5000/save_answer_key", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ group: selectedGroup, answer_key: formatted }),
    });

    const data = await response.json();

    if (data.message) {
      alert(data.message);
      // ✅ Automatically go back to HomePage
      window.location.reload();  // reloads and shows updated task list
    } else {
      alert("Something went wrong! Please try again.");
    }
  };

  return (
    <div style={{ marginTop: "30px" }}>
      <h2>{selectedGroup} - Upload Answers</h2>
      <button
        onClick={onBack}
        style={{
          marginBottom: "20px",
          padding: "8px 16px",
          backgroundColor: "#6c757d",
          color: "white",
          border: "none",
          borderRadius: "6px",
          cursor: "pointer",
        }}
      >
        ⬅ Back
      </button>

      {answers.map((item, index) => (
        <div key={index} style={{ margin: "10px" }}>
          <label>Question {item.qno}: </label>
          <select
            value={item.answer}
            onChange={(e) => updateAnswer(index, e.target.value)}
          >
            <option value="A">A</option>
            <option value="B">B</option>
            <option value="C">C</option>
            <option value="D">D</option>
          </select>
        </div>
      ))}

      <button
        onClick={addQuestion}
        style={{
          margin: "15px",
          padding: "8px 16px",
          backgroundColor: "#ffc107",
          border: "none",
          borderRadius: "6px",
          cursor: "pointer",
        }}
      >
        ➕ Add Question
      </button>

      <button
        onClick={saveAnswers}
        style={{
          backgroundColor: "green",
          color: "white",
          border: "none",
          borderRadius: "8px",
          padding: "10px 20px",
          cursor: "pointer",
        }}
      >
        💾 Save Answer Key
      </button>
    </div>
  );
}
