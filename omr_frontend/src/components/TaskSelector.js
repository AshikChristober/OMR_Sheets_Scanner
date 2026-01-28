import React from "react";

export default function TaskSelector({ onSelectGroup }) {
  const groups = ["TNPSC Group 1", "TNPSC Group 2", "TNPSC Group 3"];

  return (
    <div style={{ marginTop: "30px" }}>
      <h2>Select a Category</h2>
      <div style={{ display: "flex", justifyContent: "center", gap: "20px", marginTop: "20px" }}>
        {groups.map((group) => (
          <button
            key={group}
            onClick={() => onSelectGroup(group)}
            style={{
              padding: "15px 30px",
              borderRadius: "10px",
              border: "none",
              backgroundColor: "#17a2b8",
              color: "white",
              cursor: "pointer",
              fontSize: "16px",
            }}
          >
            {group}
          </button>
        ))}
      </div>
    </div>
  );
}
