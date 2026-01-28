import React, { useState } from "react";
import HomePage from "./components/HomePage";
import TaskSelector from "./components/TaskSelector";
import AnswerKeyBuilder from "./components/AnswerKeyBuilder";
import OMREvaluator from "./components/OMREvaluator";

function App() {
  const [stage, setStage] = useState("home");
  const [selectedGroup, setSelectedGroup] = useState("");

  return (
    <div className="App" style={{ textAlign: "center", padding: "30px" }}>
      <h1>📝 TNPSC OMR Evaluation System</h1>

      {/* ---- Stage 1: Home ---- */}
      {stage === "home" && (
        <HomePage
          onAddTask={() => setStage("select")}
          onEvaluate={() => setStage("evaluate")}
        />
      )}

      {/* ---- Stage 2: Select Group ---- */}
      {stage === "select" && (
        <TaskSelector
          onSelectGroup={(group) => {
            setSelectedGroup(group);
            setStage("builder");
          }}
          onBack={() => setStage("home")}
        />
      )}

      {/* ---- Stage 3: Build Answer Key ---- */}
      {stage === "builder" && (
        <AnswerKeyBuilder
          selectedGroup={selectedGroup}
          onBack={() => setStage("select")}
          onDone={() => setStage("evaluate")}
        />
      )}

      {/* ---- Stage 4: OMR Evaluation ---- */}
      {stage === "evaluate" && (
        <OMREvaluator onBack={() => setStage("home")} />
      )}
    </div>
  );
}

export default App;
