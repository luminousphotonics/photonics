import React, { useState, useRef, useEffect } from "react";
import ModularVisualization from "../components/ModularVisualization";
import { SimulationData } from "../types";

interface SseMessageData {
  message: string;
}

interface FormDataState {
  floor_width: string;
  floor_length: string;
  target_ppfd: string;
}

const SimulationForm: React.FC = () => {
  // Form state
  const [formData, setFormData] = useState<FormDataState>({
    floor_width: "12",
    floor_length: "12",
    target_ppfd: "1250",
  });

  // SSE and simulation state
  const [progress, setProgress] = useState<number>(0);
  const [logMessages, setLogMessages] = useState<string[]>([]);
  const [simulationResult, setSimulationResult] = useState<SimulationData | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const logOutputRef = useRef<HTMLDivElement>(null);

  // Auto-scroll log output when messages update.
  useEffect(() => {
    if (logOutputRef.current) {
      logOutputRef.current.scrollTop = logOutputRef.current.scrollHeight;
    }
  }, [logMessages]);

  // Handle form changes
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  // Start simulation on button click.
  const startSimulation = () => {
    // Reset state.
    setProgress(0);
    setLogMessages([]);
    setSimulationResult(null);

    const params = new URLSearchParams({
      start: "1",
      floor_width: formData.floor_width,
      floor_length: formData.floor_length,
      target_ppfd: formData.target_ppfd,
    });
    const url = `/api/ml_simulation/progress/?${params.toString()}`;
    const es = new EventSource(url);
    eventSourceRef.current = es;

    es.onmessage = (event) => {
      const data: SseMessageData = JSON.parse(event.data);
      const { message } = data;

      // If it's a result message, parse and set the final simulation result.
      if (message.startsWith("RESULT:")) {
        try {
          const jsonStr = message.replace("RESULT:", "");
          const result: SimulationData = JSON.parse(jsonStr);
          setSimulationResult(result);
          setLogMessages((prev) => [...prev, "[INFO] Simulation complete!"]);
        } catch (err) {
          setLogMessages((prev) => [...prev, "[ERROR] Failed to parse result JSON"]);
        }
        return;
      }

      // If it's an error message, log it.
      if (message.startsWith("ERROR:")) {
        setLogMessages((prev) => [...prev, message]);
        return;
      }

      // If the message contains progress information, update the progress bar.
      if (message.startsWith("PROGRESS:")) {
        const pctStr = message.replace("PROGRESS:", "").trim();
        const pct = parseFloat(pctStr);
        if (!isNaN(pct)) {
          setProgress(pct);
        }
        // Optionally log the progress update.
        setLogMessages((prev) => [...prev, `[INFO] ${pct}% complete`]);
      } else {
        // Otherwise, simply append the message to the log.
        setLogMessages((prev) => [...prev, message]);
      }
    };

    es.onerror = (err) => {
      setLogMessages((prev) => [...prev, "[ERROR] SSE connection failed"]);
      console.error("EventSource failed:", err);
      es.close();
    };
  };

  return (
    <div style={{ maxWidth: "800px", margin: "0 auto" }}>
      <h1>Lighting Simulation Progress</h1>

      {/* Progress Bar */}
      <div style={{ margin: "20px 0" }}>
        <div style={{ width: "100%", background: "#eee", borderRadius: "5px", overflow: "hidden", height: "20px" }}>
          <div
            style={{
              height: "100%",
              width: `${progress}%`,
              background: "#28a745",
              transition: "width 0.5s ease",
            }}
          />
        </div>
        <p style={{ textAlign: "center", marginTop: "5px", fontWeight: "bold" }}>{progress}%</p>
      </div>

      {/* Log Output */}
      <div
        ref={logOutputRef}
        style={{
          background: "#333",
          color: "#fff",
          padding: "10px",
          borderRadius: "5px",
          height: "200px",
          overflowY: "scroll",
          fontFamily: "monospace",
          fontSize: "0.9em",
          marginBottom: "20px",
        }}
      >
        {logMessages.map((line, idx) => (
          <div key={idx}>{line}</div>
        ))}
      </div>

      {/* Input Fields */}
      <div style={{ marginBottom: "20px" }}>
        <label>
          Floor Width (feet):{" "}
          <input type="number" name="floor_width" value={formData.floor_width} onChange={handleChange} style={{ marginRight: "10px" }} />
        </label>
        <label>
          Floor Length (feet):{" "}
          <input type="number" name="floor_length" value={formData.floor_length} onChange={handleChange} style={{ marginRight: "10px" }} />
        </label>
        <label>
          Target PPFD (µmol/m²/s):{" "}
          <input type="number" name="target_ppfd" value={formData.target_ppfd} onChange={handleChange} style={{ marginRight: "10px" }} />
        </label>
      </div>

      {/* Single Start Simulation Button */}
      <button
        onClick={startSimulation}
        style={{
          display: "inline-block",
          padding: "10px 20px",
          background: "#007bff",
          color: "#fff",
          border: "none",
          borderRadius: "5px",
          cursor: "pointer",
          fontSize: "1em",
        }}
      >
        Start Simulation
      </button>

      {/* Final Result & Visualization */}
      {simulationResult && (
        <div style={{ marginTop: "30px" }}>
          <h2>Simulation Results</h2>
          <p>
            <strong>Minimized MAD:</strong> {simulationResult.mad.toFixed(2)}
          </p>
          <p>
            <strong>Optimized PPFD:</strong> {simulationResult.optimized_ppfd.toFixed(2)} µmol/m²/s
          </p>
          <h3>Optimized Lumens by Layer</h3>
          <ul>
            {simulationResult.optimized_lumens_by_layer.map((lumens, i) => (
              <li key={i}>
                {i === 0 ? "Center COB" : `Layer ${i + 1}`}: {lumens.toFixed(2)} lumens
              </li>
            ))}
          </ul>
          <div style={{ marginTop: "20px" }}>
            <ModularVisualization
              mad={simulationResult.mad}
              optimized_ppfd={simulationResult.optimized_ppfd}
              floorWidth={simulationResult.floor_width}
              floorLength={simulationResult.floor_length}
              floorHeight={simulationResult.floor_height}
              optimizedLumensByLayer={simulationResult.optimized_lumens_by_layer}
              simulationResult={simulationResult}
            />
          </div>
          <div>
            <h2>Surface Graph</h2>
            <img src={`data:image/png;base64,${simulationResult.surface_graph}`} alt="Surface Graph" style={{ maxWidth: "100%" }} />
            <h2>Heatmap</h2>
            <img src={`data:image/png;base64,${simulationResult.heatmap}`} alt="Heatmap" style={{ maxWidth: "100%" }} />
          </div>
        </div>
      )}
    </div>
  );
};

export default SimulationForm;
