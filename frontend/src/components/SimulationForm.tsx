import React, { useState, useRef, useEffect } from "react";
import ModularVisualization from "../components/ModularVisualization";
import { SimulationData } from "../types";

// Immaculate jetColor function – produces a smooth jet colormap.
// This implementation is based on the MATLAB formulation:
//   r = clamp(1.5 - |4*t - 3|, 0, 1)
//   g = clamp(1.5 - |4*t - 2|, 0, 1)
//   b = clamp(1.5 - |4*t - 1|, 0, 1)
// where t is the normalized value between 0 and 1.
// The output is then scaled to 0-255 for standard RGB color strings.
function jetColor(value: number, min: number, max: number): string {
  // Normalize value to t in [0,1]
  let t = (value - min) / (max - min);
  t = Math.max(0, Math.min(1, t));

  // Compute red channel.
  // When t=0, |4*0 - 3| = 3, so r = clamp(1.5 - 3, 0,1) = 0.
  // When t=1, |4*1 - 3| = 1, so r = clamp(1.5 - 1, 0,1) = 0.5.
  const r = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 3)));
  
  // Compute green channel.
  // When t=0, |4*0 - 2| = 2, so g = clamp(1.5 - 2, 0,1) = 0.
  // When t=0.5, |4*0.5 - 2| = 0, so g = clamp(1.5 - 0, 0,1) = 1.
  // When t=1, |4*1 - 2| = 2, so g = clamp(1.5 - 2, 0,1) = 0.
  const g = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 2)));
  
  // Compute blue channel.
  // When t=0, |4*0 - 1| = 1, so b = clamp(1.5 - 1, 0,1) = 0.5.
  // When t=1, |4*1 - 1| = 3, so b = clamp(1.5 - 3, 0,1) = 0.
  const b = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 1)));

  // Scale each channel from [0,1] to [0,255].
  const R = Math.round(r * 255);
  const G = Math.round(g * 255);
  const B = Math.round(b * 255);
  return `rgb(${R}, ${G}, ${B})`;
}



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

  // State for plant growth stage selection
  const [growthStage, setGrowthStage] = useState<string>("propagation");

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

  // Handler for plant icon click to compute suggested PPFD
  const handlePlantClick = (e: React.MouseEvent<HTMLElement>) => {
    const plantType = e.currentTarget.getAttribute("data-plant");
    if (!plantType) return;

    // Updated mapping based on your provided table:
    const plantMapping: Record<string, Record<string, { dli: number; dayLength: number }>> = {
      cannabis: {
        propagation: { dli: 45, dayLength: 18 },
        vegetative: { dli: 50, dayLength: 18 },
        flowering: { dli: 50, dayLength: 12 },
      },
      tomatoes: {
        propagation: { dli: 25, dayLength: 16 },
        vegetative: { dli: 27.5, dayLength: 16 },
        flowering: { dli: 27.5, dayLength: 16 },
      },
      strawberries: {
        propagation: { dli: 20, dayLength: 8 },
        vegetative: { dli: 22.5, dayLength: 8 },
        flowering: { dli: 22.5, dayLength: 8 },
      },
      leafy: {
        propagation: { dli: 12, dayLength: 16 },
        vegetative: { dli: 16, dayLength: 16 },
        flowering: { dli: 16, dayLength: 12 },
      },
    };

    const plantInfo = plantMapping[plantType];
    if (!plantInfo) return;
    const stageData = plantInfo[growthStage];
    if (!stageData) return;
    const { dli, dayLength } = stageData;

    // Calculate PPFD correctly:
    const suggestedPPFD = ((dli * 1000000) / (dayLength * 3600)).toFixed(0);
    setFormData((prev) => ({ ...prev, target_ppfd: suggestedPPFD }));
  };
  
  const handleFloorSizeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    // Expected value format: "14 x 14" or "12 x 16"
    const [widthStr, lengthStr] = e.target.value.split(" x ");
    setFormData((prev) => ({
      ...prev,
      floor_width: widthStr,
      floor_length: lengthStr,
    }));
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
        setLogMessages((prev) => [...prev, `[INFO] ${pct}% complete`]);
      } else {
        setLogMessages((prev) => [...prev, message]);
      }
    };

    es.onerror = (err) => {
      setLogMessages((prev) => [
        ...prev,
        "[WARN] Connection to server lost. Please try again.",
      ]);
      console.error("EventSource failed:", err);
      es.close();
    };
  };

  return (
    <div style={{ maxWidth: "800px", margin: "0 auto" }}>
      <h1>Lighting Simulation Progress</h1>

      {/* Progress Bar */}
      <div style={{ margin: "20px 0" }}>
        <div
          style={{
            width: "100%",
            background: "#eee",
            borderRadius: "5px",
            overflow: "hidden",
            height: "20px",
          }}
        >
          <div
            style={{
              height: "100%",
              width: `${progress}%`,
              background: "#28a745",
              transition: "width 0.5s ease",
            }}
          />
        </div>
        <p style={{ textAlign: "center", marginTop: "5px", fontWeight: "bold" }}>
          {progress}%
        </p>
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
      {logMessages.map((line, idx) => {
        // Remove any "[DEBUG]" prefix and substitute "param=" with "Layer Intensity:"
        let modifiedLine = line.replace("[DEBUG] ", "").replace("param=", "Layer Intensity:");
        modifiedLine = modifiedLine.replace(
          "[ERROR] SSE connection failed",
          "[WARN] Connection to server lost. Please try again."
        );

        // Determine log line color based on prefix.
        let logColor = "#fff";
        if (modifiedLine.startsWith("[INFO]")) {
          logColor = "#8FBC8F"; // green
        } else if (modifiedLine.startsWith("[WARN]")) {
          logColor = "#FFA500"; // orange
        } else if (modifiedLine.startsWith("[ERROR]")) {
          logColor = "#FF6347"; // red
        }

        // Use regex to extract intensity values from a line formatted like:
        // "Layer Intensity:[ 9892.47 12362.66 6095.29 3426.29 2000. 34628.58 ], mean_ppfd=..."
        const intensityRegex = /Layer Intensity:\[\s*([^\]]+)\]/;
        const intensityMatch = modifiedLine.match(intensityRegex);

        if (intensityMatch) {
          const valuesStr = intensityMatch[1];
          // Split by whitespace and parse to numbers.
          const values = valuesStr.trim().split(/\s+/).map(Number);
          const minVal = Math.min(...values);
          const maxVal = Math.max(...values);

          // Get any text following the intensity array.
          const afterBracket = modifiedLine.indexOf("]") !== -1
            ? modifiedLine.slice(modifiedLine.indexOf("]") + 1)
            : "";

          return (
            <div key={idx} style={{ margin: "2px 0", fontFamily: "monospace", color: logColor }}>
              <span>Layer Intensity: [ </span>
              {values.map((val, i) => (
                <span
                  key={i}
                  style={{
                    backgroundColor: jetColor(val, minVal, maxVal),
                    color: "#fff", // white text for contrast
                    padding: "0 4px",
                    margin: "0 2px",
                    borderRadius: "3px",
                    fontWeight: "bold",
                  }}
                >
                  {Math.round(val)}
                </span>
              ))}
              <span> ]{afterBracket}</span>
            </div>
          );
        } else {
          return (
            <div key={idx} style={{ margin: "2px 0", fontFamily: "monospace", color: logColor }}>
              {modifiedLine}
            </div>
          );
        }
      })}

      </div>

      {/* Input Fields */}
      <div style={{ marginBottom: "20px" }}>
        <label>
          Floor Size:{" "}
          <select
            name="floor_size"
            value={`${formData.floor_width} x ${formData.floor_length}`}
            onChange={handleFloorSizeChange}
            style={{ marginRight: "10px" }}
          >
            <option value="2 x 2">2' x 2'</option>
            <option value="4 x 4">4' x 4'</option>
            <option value="6 x 6">6' x 6'</option>
            <option value="8 x 8">8' x 8'</option>
            <option value="10 x 10">10' x 10'</option>
            <option value="12 x 12">12' x 12'</option>
            <option value="14 x 14">14' x 14'</option>
            <option value="16 x 16">16' x 16'</option>
            <option value="12 x 16">12' x 16'</option>
          </select>
        </label>
        <label>
          Target PPFD (µmol/m²/s):{" "}
          <input
            type="number"
            name="target_ppfd"
            value={formData.target_ppfd}
            onChange={handleChange}
            style={{ marginRight: "10px" }}
          />
        </label>
      </div>


      {/* Plant Selector */}
      <div className="plant-selector" style={{ marginBottom: "20px" }}>
        <label>
          Growth Stage:{" "}
          <select
            value={growthStage}
            onChange={(e) => setGrowthStage(e.target.value)}
          >
            <option value="propagation">Propagation</option>
            <option value="vegetative">Vegetative</option>
            <option value="flowering">Flowering</option>
          </select>
        </label>
        <div className="plant-icons" style={{ marginTop: "10px" }}>
          <span style={{ marginRight: "10px" }}>Select Plant:</span>
          <i
            className="fa fa-leaf plant-icon"
            data-plant="leafy"
            onClick={handlePlantClick}
          ></i>
          <i
            className="fa fa-seedling plant-icon"
            data-plant="tomatoes"
            onClick={handlePlantClick}
          ></i>
          <i
            className="fa fa-cannabis plant-icon"
            data-plant="cannabis"
            onClick={handlePlantClick}
          ></i>
          <i
            className="fa fa-apple-whole plant-icon"
            data-plant="strawberries"
            onClick={handlePlantClick}
          ></i>
        </div>
      </div>

      {/* Start Simulation Button */}
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
            <img
              src={`data:image/png;base64,${simulationResult.surface_graph}`}
              alt="Surface Graph"
              style={{ maxWidth: "100%" }}
            />
            <h2>Heatmap</h2>
            <img
              src={`data:image/png;base64,${simulationResult.heatmap}`}
              alt="Heatmap"
              style={{ maxWidth: "100%" }}
            />
          </div>
        </div>
      )}

    </div>
  );
};

export default SimulationForm;
