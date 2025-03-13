import React, { useState, useRef, useEffect } from "react";
import { MathJax, MathJaxContext } from "better-react-mathjax";
import ModularVisualization from "../components/ModularVisualization";
import { SimulationData } from "../types";
import GridVisualization from "../components/GridVisualization";

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

// Mapping for friendly plant names.
const plantDisplayNames: Record<string, string> = {
  cannabis: "Cannabis",
  tomatoes: "Tomatoes",
  strawberries: "Strawberries",
  leafy: "Leafy Greens",
};

interface FormDataState {
  floor_width: string;
  floor_length: string;
  target_ppfd: string;
  light_height: string; // New field
}

const SimulationForm: React.FC = () => {

  const [formData, setFormData] = useState<FormDataState>({
    floor_width: "12",
    floor_length: "12",
    target_ppfd: "1250",
    light_height: "3", // default light height in feet
  });

  // State for plant growth stage and selected plant
  const [growthStage, setGrowthStage] = useState<string>("propagation");
  const [selectedPlant, setSelectedPlant] = useState<string>("");

  // New state: Enable side-by-side comparison
  const [enableComparison, setEnableComparison] = useState<boolean>(false);

  // SSE and simulation state
  const [progress, setProgress] = useState<number>(0);
  const [logMessages, setLogMessages] = useState<string[]>([]);
  const [simulationResult, setSimulationResult] = useState<SimulationData | null>(null);
  const logOutputRef = useRef<HTMLDivElement>(null);

  // Show Explain Metrics Button

  const [showMetricsModal, setShowMetricsModal] = useState<boolean>(false);
  const [showMethodologyModal, setShowMethodologyModal] = useState<boolean>(false);

  // Hover Effects for Start Simulation and Explain Metrics Buttons
  const [blueHover, setBlueHover] = useState<boolean>(false);
  const [metricsHover, setMetricsHover] = useState<boolean>(false);
  const [methodologyHover, setMethodologyHover] = useState<boolean>(false);

  // Check if Simulation is Complete to Avoid EventSource Error
  const simulationCompleteRef = useRef(false);


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
  
    // Save the selected plant for display.
    setSelectedPlant(plantType);

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

    const suggestedPPFD = ((dli * 1000000) / (dayLength * 3600)).toFixed(0);
    setFormData((prev) => ({ ...prev, target_ppfd: suggestedPPFD }));
  };
  
  const handleFloorSizeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const [widthStr, lengthStr] = e.target.value.split(" x ");
    setFormData((prev) => ({
      ...prev,
      floor_width: widthStr,
      floor_length: lengthStr,
    }));
  };
  
  // Start simulation on button click.
  const startSimulation = async (): Promise<void> => {
    // Reset state.
    setProgress(0);
    setLogMessages([]);
    setSimulationResult(null);
    simulationCompleteRef.current = false;
  
    // Build simulation parameters.
    const params = {
      floor_width: formData.floor_width,
      floor_length: formData.floor_length,
      target_ppfd: formData.target_ppfd,
      floor_height: formData.light_height,
      compare: enableComparison ? "1" : "0",
    };
  
    try {
      // Start the simulation asynchronously; get back a unique job ID.
      const response = await fetch("/api/ml_simulation/start/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
      });
      const data = await response.json();
      const jobId: string = data.job_id;
      setLogMessages((prev) => [...prev, `[INFO] Simulation started. Job ID: ${jobId}`]);
  
      // Poll for progress updates using the job ID.
      const pollProgress = async (): Promise<void> => {
        try {
          const progressRes = await fetch(`/api/ml_simulation/progress/${jobId}/`);
          const progressData: { status: string; progress: string[] } = await progressRes.json();
          
          // Ensure progressData.progress is an array.
          const progressArray: string[] = Array.isArray(progressData.progress)
            ? progressData.progress
            : [];
          setLogMessages(progressArray);
          
          // Filter for progress messages that begin with "PROGRESS:".
          const progressMsgs = progressArray.filter((msg: string) => msg.startsWith("PROGRESS:"));
          if (progressMsgs.length > 0) {
            const lastProgressMsg = progressMsgs[progressMsgs.length - 1];
            const pctStr = lastProgressMsg.replace("PROGRESS:", "").trim();
            const pct = parseFloat(pctStr);
            if (!isNaN(pct)) {
              setProgress(pct);
            }
          }
          
          // Instead of solely relying on progress messages, check the status.
          if (progressData.status === "done") {
            const resultRes = await fetch(`/api/ml_simulation/result/${jobId}/`);
            const resultData: SimulationData = await resultRes.json();
            setSimulationResult(resultData);
            setLogMessages(prev => [...prev, "[INFO] Simulation complete!"]);
            simulationCompleteRef.current = true;
          } else {
            setTimeout(pollProgress, 3000);
          }
        } catch (error: any) {
          setLogMessages(prev => [...prev, "[ERROR] Error polling progress: " + error]);
          setTimeout(pollProgress, 3000);
        }
      };
      
      
  
      pollProgress();
    } catch (error: any) {
      setLogMessages((prev) => [...prev, "[ERROR] Failed to start simulation: " + error]);
    }
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
            const afterBracket =
              modifiedLine.indexOf("]") !== -1
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
          <select value={growthStage} onChange={(e) => setGrowthStage(e.target.value)}>
            <option value="propagation">Propagation</option>
            <option value="vegetative">Vegetative</option>
            <option value="flowering">Flowering</option>
          </select>
          {selectedPlant && (
            <span style={{ marginLeft: "10px", fontWeight: "bold" }}>
              Plant: {plantDisplayNames[selectedPlant] || selectedPlant}
            </span>
          )}
        </label>
        <div className="plant-icons" style={{ marginTop: "10px" }}>
          <span style={{ marginRight: "10px" }}>Select Plant:</span>
          <i className="fa fa-leaf plant-icon" data-plant="leafy" onClick={handlePlantClick} title="Leafy Greens"></i>
          <i className="fa fa-seedling plant-icon" data-plant="tomatoes" onClick={handlePlantClick} title="Tomatoes"></i>
          <i className="fa fa-cannabis plant-icon" data-plant="cannabis" onClick={handlePlantClick} title="Cannabis"></i>
          <i className="fa fa-apple-whole plant-icon" data-plant="strawberries" onClick={handlePlantClick} title="Strawberries"></i>
        </div>
      </div>

      {/* New: Side-by-Side Comparison Checkbox */}
      <div style={{ marginBottom: "20px" }}>
        <label>
          <input
            type="checkbox"
            checked={enableComparison}
            onChange={(e) => setEnableComparison(e.target.checked)}
            style={{ marginRight: "5px" }}
          />
          Enable Side-By-Side Comparison
        </label>
      </div>

        {/* Start Simulation Button */}
        <button
          onClick={startSimulation}
          onMouseEnter={() => setBlueHover(true)}
          onMouseLeave={() => setBlueHover(false)}
          style={{
            display: "inline-block",
            padding: "12px 25px",
            backgroundColor: blueHover ? "#F0F0F0" : "#0078BE",
            color: blueHover ? "#0078BE" : "#F0F0F0",
            textDecoration: "none",
            borderRadius: "5px",
            fontSize: "1.1em",
            transition: "background-color 0.3s ease",
            border: "none",
            cursor: "pointer",
          }}
        >
          Start Simulation
        </button>


        {/* Final Result & Visualization */}
        <div style={{ display: "flex", alignItems: "center" }}>
          <h2 style={{ marginRight: "10px" }}>Simulation Results</h2>
          {simulationResult && (
            <>
            <button
              onClick={() => setShowMetricsModal(true)}
              onMouseEnter={() => setMetricsHover(true)}
              onMouseLeave={() => setMetricsHover(false)}
              style={{
                display: "inline-block",
                padding: "12px 25px",
                backgroundColor: metricsHover ? "#F0F0F0" : "red",
                color: metricsHover ? "red" : "#F0F0F0",
                textDecoration: "none",
                borderRadius: "5px",
                fontSize: "1.1em",
                transition: "background-color 0.3s ease",
                border: "none",
                cursor: "pointer",
                marginRight: "10px",
              }}
            >
              Explain the Metrics
            </button>
            <button
              onClick={() => setShowMethodologyModal(true)}
              onMouseEnter={() => setMethodologyHover(true)}
              onMouseLeave={() => setMethodologyHover(false)}
              style={{
                display: "inline-block",
                padding: "12px 25px",
                backgroundColor: methodologyHover ? "#F0F0F0" : "red",
                color: methodologyHover ? "red" : "#F0F0F0",
                textDecoration: "none",
                borderRadius: "5px",
                fontSize: "1.1em",
                transition: "background-color 0.3s ease",
                border: "none",
                cursor: "pointer",
              }}
            >
              Explain the Methodology
            </button>

            </>
          )}
        </div>

        {/* Metrics Modal */}
        {showMetricsModal && (
          <div
            className="modal-overlay"
            onClick={() => setShowMetricsModal(false)}
            style={{
              position: "fixed",
              top: 0,
              left: 0,
              width: "100%",
              height: "100%",
              backgroundColor: "rgba(0,0,0,0.5)",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              zIndex: 1000,
              transition: "opacity 0.3s ease",
            }}
          >
            <div
              className="modal-content"
              onClick={(e) => e.stopPropagation()}
              style={{
                background: "#fff",
                padding: "20px",
                borderRadius: "5px",
                maxWidth: "600px",
                maxHeight: "80%",
                overflowY: "auto",
                boxShadow: "0 2px 10px rgba(0,0,0,0.3)",
              }}
            >
              <MathJaxContext>
                <div className="metrics-explanation" style={{ marginTop: "20px" }}>
                  <ol>
                    <li style={{ marginBottom: "1em" }}>
                      <strong>
                        Average PPFD (<MathJax inline>{"\\(\\text{PPFD}_{\\text{avg}}\\)"}</MathJax>):
                      </strong>{" "}
                      The average PPFD is simply the arithmetic mean of all PPFD measurements:
                      <MathJax>{"\\(\\text{PPFD}_{\\text{avg}} = \\frac{1}{n} \\sum_{i=1}^{n} P_i\\)"}</MathJax>
                    </li>
                    <li style={{ marginBottom: "1em" }}>
                      <strong>Root Mean Squared Error (RMSE):</strong>{" "}
                      RMSE quantifies the difference between the measured PPFD values and the average PPFD:
                      <MathJax>{"\\(\\text{RMSE} = \\sqrt{\\frac{\\sum_{i=1}^{n} (P_i - \\text{PPFD}_{\\text{avg}})^2}{n}}\\)"}</MathJax>
                    </li>
                    <li style={{ marginBottom: "1em" }}>
                      <strong>Degree of Uniformity (DOU):</strong>{" "}
                      DOU is calculated based on the RMSE and the average PPFD:
                      <MathJax>{"\\(\\text{DOU} = 100 \\times \\left(1 - \\frac{\\text{RMSE}}{\\text{PPFD}_{\\text{avg}}}\\right)\\)"}</MathJax>
                    </li>
                    <li style={{ marginBottom: "1em" }}>
                      <strong>Mean Absolute Deviation (MAD):</strong>{" "}
                      MAD measures the average absolute difference between each PPFD value and the average PPFD:
                      <MathJax>{"\\(\\text{MAD} = \\frac{1}{n} \\sum_{i=1}^{n} |P_i - \\text{PPFD}_{\\text{avg}}|\\)"}</MathJax>
                    </li>
                    <li style={{ marginBottom: "1em" }}>
                      <strong>Coefficient of Variation (CV):</strong>{" "}
                      CV is the ratio of the standard deviation (
                      <MathJax inline>{"\\(\\sigma\\)"}</MathJax>
                      ) to the average PPFD, expressed as a percentage:
                      <MathJax>{"\\(\\text{CV} = 100 \\times \\frac{\\sigma}{\\text{PPFD}_{\\text{avg}}}\\)"}</MathJax>
                      where the standard deviation, <MathJax inline>{"\\(\\sigma\\)"}</MathJax>, is calculated as:
                      <MathJax>{"\\(\\sigma = \\sqrt{\\frac{\\sum_{i=1}^{n} (P_i - \\text{PPFD}_{\\text{avg}})^2}{n}}\\)"}</MathJax>
                      <em>Note:</em> In this specific case, since we are calculating sample statistics and using all data points, the sample and population standard deviations are equivalent.
                    </li>
                  </ol>
                </div>
              </MathJaxContext>
            </div>
          </div>
        )}

        {/* Methodology Modal */}
        {showMethodologyModal && (
          <div
            className="modal-overlay"
            onClick={() => setShowMethodologyModal(false)}
            style={{
              position: "fixed",
              top: 0,
              left: 0,
              width: "100%",
              height: "100%",
              backgroundColor: "rgba(0,0,0,0.5)",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              zIndex: 1000,
              transition: "opacity 0.3s ease",
            }}
          >
            <div
              className="modal-content"
              onClick={(e) => e.stopPropagation()}
              style={{
                background: "#fff",
                padding: "20px",
                borderRadius: "5px",
                maxWidth: "600px",
                maxHeight: "80%",
                overflowY: "auto",
                boxShadow: "0 2px 10px rgba(0,0,0,0.3)",
              }}
            >
              <div style={{ marginTop: "20px" }}>
                <p>
                  The <b>Chip-On-Board</b> (COB) LEDs are arranged in a centered square pattern sequence, which enables the separation of the COBs into concentric square layers for the application of layer-wise intensity assignments.
                </p>
                <p>
                  Our <b>differential evolution-based global optimization algorithm</b> finds the optimal intensities to apply to each layer of COBs for the purpose of maximizing the <b>Degree of Uniformity</b> (DOU) of <b>Photosynthetic Photon Flux Density</b> (PPFD) while meeting the target PPFD.
                </p>
                <p>
                  The reason the algorithm converges on a solution where there’s a large concentration of luminous flux assigned to outer layers of COBs with inner layers receiving lower luminous flux assignments is quite simple:
                  The algorithm converges on a solution with higher luminous flux assigned to the outer COB layers because of the combined effects of the <b>inverse square law</b> and the overlapping emission patterns of the COBs. The inverse square law dictates that the light intensity from each individual COB decreases with the square of the distance. Because the inner COBs have more neighboring COBs, their overlapping light contributions, each diminishing according to the inverse square law, create a higher overall PPFD in the center of the illuminated plane. Conversely, the outer COBs have fewer neighbors, and some of their light is emitted outside the target area, leading to a lower PPFD at the edges.
                </p>
                <p>
                  Although COBs have a circular <b>Light Emitting Surface</b> (LES), they typically do not emit light omnidirectionally. Many COBs have an emission pattern that approximates a <b>Lambertian distribution</b>, where the intensity is highest perpendicular to the surface and decreases with the cosine of the viewing angle. This, along with the physical arrangement of the COBs, leads to overlapping light patterns, with light traveling from the inner layers toward the outer layers, and vice versa, due to overlapping emission cones.
                </p>
                <p>
                  Therefore, we can take advantage of these natural phenomena by assigning a concentration of luminous flux to the outer perimeter layers of the lighting array, which fills in the deficit of photons along the perimeter, and flattens out the photon distribution across the entire illuminated plane.
                </p>
                <p>
                  <b>In summation</b>:
                  By separating the COBs into concentric square layers via the centered square pattern sequence, and assigning a concentration of luminous flux to outer perimeter layers of COBs, light fills in the naturally occurring deficit of photons at the perimeter layers of the illuminated plane, with the remainder of this intensity concentration traveling inward to flatten out the PPFD across the entire illuminated plane.
                </p>
              </div>
            </div>
          </div>
        )}

        {simulationResult && (
          <div style={{ marginTop: "30px" }}>
            <h2>Simulation Results</h2>
            <table>
              <tbody>
                <tr>
                  <td><strong>Minimized MAD:</strong></td>
                  <td>{simulationResult.mad.toFixed(2)}</td>
                </tr>
                <tr>
                  <td><strong>Average PPFD:</strong></td>
                  <td>{simulationResult.optimized_ppfd.toFixed(2)} µmol/m²/s</td>
                </tr>
                <tr>
                  <td><strong>Minimized RMSE:</strong></td>
                  <td>{simulationResult.rmse.toFixed(2)}</td>
                </tr>
                <tr>
                  <td><strong>Maximized DOU:</strong></td>
                  <td>{simulationResult.dou.toFixed(2)}%</td>
                </tr>
                <tr>
                  <td><strong>Minimized CV:</strong></td>
                  <td>{simulationResult.cv.toFixed(2)}%</td>
                </tr>
              </tbody>
            </table>

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

            <div style={{ marginTop: "30px" }}>
              <h2>Surface Graph &amp; Heatmap</h2>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "20px" }}>
                <div style={{ flex: "1 1 300px" }}>
                  <h3>Optimized (Staggered) Simulation</h3>
                  <img
                    src={`data:image/png;base64,${simulationResult.surface_graph}`}
                    alt="Surface Graph"
                    style={{ maxWidth: "100%" }}
                  />
                  <img
                    src={`data:image/png;base64,${simulationResult.heatmap}`}
                    alt="Heatmap"
                    style={{ maxWidth: "100%", marginTop: "10px" }}
                  />
                </div>
                {simulationResult.grid_surface_graph && simulationResult.grid_heatmap && (
                  <div style={{ flex: "1 1 300px" }}>
                    <h3>Uniform Grid Simulation</h3>
                    <img
                      src={`data:image/png;base64,${simulationResult.grid_surface_graph}`}
                      alt="Grid Surface Graph"
                      style={{ maxWidth: "100%" }}
                    />
                    <img
                      src={`data:image/png;base64,${simulationResult.grid_heatmap}`}
                      alt="Grid Heatmap"
                      style={{ maxWidth: "100%", marginTop: "10px" }}
                    />
                  </div>
                )}
              </div>
            </div>

            {simulationResult.grid_cob_arrangement &&
              simulationResult.grid_uniform_flux !== undefined &&
              simulationResult.grid_rmse !== undefined &&
              simulationResult.grid_dou !== undefined &&
              simulationResult.grid_cv !== undefined &&
              simulationResult.grid_mad !== undefined &&
              simulationResult.grid_ppfd !== undefined && (
                <div style={{ marginTop: "30px" }}>
                  <h3>Uniform Grid Simulation Results</h3>
                  <table>
                    <tbody>
                      <tr>
                        <td><strong>Lumens Per COB:</strong></td>
                        <td>{simulationResult.grid_uniform_flux.toFixed(2)} lumens</td>
                      </tr>
                      <tr>
                        <td><strong>Average PPFD:</strong></td>
                        <td>{simulationResult.grid_ppfd.toFixed(2)} µmol/m²/s</td>
                      </tr>
                      <tr>
                        <td><strong>MAD:</strong></td>
                        <td>{simulationResult.grid_mad.toFixed(2)}</td>
                      </tr>
                      <tr>
                        <td><strong>RMSE:</strong></td>
                        <td>{simulationResult.grid_rmse.toFixed(2)}</td>
                      </tr>
                      <tr>
                        <td><strong>DOU:</strong></td>
                        <td>{simulationResult.grid_dou.toFixed(2)}%</td>
                      </tr>
                      <tr>
                        <td><strong>CV:</strong></td>
                        <td>{simulationResult.grid_cv.toFixed(2)}%</td>
                      </tr>
                    </tbody>
                  </table>
                  <div style={{ marginTop: "30px" }}>
                    <GridVisualization
                      floorWidth={simulationResult.floor_width}
                      floorLength={simulationResult.floor_length}
                      floorHeight={simulationResult.floor_height}
                      uniformFlux={simulationResult.grid_uniform_flux}
                      gridArrangement={simulationResult.grid_cob_arrangement}
                      gridPPFD={simulationResult.grid_ppfd}
                    />
                  </div>
                </div>
              )}
          </div>
        )}


    </div>
  );
};

export default SimulationForm;
