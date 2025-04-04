import React, { useState, useRef, useEffect } from "react";
import { MathJax, MathJaxContext } from "better-react-mathjax";
import ModularVisualization from "../components/ModularVisualization";
import { SimulationData } from "../types"; // Assuming this might need extension or ppfd_grid_data is already part of it
import GridVisualization from "../components/GridVisualization";

// --- Helper function to send messages to the parent window ---
const notifyParentBuilder = (messagePayload: object, error?: string) => {
  // --- IMPORTANT: Specify the Builder App's origin for security ---
  // Replace with the actual origin where your builder app runs in production
  // Using window.location.origin is generally okay if both apps are served from the same domain.
  // If they are on different domains, hardcode the builder's origin.
  const builderOrigin = window.location.origin; // Or 'http://localhost:8000', 'https://yourdomain.com'

  const message = {
    type: 'simulationResult', // Consistent type identifier used by the builder listener
    payload: error ? { error: error } : messagePayload
  };

  try {
    // Check if running inside an iframe to avoid errors when run standalone
    if (window.parent !== window) {
      console.log("SIMULATION FORM: Sending message to parent:", message, "Target Origin:", builderOrigin);
      window.parent.postMessage(message, builderOrigin);
      console.log("SIMULATION FORM: Message sent.");
    } else {
      console.log("SIMULATION FORM: Not running in iframe, skipping postMessage.");
    }
  } catch (e) {
    console.error("SIMULATION FORM: Error sending postMessage:", e);
    // This might happen due to cross-origin restrictions if origins don't match,
    // or other browser security features.
  }
};


// Immaculate jetColor function – produces a smooth jet colormap.
function jetColor(value: number, min: number, max: number): string {
  let t = (value - min) / (max - min);
  t = Math.max(0, Math.min(1, t)); // Clamp t to [0, 1]
  // These formulas approximate the jet colormap segments
  const r = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 3)));
  const g = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 2)));
  const b = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 1)));
  const R = Math.round(r * 255);
  const G = Math.round(g * 255);
  const B = Math.round(b * 255);
  return `rgb(${R}, ${G}, ${B})`;
}

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
  light_height: string; // Note: Builder calls this 'floor_height' in params, ensure consistency
}

// Extend SimulationData type if needed, or ensure SimulationData from "../types"
// includes ppfd_grid_data if your backend already provides it.
// If not, define it here.
interface ExtendedSimulationData extends SimulationData {
    floor_width: number; // Assuming backend returns this in feet
    floor_length: number; // Assuming backend returns this in feet
    floor_height: number; // Added for consistency if backend returns it
    // This structure MUST match what your backend sends for the builder heatmap
    ppfd_grid_data?: {
        values: number[][];      // 2D array of PPFD values
        resolution_x: number;   // Number of grid points along width
        resolution_y: number;   // Number of grid points along length
        width_m: number;        // Actual width dimension in meters for the grid
        length_m: number;       // Actual length dimension in meters for the grid
        height_m: number;       // Height above floor (in meters) where grid is calculated
    };
    // Add error field if backend can return specific errors in the result object
    error?: string;
    // Ensure all fields from the original SimulationData are included implicitly or explicitly
    // mad: number; optimized_ppfd: number; rmse: number; dou: number; cv: number;
    // optimized_lumens_by_layer: number[]; surface_graph: string; heatmap: string;
    // grid_surface_graph?: string; grid_heatmap?: string; grid_cob_arrangement?: any;
    // grid_uniform_flux?: number; grid_rmse?: number; grid_dou?: number;
    // grid_cv?: number; grid_mad?: number; grid_ppfd?: number;
}


const SimulationForm: React.FC = () => {
  // --- State Variables ---
  const [formData, setFormData] = useState<FormDataState>({
    floor_width: "12",
    floor_length: "12",
    target_ppfd: "1250",
    light_height: "3", // Corresponds to floor_height param in API call
  });

  const [growthStage, setGrowthStage] = useState<string>("propagation");
  const [selectedPlant, setSelectedPlant] = useState<string>("");
  const [enableComparison, setEnableComparison] = useState<boolean>(false);

  const [progress, setProgress] = useState<number>(0);
  const [targetProgress, setTargetProgress] = useState<number>(0);
  const [isSimulating, setIsSimulating] = useState<boolean>(false);
  const [logMessages, setLogMessages] = useState<string[]>([]);
  const [simulationResult, setSimulationResult] = useState<ExtendedSimulationData | null>(null); // Use extended type

  // Refs
  const logOutputRef = useRef<HTMLDivElement>(null);
  const simulationCompleteRef = useRef(false); // Tracks if polling should stop

  // Modal Visibility States
  const [showMetricsModal, setShowMetricsModal] = useState<boolean>(false);
  const [showMethodologyModal, setShowMethodologyModal] = useState<boolean>(false);

  // Hover States for Buttons
  const [blueHover, setBlueHover] = useState<boolean>(false);
  const [metricsHover, setMetricsHover] = useState<boolean>(false);
  const [methodologyHover, setMethodologyHover] = useState<boolean>(false);

  // --- Effects ---

  // Auto-scroll log output.
  useEffect(() => {
    if (logOutputRef.current) {
      logOutputRef.current.scrollTop = logOutputRef.current.scrollHeight;
    }
  }, [logMessages]);

  // Animate progress smoothly.
  useEffect(() => {
    let timer: NodeJS.Timeout | null = null;
    if (progress < targetProgress) {
      const diff = targetProgress - progress;
      // Adjust step for smoother/faster animation if needed
      const step = Math.max(0.1, diff / 50); // Ensure minimum step, avoid too slow near end
      timer = setTimeout(() => {
        const newProgress = Math.min(progress + step, targetProgress);
        setProgress(parseFloat(newProgress.toFixed(2))); // Keep 2 decimal places
      }, 30); // Update slightly faster (e.g., 30ms)
    }
    // Cleanup function to clear timeout if component unmounts or progress changes
    return () => {
        if (timer) clearTimeout(timer);
    };
  }, [progress, targetProgress]);

  // Handle initial parameters passed from parent iframe src
  useEffect(() => {
    const queryParams = new URLSearchParams(window.location.search);
    const initialW = queryParams.get('initialW');
    const initialL = queryParams.get('initialL');
    // Could also get initial H, light_h etc. if builder sends them

    if (initialW && initialL) {
        console.log("SIMULATION FORM: Received initial dimensions from parent:", { initialW, initialL });
        setFormData(prev => ({
            ...prev,
            floor_width: initialW,
            floor_length: initialL
            // Potentially update light_height/target_ppfd too if desired
        }));
    }
  }, []); // Run only once on mount


  // --- Event Handlers ---

  // Handle generic input changes.
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  // Handle plant icon click to set plant type and suggest PPFD.
  const handlePlantClick = (e: React.MouseEvent<HTMLElement>) => {
    const plantType = e.currentTarget.getAttribute("data-plant");
    if (!plantType) return;
    setSelectedPlant(plantType);
    // DLI and PPFD calculation logic (remains the same)
    const plantMapping: Record<string, Record<string, { dli: number; dayLength: number }>> = {
        cannabis: { propagation: { dli: 45, dayLength: 18 }, vegetative: { dli: 50, dayLength: 18 }, flowering: { dli: 50, dayLength: 12 } },
        tomatoes: { propagation: { dli: 25, dayLength: 16 }, vegetative: { dli: 27.5, dayLength: 16 }, flowering: { dli: 27.5, dayLength: 16 } },
        strawberries: { propagation: { dli: 20, dayLength: 8 }, vegetative: { dli: 22.5, dayLength: 8 }, flowering: { dli: 22.5, dayLength: 8 } },
        leafy: { propagation: { dli: 12, dayLength: 16 }, vegetative: { dli: 16, dayLength: 16 }, flowering: { dli: 16, dayLength: 12 } },
    };
    const plantInfo = plantMapping[plantType];
    if (!plantInfo) return;
    const stageData = plantInfo[growthStage];
    if (!stageData) return;
    const { dli, dayLength } = stageData;
    const suggestedPPFD = ((dli * 1000000) / (dayLength * 3600)).toFixed(0);
    setFormData((prev) => ({ ...prev, target_ppfd: suggestedPPFD }));
  };

  // Handle floor size dropdown change.
  const handleFloorSizeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const [widthStr, lengthStr] = e.target.value.split(" x ");
    setFormData((prev) => ({ ...prev, floor_width: widthStr, floor_length: lengthStr }));
  };

  // --- Simulation Logic ---

  // Start simulation on button click.
  const startSimulation = async (): Promise<void> => {
    // Reset state for a new simulation run
    setProgress(0);
    setTargetProgress(0);
    setLogMessages([]);
    setSimulationResult(null);
    simulationCompleteRef.current = false; // Allow polling again
    setIsSimulating(true); // Disable button, show progress

    // Build simulation parameters for the backend API call
    const params = {
      floor_width: formData.floor_width,
      floor_length: formData.floor_length,
      target_ppfd: formData.target_ppfd,
      floor_height: formData.light_height, // Sending light_height as floor_height param
      compare: enableComparison ? "1" : "0", // Include comparison flag
    };

    try {
      // Call the backend endpoint to start the job
      const response = await fetch("/api/ml_simulation/start/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
      });

      if (!response.ok) {
          // Handle non-2xx responses from the start endpoint
          const errorData = await response.json().catch(() => ({ error: 'Failed to parse error response' }));
          throw new Error(errorData.error || `Failed to start simulation: ${response.statusText}`);
      }

      const data = await response.json();
      const jobId: string = data.job_id;
      if (!jobId) {
          throw new Error("Backend did not return a valid job ID.");
      }

      setLogMessages((prev) => [...prev, `[INFO] Simulation started. Job ID: ${jobId}`]);

      // --- Polling function ---
      const pollProgress = async (): Promise<void> => {
        // Stop polling if simulation is marked complete by fetching result
        if (simulationCompleteRef.current) return;

        try {
          // Fetch progress and status
          const progressRes = await fetch(`/api/ml_simulation/progress/${jobId}/`);
          if (!progressRes.ok) {
              // Handle errors fetching progress (e.g., job not found after a delay?)
              console.warn(`Polling warning: Failed to fetch progress (${progressRes.status})`);
              // Decide whether to retry or stop based on status code
              if (progressRes.status === 404 && logMessages.length > 1) { // Check if job likely existed before
                   throw new Error("Job progress endpoint returned 404. Job might have expired or been deleted.");
              }
               // Continue polling for temporary errors
               setTimeout(pollProgress, 5000); // Retry after longer delay
               return;
          }

          const progressData: { status: string; progress: string[] } = await progressRes.json();
          const progressArray: string[] = Array.isArray(progressData.progress) ? progressData.progress : [];

          // Update logs displayed to the user
          setLogMessages(progressArray);

          // Extract the latest progress percentage from log messages
          const progressMsgs = progressArray.filter((msg: string) => msg.startsWith("PROGRESS:"));
          if (progressMsgs.length > 0) {
            const lastProgressMsg = progressMsgs[progressMsgs.length - 1];
            const pctStr = lastProgressMsg.split(":")[1]?.trim(); // Safer parsing
            if (pctStr) {
                const pct = parseFloat(pctStr);
                if (!isNaN(pct)) {
                    // Update target progress smoothly
                    setTargetProgress(prev => Math.max(prev, Math.min(pct, 100))); // Ensure target doesn't exceed 100
                }
            }
          }

          // Check if the job status is "done"
          if (progressData.status === "done") {
            simulationCompleteRef.current = true; // Mark as complete to stop polling loop
            setTargetProgress(100); // Ensure visual progress hits 100%
            setProgress(100);       // Force immediate visual update to 100%
            setIsSimulating(false); // Re-enable the start button

            setLogMessages((prev) => [...prev, "[INFO] Simulation complete! Fetching final result..."]);

            // Fetch the final result data
            try {
                const resultRes = await fetch(`/api/ml_simulation/result/${jobId}/`);
                if (!resultRes.ok) {
                    const errorData = await resultRes.json().catch(() => ({error: `Failed to fetch result: ${resultRes.statusText}`}));
                    throw new Error(errorData.error || `Failed to fetch result: ${resultRes.statusText}`);
                }
                const resultData: ExtendedSimulationData = await resultRes.json(); // <-- Final result

                // Update local state to display results in this simulation form
                setSimulationResult(resultData);
                setLogMessages((prev) => [...prev, "[INFO] Final results received."]);

                // ***** SEND RESULT (OR ERROR) TO PARENT BUILDER *****
                if (resultData.error) {
                    // Handle errors reported by the simulation itself within the result data
                    console.error("SIMULATION FORM: Simulation backend reported an error:", resultData.error);
                    notifyParentBuilder({}, `Simulation Error: ${resultData.error}`); // Send error payload
                } else if (resultData.ppfd_grid_data && resultData.floor_width && resultData.floor_length) {
                    // Prepare successful payload if ppfd_grid_data is present
                    const messagePayload = {
                        ppfd_grid: resultData.ppfd_grid_data, // Send the crucial grid data
                        dimensions: { // Send dimensions used for *this* specific simulation run
                            W_ft: resultData.floor_width.toFixed(1),
                            L_ft: resultData.floor_length.toFixed(1),
                            // Include meters if the backend provides them within ppfd_grid_data
                            W_m: resultData.ppfd_grid_data.width_m,
                            L_m: resultData.ppfd_grid_data.length_m,
                        }
                        // Optionally include other results if the builder needs them:
                        // average_ppfd: resultData.optimized_ppfd,
                        // dou: resultData.dou,
                    };
                    notifyParentBuilder(messagePayload); // Send success payload
                } else {
                    // Handle case where backend result is missing necessary data for the builder
                    console.error("SIMULATION FORM: Result data from backend is missing ppfd_grid_data or dimensions needed for builder.");
                    // Still display results locally, but notify builder of the issue
                    notifyParentBuilder({}, "Error: Simulation result missing data needed for builder heatmap.");
                }
                 // ***** END OF POSTMESSAGE CALL *****

            } catch (resultError: any) {
                 // Handle errors during the *final result fetch* itself
                 console.error("SIMULATION FORM: Error fetching or processing simulation result:", resultError);
                 setLogMessages((prev) => [...prev, "[ERROR] Failed to get final result: " + resultError.message]);
                 setIsSimulating(false); // Ensure button is enabled on error
                 // Notify parent about the fetch error
                 notifyParentBuilder({}, `Error fetching simulation result: ${resultError.message}`);
            }

          } else {
            // Job is not "done" yet, schedule the next poll check
            setTimeout(pollProgress, 2000); // Poll every 2 seconds (adjust interval as needed)
          }
        } catch (error: any) {
          // Handle errors during the polling process (e.g., network issues)
          setLogMessages((prev) => [...prev, "[ERROR] Error polling progress: " + error.message]);
          // Don't necessarily stop simulation on polling error, could be temporary
          if (!simulationCompleteRef.current) {
               // Retry polling after a longer delay if not already marked complete
               setTimeout(pollProgress, 5000);
          } else {
               // If it was already complete but polling failed, stop trying
               setIsSimulating(false);
               notifyParentBuilder({}, `Error polling for progress: ${error.message}`);
          }
        }
      }; // End of pollProgress function

      pollProgress(); // Start the polling loop

    } catch (error: any) {
      // Handle errors during the *initial start* API call
      console.error("SIMULATION FORM: Error starting simulation:", error);
      setLogMessages((prev) => [...prev, "[ERROR] Failed to start simulation: " + error.message]);
      setIsSimulating(false); // Ensure button is enabled
      // Notify parent if simulation couldn't even start
      notifyParentBuilder({}, `Error starting simulation: ${error.message}`);
    }
  }; // End of startSimulation function

  // --- Render Logic ---
  return (
    // Use MathJaxContext for rendering math formulas in modals
    <MathJaxContext>
      {/* Adjust container style for better iframe viewing */}
      <div style={{ maxWidth: "95%", margin: "10px auto", padding: "15px" }}>
        <h1>Lighting Simulation</h1> {/* Changed title slightly */}

        {/* Progress Bar */}
        <div style={{ margin: "20px 0" }}>
          <div style={{ width: "100%", background: "#eee", borderRadius: "5px", overflow: "hidden", height: "20px" }}>
            <div style={{ height: "100%", width: `${progress}%`, background: "#28a745", transition: "width 0.3s ease-out" }} />
          </div>
          <p style={{ textAlign: "center", marginTop: "5px", fontWeight: "bold" }}>{progress}% Complete</p>
        </div>

        {/* Log Output Section */}
        <div
          ref={logOutputRef}
          style={{
            background: "#222", // Slightly darker background
            color: "#e0e0e0", // Lighter text
            padding: "15px",
            borderRadius: "5px",
            height: "250px", // Increased height
            overflowY: "scroll",
            fontFamily: "Consolas, 'Courier New', monospace", // Monospace font
            fontSize: "0.85em", // Slightly smaller font
            marginBottom: "25px",
            border: "1px solid #444",
            whiteSpace: "pre-wrap", // Preserve whitespace and wrap lines
          }}
        >
          {logMessages.map((line, idx) => {
            // Log line processing and coloring (remains the same)
            let modifiedLine = line.replace("[DEBUG] ", "").replace("param=", "Layer Intensity:");
            modifiedLine = modifiedLine.replace(
              "[ERROR] SSE connection failed",
              "[WARN] Connection to server lost. Please try again."
            );
            let logColor = "#e0e0e0"; // Default light text
            if (modifiedLine.startsWith("[INFO]")) {
                logColor = "#8FBC8F"; // Soft green
            } else if (modifiedLine.startsWith("[WARN]")) {
                logColor = "#FFA500"; // Orange
            } else if (modifiedLine.startsWith("[ERROR]")) {
                logColor = "#FF6347"; // Tomato red
            }

            const intensityRegex = /Layer Intensity:\[\s*([^\]]+)\]/;
            const intensityMatch = modifiedLine.match(intensityRegex);

            if (intensityMatch) {
                // Intensity value coloring logic (remains the same)
                const valuesStr = intensityMatch[1];
                const values = valuesStr.trim().split(/\s+/).map(Number).filter(v => !isNaN(v)); // Filter out NaN
                if (values.length === 0) return <div key={idx} style={{ margin: "2px 0", color: logColor }}>{modifiedLine}</div>; // Skip if no valid numbers
                const minVal = Math.min(...values);
                const maxVal = Math.max(...values);
                const afterBracket =
                modifiedLine.indexOf("]") !== -1
                    ? modifiedLine.slice(modifiedLine.indexOf("]") + 1)
                    : "";

                return (
                <div key={idx} style={{ margin: "3px 0", color: logColor }}> {/* Added line height */}
                    <span>Layer Intensity: [ </span>
                    {values.map((val, i) => (
                    <span
                        key={i}
                        style={{
                        backgroundColor: jetColor(val, minVal, maxVal),
                        color: (val > (minVal + maxVal) / 1.5) ? "#000" : "#fff", // Dynamic text color based on bg
                        padding: "1px 5px", // Adjusted padding
                        margin: "0 2px",
                        borderRadius: "3px",
                        fontWeight: "bold",
                        display: "inline-block", // Ensure bg color covers number
                        minWidth: "2em", // Ensure minimum width for small numbers
                        textAlign: "center" // Center text in colored span
                        }}
                    >
                        {Math.round(val)}
                    </span>
                    ))}
                    <span> ]{afterBracket}</span>
                </div>
                );
            } else {
                // Render normal log lines
                return (
                <div key={idx} style={{ margin: "3px 0", color: logColor }}>
                    {modifiedLine}
                </div>
                );
            }
          })}
        </div>

        {/* --- Input Form Sections --- */}
        <div style={{ display: "flex", flexWrap: "wrap", gap: "20px", marginBottom: "20px" }}>

            {/* Floor Size & PPFD */}
            <div style={{ flex: "1 1 300px", minWidth: "280px" }}>
                 <h4>Room Setup</h4>
                 <label style={{ display: 'block', marginBottom: '10px' }}>
                     Floor Size (ft):{" "}
                     <select
                         name="floor_size"
                         value={`${formData.floor_width} x ${formData.floor_length}`}
                         onChange={handleFloorSizeChange}
                         style={{ marginLeft: '5px', padding: '5px' }}
                         disabled={isSimulating}
                     >
                         {/* Add more options if needed */}
                         <option value="2 x 2">2' x 2'</option>
                         <option value="4 x 4">4' x 4'</option>
                         <option value="6 x 6">6' x 6'</option>
                         <option value="8 x 8">8' x 8'</option>
                         <option value="10 x 10">10' x 10'</option>
                         <option value="12 x 12">12' x 12'</option>
                         <option value="14 x 14">14' x 14'</option>
                         <option value="16 x 16">16' x 16'</option>
                         <option value="12 x 16">12' x 16'</option>
                         <option value="20 x 20">20' x 20'</option>
                     </select>
                 </label>
                 <label style={{ display: 'block', marginBottom: '10px' }}>
                     Target PPFD (µmol/m²/s):{" "}
                     <input
                         type="number"
                         name="target_ppfd"
                         value={formData.target_ppfd}
                         onChange={handleChange}
                         style={{ marginLeft: '5px', padding: '5px', width: '80px' }}
                         disabled={isSimulating}
                         min="100" // Example min/max
                         max="2500"
                     />
                 </label>
                  <label style={{ display: 'block', marginBottom: '10px' }}>
                     Light Mounting Height (ft):{" "}
                     <input
                         type="number"
                         name="light_height" // Matches state key
                         value={formData.light_height}
                         onChange={handleChange}
                         style={{ marginLeft: '5px', padding: '5px', width: '80px' }}
                         disabled={isSimulating}
                         min="1" // Example min/max
                         max="20"
                         step="0.5"
                     />
                 </label>
            </div>

             {/* Plant Selector */}
            <div style={{ flex: "1 1 300px", minWidth: "280px" }}>
                <h4>Plant & Stage (Optional)</h4>
                 <label style={{ display: 'block', marginBottom: '10px' }}>
                 Growth Stage:{" "}
                 <select
                    value={growthStage}
                    onChange={(e) => setGrowthStage(e.target.value)}
                    style={{ marginLeft: '5px', padding: '5px' }}
                    disabled={isSimulating}
                 >
                     <option value="propagation">Propagation</option>
                     <option value="vegetative">Vegetative</option>
                     <option value="flowering">Flowering</option>
                 </select>
                 {selectedPlant && (
                     <span style={{ marginLeft: "15px", fontWeight: "bold", fontStyle: "italic" }}>
                     {plantDisplayNames[selectedPlant] || selectedPlant} Selected
                     </span>
                 )}
                 </label>
                 <div style={{ marginTop: "15px" }}>
                 <span style={{ marginRight: "10px", fontWeight: "500" }}>Select Plant Type (Suggests PPFD):</span>
                 <div className="plant-icons" style={{ marginTop: "8px", display: 'flex', gap: '15px', fontSize: '1.5em' }}>
                     {/* Using font-awesome icons requires font-awesome setup */}
                     {/* Replace with actual icons or images if needed */}
                     <i className="fa fa-leaf plant-icon" data-plant="leafy" onClick={handlePlantClick} title="Leafy Greens" style={{ cursor: 'pointer', color: selectedPlant === 'leafy' ? '#28a745' : '#555' }}></i>
                     <i className="fa fa-seedling plant-icon" data-plant="tomatoes" onClick={handlePlantClick} title="Tomatoes" style={{ cursor: 'pointer', color: selectedPlant === 'tomatoes' ? '#dc3545' : '#555' }}></i>
                     <i className="fa fa-cannabis plant-icon" data-plant="cannabis" onClick={handlePlantClick} title="Cannabis" style={{ cursor: 'pointer', color: selectedPlant === 'cannabis' ? '#17a2b8' : '#555' }}></i>
                     <i className="fa fa-apple-whole plant-icon" data-plant="strawberries" onClick={handlePlantClick} title="Strawberries" style={{ cursor: 'pointer', color: selectedPlant === 'strawberries' ? '#ffc107' : '#555' }}></i>
                 </div>
                 </div>
            </div>
        </div>


        {/* Side-by-Side Comparison Checkbox */}
        <div style={{ marginBottom: "25px", padding: "10px", background: "#f8f9fa", borderRadius: "4px", border: "1px solid #eee" }}>
            <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                <input
                    type="checkbox"
                    checked={enableComparison}
                    onChange={(e) => setEnableComparison(e.target.checked)}
                    style={{ marginRight: "10px", transform: 'scale(1.2)' }}
                    disabled={isSimulating}
                />
                Enable Side-By-Side Comparison (Uniform Grid vs Optimized)
            </label>
        </div>

        {/* Start Simulation Button */}
        <div style={{ textAlign: "center", marginBottom: "30px" }}>
             <button
                 onClick={startSimulation}
                 disabled={isSimulating}
                 onMouseEnter={() => { if (!isSimulating) setBlueHover(true); }}
                 onMouseLeave={() => { if (!isSimulating) setBlueHover(false); }}
                 style={{
                 position: "relative", // Needed for progress overlay
                 display: "inline-block",
                 padding: "12px 30px", // Slightly more padding
                 backgroundColor: isSimulating ? "#adb5bd" : (blueHover ? "#0056b3" : "#007bff"), // Standard bootstrap colors
                 color: "#fff",
                 textDecoration: "none",
                 borderRadius: "5px",
                 fontSize: "1.2em", // Larger font
                 fontWeight: "bold",
                 transition: "background-color 0.2s ease, transform 0.1s ease",
                 border: "none",
                 cursor: isSimulating ? "wait" : "pointer", // Indicate waiting state
                 overflow: "hidden", // Clip the progress bar
                 boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
                 }}
                 // Add slight press effect
                 onMouseDown={e => { if (!isSimulating) (e.target as HTMLButtonElement).style.transform = 'scale(0.98)'; }}
                 onMouseUp={e => { if (!isSimulating) (e.target as HTMLButtonElement).style.transform = 'scale(1)'; }}
             >
                 {/* Progress Bar Overlay */}
                 {isSimulating && (
                 <div
                     style={{
                     position: "absolute",
                     top: 0,
                     left: 0,
                     height: "100%",
                     width: `${progress}%`,
                     // Use a semi-transparent overlay color
                     background: "rgba(40, 167, 69, 0.7)", // Semi-transparent green
                     transition: "width 0.3s ease-out", // Match progress animation
                     zIndex: 0, // Behind the text
                     }}
                 ></div>
                 )}
                 {/* Button Text */}
                 <span style={{ position: "relative", zIndex: 1 }}>
                 {isSimulating ? `Simulating... ${progress.toFixed(0)}%` : "Start Simulation"}
                 </span>
             </button>
        </div>


        {/* --- Final Result Display Section --- */}
        {/* Conditionally render based on simulationResult */}
        {simulationResult && (
          <div style={{ marginTop: "30px", borderTop: "2px solid #eee", paddingTop: "20px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "20px" }}>
                 <h2 style={{ margin: 0 }}>Simulation Results</h2>
                 {/* Buttons to open modals */}
                 <div>
                    <button
                        onClick={() => setShowMetricsModal(true)}
                        onMouseEnter={() => setMetricsHover(true)}
                        onMouseLeave={() => setMetricsHover(false)}
                        style={{ /* Button styles */
                            padding: "8px 15px", marginRight: "10px", cursor: "pointer",
                            backgroundColor: metricsHover ? "#e2e6ea" : "#6c757d", color: metricsHover ? "#343a40" : "#fff",
                            border: "none", borderRadius: "4px", transition: "background-color 0.2s ease"
                        }}
                    >
                        Explain Metrics
                    </button>
                    <button
                        onClick={() => setShowMethodologyModal(true)}
                        onMouseEnter={() => setMethodologyHover(true)}
                        onMouseLeave={() => setMethodologyHover(false)}
                         style={{ /* Button styles */
                            padding: "8px 15px", cursor: "pointer",
                            backgroundColor: methodologyHover ? "#e2e6ea" : "#6c757d", color: methodologyHover ? "#343a40" : "#fff",
                            border: "none", borderRadius: "4px", transition: "background-color 0.2s ease"
                        }}
                    >
                        Explain Methodology
                    </button>
                 </div>
            </div>

             {/* Display error message from result if it exists */}
             {simulationResult.error && (
                <div style={{color: 'red', marginBottom: '20px', fontWeight: 'bold', border: '1px solid red', padding: '10px', borderRadius: '4px'}}>
                    Simulation Error: {simulationResult.error}
                </div>
             )}

             {/* Only display results table and viz if NO error */}
             {!simulationResult.error && (
                <>
                    {/* Results Table */}
                    <table style={{ width: "100%", marginBottom: "20px", borderCollapse: "collapse" }}>
                        <tbody>
                        {/* Use conditional rendering for each metric */}
                        {typeof simulationResult.optimized_ppfd === 'number' && <tr><td style={tableCellStyle}><strong>Average PPFD:</strong></td><td style={tableCellStyle}>{simulationResult.optimized_ppfd.toFixed(2)} µmol/m²/s</td></tr>}
                        {typeof simulationResult.dou === 'number' && <tr><td style={tableCellStyle}><strong>Maximized DOU:</strong></td><td style={tableCellStyle}>{simulationResult.dou.toFixed(2)}%</td></tr>}
                        {typeof simulationResult.mad === 'number' && <tr><td style={tableCellStyle}><strong>Minimized MAD:</strong></td><td style={tableCellStyle}>{simulationResult.mad.toFixed(2)}</td></tr>}
                        {typeof simulationResult.rmse === 'number' && <tr><td style={tableCellStyle}><strong>Minimized RMSE:</strong></td><td style={tableCellStyle}>{simulationResult.rmse.toFixed(2)}</td></tr>}
                        {typeof simulationResult.cv === 'number' && <tr><td style={tableCellStyle}><strong>Minimized CV:</strong></td><td style={tableCellStyle}>{simulationResult.cv.toFixed(2)}%</td></tr>}
                        </tbody>
                    </table>

                    {/* Optimized Lumens List */}
                    {simulationResult.optimized_lumens_by_layer && simulationResult.optimized_lumens_by_layer.length > 0 && (
                        <>
                        <h4 style={{marginTop: "25px"}}>Optimized Lumens by Layer</h4>
                        <ul style={{ listStyle: "none", paddingLeft: 0 }}>
                            {simulationResult.optimized_lumens_by_layer.map((lumens, i) => (
                            <li key={i} style={{ marginBottom: "5px", background: "#f8f9fa", padding: "5px 10px", borderRadius: "3px" }}>
                                {i === 0 ? "Center COB" : `Layer ${i}`}: {lumens.toFixed(2)} lumens {/* Adjusted layer numbering to start from 0 */}
                            </li>
                            ))}
                        </ul>
                        </>
                    )}

                    {/* Modular Visualization Component */}
                    {simulationResult.ppfd_grid_data && ( // Check if needed data is available
                         <div style={{ marginTop: "20px" }}>
                              <h3>Optimized Layout Visualization</h3>
                              <ModularVisualization
                                   // Pass all necessary props from simulationResult
                                   mad={simulationResult.mad}
                                   optimized_ppfd={simulationResult.optimized_ppfd}
                                   floorWidth={simulationResult.floor_width} // Assuming feet
                                   floorLength={simulationResult.floor_length} // Assuming feet
                                   floorHeight={simulationResult.floor_height || 3} // Assuming feet, provide default if missing
                                   optimizedLumensByLayer={simulationResult.optimized_lumens_by_layer}
                                   simulationResult={simulationResult} // Pass the whole object if easier
                                   // Ensure ModularVisualization is adapted for these props
                                   // It might need ppfd_grid_data as well
                                   //ppfdGridData={simulationResult.ppfd_grid_data}
                              />
                         </div>
                    )}


                    {/* Surface Graph & Heatmap Images */}
                    <div style={{ marginTop: "30px" }}>
                        <h3 style={{ textAlign: 'center', marginBottom: '20px' }}>PPFD Distribution Analysis</h3>
                        <div style={{ display: "flex", flexWrap: "wrap", gap: "20px", justifyContent: 'center' }}>
                        {/* Optimized (Staggered) Results */}
                        {simulationResult.surface_graph && simulationResult.heatmap && (
                            <div style={{ flex: "1 1 45%", minWidth: "300px", textAlign: 'center', padding: '10px', border: '1px solid #eee', borderRadius: '5px' }}>
                                <h4>Optimized Simulation</h4>
                                <img src={`data:image/png;base64,${simulationResult.surface_graph}`} alt="Optimized Surface Graph" style={{ maxWidth: "100%", height: 'auto', marginBottom: '10px' }} />
                                <img src={`data:image/png;base64,${simulationResult.heatmap}`} alt="Optimized Heatmap" style={{ maxWidth: "100%", height: 'auto' }} />
                            </div>
                        )}
                        {/* Comparison (Uniform Grid) Results */}
                        {simulationResult.grid_surface_graph && simulationResult.grid_heatmap && (
                            <div style={{ flex: "1 1 45%", minWidth: "300px", textAlign: 'center', padding: '10px', border: '1px solid #eee', borderRadius: '5px' }}>
                            <h4>Comparison: Uniform Grid</h4>
                            <img src={`data:image/png;base64,${simulationResult.grid_surface_graph}`} alt="Grid Surface Graph" style={{ maxWidth: "100%", height: 'auto', marginBottom: '10px' }}/>
                            <img src={`data:image/png;base64,${simulationResult.grid_heatmap}`} alt="Grid Heatmap" style={{ maxWidth: "100%", height: 'auto' }} />
                            </div>
                        )}
                        </div>
                    </div>

                    {/* Uniform Grid Details & Visualization (if comparison enabled/data available) */}
                    {simulationResult.grid_cob_arrangement &&
                        simulationResult.grid_uniform_flux !== undefined && ( // Check essential fields
                        <div style={{ marginTop: "30px", borderTop: "1px solid #ddd", paddingTop: "20px" }}>
                            <h3>Comparison: Uniform Grid Simulation Details</h3>
                             <table style={{ width: "100%", marginBottom: "20px", borderCollapse: "collapse" }}>
                                <tbody>
                                    {/* Add checks for each optional grid metric */}
                                    {typeof simulationResult.grid_uniform_flux === 'number' && <tr><td style={tableCellStyle}><strong>Lumens Per COB:</strong></td><td style={tableCellStyle}>{simulationResult.grid_uniform_flux.toFixed(2)} lumens</td></tr>}
                                    {typeof simulationResult.grid_ppfd === 'number' && <tr><td style={tableCellStyle}><strong>Average PPFD:</strong></td><td style={tableCellStyle}>{simulationResult.grid_ppfd.toFixed(2)} µmol/m²/s</td></tr>}
                                    {typeof simulationResult.grid_mad === 'number' && <tr><td style={tableCellStyle}><strong>MAD:</strong></td><td style={tableCellStyle}>{simulationResult.grid_mad.toFixed(2)}</td></tr>}
                                    {typeof simulationResult.grid_rmse === 'number' && <tr><td style={tableCellStyle}><strong>RMSE:</strong></td><td style={tableCellStyle}>{simulationResult.grid_rmse.toFixed(2)}</td></tr>}
                                    {typeof simulationResult.grid_dou === 'number' && <tr><td style={tableCellStyle}><strong>DOU:</strong></td><td style={tableCellStyle}>{simulationResult.grid_dou.toFixed(2)}%</td></tr>}
                                    {typeof simulationResult.grid_cv === 'number' && <tr><td style={tableCellStyle}><strong>CV:</strong></td><td style={tableCellStyle}>{simulationResult.grid_cv.toFixed(2)}%</td></tr>}
                                </tbody>
                             </table>
                            {/* Grid Visualization Component */}
                            {simulationResult.grid_cob_arrangement && simulationResult.grid_ppfd !== undefined &&(
                                <div style={{ marginTop: "20px" }}>
                                     <h4>Grid Layout Visualization</h4>
                                     <GridVisualization
                                        floorWidth={simulationResult.floor_width} // Assuming feet
                                        floorLength={simulationResult.floor_length} // Assuming feet
                                        floorHeight={simulationResult.floor_height || 3} // Assuming feet
                                        uniformFlux={simulationResult.grid_uniform_flux}
                                        gridArrangement={simulationResult.grid_cob_arrangement}
                                        gridPPFD={simulationResult.grid_ppfd}
                                        // Ensure GridVisualization is adapted for these props
                                     />
                                </div>
                            )}
                        </div>
                    )}
                </> // End fragment for non-error results
             )}
          </div> // End simulationResult conditional block
        )}

        {/* --- Modals --- */}
        {/* Metrics Modal */}
        {showMetricsModal && (
          <div
            className="modal-overlay"
            onClick={() => setShowMetricsModal(false)}
            style={modalOverlayStyle} // Use style objects
          >
            <div className="modal-content" onClick={(e) => e.stopPropagation()} style={modalContentStyle}>
              <span style={modalCloseButtonStyle} onClick={() => setShowMetricsModal(false)}>×</span>
              <h3 style={{ marginTop: 0 }}>Explanation of Metrics</h3>
               {/* Metrics explanation content (remains the same) */}
               <div className="metrics-explanation" style={{ marginTop: "20px" }}>
                 <ol>
                   <li style={{ marginBottom: "1em" }}><strong>Average PPFD (<MathJax inline>{"\\(\\text{PPFD}_{\\text{avg}}\\)"}</MathJax>):</strong> The average PPFD is simply the arithmetic mean of all PPFD measurements: <MathJax>{"\\(\\text{PPFD}_{\\text{avg}} = \\frac{1}{n} \\sum_{i=1}^{n} P_i\\)"}</MathJax></li>
                   <li style={{ marginBottom: "1em" }}><strong>Root Mean Squared Error (RMSE):</strong> RMSE quantifies the difference between the measured PPFD values and the average PPFD: <MathJax>{"\\(\\text{RMSE} = \\sqrt{\\frac{\\sum_{i=1}^{n} (P_i - \\text{PPFD}_{\\text{avg}})^2}{n}}\\)"}</MathJax></li>
                   <li style={{ marginBottom: "1em" }}><strong>Degree of Uniformity (DOU):</strong> DOU is calculated based on the RMSE and the average PPFD: <MathJax>{"\\(\\text{DOU} = 100 \\times \\left(1 - \\frac{\\text{RMSE}}{\\text{PPFD}_{\\text{avg}}}\\right)\\)"}</MathJax></li>
                   <li style={{ marginBottom: "1em" }}><strong>Mean Absolute Deviation (MAD):</strong> MAD measures the average absolute difference between each PPFD value and the average PPFD: <MathJax>{"\\(\\text{MAD} = \\frac{1}{n} \\sum_{i=1}^{n} |P_i - \\text{PPFD}_{\\text{avg}}|\\)"}</MathJax></li>
                   <li style={{ marginBottom: "1em" }}><strong>Coefficient of Variation (CV):</strong> CV is the ratio of the standard deviation (<MathJax inline>{"\\(\\sigma\\)"}</MathJax>) to the average PPFD, expressed as a percentage: <MathJax>{"\\(\\text{CV} = 100 \\times \\frac{\\sigma}{\\text{PPFD}_{\\text{avg}}}\\)"}</MathJax> where the standard deviation, <MathJax inline>{"\\(\\sigma\\)"}</MathJax>, is calculated as: <MathJax>{"\\(\\sigma = \\sqrt{\\frac{\\sum_{i=1}^{n} (P_i - \\text{PPFD}_{\\text{avg}})^2}{n}}\\)"}</MathJax> <em>Note:</em> In this specific case, since we are calculating sample statistics and using all data points, the sample and population standard deviations are equivalent.</li>
                 </ol>
               </div>
            </div>
          </div>
        )}

        {/* Methodology Modal */}
        {showMethodologyModal && (
          <div
            className="modal-overlay"
            onClick={() => setShowMethodologyModal(false)}
            style={modalOverlayStyle} // Use style objects
          >
            <div className="modal-content" onClick={(e) => e.stopPropagation()} style={modalContentStyle}>
              <span style={modalCloseButtonStyle} onClick={() => setShowMethodologyModal(false)}>×</span>
              <h3 style={{ marginTop: 0 }}>Explanation of Methodology</h3>
               {/* Methodology explanation content (remains the same) */}
               <div style={{ marginTop: "20px" }}>
                 <p>The <b>Chip-On-Board</b> (COB) LEDs are arranged in a centered square pattern sequence, which enables the separation of the COBs into concentric square layers for the application of layer-wise intensity assignments.</p>
                 <p>Our <b>differential evolution-based global optimization algorithm</b> finds the optimal intensities to apply to each layer of COBs for the purpose of maximizing the <b>Degree of Uniformity</b> (DOU) of <b>Photosynthetic Photon Flux Density</b> (PPFD) while meeting the target PPFD.</p>
                 <p>The reason the algorithm converges on a solution where there’s a large concentration of luminous flux assigned to outer layers of COBs with inner layers receiving lower luminous flux assignments is quite simple: The algorithm converges on a solution with higher luminous flux assigned to the outer COB layers because of the combined effects of the <b>inverse square law</b> and the overlapping emission patterns of the COBs.</p>
                 <p>Although COBs have a circular <b>Light Emitting Surface</b> (LES), they typically do not emit light omnidirectionally. Many COBs have an emission pattern that approximates a <b>Lambertian distribution</b>.</p>
                 <p>In summation: By separating the COBs into concentric square layers and assigning a concentration of luminous flux to outer perimeter layers, light fills in the naturally occurring deficit of photons at the perimeter.</p>
               </div>
            </div>
          </div>
        )}

      </div> {/* End main container div */}
    </MathJaxContext> // End MathJax context
  ); // End return
}; // End SimulationForm component

// Define some inline styles for reuse
const tableCellStyle: React.CSSProperties = {
     padding: "8px 12px",
     borderBottom: "1px solid #eee",
     textAlign: "left"
};

const modalOverlayStyle: React.CSSProperties = {
    position: "fixed", top: 0, left: 0, width: "100%", height: "100%",
    backgroundColor: "rgba(0,0,0,0.6)", display: "flex",
    justifyContent: "center", alignItems: "center", zIndex: 1050, // Ensure high z-index
    backdropFilter: "blur(3px)"
};

const modalContentStyle: React.CSSProperties = {
    background: "#fff", padding: "25px", borderRadius: "8px",
    maxWidth: "700px", maxHeight: "90%", overflowY: "auto",
    boxShadow: "0 5px 15px rgba(0,0,0,0.3)", position: "relative"
};

const modalCloseButtonStyle: React.CSSProperties = {
    position: "absolute", top: "10px", right: "15px", fontSize: "2rem",
    fontWeight: "bold", color: "#888", cursor: "pointer", lineHeight: 1
};


export default SimulationForm;