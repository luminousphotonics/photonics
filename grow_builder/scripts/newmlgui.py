import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading, sys, queue
from newml import iterative_optimization
# Add a check for the simulation module early on
try:
    from newml import solve_single_scenario, HIGH_LOSS  # Import HIGH_LOSS for result formatting
except ImportError as e:
    messagebox.showerror("Import Error", f"Failed to import 'newml':\n{e}\n\nPlease ensure newml.py and its dependencies are available.")
    sys.exit(1)
except Exception as e:
    messagebox.showerror("Error", f"An unexpected error occurred during import:\n{e}")
    sys.exit(1)

# A thread-safe queue for log messages
log_queue = queue.Queue()

# --- Main Application Class ---
class OptimizerApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("NewML Optimizer UI")
        self._build_ui()
        # Start polling the log queue
        self.root.after(100, self.poll_queue)

    def _build_ui(self):
        # Input fields frame
        input_frame = tk.Frame(self.root)
        input_frame.pack(padx=10, pady=10)

        tk.Label(input_frame, text="Layer Count:").grid(row=0, column=0, sticky="w")
        self.layer_entry = tk.Entry(input_frame, width=10)
        self.layer_entry.insert(0, "19")
        self.layer_entry.grid(row=0, column=1, padx=5)

        tk.Label(input_frame, text="Height (ft):").grid(row=1, column=0, sticky="w")
        self.height_entry = tk.Entry(input_frame, width=10)
        self.height_entry.insert(0, "3.0")
        self.height_entry.grid(row=1, column=1, padx=5)

        tk.Label(input_frame, text="Target PPFD:").grid(row=2, column=0, sticky="w")
        self.ppfd_entry = tk.Entry(input_frame, width=10)
        self.ppfd_entry.insert(0, "1250")
        self.ppfd_entry.grid(row=2, column=1, padx=5)

        # Button command now refers to the instance method
        self.run_button = tk.Button(input_frame, text="Run Scenario", command=self.run_scenario)
        self.run_button.grid(row=3, column=0, columnspan=2, pady=5)

        # Iterative Optimization Parameters Frame
        iter_frame = tk.LabelFrame(input_frame, text="Iterative Optimization Settings", padx=5, pady=5)
        iter_frame.grid(row=4, column=0, columnspan=2, pady=10, sticky="w")

        tk.Label(iter_frame, text="Start Layers:").grid(row=0, column=0, sticky="w")
        self.start_layers_entry = tk.Entry(iter_frame, width=10)
        self.start_layers_entry.insert(0, "5")
        self.start_layers_entry.grid(row=0, column=1, padx=5)

        tk.Label(iter_frame, text="Max Layers:").grid(row=1, column=0, sticky="w")
        self.max_layers_entry = tk.Entry(iter_frame, width=10)
        self.max_layers_entry.insert(0, "20")
        self.max_layers_entry.grid(row=1, column=1, padx=5)

        tk.Label(iter_frame, text="Max Iterations/Layer:").grid(row=2, column=0, sticky="w")
        self.max_iter_entry = tk.Entry(iter_frame, width=10)
        self.max_iter_entry.insert(0, "10")
        self.max_iter_entry.grid(row=2, column=1, padx=5)

        # Button to run iterative optimization
        self.run_iterative_button = tk.Button(iter_frame, text="Run Iterative Optimization", command=self.run_iterative_optimization)
        self.run_iterative_button.grid(row=3, column=0, columnspan=2, pady=5)

        # Logs display frame
        logs_label = tk.Label(self.root, text="Optimizer Log:")
        logs_label.pack(anchor="w", padx=10)
        self.logs_text = scrolledtext.ScrolledText(self.root, width=80, height=20, font=("Courier", 10), state=tk.DISABLED)
        self.logs_text.pack(fill="both", expand=True, padx=10, pady=5)

        # Final result display frame
        result_label = tk.Label(self.root, text="Final Result:")
        result_label.pack(anchor="w", padx=10)
        self.result_text = scrolledtext.ScrolledText(self.root, width=80, height=6, font=("Courier", 10))
        self.result_text.pack(fill="x", expand=False, padx=10, pady=5)

    def poll_queue(self):
        """Poll the log_queue and append any messages to the logs_text widget."""
        try:
            while not log_queue.empty():
                s = log_queue.get_nowait()
                self.logs_text.config(state=tk.NORMAL)
                self.logs_text.insert(tk.END, s)
                self.logs_text.see(tk.END)
                self.logs_text.config(state=tk.DISABLED)
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error updating log text: {e}")
        finally:
            self.root.after(100, self.poll_queue)

    def run_scenario(self):
        """Callback function for the 'Run Scenario' button."""
        try:
            layer = int(self.layer_entry.get())
            height = float(self.height_entry.get())
            ppfd = float(self.ppfd_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Invalid input values. Please enter numbers.")
            return
        except Exception as e:
            messagebox.showerror("Input Error", f"Error reading inputs: {e}")
            return

        self.run_button.config(state=tk.DISABLED, text="Running...")
        self.logs_text.config(state=tk.NORMAL)
        self.logs_text.delete('1.0', tk.END)
        self.logs_text.config(state=tk.DISABLED)
        self.result_text.delete('1.0', tk.END)
        self.result_text.insert(tk.END, "Starting optimization...\n")

        def task():
            final_result = {}
            try:
                final_result = solve_single_scenario(layer, height, ppfd, log_queue=log_queue)
            except Exception as e:
                error_msg = f"\n--- ERROR DURING TASK EXECUTION ---\n{type(e).__name__}: {e}\n"
                log_queue.put(error_msg)
                import traceback
                tb_msg = traceback.format_exc() + "\n"
                log_queue.put(tb_msg)
                final_result = {"error": f"Task failed: {e}", "logs": error_msg + tb_msg}
            finally:
                self.root.after(0, lambda res=final_result: self.final_ui_updates(res))
        
        threading.Thread(target=task, daemon=True).start()

    def run_iterative_optimization(self):
        """Callback function for the 'Run Iterative Optimization' button.
        Reads the iterative optimization parameters from the GUI and starts the process."""
        try:
            start_layers = int(self.start_layers_entry.get())
            max_layers = int(self.max_layers_entry.get())
            max_iter = int(self.max_iter_entry.get())
            target_ppfd = float(self.ppfd_entry.get())  # Reusing existing field
            height = float(self.height_entry.get())       # Reusing existing field
        except ValueError:
            messagebox.showerror("Input Error", "Invalid iterative optimization parameters.")
            return

        self.run_iterative_button.config(state=tk.DISABLED, text="Running Iterative...")
        self.logs_text.config(state=tk.NORMAL)
        self.logs_text.delete('1.0', tk.END)
        self.logs_text.config(state=tk.DISABLED)

        def task():
            iterative_optimization(start_layers=start_layers, max_layers=max_layers, 
                                   target_ppfd=target_ppfd, height_m=height, 
                                   max_iterations_per_layer=max_iter, queue_instance=log_queue)
            self.root.after(0, lambda: self.run_iterative_button.config(state=tk.NORMAL, text="Run Iterative Optimization"))
        
        threading.Thread(target=task, daemon=True).start()

    def final_ui_updates(self, final_result):
        """Updates the UI after the background task finishes."""
        self.run_button.config(state=tk.NORMAL, text="Run Scenario")
        self.result_text.delete('1.0', tk.END)
        error_msg = final_result.get("error")
        if error_msg:
            res_str = f"Error: {error_msg}"
        elif not final_result:
            res_str = "Error: Optimization task returned no result."
        else:
            ppfd_val = final_result.get('final_ppfd', 'N/A')
            mdou_val = final_result.get('mDOU', 'N/A')
            loss_val = final_result.get('loss', 'N/A')
            ppfd_str = f"{ppfd_val:.1f}" if isinstance(ppfd_val, (int, float)) and ppfd_val != -1.0 else "N/A"
            mdou_str = f"{mdou_val:.1f}%" if isinstance(mdou_val, (int, float)) and mdou_val != -1.0 else "N/A"
            loss_str = "N/A"
            if isinstance(loss_val, (int, float)):
                if loss_val >= HIGH_LOSS: 
                    loss_str = f"{loss_val:.3e}"
                elif loss_val != -1.0: 
                    loss_str = f"{loss_val:.3f}"
            res_str = (
                f"Action: {final_result.get('action', 'N/A')}\n"
                f"Reason: {final_result.get('reason', 'N/A')}\n"
                f"Final PPFD: {ppfd_str}\n"
                f"mDOU: {mdou_str}\n"
                f"Loss: {loss_str}\n"
            )
        self.result_text.insert(tk.END, res_str)

# --- Script Entry Point ---
if __name__ == "__main__":
    root = tk.Tk()
    app = OptimizerApp(root)
    root.mainloop()
