import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Gallery } from "./pages/Gallery";
import { Landing } from "./pages/Landing";
import { RunConsole } from "./pages/RunConsole";

/**
 * Client-side routing. FastAPI serves index.html for every non-/api path, so
 * a deep link to /run/<id> survives a refresh — which matters, because a
 * ten-minute run is exactly the kind of page someone reloads.
 */
export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/run/:runId" element={<RunConsole />} />
        <Route path="/runs" element={<Gallery />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
