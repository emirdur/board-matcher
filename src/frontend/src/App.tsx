import { useState, useRef } from "react";
import { Upload, CircleHelp, File, X, Download } from "lucide-react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

interface Match {
  name: string;
  employment: string;
  board_service: string;
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [query, setQuery] = useState("");
  const [matches, setMatches] = useState<Match[]>([]);
  const [datasetLoaded, setDatasetLoaded] = useState(false);
  const [showTooltip, setShowTooltip] = useState(false);
  const [toast, setToast] = useState<{
    message: string;
    type: "success" | "error";
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const tooltipTimeout = useRef<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const showToast = (message: string, type: "success" | "error") => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 3000);
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      await uploadFile(selectedFile);
    }
  };

  const uploadFile = async (file: File) => {
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Upload failed");
      }

      await response.json();
      setDatasetLoaded(true);
      showToast("Dataset loaded successfully", "success");
    } catch (err) {
      showToast(err instanceof Error ? err.message : "Upload failed", "error");
    }
  };

  const handleMatch = async () => {
    setLoading(true);
    setMatches([]);

    try {
      const response = await fetch(`${API_URL}/match`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: 10 }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Match failed");
      }

      const data = await response.json();
      setMatches(data.matches);
    } catch (err) {
      showToast(err instanceof Error ? err.message : "Match failed", "error");
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async () => {
    try {
      const response = await fetch(`${API_URL}/export`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ matches }),
      });

      if (!response.ok) {
        throw new Error("Export failed");
      }

      // Create download link
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "board_matches.xlsx";
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      showToast("Export successful", "success");
    } catch (err) {
      showToast(err instanceof Error ? err.message : "Export failed", "error");
    }
  };

  return (
    <div className="max-w-3xl mx-auto px-4 py-12 font-sans text-black">
      {/* Toast */}
      {toast && (
        <div
          className={`fixed top-6 left-1/2 transform -translate-x-1/2 px-4 py-2 rounded-md shadow-md text-white transition-all duration-500 ${
            toast.type === "success" ? "bg-green-600" : "bg-red-600"
          }`}
        >
          {toast.message}
        </div>
      )}

      {/* Header */}
      <div className="text-left mb-10">
        <p className="tracking-widest font-semibold text-sm uppercase">
          A Healthier Democracy
        </p>
        <h1 className="text-4xl mt-1">Board Matcher</h1>
      </div>

      {/* Upload Section */}
      <div className="mb-10">
        <div className="flex items-center gap-3 mb-3">
          <h2 className="text-2xl">Upload Your Dataset</h2>
          <div
            className="relative"
            onMouseEnter={() => setShowTooltip(true)}
            onMouseLeave={() => {
              if (tooltipTimeout.current) clearTimeout(tooltipTimeout.current);
              tooltipTimeout.current = window.setTimeout(
                () => setShowTooltip(false),
                150
              );
            }}
            onClick={() => setShowTooltip(!showTooltip)}
          >
            <CircleHelp className="w-4 h-4 text-black cursor-pointer relative top-[0.75px]" />
            {showTooltip && (
              <div className="absolute -top-23 left-1/2 -translate-x-1/2 w-64 text-sm bg-white border border-gray-200 text-black p-3 rounded-lg shadow-md z-10">
                Your dataset should include the following columns:
                <br />
                <span className="font-medium">
                  Name, Employment, Board Service
                </span>
              </div>
            )}
          </div>
        </div>

        <label
          htmlFor="file-upload"
          className="flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-lg py-10 cursor-pointer transition w-full"
        >
          <input
            ref={fileInputRef}
            id="file-upload"
            type="file"
            accept=".xlsx,.xls"
            className="hidden"
            onChange={handleFileChange}
          />
          {!file ? (
            <>
              <Upload className="w-6 h-6 mb-2 text-gray-500" />
              <p className="text-gray-500">
                Click to choose a file or drag here
              </p>
            </>
          ) : (
            <div className="relative flex flex-col items-center justify-center bg-white px-10 py-6 rounded-xl shadow-md">
              <File className="w-8 h-8 mb-2 text-black" />
              <p className="text-black">{file.name}</p>
              <button
                type="button"
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  setFile(null);
                  setDatasetLoaded(false);
                  if (fileInputRef.current) {
                    fileInputRef.current.value = "";
                  }
                }}
                className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1 hover:bg-red-600 transition cursor-pointer shadow-sm"
              >
                <X className="w-3 h-3" strokeWidth={3} />
              </button>
            </div>
          )}
        </label>
      </div>

      {/* Search Section */}
      <div className="mb-10">
        <h2 className="text-2xl mb-3">Search for Matches</h2>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ex. Mr. John A. Doe (Johnny)     AlphaTech Ltd., CFO, 2002 - 2007, Smart Solutions LLC, CEO, 2015 - 2025"
          rows={3}
          className="w-full border border-gray-300 rounded-lg p-3 mb-4 text-gray-600 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[#3464DE]"
        />
        <button
          onClick={handleMatch}
          disabled={loading || !datasetLoaded || !query.trim()}
          className="w-full bg-[#3464DE] text-white py-3 rounded-lg shadow-md hover:shadow-lg transition disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
        >
          {loading ? "Matching..." : "Match Connections"}
        </button>
      </div>

      {/* Results Section */}
      {matches.length > 0 && (
        <div>
          <div>
            <h2 className="text-2xl mb-4">Likely Connections</h2>
            <div className="flex flex-wrap gap-4">
              {matches.map((m, idx) => (
                <div
                  key={idx}
                  className="w-full sm:w-[48%] lg:w-[31%] bg-white rounded-xl shadow-md p-4"
                >
                  <h3 className="font-semibold mb-1">{m.name}</h3>
                  <p className="text-sm text-gray-600">
                    {m.employment || m.board_service || "No additional info"}
                  </p>
                </div>
              ))}
            </div>
          </div>
          <button
            onClick={handleExport}
            className="mt-6 w-full border-2 border-gray-300 text-gray-400 py-3 rounded-lg shadow-md hover:shadow-lg transition cursor-pointer flex items-center justify-center gap-2"
          >
            <Download className="w-5 h-5" />
            Export to Excel
          </button>
        </div>
      )}
      <div className="mt-12 text-center text-xs">
        © 2025 A Healthier Democracy. All rights reserved.
      </div>
    </div>
  );
}
