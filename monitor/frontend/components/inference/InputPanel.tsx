"use client";

import { ImageIcon, Type, Upload } from "lucide-react";
import { useCallback, useRef, useState } from "react";

interface Props {
  label: string;
  onEncode: (codes: HashCode[]) => void;
}

interface HashCode {
  bits: number;
  binary: number[];
  continuous: number[];
}

export default function InputPanel({ label, onEncode }: Props) {
  const [mode, setMode] = useState<"image" | "text">("image");
  const [text, setText] = useState("");
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target?.result as string);
      setError(null);
    };
    reader.readAsDataURL(file);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const encode = async () => {
    setLoading(true);
    setError(null);
    try {
      const body: Record<string, string> = {};
      if (mode === "image" && preview) {
        body.image_base64 = preview;
      } else if (mode === "text" && text.trim()) {
        body.text = text.trim();
      } else {
        setError("Please provide an input");
        setLoading(false);
        return;
      }

      const res = await fetch("/api/inference/encode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || `HTTP ${res.status}`);
      }

      const data = await res.json();
      onEncode(data.codes);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Encode failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="rounded-xl bg-gray-900 border border-gray-800 p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-300">{label}</h3>
        <div className="flex gap-1">
          <button
            onClick={() => setMode("image")}
            className={`flex items-center gap-1 text-xs px-2 py-1 rounded ${
              mode === "image"
                ? "bg-blue-900/50 text-blue-400"
                : "text-gray-500 hover:text-gray-300"
            }`}
          >
            <ImageIcon className="w-3 h-3" />
            Image
          </button>
          <button
            onClick={() => setMode("text")}
            className={`flex items-center gap-1 text-xs px-2 py-1 rounded ${
              mode === "text"
                ? "bg-blue-900/50 text-blue-400"
                : "text-gray-500 hover:text-gray-300"
            }`}
          >
            <Type className="w-3 h-3" />
            Text
          </button>
        </div>
      </div>

      {/* Input area */}
      {mode === "image" ? (
        <div
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => fileRef.current?.click()}
          className="border-2 border-dashed border-gray-700 rounded-lg p-4 cursor-pointer
                     hover:border-gray-500 transition-colors min-h-[140px]
                     flex flex-col items-center justify-center gap-2"
        >
          {preview ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={preview}
              alt="Preview"
              className="max-h-[120px] rounded object-contain"
            />
          ) : (
            <>
              <Upload className="w-6 h-6 text-gray-600" />
              <p className="text-xs text-gray-500">
                Drop image or click to upload
              </p>
            </>
          )}
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleFile(file);
            }}
          />
        </div>
      ) : (
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter text to encode..."
          className="w-full h-[140px] bg-gray-800 border border-gray-700 rounded-lg p-3
                     text-sm text-gray-200 placeholder-gray-600 resize-none
                     focus:outline-none focus:border-gray-500"
        />
      )}

      {/* Encode button */}
      <button
        onClick={encode}
        disabled={loading}
        className="mt-3 w-full py-2 rounded-lg text-xs font-medium
                   bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700
                   disabled:text-gray-500 text-white transition-colors"
      >
        {loading ? "Encoding..." : "Encode"}
      </button>

      {error && (
        <p className="mt-2 text-xs text-red-400">{error}</p>
      )}
    </div>
  );
}
