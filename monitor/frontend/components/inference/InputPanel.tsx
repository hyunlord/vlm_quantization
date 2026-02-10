"use client";

import { Globe, ImageIcon, Loader2, Type, Upload } from "lucide-react";
import { useCallback, useRef, useState } from "react";

interface Props {
  label: string;
  onEncode: (codes: HashCode[]) => void;
  onBackboneEncode?: (embedding: number[]) => void;
  backboneOnly?: boolean;
}

interface HashCode {
  bits: number;
  binary: number[];
  continuous: number[];
}

type InputMode = "upload" | "url" | "text";

export default function InputPanel({ label, onEncode, onBackboneEncode, backboneOnly }: Props) {
  const [mode, setMode] = useState<InputMode>("upload");
  const [text, setText] = useState("");
  const [imageUrl, setImageUrl] = useState("");
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

  const loadUrlPreview = () => {
    if (!imageUrl.trim()) return;
    setPreview(imageUrl.trim());
    setError(null);
  };

  const encode = async () => {
    setLoading(true);
    setError(null);
    try {
      const body: Record<string, string> = {};
      if (mode === "upload" && preview) {
        body.image_base64 = preview;
      } else if (mode === "url" && imageUrl.trim()) {
        body.image_url = imageUrl.trim();
      } else if (mode === "text" && text.trim()) {
        body.text = text.trim();
      } else {
        setError("Please provide an input");
        setLoading(false);
        return;
      }

      // Fetch hash codes (unless backbone-only mode)
      if (!backboneOnly) {
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
      }

      // Also fetch backbone embedding if callback provided
      if (onBackboneEncode) {
        const bbRes = await fetch("/api/inference/encode-backbone", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (bbRes.ok) {
          const bbData = await bbRes.json();
          if (bbData.embedding) {
            onBackboneEncode(bbData.embedding);
          }
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Encode failed");
    } finally {
      setLoading(false);
    }
  };

  const tabs: { key: InputMode; icon: typeof Upload; label: string }[] = [
    { key: "upload", icon: Upload, label: "Upload" },
    { key: "url", icon: Globe, label: "URL" },
    { key: "text", icon: Type, label: "Text" },
  ];

  return (
    <div className="rounded-xl bg-gray-900 border border-gray-800 p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-300">{label}</h3>
        <div className="flex gap-0.5 bg-gray-800 rounded-lg p-0.5">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.key}
                onClick={() => {
                  setMode(tab.key);
                  setError(null);
                }}
                className={`flex items-center gap-1 text-[11px] px-2.5 py-1 rounded-md transition-colors ${
                  mode === tab.key
                    ? "bg-blue-600 text-white"
                    : "text-gray-500 hover:text-gray-300"
                }`}
              >
                <Icon className="w-3 h-3" />
                {tab.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Upload mode */}
      {mode === "upload" && (
        <div
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => fileRef.current?.click()}
          className="border-2 border-dashed border-gray-700 rounded-lg p-4 cursor-pointer
                     hover:border-gray-500 transition-colors min-h-[160px]
                     flex flex-col items-center justify-center gap-2"
        >
          {preview ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={preview}
              alt="Preview"
              className="max-h-[130px] rounded object-contain"
            />
          ) : (
            <>
              <ImageIcon className="w-8 h-8 text-gray-600" />
              <p className="text-xs text-gray-500 text-center">
                Drop image here or click to browse
              </p>
              <p className="text-[10px] text-gray-600">
                JPG, PNG, WebP supported
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
      )}

      {/* URL mode */}
      {mode === "url" && (
        <div className="space-y-2">
          <div className="flex gap-2">
            <input
              value={imageUrl}
              onChange={(e) => setImageUrl(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && loadUrlPreview()}
              placeholder="https://example.com/image.jpg"
              className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2
                         text-sm text-gray-200 placeholder-gray-600
                         focus:outline-none focus:border-gray-500"
            />
            <button
              onClick={loadUrlPreview}
              disabled={!imageUrl.trim()}
              className="px-3 py-2 rounded-lg text-[11px] font-medium
                         bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800
                         disabled:text-gray-600 text-gray-300 transition-colors"
            >
              Preview
            </button>
          </div>
          <div className="border border-gray-800 rounded-lg min-h-[120px] flex items-center justify-center bg-gray-800/30">
            {preview && mode === "url" ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={preview}
                alt="URL Preview"
                className="max-h-[120px] rounded object-contain"
                onError={() => {
                  setPreview(null);
                  setError("Failed to load image from URL");
                }}
              />
            ) : (
              <p className="text-[10px] text-gray-600">
                Enter a URL and click Preview
              </p>
            )}
          </div>
        </div>
      )}

      {/* Text mode */}
      {mode === "text" && (
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder={'Describe an image, e.g. "A golden retriever playing in a sunny park"'}
          className="w-full h-[160px] bg-gray-800 border border-gray-700 rounded-lg p-3
                     text-sm text-gray-200 placeholder-gray-600 resize-none
                     focus:outline-none focus:border-gray-500"
        />
      )}

      {/* Encode button */}
      <button
        onClick={encode}
        disabled={loading}
        className="mt-3 w-full py-2.5 rounded-lg text-xs font-medium
                   bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700
                   disabled:text-gray-500 text-white transition-colors
                   flex items-center justify-center gap-2"
      >
        {loading ? (
          <>
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
            Encoding...
          </>
        ) : (
          "Encode"
        )}
      </button>

      {error && <p className="mt-2 text-xs text-red-400">{error}</p>}
    </div>
  );
}
