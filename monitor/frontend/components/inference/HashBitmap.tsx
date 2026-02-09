"use client";

interface Props {
  binary: number[];
  highlights?: Set<number>;
  label?: string;
}

export default function HashBitmap({ binary, highlights, label }: Props) {
  // Determine grid columns based on bit count
  const cols = binary.length <= 32 ? 16 : 32;

  return (
    <div>
      {label && (
        <p className="text-[10px] text-gray-500 mb-1 font-mono">{label}</p>
      )}
      <div
        className="inline-grid gap-[1px]"
        style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}
      >
        {binary.map((bit, i) => {
          const isDiff = highlights?.has(i);
          let bg: string;
          if (isDiff) {
            bg = bit === 1 ? "bg-red-500" : "bg-red-900";
          } else {
            bg = bit === 1 ? "bg-emerald-400" : "bg-gray-800";
          }
          return (
            <div
              key={i}
              className={`w-2 h-2 rounded-[1px] ${bg}`}
              title={`bit ${i}: ${bit}`}
            />
          );
        })}
      </div>
    </div>
  );
}
