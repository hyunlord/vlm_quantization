import type { Metadata } from "next";
import { Geist_Mono } from "next/font/google";
import { RunProvider } from "@/contexts/RunContext";
import "./globals.css";

const mono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "VLM Quantization Monitor",
  description: "Real-time training dashboard for cross-modal deep hashing",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${mono.variable} font-mono antialiased bg-gray-950 text-gray-100`}>
        <RunProvider>{children}</RunProvider>
      </body>
    </html>
  );
}
