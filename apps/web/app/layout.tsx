import type { ReactNode } from "react";
import "../styles/globals.css";
import React from "react";

export const metadata = {
  title: "Approps Parser",
  description: "Health + Upload shell",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-950 text-slate-200">
        {children}
      </body>
    </html>
  );
}
