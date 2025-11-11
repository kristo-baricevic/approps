"use client";
import React from "react";
import { useEffect, useState } from "react";

export default function Page() {
  const [status, setStatus] = useState<string>("checking...");
  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_BASE_URL}/health`)
      .then((r) => r.json())
      .then((j) => setStatus(j.status ?? "unknown"))
      .catch(() => setStatus("down"));
  }, []);
  return (
    <main className="p-6">
      <h1 className="text-2xl font-semibold mb-2">Appropriations Parser</h1>
      <p className="mb-6">
        API health: <span className="font-mono">{status}</span>
      </p>
      <div className="rounded-2xl border border-slate-700 p-6">
        <p>Day 1 skeleton is up. Upload UI comes next.</p>
      </div>
    </main>
  );
}
