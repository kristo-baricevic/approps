"use client";
import { useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL as string;
if (!API) {
  throw new Error("Missing NEXT_PUBLIC_API_URL in apps/web/.env.local");
}

const fd = new FormData();
const r = await fetch(`${API}/upload`, { method: "POST", body: fd });

async function sha256Hex(file: File) {
  const buf = await file.arrayBuffer();
  const hashBuf = await crypto.subtle.digest("SHA-256", buf);
  const arr = Array.from(new Uint8Array(hashBuf));
  return arr.map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function presign(sha256: string, file: File) {
  const r = await fetch(process.env.NEXT_PUBLIC_API_URL + "/upload/presign", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sha256,
      filename: file.name,
      content_type: file.type || "application/pdf",
    }),
  });
  if (!r.ok) throw new Error("presign failed");
  return r.json();
}

async function uploadDirect(presigned: any, file: File) {
  const form = new FormData();
  Object.entries(presigned.fields).forEach(([k, v]) =>
    form.append(k, String(v))
  );
  form.append("file", file);
  const r = await fetch(presigned.url, { method: "POST", body: form });
  if (!r.ok) throw new Error("s3 upload failed");
}

async function register(sha256: string, file: File, key: string) {
  const r = await fetch(process.env.NEXT_PUBLIC_API_URL + "/files/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sha256, filename: file.name, key }),
  });
  if (!r.ok) throw new Error("register failed");
  return r.json();
}

export default function UploadPage() {
  const [a, setA] = useState<File | null>(null);
  const [b, setB] = useState<File | null>(null);
  const [result, setResult] = useState<any>(null);

  async function handle(file: File) {
    console.log("handle");

    const sha = await sha256Hex(file);
    console.log("sha", sha);
    const p = await presign(sha, file);
    console.log("p", p);

    await uploadDirect(p, file);
    console.log("uploadDirect");

    const reg = await register(sha, file, p.key);
    return { ...reg, sha256: sha };
  }

  async function go() {
    if (!a || !b) return;
    const last = await handle(a);
    const curr = await handle(b);
    setResult({ last, curr });
  }

  return (
    <div className="max-w-xl mx-auto p-6 space-y-4">
      <h1 className="text-xl font-semibold">Upload two PDFs</h1>
      <div className="flex flex-col">
        Last
        <input
          type="file"
          className="bg-slate-500 p-2 rounded-md flex cursor-pointer!"
          accept="application/pdf"
          onChange={(e) => setA(e.target.files?.[0] ?? null)}
        />
      </div>
      <div className="flex flex-col">
        Current
        <input
          type="file"
          className="bg-slate-500 p-2 rounded-md cursor-pointer!"
          accept="application/pdf"
          onChange={(e) => setB(e.target.files?.[0] ?? null)}
        />
      </div>
      <button
        onClick={go}
        className="relative hover:bg-slate-500 z-10 px-4 py-2 rounded bg-slate-700 text-white pointer-events-auto cursor-pointer! hover:opacity-90"
      >
        Upload
      </button>

      {result && (
        <pre className="bg-slate-800 p-3 text-white rounded text-sm overflow-auto">
          {JSON.stringify(result, null, 2)}
        </pre>
      )}
    </div>
  );
}
