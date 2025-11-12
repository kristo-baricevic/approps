"use client";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL as string;
if (!API) {
  throw new Error("Missing NEXT_PUBLIC_API_URL in apps/web/.env.local");
}

const fd = new FormData();

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
  const contentType = file.type || "application/pdf";

  console.log("Starting upload to:", presigned.url);
  console.log("File size:", file.size);
  console.log("Content-Type:", contentType);

  try {
    const r = await fetch(presigned.url, {
      method: "PUT",
      mode: "cors",
      headers: {
        "Content-Type": contentType,
      },
      body: file,
    });

    console.log("Response received:");
    console.log("- Status:", r.status);
    console.log("- StatusText:", r.statusText);
    console.log("- OK:", r.ok);
    console.log("- Headers:", Object.fromEntries(r.headers.entries()));

    // MinIO returns 200 for successful PUT
    if (r.status === 200) {
      console.log("✅ Upload successful!");
      return;
    }

    // Try to get response body
    const responseText = await r
      .text()
      .catch((e) => `Failed to read response: ${e}`);
    console.log("Response body:", responseText);

    // Only throw if not 2xx
    if (!r.ok) {
      throw new Error(`Upload failed: ${r.status} ${r.statusText}`);
    }
  } catch (error) {
    console.error("Upload error details:", {
      name: error.name,
      message: error.message,
      stack: error.stack,
    });

    // Check if it's a network error
    if (error instanceof TypeError && error.message === "Failed to fetch") {
      console.error("Network error - this could be:");
      console.error("1. CORS (but we verified this works)");
      console.error("2. Network timeout");
      console.error("3. Connection reset");
      console.error("4. Response parsing issue");
    }

    throw error;
  }
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

async function parseFile(fileId: string, tableLabel: string) {
  try {
    const r = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/parse`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ file_id: fileId, table_label: tableLabel }),
    });
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  } catch (e) {
    console.error("parseFile fetch failed:", e);
    throw e;
  }
}

export default function UploadPage() {
  const [a, setA] = useState<File | null>(null);
  const [b, setB] = useState<File | null>(null);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  const router = useRouter();

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
    try {
      const last = await handle(a);
      const curr = await handle(b);

      // parse both
      const lastParse = await parseFile(last.file_id, "Labor-HHS-Education");
      const currParse = await parseFile(curr.file_id, "Labor-HHS-Education");

      console.log("lastParse", lastParse);
      console.log("currParse", currParse);

      if (!currParse.audit || currParse.audit.passed !== true) {
        setResult({ last, curr, lastParse, currParse });
        setStatus(null);
        setError("We could not reliably process the current document.");
        return;
      }

      // preview rows for the current file (and last if you want)
      const currPreview = await fetch(
        `${API}/tables/${currParse.table_id}/preview`
      ).then((r) => r.json());

      const lastPreview = await fetch(
        `${API}/tables/${lastParse.table_id}/preview`
      ).then((r) => r.json());

      const diffRes = await fetch(`${API}/diff`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prev_table_id: lastParse.table_id,
          curr_table_id: currParse.table_id,
        }),
      }).then((r) => r.json());

      // optional: still keep previews in `result` if you like
      setResult({
        last,
        curr,
        lastParse,
        currParse,
        lastPreview,
        currPreview,
        diffId: diffRes.diff_id,
      });
      setStatus(null);
      // send user straight to the Diff view
      router.push(`/diff/${diffRes.diff_id}`);
    } catch (e) {
      console.error(e);
      setStatus(null);
      setError(
        e?.message || "Something went wrong while processing the files."
      );
    }
  }

  useEffect(() => {
    console.log("result ===> ", result);
  }, [result]);

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
      <div>
        {result && (
          <pre className="bg-slate-800 p-3 text-white rounded text-sm overflow-auto">
            {JSON.stringify(result, null, 2)}
          </pre>
        )}
      </div>
      {result?.currPreview && result.currPreview.length > 0 && (
        <table className="w-full text-sm border mt-4">
          <thead className="bg-slate-600">
            <tr>
              <th className="p-2 text-left">Program</th>
              <th className="p-2">FY</th>
              <th className="p-2">Amount</th>
              <th className="p-2">Page</th>
            </tr>
          </thead>
          <tbody>
            {result.currPreview.map((r: any, i: number) => (
              <tr key={i} className="border-t">
                <td className="p-2">{r.program_name}</td>
                <td className="p-2 text-center">{r.fy ?? ""}</td>
                <td className="p-2 text-right">{r.amount}</td>
                <td className="p-2 text-center">{r.page}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
