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

    if (r.status === 200) {
      console.log("✅ Upload successful!");
      return;
    }

    const responseText = await r
      .text()
      .catch((e) => `Failed to read response: ${e}`);
    console.log("Response body:", responseText);

    if (!r.ok) {
      throw new Error(`Upload failed: ${r.status} ${r.statusText}`);
    }
  } catch (error: any) {
    console.error("Upload error details:", {
      name: error.name,
      message: error.message,
      stack: error.stack,
    });

    if (error instanceof TypeError && error.message === "Failed to fetch") {
      console.error("Network error, possible causes:");
      console.error("1. CORS");
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
    if (!a && !b) return;

    try {
      setStatus("Uploading and processing...");
      setError(null);

      if (a && b) {
        const last = await handle(a);
        const curr = await handle(b);

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
        router.push(`/diff/${diffRes.diff_id}`);
      } else {
        const file = (b ?? a)!;
        const reg = await handle(file);
        const parseRes = await parseFile(reg.file_id, "Labor-HHS-Education");

        console.log("singleParse", parseRes);

        if (!parseRes.table_id) {
          setStatus(null);
          setError("Could not determine table for this document.");
          return;
        }

        setResult({ single: reg, parse: parseRes });
        setStatus(null);
        router.push(`/tables/${parseRes.table_id}`);
      }
    } catch (e: any) {
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
    <main className="min-h-screen bg-slate-950 px-4 py-8">
      <div className="mx-auto max-w-3xl space-y-6">
        <header>
          <h1 className="text-3xl font-semibold tracking-tight bg-linear-to-r from-blue-400 to-slate-300 bg-clip-text text-transparent">
            Upload appropriations PDFs
          </h1>
          <p className="mt-2 text-sm text-slate-400">
            Upload one bill to parse it, or upload two fiscal years to see a
            diff.
          </p>
        </header>

        <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-6 shadow-md shadow-black/40 space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-slate-200">
              Primary file
            </label>
            <p className="text-xs text-slate-400">
              This is the version you want to explore or compare against another
              year.
            </p>
            <input
              type="file"
              className="block w-full cursor-pointer rounded-lg border border-slate-600 bg-slate-900/70 px-3 py-2 text-sm text-slate-100 file:mr-3 file:rounded-md file:border-0 file:bg-blue-400 file:px-3 file:py-1 file:text-xs file:font-medium file:text-slate-900 hover:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
              accept="application/pdf"
              onChange={(e) => setA(e.target.files?.[0] ?? null)}
            />
          </div>

          {a && (
            <div className="mt-4 space-y-3 border-t border-slate-800 pt-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-slate-200">
                  Optional previous year
                </label>
                <p className="text-xs text-slate-400">
                  Add last year&apos;s version of this bill to generate a side
                  by side diff. You can leave this blank to just parse the
                  primary file.
                </p>
                <input
                  type="file"
                  className="block w-full cursor-pointer rounded-lg border border-slate-600 bg-slate-900/70 px-3 py-2 text-sm text-slate-100 file:mr-3 file:rounded-md file:border-0 file:bg-blue-400 file:px-3 file:py-1 file:text-xs file:font-medium file:text-slate-900 hover:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  accept="application/pdf"
                  onChange={(e) => setB(e.target.files?.[0] ?? null)}
                />
              </div>

              <button
                onClick={go}
                className="mt-2 inline-flex items-center rounded-lg bg-blue-500 px-4 py-2 text-sm font-medium text-slate-900 shadow hover:bg-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer"
              >
                Upload and process
              </button>
            </div>
          )}

          {status && (
            <div className="text-xs text-blue-300 bg-blue-500/10 border border-blue-500/40 rounded-md px-3 py-2">
              {status}
            </div>
          )}
          {error && (
            <div className="text-xs text-red-300 bg-red-500/10 border border-red-500/40 rounded-md px-3 py-2">
              {error}
            </div>
          )}
        </div>

        {result && (
          <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4 shadow-md shadow-black/30">
            <h2 className="mb-3 text-sm font-semibold text-slate-200">
              Debug output
            </h2>
            <pre className="bg-slate-950/80 p-3 text-xs text-slate-100 rounded-lg overflow-auto max-h-80">
              {JSON.stringify(result, null, 2)}
            </pre>
          </div>
        )}

        {result?.currPreview && result.currPreview.length > 0 && (
          <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4 shadow-md shadow-black/30">
            <h2 className="mb-3 text-sm font-semibold text-slate-200">
              Current preview
            </h2>
            <table className="w-full text-xs border border-slate-800 rounded-lg overflow-hidden">
              <thead className="bg-slate-800/80 text-slate-100">
                <tr>
                  <th className="p-2 text-left">Program</th>
                  <th className="p-2 text-center">FY</th>
                  <th className="p-2 text-right">Amount</th>
                  <th className="p-2 text-center">Page</th>
                </tr>
              </thead>
              <tbody>
                {result.currPreview.map((r: any, i: number) => (
                  <tr
                    key={i}
                    className="border-t border-slate-800 hover:bg-slate-800/60"
                  >
                    <td className="p-2 align-top text-slate-100">
                      {r.program_name}
                    </td>
                    <td className="p-2 text-center text-slate-300">
                      {r.fy ?? ""}
                    </td>
                    <td className="p-2 text-right text-slate-300">
                      {r.amount}
                    </td>
                    <td className="p-2 text-center text-slate-300">{r.page}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </main>
  );
}
