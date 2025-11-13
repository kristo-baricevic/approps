"use client";

import Link from "next/link";
import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

type LatestTable = {
  table_id: string;
  file_id: string;
  file_name: string;
  table_label: string;
  file_created_at: string;
  table_created_at: string;
};

function formatDateTime(iso: string | null | undefined) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString();
}

export default function Page() {
  const [status, setStatus] = useState<string>("checking...");
  const [docs, setDocs] = useState<LatestTable[]>([]);
  const [docsError, setDocsError] = useState<string | null>(null);
  const [docsLoading, setDocsLoading] = useState<boolean>(true);

  const [firstCompareId, setFirstCompareId] = useState<string | null>(null);
  const [firstCompareLabel, setFirstCompareLabel] = useState<string | null>(
    null
  );
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareError, setCompareError] = useState<string | null>(null);

  const router = useRouter();
  const links = [{ href: "/upload", label: "Upload" }];

  useEffect(() => {
    const base = process.env.NEXT_PUBLIC_API_URL;
    if (!base) {
      setStatus("API URL not configured");
      return;
    }
    fetch(`${base}/health`)
      .then((r) => r.json())
      .then((j) => setStatus(j.status ?? "unknown"))
      .catch(() => setStatus("down"));
  }, []);

  useEffect(() => {
    const base = process.env.NEXT_PUBLIC_API_URL;
    if (!base) {
      setDocsError("API URL not configured");
      setDocsLoading(false);
      return;
    }

    setDocsLoading(true);
    fetch(`${base}/tables/latest`)
      .then((r) => {
        if (!r.ok) throw new Error("failed");
        return r.json();
      })
      .then((data: LatestTable[]) => {
        setDocs(data);
        setDocsLoading(false);
      })
      .catch(() => {
        setDocsError("failed to load documents");
        setDocsLoading(false);
      });
  }, []);

  async function createDiff(prevTableId: string, currTableId: string) {
    const base = process.env.NEXT_PUBLIC_API_URL;
    if (!base) {
      setCompareError("API URL not configured");
      return;
    }

    try {
      setCompareLoading(true);
      setCompareError(null);

      const res = await fetch(`${base}/diff`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prev_table_id: prevTableId,
          curr_table_id: currTableId,
        }),
      });

      if (!res.ok) {
        const msg = await res.text();
        throw new Error(msg || "diff failed");
      }

      const { diff_id } = await res.json();
      router.push(`/diff/${diff_id}`);
    } catch (e) {
      console.error("diff failed", e);
      setCompareError("Failed to create diff");
    } finally {
      setCompareLoading(false);
    }
  }

  async function handleCompareClick(doc: LatestTable) {
    if (!doc.table_id) return;

    // first selection
    if (!firstCompareId) {
      setFirstCompareId(doc.table_id);
      setFirstCompareLabel(doc.file_name);
      setCompareError(null);
      return;
    }

    // clicking the same again cancels
    if (firstCompareId === doc.table_id) {
      setFirstCompareId(null);
      setFirstCompareLabel(null);
      setCompareError(null);
      return;
    }

    // second selection → create diff
    await createDiff(firstCompareId, doc.table_id);
    setFirstCompareId(null);
    setFirstCompareLabel(null);
  }

  return (
    <main className="p-6">
      <h1 className="text-2xl font-semibold mb-2">Appropriations Parser</h1>
      <p className="mb-6">
        API health: <span className="font-mono">{status}</span>
      </p>

      <div className="rounded-2xl border border-slate-700 p-6 mb-6">
        <nav className="border-b border-black/10">
          <ul className="flex sm:space-x-4 justify-center">
            {links.map((link) => (
              <li
                className="hover:bg-blue-300 hover:text-slate-700 hover:rounded-lg sm:px-4 p-2 text-xs sm:text-base"
                key={link.href}
              >
                <Link href={link.href}>
                  <span className="text-mono">{link.label}</span>
                </Link>
              </li>
            ))}
          </ul>
        </nav>
      </div>

      <section className="mt-4">
        <h2 className="text-xl font-semibold mb-3">Latest Budget Documents</h2>

        {firstCompareId && (
          <p className="text-xs text-slate-300 mb-2">
            Selected first document:{" "}
            <span className="font-mono">{firstCompareLabel}</span>. Click
            “Compare” on another document to create a diff. Click “Compare” on
            the same row again to clear.
          </p>
        )}

        {compareError && (
          <p className="text-xs text-red-400 mb-2">{compareError}</p>
        )}

        {docsLoading && <p className="text-sm text-slate-400">Loading…</p>}
        {docsError && (
          <p className="text-sm text-red-400">Error: {docsError}</p>
        )}

        {!docsLoading && !docsError && docs.length === 0 && (
          <p className="text-sm text-slate-400">
            No parsed documents yet. Run the seeding script or upload a PDF.
          </p>
        )}

        {!docsLoading && !docsError && docs.length > 0 && (
          <div className="overflow-x-auto rounded-xl border border-slate-700">
            <table className="min-w-full text-sm">
              <thead className="bg-slate-900/60">
                <tr>
                  <th className="px-4 py-2 text-left">Title</th>
                  <th className="px-4 py-2 text-left">Subcommittee</th>
                  <th className="px-4 py-2 text-left">Parsed At</th>
                  <th className="px-4 py-2 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {docs.map((f) => {
                  const isSelected =
                    firstCompareId && firstCompareId === f.table_id;
                  return (
                    <tr
                      key={f.table_id}
                      className={`border-t border-slate-800 ${
                        isSelected ? "bg-slate-800/60" : "hover:bg-slate-800/40"
                      }`}
                    >
                      <td className="px-3 py-2">{f.file_name}</td>
                      <td className="px-3 py-2 text-xs text-slate-300">
                        {f.table_label || "—"}
                      </td>
                      <td className="px-3 py-2 text-xs text-slate-300">
                        {formatDateTime(f.table_created_at)}
                      </td>
                      <td className="px-3 py-2 text-right space-x-2">
                        {f.table_id && (
                          <Link
                            href={`/tables/${f.table_id}`}
                            className="inline-block rounded-lg border border-slate-700 px-2 py-1 text-xs hover:bg-slate-200 hover:text-slate-900"
                          >
                            View
                          </Link>
                        )}
                        <button
                          type="button"
                          onClick={() => handleCompareClick(f)}
                          className="inline-block rounded-lg border border-blue-600 px-2 py-1 text-xs hover:bg-blue-500 hover:text-white disabled:opacity-40"
                          disabled={!f.table_id || compareLoading}
                        >
                          {isSelected ? "Selected…" : "Compare"}
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </main>
  );
}
