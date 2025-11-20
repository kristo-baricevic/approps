"use client";

import Link from "next/link";
import React, { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";

type LatestTable = {
  table_id: string;
  file_id: string;
  file_name: string;
  table_label: string;
  created_at: string;
  file_created_at: string;
};

type LatestTableWithFy = LatestTable & {
  fy: number | null;
};

const API = process.env.NEXT_PUBLIC_API_URL as string | undefined;

function extractFiscalYear(name: string): number | null {
  const m4 = name.match(/\bFY\s*?(20\d{2})\b/i);
  if (m4) return parseInt(m4[1], 10);

  const m4b = name.match(/\b(20\d{2})\b/);
  if (m4b) return parseInt(m4b[1], 10);

  const m2 = name.match(/\bFY\s*?(\d{2})\b/i);
  if (m2) {
    const yy = parseInt(m2[1], 10);
    return yy >= 90 ? 1900 + yy : 2000 + yy;
  }

  return null;
}

function formatDateTime(s: string) {
  return new Date(s).toLocaleString();
}

function sortableTitle(name: string): string {
  return name
    .replace(/^FY\s*\d{2,4}\s*/i, "")
    .trim()
    .toLowerCase();
}

type SortKey = "title" | "subcommittee" | "fy" | null;

export default function Page() {
  const [status, setStatus] = useState<string>("checking...");
  const [docs, setDocs] = useState<LatestTableWithFy[]>([]);
  const [docsError, setDocsError] = useState<string | null>(null);
  const [docsLoading, setDocsLoading] = useState<boolean>(true);
  const [firstCompare, setFirstCompare] = useState<LatestTableWithFy | null>(
    null
  );
  const [sortKey, setSortKey] = useState<SortKey>(null);
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);
  const [apiHealthy, setApiHealthy] = useState<boolean>(false);

  const router = useRouter();

  const links = [
    { href: "/upload", label: "Upload" },
    { href: "/about", label: "About" },
  ];

  useEffect(() => {
    if (!API) {
      setStatus("API URL not configured");
      return;
    }
    fetch(`${API}/health`)
      .then((r) => r.json())
      .then((j) => {
        setStatus(j.status ?? "unknown");
        setApiHealthy(true);
      })
      .catch(() => {
        setStatus("down");
        setApiHealthy(false);
      });
  }, []);

  useEffect(() => {
    if (!API) {
      setDocsError("API URL not configured");
      setDocsLoading(false);
      return;
    }

    setDocsLoading(true);
    fetch(`${API}/tables/latest`)
      .then((r) => {
        if (!r.ok) throw new Error("failed");
        return r.json();
      })
      .then((data: LatestTable[]) => {
        const withFy: LatestTableWithFy[] = data.map((d) => ({
          ...d,
          fy: extractFiscalYear(d.file_name),
        }));
        setDocs(withFy);
        setDocsLoading(false);
      })
      .catch(() => {
        setDocsError("failed to load documents");
        setDocsLoading(false);
      });
  }, []);

  function handleSort(nextKey: SortKey) {
    if (!nextKey) return;
    if (sortKey === nextKey) {
      setSortDir((prev) => (prev === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(nextKey);
      setSortDir("asc");
    }
  }

  async function handleDelete(tableId: string) {
    if (!API) return;

    const res = await fetch(`${API}/delete/${tableId}`, {
      method: "DELETE",
    });

    if (res.ok) {
      setDocs(docs.filter((d) => d.table_id !== tableId));
    } else {
      console.error("delete failed", await res.text());
    }
  }

  const sortedDocs = useMemo(() => {
    if (!sortKey) return docs;

    const sorted = [...docs].sort((a, b) => {
      let cmp = 0;

      if (sortKey === "title") {
        const aTitle = sortableTitle(a.file_name);
        const bTitle = sortableTitle(b.file_name);
        cmp = aTitle.localeCompare(bTitle);
      } else if (sortKey === "subcommittee") {
        const aLabel = (a.table_label || "").toLowerCase();
        const bLabel = (b.table_label || "").toLowerCase();
        cmp = aLabel.localeCompare(bLabel);
      } else if (sortKey === "fy") {
        const aFy = a.fy;
        const bFy = b.fy;
        if (aFy == null && bFy == null) cmp = 0;
        else if (aFy == null) cmp = 1;
        else if (bFy == null) cmp = -1;
        else cmp = aFy - bFy;
      }

      return sortDir === "asc" ? cmp : -cmp;
    });

    return sorted;
  }, [docs, sortKey, sortDir]);

  async function handleDiff(prevTableId: string, currTableId: string) {
    if (!API) return;
    const res = await fetch(`${API}/diff`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prev_table_id: prevTableId,
        curr_table_id: currTableId,
      }),
    });

    if (!res.ok) {
      const msg = await res.text();
      console.error("diff failed", msg);
      return;
    }

    const { diff_id } = await res.json();
    router.push(`/diff/${diff_id}`);
  }

  function handleCompareClick(doc: LatestTableWithFy) {
    if (!doc.fy) {
      console.warn("Document has no FY parsed; skipping compare");
      return;
    }

    if (!firstCompare) {
      setFirstCompare(doc);
      return;
    }

    if (firstCompare.table_label !== doc.table_label) {
      console.warn("Documents must be from same subcommittee to compare");
      setFirstCompare(null);
      return;
    }

    if (!firstCompare.fy || firstCompare.fy === doc.fy) {
      console.warn("Documents must be different fiscal years to compare");
      setFirstCompare(null);
      return;
    }

    const prev =
      firstCompare.fy < doc.fy ? firstCompare.table_id : doc.table_id;
    const curr =
      firstCompare.fy < doc.fy ? doc.table_id : firstCompare.table_id;

    handleDiff(prev, curr);
    setFirstCompare(null);
  }

  const sortIndicator = (key: SortKey) =>
    sortKey === key ? (sortDir === "asc" ? " ↑" : " ↓") : "";

  return (
    <main className="min-h-screen bg-slate-950 px-4 py-8">
      <div className="mx-auto max-w-5xl">
        <header className="mb-8 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 className="text-3xl font-semibold tracking-tight bg-linear-to-r from-blue-400 to-slate-300 bg-clip-text text-transparent">
              Appropriations Parser
            </h1>
            <p className="mt-2 text-sm text-slate-400">
              Upload, parse, and compare federal appropriations tables from PDF
              bills.
            </p>
          </div>

          <div className="flex items-center space-x-3 rounded-full border border-slate-700 bg-slate-900/70 px-3 py-2 shadow-sm shadow-black/40">
            <span className="text-xs uppercase tracking-wide text-slate-400">
              API health
            </span>
            <div className="relative group">
              <div
                className={`h-3 w-3 rounded-full transition-shadow ${
                  apiHealthy
                    ? "bg-green-500 shadow-[0_0_10px_rgba(34,197,94,0.9)]"
                    : "bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.9)]"
                }`}
              />
              <div className="absolute left-1/2 top-6 hidden -translate-x-1/2 whitespace-nowrap rounded-md bg-slate-800 px-2 py-1 text-xs text-slate-200 shadow-lg group-hover:block">
                {status}
              </div>
            </div>
          </div>
        </header>

        <div className="rounded-2xl border border-slate-700 bg-slate-900/70 p-2 mb-6 backdrop-blur shadow-md shadow-black/30">
          <nav>
            <ul className="flex justify-center sm:space-x-4">
              {links.map((link) => (
                <li className="sm:px-4 p-2 text-xs sm:text-sm" key={link.href}>
                  <Link
                    href={link.href}
                    className="inline-flex items-center rounded-lg px-3 py-1 text-slate-200 transition-colors hover:bg-sky-400 hover:text-slate-900"
                  >
                    <span className="font-medium">{link.label}</span>
                  </Link>
                </li>
              ))}
            </ul>
          </nav>
        </div>

        <section className="mt-4">
          <h2 className="text-xl font-semibold mb-3 text-slate-100">
            Latest Budget Documents
          </h2>

          {docsLoading && <p className="text-sm text-slate-400">Loading…</p>}
          {docsError && (
            <p className="text-sm text-red-400">Error: {docsError}</p>
          )}

          {!docsLoading && !docsError && sortedDocs.length === 0 && (
            <p className="text-sm text-slate-400">
              No parsed documents yet. Run the seeding script or upload a PDF.
            </p>
          )}

          {!docsLoading && !docsError && sortedDocs.length > 0 && (
            <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-900/60 shadow-md shadow-black/40">
              <table className="min-w-full text-sm">
                <thead className="bg-slate-900/80 text-slate-200">
                  <tr className="text-xs uppercase tracking-wide">
                    <th
                      className="px-4 py-3 text-left cursor-pointer select-none hover:text-sky-300"
                      onClick={() => handleSort("title")}
                    >
                      Title{sortIndicator("title")}
                    </th>
                    <th
                      className="px-4 py-3 text-left cursor-pointer select-none hover:text-sky-300"
                      onClick={() => handleSort("subcommittee")}
                    >
                      Subcommittee{sortIndicator("subcommittee")}
                    </th>
                    <th
                      className="px-4 py-3 text-left cursor-pointer select-none hover:text-sky-300"
                      onClick={() => handleSort("fy")}
                    >
                      Fiscal Year{sortIndicator("fy")}
                    </th>
                    <th className="px-4 py-3 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedDocs.map((f) => {
                    const isSelected =
                      firstCompare && firstCompare.table_id === f.table_id;

                    return (
                      <tr
                        key={f.table_id}
                        className={`border-t border-slate-800/70 transition-colors ${
                          isSelected
                            ? "bg-sky-950/40 border-sky-700"
                            : "hover:bg-slate-800/50"
                        }`}
                      >
                        <td className="px-3 py-2 align-middle">
                          <div className="font-medium text-slate-100">
                            {f.file_name}
                          </div>
                          <div className="mt-0.5 text-[11px] text-slate-500">
                            Parsed {formatDateTime(f.created_at)}
                          </div>
                        </td>
                        <td className="px-3 py-2 align-middle text-xs text-slate-300">
                          <span className="inline-flex items-center rounded-full border border-slate-700 bg-slate-900/70 px-2 py-0.5 text-[11px]">
                            {f.table_label || "-"}
                          </span>
                        </td>
                        <td className="px-3 py-2 align-middle text-xs text-slate-300">
                          <span className="inline-flex items-center rounded-full border border-slate-700 bg-slate-900/70 px-2 py-0.5 text-[11px]">
                            {f.fy ?? "2025"}
                          </span>
                        </td>
                        <td className="px-3 py-2 align-middle text-right space-x-2">
                          <Link
                            href={`/tables/${f.table_id}`}
                            className="inline-flex items-center rounded-lg border border-slate-600 bg-slate-900/70 px-2 py-1 text-xs text-slate-100 transition-colors hover:bg-slate-100 hover:text-slate-900"
                          >
                            View
                          </Link>
                          <button
                            type="button"
                            onClick={() => handleCompareClick(f)}
                            className="inline-flex items-center rounded-lg border border-sky-500 bg-sky-500/10 px-2 py-1 text-xs text-sky-300 transition-colors hover:bg-sky-500 hover:text-white disabled:opacity-40 cursor-pointer"
                            disabled={!f.table_id}
                          >
                            {isSelected ? "Selected…" : "Compare"}
                          </button>
                          <button
                            type="button"
                            onClick={() => {
                              setPendingDeleteId(f.table_id);
                              setConfirmOpen(true);
                            }}
                            className="inline-flex items-center rounded-lg border border-red-600 bg-red-500/10 px-2 py-1 text-xs text-red-300 transition-colors hover:bg-red-600 hover:text-white cursor-pointer"
                          >
                            Delete
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>

              {confirmOpen && (
                <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50">
                  <div className="bg-slate-900 border border-slate-700 p-6 rounded-xl w-72 text-sm shadow-2xl shadow-black/50">
                    <p className="mb-4 text-slate-100">
                      Delete this table and its rows?
                    </p>

                    <div className="flex justify-end space-x-3">
                      <button
                        className="px-3 py-1 rounded-lg border border-slate-600 bg-slate-900/70 text-slate-200 text-xs transition-colors hover:bg-slate-700 cursor-pointer"
                        onClick={() => {
                          setConfirmOpen(false);
                          setPendingDeleteId(null);
                        }}
                      >
                        Cancel
                      </button>

                      <button
                        className="px-3 py-1 rounded-lg border border-red-600 bg-red-600/80 text-white text-xs transition-colors hover:bg-red-700 cursor-pointer"
                        onClick={async () => {
                          if (pendingDeleteId) {
                            await handleDelete(pendingDeleteId);
                          }
                          setConfirmOpen(false);
                          setPendingDeleteId(null);
                        }}
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
