"use client";

import Link from "next/link";
import React, { useEffect, useState } from "react";
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
  // Try "FY 2025" / "FY2025"
  const m4 = name.match(/\bFY\s*?(20\d{2})\b/i);
  if (m4) return parseInt(m4[1], 10);

  // Try standalone 4-digit 20xx
  const m4b = name.match(/\b(20\d{2})\b/);
  if (m4b) return parseInt(m4b[1], 10);

  // Try "FY25" → 2025 (naive but fine for our seeded docs)
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

export default function Page() {
  const [status, setStatus] = useState<string>("checking...");
  const [docs, setDocs] = useState<LatestTableWithFy[]>([]);
  const [docsError, setDocsError] = useState<string | null>(null);
  const [docsLoading, setDocsLoading] = useState<boolean>(true);
  const [firstCompare, setFirstCompare] = useState<LatestTableWithFy | null>(
    null
  );

  const router = useRouter();

  const links = [{ href: "/upload", label: "Upload" }];

  useEffect(() => {
    if (!API) {
      setStatus("API URL not configured");
      return;
    }
    fetch(`${API}/health`)
      .then((r) => r.json())
      .then((j) => setStatus(j.status ?? "unknown"))
      .catch(() => setStatus("down"));
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
    // No FY? just bail for now; could show a toast later.
    if (!doc.fy) {
      console.warn("Document has no FY parsed; skipping compare");
      return;
    }

    // First selection
    if (!firstCompare) {
      setFirstCompare(doc);
      return;
    }

    // Second selection:
    // enforce same subcommittee
    if (firstCompare.table_label !== doc.table_label) {
      console.warn("Documents must be from same subcommittee to compare");
      setFirstCompare(null);
      return;
    }

    // and different years
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
                  <th className="px-4 py-2 text-left">FY</th>
                  <th className="px-4 py-2 text-left">Parsed At</th>
                  <th className="px-4 py-2 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {docs.map((f) => {
                  const isSelected =
                    firstCompare && firstCompare.table_id === f.table_id;

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
                        {f.fy ?? "—"}
                      </td>
                      <td className="px-3 py-2 text-xs text-slate-300">
                        {formatDateTime(f.file_created_at)}
                      </td>
                      <td className="px-3 py-2 text-right space-x-2">
                        <Link
                          href={`/tables/${f.table_id}`}
                          className="inline-block rounded-lg border border-slate-700 px-2 py-1 text-xs hover:bg-slate-200 hover:text-slate-900"
                        >
                          View
                        </Link>
                        <button
                          type="button"
                          onClick={() => handleCompareClick(f)}
                          className="inline-block rounded-lg border border-blue-600 px-2 py-1 text-xs hover:bg-blue-500 hover:text-white disabled:opacity-40"
                          disabled={!f.table_id}
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
