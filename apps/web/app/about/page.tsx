"use client";

import Link from "next/link";
import React from "react";

export default function About() {
  return (
    <main className="p-6 text-slate-200">
      <h1 className="text-3xl font-semibold mb-6">About</h1>

      <div className="space-y-6">
        <section className="bg-slate-900/40 border border-slate-700 rounded-xl p-5">
          <h2 className="text-xl font-semibold mb-2">
            About the Appropriations Parser
          </h2>
          <p className="leading-relaxed">
            This project analyzes federal appropriations bills and their
            explanatory statements, converting complex budget language into
            structured, searchable data. It takes unstructured PDF documents and
            breaks them down into clean program entries with funding amounts,
            fiscal years, identifiers, and context, making it far easier to
            explore how government money is allocated and how those allocations
            change over time.
          </p>
        </section>

        <section className="bg-slate-900/40 border border-slate-700 rounded-xl p-5">
          <h2 className="text-xl font-semibold mb-2">How It Works</h2>
          <p className="leading-relaxed">
            The system is built around a FastAPI backend, a PostgreSQL database,
            and MinIO object storage. Uploaded bills are stored in MinIO using
            secure presigned URLs, while the backend manages parsing, AI
            labeling, diff generation, and data integrity checks. Parsed tables
            and their rows are stored in Postgres with full relationships,
            audits, and deduplication. A Next.js frontend queries the API to
            display parsed entries, compare fiscal years, render source
            highlights, and export CSV or PDF summaries.
          </p>
        </section>

        <section className="bg-slate-900/40 border border-slate-700 rounded-xl p-5">
          <h2 className="text-xl font-semibold mb-2">Deployment</h2>
          <p className="leading-relaxed">
            The entire environment runs inside Docker Compose. Postgres handles
            relational data, MinIO provides S3-compatible storage for PDFs and
            rendered images, FastAPI processes uploads and generates structured
            output, and Nginx serves as the single public entry point. This
            setup ensures predictable, isolated services and makes the system
            easy to deploy on any server.
          </p>
        </section>

        <section className="bg-slate-900/40 border border-slate-700 rounded-xl p-5">
          <h2 className="text-xl font-semibold mb-2">Parsing Pipeline</h2>
          <p className="leading-relaxed">
            When a PDF is uploaded, the frontend obtains a presigned URL and
            places the file directly into storage. The backend pulls the file,
            extracts program lines and amounts, merges table and prose text,
            collects bounding boxes, detects the fiscal year, and inserts all
            cleaned rows into the database. If AI is enabled, the system
            generates clearer program names and short human summaries. Each
            table is automatically audited, and users can generate comparisons,
            exports, or highlighted views of specific entries.
          </p>
        </section>
        <div className="flex flex-row justify-center items-center">
          <Link
            href="https://github.com/kristo-baricevic/approps"
            target="_blank"
            className="inline-flex items-center rounded-lg border border-slate-600 bg-slate-900/70 px-3 py-2 text-sm text-slate-200 hover:bg-slate-700 transition"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="currentColor"
              className="w-5 h-5 mr-2"
            >
              <path
                fillRule="evenodd"
                d="M12 2C6.477 2 2 6.486 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483
      0-.237-.009-.868-.014-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608
      1.004.07 1.532 1.033 1.532 1.033.892 1.53 2.341 1.087 2.91.832.092-.647.35-1.087.636-1.337-2.22-.255-4.555-1.113-4.555-4.951
      0-1.093.39-1.988 1.029-2.688-.103-.255-.446-1.285.098-2.677 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0 1 12 6.844c.85.004
      1.705.115 2.504.337 1.909-1.296 2.748-1.026 2.748-1.026.546 1.392.203 2.422.1 2.677.64.7 1.028 1.595 1.028 2.688
      0 3.848-2.339 4.692-4.566 4.942.359.31.678.92.678 1.855 0 1.338-.012 2.419-.012 2.749
      0 .268.18.58.687.482A10.019 10.019 0 0 0 22 12.017C22 6.486 17.522 2 12 2z"
                clipRule="evenodd"
              />
            </svg>
            GitHub Repository
          </Link>
          <Link
            href="https://kristo-portfolio.vercel.app/"
            target="_blank"
            className="inline-flex items-center rounded-lg border border-slate-600 bg-slate-900/70 px-3 py-2 text-sm text-slate-200 hover:bg-slate-700 transition ml-2"
          >
            Portfolio
          </Link>
        </div>
      </div>
    </main>
  );
}
