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
      </div>
    </main>
  );
}
