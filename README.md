# Overview

This project parses federal appropriations bills and their explanatory statements into a structured database, turning unstructured PDF budget language into clean, queryable program entries with amounts, fiscal years, and metadata. The goal is to surface funding lines in a consistent, searchable format and make it easy to compare changes across fiscal years, generate diffs, and attach AI-generated briefs to each budget line.

## Architecture

The system is built around three components:

**FastAPI backend:** Handles file registration, PDF parsing, AI refinement, table storage, diff generation, CSV/PDF exports, and row-level rendering.

**PostgreSQL database:** Stores files, parsed tables, individual rows, audits, and diffs. All relationships cascade and each row carries semantic fields like program names, AI labels, amounts, bounding boxes, and MD5 deduplication checks.

**MinIO object storage:** Holds PDFs and rendered highlight images. The API generates presigned PUT/GET URLs so the frontend uploads and retrieves files directly without routing the binary data through FastAPI.

The backend manages all ingestion logic and interacts with storage via presigned URLs. The frontend (Next.js) queries the API, displays parsed tables, runs comparisons, and manages uploads.

## Deployment Environment

The full environment is orchestrated with Docker Compose. It runs:

### Postgres 16 for relational storage

### MinIO for S3-compatible object storage

### FastAPI via Uvicorn for the API

### Nginx as the public-facing reverse proxy that forwards traffic to the API

Every service is health-checked. The API connects to Postgres at `db:5432`, connects to MinIO at `http://minio:9000`, and exposes its application port only internally. Nginx handles all external traffic, serving as the single public entry point for requests.

## Parsing Pipeline

### Upload

The frontend requests a presigned upload URL and uploads the PDF directly to MinIO.

The backend records the file metadata in the `files` table.

### Parse

The API downloads the PDF from MinIO to a temporary file.

It runs the combined parser, which extracts structured rows, prose-based amounts, bounding boxes, page numbers, and inferred fiscal years.

Rows are cleaned, deduplicated by checksum, and inserted into the `rows` table under a `tables` entry.

### AI refinement

If enabled, OpenAI is used to generate a cleaner program name and a short human-readable description for each line, based on the surrounding PDF context.

### Audit

After parsing, an audit record is created summarizing counts, totals, negative values, and anomalies.

## Usage

The frontend can preview tables, compare fiscal year versions, export CSVs, generate PDF briefs, or request row-level highlighted renders of the source PDF.
