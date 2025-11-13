import asyncio
import hashlib
import os

import httpx

print(">>> seed_latest_docs.py imported") 

# BASE_URL = os.getenv("APPROPS_API_BASE_URL", "http://localhost:8000")
BASE_URL = os.getenv("APPROPS_API_BASE_URL", "http://64.225.3.60:8000")


DOCS = {
    "FY25 House Labor–HHS Draft": {
        # House Labor-HHS subcommittee mark for FY 2025
        # docs.house.gov meeting AP07, June 27, 2024
        "url": "https://docs.house.gov/meetings/AP/AP07/20240627/117475/BILLS-118-SC-AP-FY2025-LaborHHS-FY25LHHSSubcommitteeMark.pdf",
        "label": "Labor-HHS-Education",
    },

    "FY25 Senate Labor–HHS Draft": {
        # Senate Appropriations FY 2025 LHHS bill text
        "url": "https://www.appropriations.senate.gov/imo/media/doc/fy25_lhhs_bill_text6.pdf",
        "label": "Labor-HHS-Education",
    },

    "FY24 Enacted Labor–HHS": {
        # Division D — Labor, Health and Human Services, Education,
        # from the 2024 omnibus (“Further Consolidated Appropriations Act, 2024”)
        "url": "https://docs.house.gov/billsthisweek/20240318/Division%20D%20LHHS.pdf",
        "label": "Labor-HHS-Education",
    },

    "FY24 Enacted Defense": {
        # Division A — Department of Defense Appropriations Act, 2024
        "url": "https://docs.house.gov/billsthisweek/20240318/Division%20A%20Defense.PDF",
        "label": "Defense",
    },

    "FY24 CJS Bill Text (Senate)": {
        # Commerce, Justice, Science, and Related Agencies Appropriations Act, 2024
        "url": "https://www.appropriations.senate.gov/imo/media/doc/fy24_cjs_bill_text.pdf",
        "label": "Commerce-Justice-Science",
    },
}


DOC_LABELS = {
    "FY25 House Labor–HHS Draft": "Labor-HHS",
    "FY25 Senate Labor–HHS Draft": "Labor-HHS",
    "FY24 Enacted Labor–HHS": "Labor-HHS",
    "FY24 Enacted Defense": "Defense",
    "FY23 House Commerce–Justice–Science": "CJS",
}

DEMO_DIFFS = [
    # ("FY24 Enacted Labor–HHS", "FY25 House Labor–HHS Draft"),
]

print(">>> seed_latest_docs.py running")

async def seed_one_doc(client: httpx.AsyncClient, title: str, meta: dict) -> dict:
    url = meta["url"]
    label = meta["label"]
    
    print(f"\n=== Seeding {title} from {url} ===")

    # 1) download PDF
    r = await client.get(url)
    r.raise_for_status()
    data = r.content
    print(f"  downloaded {len(data)} bytes")

    sha256 = hashlib.sha256(data).hexdigest()
    filename = url.rsplit("/", 1)[-1] or "document.pdf"

    # 2) presign
    presign_resp = await client.post(
        f"{BASE_URL}/upload/presign",
        json={
            "sha256": sha256,
            "filename": filename,
            "content_type": "application/pdf",
        },
    )
    presign_resp.raise_for_status()
    presign = presign_resp.json()
    put_url = presign["url"]
    key = presign["key"]
    print(f"  got presigned URL, key={key}")

    # 3) upload to S3
    put_resp = await client.put(put_url, content=data, headers={"Content-Type": "application/pdf"})
    put_resp.raise_for_status()
    print("  uploaded to S3 via presigned URL")

    # 4) register file
    register_resp = await client.post(
        f"{BASE_URL}/files/register",
        json={
            "sha256": sha256,
            "filename": title,
            "key": key,
        },
    )
    register_resp.raise_for_status()
    file_info = register_resp.json()
    file_id = file_info["file_id"]
    print(f"  registered file_id={file_id}")

    # 5) parse
    label = DOC_LABELS[title]
    parse_resp = await client.post(
        f"{BASE_URL}/parse",
        json={
            "file_id": file_id,
            "table_label": label,
        },
    )
    parse_resp.raise_for_status()
    parsed = parse_resp.json()
    table_id = parsed["table_id"]
    count = parsed["count"]
    print(f"  parsed table_id={table_id}, rows={count}")

    return {"file_id": file_id, "table_id": table_id}


async def main():
    print(f">>> BASE_URL = {BASE_URL}")
    async with httpx.AsyncClient(timeout=120) as client:
        results: dict[str, dict] = {}

        for title, meta in DOCS.items():
            try:
                res = await seed_one_doc(client, title, meta)
                results[title] = res
            except Exception as e:
                print(f"!!! error while seeding {title}: {e}")

        # optional diffs
        for prev_title, curr_title in DEMO_DIFFS:
            if prev_title not in results or curr_title not in results:
                print(f"!!! skipping diff {prev_title} -> {curr_title} (missing tables)")
                continue

            prev_table_id = results[prev_title]["table_id"]
            curr_table_id = results[curr_title]["table_id"]
            
            resp = await client.post(
                f"{BASE_URL}/diff",
                json={
                    "prev_table_id": prev_table_id,
                    "curr_table_id": curr_table_id,
                },
            )
            resp.raise_for_status()
            diff_id = resp.json()["diff_id"]
            print(f"Demo diff {prev_title} → {curr_title}: diff_id = {diff_id}")

    print(">>> seeding complete")


# async def main():
#     print(">>> seed_latest_docs.py main")
#     print(f">>> BASE_URL = {BASE_URL}")

#     async with httpx.AsyncClient(timeout=120) as client:
#         results: dict[str, dict] = {}

#         for title, meta in DOCS.items():
#             try:
#                 res = await seed_one_doc(client, title, meta)
#                 results[title] = res
#             except Exception as e:
#                 print(f"!!! error while seeding {title}: {e}")

#         # optional diffs
#         for prev_title, curr_title in DEMO_DIFFS:
#             if prev_title not in results or curr_title not in results:
#                 print(f"!!! skipping diff {prev_title} -> {curr_title} (missing tables)")
#                 continue

#             prev_table_id = results[prev_title]["table_id"]
#             curr_table_id = results[curr_title]["table_id"]
#             resp = await client.post(
#                 f"{BASE_URL}/diff",
#                 json={
#                     "prev_table_id": prev_table_id,
#                     "curr_table_id": curr_table_id,
#                 },
#             )
#             resp.raise_for_status()
#             diff_id = resp.json()["diff_id"]
#             print(f"Demo diff {prev_title} → {curr_title}: diff_id = {diff_id}")

    print(">>> seeding complete")


if __name__ == "__main__":
    print(">>> running main()")
    asyncio.run(main())
