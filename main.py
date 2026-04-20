import argparse
from pathlib import Path

from data.raw_input import ingest_report, ingest_scan
from data_pipeline.qdrant_vdata import ingest_data, delete_patient_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Med-Sight | Project entrypoint for ingestion and indexing tasks"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    report_parser = subparsers.add_parser("report", help="Ingest one or more clinical PDF reports")
    report_parser.add_argument("pdfs", nargs="+", help="PDF file path(s)")
    report_parser.add_argument("--patient-id", default="anonymous", help="Patient identifier")
    report_parser.add_argument("--db", default="data/database/medsight_personal.db", help="SQLite DB path")
    report_parser.add_argument("--no-qdrant", action="store_true", help="Skip Qdrant upload")

    scan_parser = subparsers.add_parser("scan", help="Ingest one or more medical scan images")
    scan_parser.add_argument("images", nargs="+", help="Image file path(s)")
    scan_parser.add_argument("--patient-id", default="anonymous", help="Patient identifier")
    scan_parser.add_argument("--caption", default="", help="Shared caption for all images")
    scan_parser.add_argument("--device", default="cpu", help="Torch device (cpu / cuda)")
    scan_parser.add_argument("--db", default="data/database/medsight_personal.db", help="SQLite DB path")
    scan_parser.add_argument("--no-qdrant", action="store_true", help="Skip Qdrant upload")

    subparsers.add_parser("index", help="Build or refresh the public Qdrant medical corpus")

    delete_parser = subparsers.add_parser("delete-patient", help="Delete personal scan data for a patient")
    delete_parser.add_argument("patient_id", help="Patient identifier")

    args = parser.parse_args()

    if args.command == "report":
        for pdf in args.pdfs:
            result = ingest_report(
                pdf,
                patient_id=args.patient_id,
                db_path=args.db,
                skip_qdrant=args.no_qdrant,
            )
            print(result)

    elif args.command == "scan":
        for image in args.images:
            result = ingest_scan(
                image,
                patient_id=args.patient_id,
                caption=args.caption,
                device=args.device,
                db_path=args.db,
                skip_qdrant=args.no_qdrant,
            )
            print(result)

    elif args.command == "index":
        ingest_data()
        print("Completed base corpus ingestion.")

    elif args.command == "delete-patient":
        delete_patient_data(args.patient_id)


if __name__ == "__main__":
    main()
