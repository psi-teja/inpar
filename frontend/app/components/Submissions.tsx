"use client";
import { useState, useEffect } from "react";
import Link from "next/link";
import BACKEND_URLS from "../BackendUrls";

interface SubTableRow {
  doc_id: string;
  status: string;
  local_file: string;
  inserted_time: string;
}

function Submissions() {
  const [data, setData] = useState<SubTableRow[]>([]);
  const [isLoading, setLoading] = useState<boolean>(true);
  const [hoveredRow, setHoveredRow] = useState<string | null>(null);

  const getTextOnHover = (item: SubTableRow) => {
    if (hoveredRow === item.doc_id) {
      return item.status === "processed" ? "Verify" : "View Docmuent";
    } else {
      return item.status;
    }
  };

  useEffect(() => {
    fetch(
      BACKEND_URLS.getSubTableUrl
    )
      .then((res) => res.json())
      .then((data: SubTableRow[]) => {
        setData(data);
        setLoading(false);
      });
  }, []);

  if (isLoading)
    return (
      <p className="m-5 text-xl text-center">Loading data from sub_table...</p>
    );
  if (!data) return <p>No profile data</p>;

  return (
    <div className="max-w-2xl mx-auto mt-4">
      <div className="bg-slate-500 text-white rounded-md p-4">
        <div className="grid grid-cols-4 gap-3 text-xl">
          <div className="text-left">
            <h1>Document ID</h1>
          </div>
          <div className="text-left">
            <h1>Uploaded File</h1>
          </div>
          <div className="text-center">
            <h1>Status</h1>
          </div>
          <div className="text-right">
            <h1>Inserted Time (IST)</h1>
          </div>
        </div>
      </div>
      <div className="h-[75vh] overflow-y-auto">
        {data.map((item) => (
          <div
            className={`my-4 p-4 bg-white border border-gray-300 rounded-md shadow-md hover:shadow-lg transition duration-300 ease-in-out`}
            key={item.doc_id}
          >
            <div className="grid grid-cols-4 gap-4 items-center">
              <div className="text-left">
                <p className="text-sm font-semibold">{item.doc_id}</p>
              </div>
              <div className="text-center">
                <p className="text-sm font-semibold">{item.local_file}</p>
              </div>
              <div className="text-center">
                <Link
                  href={{
                    pathname: "/triage",
                    query: {
                      doc_id: item.doc_id,
                      json_type: (item.status == 'GT exists'? "gt_json": "ai_json"),
                    },
                  }}
                  onMouseEnter={() => setHoveredRow(item.doc_id)}
                  onMouseLeave={() => setHoveredRow(null)}
                  className={`text rounded-lg p-2 cursor-pointer ${
                    item.status === "processed"
                      ? "bg-green-200"
                      : item.status === "inqueue"
                      ? "bg-yellow-200"
                      : item.status === 'processing'
                      ? "bg-orange-200"
                      : item.status === "failed"
                      ? "bg-red-200"
                      : ""
                  }`}
                >
                  {getTextOnHover(item)}
                </Link>
              </div>
              <div className="text-right">
                <p className="text-sm">{item.inserted_time}</p>
              </div>
            </div>
          </div>
        ))}
        {!data.length && <a>No Data</a>}
      </div>
    </div>
  );
}

export default Submissions;
