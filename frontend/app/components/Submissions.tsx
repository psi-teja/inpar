"use client";
import { useState, useEffect } from "react";
import Link from "next/link";

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
      return item.status === "processed" ? "Verify" : "View";
    } else {
      return item.status;
    }
  };

  useEffect(() => {
    fetch(
      `http://127.0.0.1:8000/db_connect/sub_table`
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
      <div className="bg-yellow-200 text-black rounded-md p-4">
        <div className="grid grid-cols-3 gap-4 text-xl">
          <div>
            <h1 className="text-left">Document ID</h1>
          </div>
          <div>
            <h1 className="text-center">Status</h1>
          </div>
          <div>
            <h1 className="text-right">Inserted Time (IST)</h1>
          </div>
        </div>
      </div>
      <div className="h-[75vh] overflow-y-auto">
        {data &&
          data.map((item) => (
            <div
              className={`my-4 p-4 bg-white border border-gray-300 rounded-md shadow-md hover:shadow-lg transition duration-300 ease-in-out`}
              key={item.doc_id}
            >
              <div className="grid grid-cols-3 gap-4 items-center">
                <div>
                  <p className="text-sm font-semibold">{item.doc_id}</p>
                </div>
                <div className="text-center">
                  <Link
                    href={{
                      pathname: "/triage",
                      query: {
                        doc_id: item.doc_id,
                      },
                    }}
                    onMouseEnter={() => setHoveredRow(item.doc_id)}
                    onMouseLeave={() => setHoveredRow(null)}
                    className={`text rounded-lg p-2 ${
                      item.status === "processed"
                        ? "bg-green-400  cursor-pointer"
                        : "cursor-pointer hover:bg-blue-200"
                    }`}
                  >
                    {getTextOnHover(item)}
                  </Link>
                </div>
                <div>
                  <p className="text-right">{item.inserted_time}</p>
                </div>
              </div>
            </div>
          ))}
        {!data && <a>No Data</a>}
      </div>
    </div>
  );
}

export default Submissions;
