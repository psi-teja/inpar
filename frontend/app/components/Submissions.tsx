"use client";
import { useState, useEffect } from "react";
import Link from "next/link";
import Select from "react-select";
import BACKEND_URLS from "../BackendUrls";
import { FaSpinner, FaExclamationCircle } from "react-icons/fa";

interface SubTableRow {
  doc_id: string;
  status: string;
  local_file: string;
  inserted_time: string;
}

interface Option {
  value: string;
  label: string;
}

const statusClass = (status: string) => {
  switch (status) {
    case "processed":
      return "bg-blue-400 hover:bg-blue-300";
    case "inqueue":
      return "bg-yellow-500 hover:bg-yellow-400";
    case "processing":
      return "bg-blue-500";
    case "failed":
      return "bg-red-400 hover:bg-red-300";
    case "verified":
      return "bg-green-500 hover:bg-green-400";
    default:
      return "bg-gray-400";
  }
};

function Submissions() {
  const [data, setData] = useState<SubTableRow[]>([]);
  const [isLoading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [hoveredRow, setHoveredRow] = useState<string | null>(null);
  const [triageReady, setTriageReady] = useState<boolean>(false);

  // Filter states for each column
  const [docIdFilter, setDocIdFilter] = useState<Option[]>([]);
  const [statusFilter, setStatusFilter] = useState<Option[]>([]);
  const [localFileFilter, setLocalFileFilter] = useState<Option[]>([]);
  const [insertedTimeFilter, setInsertedTimeFilter] = useState<Option[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(BACKEND_URLS.getSubTableUrl);
        if (!response.ok) {
          throw new Error("Failed to fetch data");
        }
        const data: SubTableRow[] = await response.json();
        setData(data);
        setLoading(false);
      } catch (error: any) {
        setError(error.message);
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // Extract unique values for dropdowns
  const uniqueDocIds = Array.from(new Set(data.map((item) => item.doc_id)));
  const uniqueStatuses = Array.from(new Set(data.map((item) => item.status)));
  const uniqueLocalFiles = Array.from(new Set(data.map((item) => item.local_file)));
  const uniqueInsertedTimes = Array.from(new Set(data.map((item) => item.inserted_time)));

  const toOption = (value: string): Option => ({ value, label: value });

  // Filtering logic based on column values
  const filteredData = data.filter(
    (item) =>
      (docIdFilter.length === 0 || docIdFilter.some((filter) => filter.value === item.doc_id)) &&
      (statusFilter.length === 0 || statusFilter.some((filter) => filter.value === item.status)) &&
      (localFileFilter.length === 0 || localFileFilter.some((filter) => filter.value === item.local_file)) &&
      (insertedTimeFilter.length === 0 || insertedTimeFilter.some((filter) => filter.value === item.inserted_time))
  );

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <FaSpinner className="animate-spin text-4xl text-gray-600" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-screen">
        <p className="text-xl text-red-500 mb-4">Error: {error}</p>
        <button
          onClick={() => window.location.reload()}
          className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
        >
          Refresh
        </button>
      </div>
    );
  }

  if (filteredData.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-screen">
        <FaExclamationCircle className="text-5xl text-gray-400 mb-4" />
        <p className="text-xl text-gray-600">No data available</p>
      </div>
    );
  }

  if (triageReady) {
    return (
      <div className="flex items-center justify-center h-screen">
        <FaSpinner className="animate-spin text-4xl text-gray-600" />
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto mt-8">
      <div className="bg-gradient-to-br from-blue-500 to-cyan-400 text-gray-800 rounded-md p-6 shadow-lg">
        <div className="grid grid-cols-4 gap-6 text-lg">
          <div className="text-left">
            <h1 className="text-white font-bold mb-3">Document ID</h1>
            <Select
              isMulti
              value={docIdFilter}
              onChange={(selected) => setDocIdFilter(selected as Option[])}
              options={uniqueDocIds.map(toOption)}
              className="mt-2"
            />
          </div>
          <div className="text-left">
            <h1 className="text-white font-bold mb-3">Uploaded File</h1>
            <Select
              isMulti
              value={localFileFilter}
              onChange={(selected) => setLocalFileFilter(selected as Option[])}
              options={uniqueLocalFiles.map(toOption)}
              className="mt-2"
            />
          </div>
          <div className="text-center">
            <h1 className="text-white font-bold mb-3">Status</h1>
            <Select
              isMulti
              value={statusFilter}
              onChange={(selected) => setStatusFilter(selected as Option[])}
              options={uniqueStatuses.map(toOption)}
              className="mt-2"
            />
          </div>
          <div className="text-right">
            <h1 className="text-white font-bold mb-3">Inserted Time (IST)</h1>
            <Select
              isMulti
              value={insertedTimeFilter}
              onChange={(selected) => setInsertedTimeFilter(selected as Option[])}
              options={uniqueInsertedTimes.map(toOption)}
              className="mt-2"
            />
          </div>
        </div>
      </div>

      <div className="h-[75vh] overflow-y-auto mt-6">
        {filteredData.map((item) => (
          <div
            className={`my-4 p-6 bg-white border border-gray-300 rounded-md shadow-md hover:shadow-lg transition duration-300 ease-in-out ${hoveredRow === item.doc_id ? 'bg-blue-100' : ''}`}
            key={item.doc_id}
            onMouseEnter={() => setHoveredRow(item.doc_id)}
            onMouseLeave={() => setHoveredRow(null)}
          >
            <div className="grid grid-cols-4 gap-4 items-center">
              <div className="text-left">
                <p className="text-sm font-semibold">{item.doc_id}</p>
              </div>
              <div className="text-center">
                <p className="text-sm font-semibold">{item.local_file}</p>
              </div>
              <div className="text-center ">
                <Link
                  href={{
                    pathname: "/triage",
                    query: {
                      doc_id: item.doc_id,
                      json_type: item.status === "verified" ? "gt_json" : "ai_json",
                    },
                  }}
                  onClick={() => setTriageReady(true)}
                  className={`inline-block px-4 py-2 font-bold rounded-lg transition-colors duration-300 ease-in-out ${statusClass(item.status)}`}
                >
                  {item.status}
                </Link>

              </div>
              <div className="text-right">
                <p className="text-sm">{item.inserted_time}</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default Submissions;
