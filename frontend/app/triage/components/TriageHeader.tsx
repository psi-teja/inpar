import Link from "next/link";
import React from "react";
import { HiOutlineArrowLeft } from "react-icons/hi";

function TriageHeader({ doc_id }) {
  return (
    <div className="flex justify-between items-center bg-gradient-to-r from-blue-300 to-blue-700 sm:p-1 md:p-2 lg:p-3 xl:p-4">
      <Link href="/" className="text-black hover:underline flex items-center">
        <HiOutlineArrowLeft className="m-2 h-3 w-3" />
      </Link>

      <h1 className="sm:text-xs md:text-md lg:text-lg xl:text-xl font-bold text-black-900 tracking-wide">
        Triage
      </h1>

      <div className="flex items-center">
        <p className="text-gray-800 text-xs mr-2">Document ID:</p>
        <p className="text-blue-100 text-xs font-semibold">{doc_id}</p>
      </div>
    </div>
  );
}

export default TriageHeader;
